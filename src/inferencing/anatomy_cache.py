import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
import nibabel as nib
import torchio as tio
from tqdm import tqdm
from torch.amp import autocast
import pdb
from utils import downsample_upsample
from metrics import evaluate_image_quality
import xformers
import xformers.ops
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torch.nn.functional as F
from skimage.filters import sobel
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union
import time

def crop_mri_volume(data, threshold=0.01, padding=1, divisible_by=16):
    """
    Crop the 3D MRI volume to remove empty regions and ensure dimensions are divisible by divisible_by.
    """
    # Find non-empty voxels in each dimension
    x_profile = torch.mean(data, dim=(1, 2)) > threshold
    y_profile = torch.mean(data, dim=(0, 2)) > threshold
    z_profile = torch.mean(data, dim=(0, 1)) > threshold
    
    # Get the bounds of non-empty regions
    x_bounds = torch.where(x_profile)[0]
    y_bounds = torch.where(y_profile)[0]
    z_bounds = torch.where(z_profile)[0]
    
    # Get start and end indices with padding
    x_start = max(0, x_bounds[0] - padding)
    x_end = min(data.shape[0], x_bounds[-1] + padding + 1)
    y_start = max(0, y_bounds[0] - padding)
    y_end = min(data.shape[1], y_bounds[-1] + padding + 1)
    z_start = max(0, z_bounds[0] - padding)
    z_end = min(data.shape[2], z_bounds[-1] + padding + 1)
    
    # Make dimensions divisible by divisible_by (round up)
    x_size = x_end - x_start
    y_size = y_end - y_start
    z_size = z_end - z_start
    
    # Round up to nearest divisible value
    x_target = ((x_size + divisible_by - 1) // divisible_by) * divisible_by
    y_target = ((y_size + divisible_by - 1) // divisible_by) * divisible_by
    z_target = ((z_size + divisible_by - 1) // divisible_by) * divisible_by
    
    # Add padding to reach target sizes, prioritizing end padding if bounds allow
    x_diff = x_target - x_size
    y_diff = y_target - y_size
    z_diff = z_target - z_size
    
    # Try to extend right/bottom boundaries first
    x_end = min(data.shape[0], x_end + x_diff)
    y_end = min(data.shape[1], y_end + y_diff)
    z_end = min(data.shape[2], z_end + z_diff)
    
    # If still not reached target, extend left/top boundaries
    x_size = x_end - x_start
    if x_size < x_target and x_start > 0:
        x_start = max(0, x_start - (x_target - x_size))
    
    y_size = y_end - y_start
    if y_size < y_target and y_start > 0:
        y_start = max(0, y_start - (y_target - y_size))
    
    z_size = z_end - z_start
    if z_size < z_target and z_start > 0:
        z_start = max(0, z_start - (z_target - z_size))
    
    # Crop the data
    cropped_data = data[x_start:x_end, y_start:y_end, z_start:z_end]
    
    # Final adjustment if still not divisible (crop from the end)
    if cropped_data.shape[0] % divisible_by != 0:
        cropped_data = cropped_data[:-(cropped_data.shape[0] % divisible_by)]
    if cropped_data.shape[1] % divisible_by != 0:
        cropped_data = cropped_data[:, :-(cropped_data.shape[1] % divisible_by)]
    if cropped_data.shape[2] % divisible_by != 0:
        cropped_data = cropped_data[:, :, :-(cropped_data.shape[2] % divisible_by)]
    
    return cropped_data

def gpu_memory_tracker(
    device: Optional[Union[int, torch.device]] = None,
    clear_cache: bool = False,
    log: bool = True,
    return_stats: bool = False,
    sleep_interval: Optional[float] = None
) -> Optional[dict]:
    """
    Track GPU memory usage and optionally clear cache.
    
    Args:
        device: GPU device ID or torch.device (default: current device).
        clear_cache: If True, clears unused GPU memory cache.
        log: If True, prints memory stats.
        return_stats: If True, returns a dictionary of memory stats.
        sleep_interval: If set, continuously monitors memory at this interval (seconds).
    
    Returns:
        dict (if return_stats=True): {
            "total_gb": float,
            "allocated_gb": float,
            "reserved_gb": float,
            "free_gb": float
        }
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU memory tracking.")
        return None
    
    # Set device
    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device = device.index
    
    # Get memory stats
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total - allocated
    
    # Convert to GB
    stats = {
        "total_gb": total / (1024 ** 3),
        "allocated_gb": allocated / (1024 ** 3),
        "reserved_gb": reserved / (1024 ** 3),
        "free_gb": free / (1024 ** 3)
    }
    
    # Clear cache if requested
    if clear_cache:
        torch.cuda.empty_cache()
        if log:
            print("GPU cache cleared.")
    
    # Log results
    if log:
        print(
            f"GPU Memory Stats (Device {device}):\n"
            f"  Total: {stats['total_gb']:.2f} GB\n"
            f"  Allocated: {stats['allocated_gb']:.2f} GB\n"
            f"  Reserved (Cached): {stats['reserved_gb']:.2f} GB\n"
            f"  Free: {stats['free_gb']:.2f} GB"
        )
    
    # Continuous monitoring
    if sleep_interval is not None:
        try:
            while True:
                gpu_memory_tracker(device, clear_cache, log, False)
                time.sleep(sleep_interval)
        except KeyboardInterrupt:
            print("Stopped continuous monitoring.")
    
    return stats if return_stats else None

class VolumeDataset(Dataset):
    """Dataset for 3D volume data"""
    
    def __init__(self, volume_array, dimension, transform=None):
        """
        Args:
            volume_array (numpy.ndarray): 3D volume data of shape (D, H, W)
            transform (callable, optional): Optional transform to apply to samples
        """
        self.volume = volume_array
        self.transform = transform
        self.dimension = dimension
        if self.dimension == 'axial':
            self.depth = volume_array.shape[-1]
        elif self.dimension == 'coronal':
            self.depth = volume_array.shape[1]
        elif self.dimension == 'sagittal':
            self.depth = volume_array.shape[0]

        
    def __len__(self):
        """Return the number of 2D slices in the volume"""
        return self.depth
    
    def __getitem__(self, idx):
        """Get a single 2D slice from the volume"""
        # Extract a 2D slice from the 3D volume
        if self.dimension == 'axial':
            slice_2d = torch.rot90(self.volume[:,:,idx], 1)
        elif self.dimension == 'coronal':
            slice_2d = torch.rot90(self.volume[:,idx,:], 1)
        elif self.dimension == 'sagittal':
            slice_2d = torch.rot90(self.volume[idx,:,:], 1)
  
        # Add channel dimension
        slice_tensor = slice_2d.unsqueeze(0)  # Shape becomes (1, H, W)
        slice_tensor_original = slice_tensor.clone()
        
        if self.transform:
            slice_tensor = self.transform(slice_tensor)
        slice_tensor = downsample_upsample(slice_tensor.unsqueeze(0), scale_factor=0.5, mode='bilinear')
        
        return slice_tensor.squeeze(0), slice_tensor_original

def plot_patch_heatmap(image, positions, overlap_map):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original image with patch grid
    ax1.imshow(image, cmap='gray')
    for y, x in positions:
        rect = plt.Rectangle((x, y), 128, 128, linewidth=1, edgecolor='r', facecolor='none')
        ax1.add_patch(rect)
    ax1.set_title("Patch Locations")
    
    # Overlap heatmap
    ax2.imshow(overlap_map, cmap='hot')
    ax2.set_title("Overlap Density (Hot = More Overlap)")
    plt.savefig('adaptive_patch_heatmap.png')
    
def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    if timesteps.ndim != 1:
        raise ValueError("Timesteps should be a 1d-array")

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    freqs = torch.exp(exponent / half_dim)

    args = timesteps[:, None].float() * freqs[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        embedding = torch.nn.functional.pad(embedding, (0, 1, 0, 0))

    return embedding

class MonaiDeepCache:
    """
    DeepCache implementation for MONAI UNet following the methodology in the DeepCache paper.
    
    Caches and reuses high-level features from the UNet while computing low-level features.
    """
    def __init__(self, model, cache_interval=5, cache_branch_id=0):
        """
        Initialize DeepCache for a MONAI diffusion model.
        
        Args:
            model: MONAI diffusion model
            cache_interval: How frequently to update the cache (N value)
            cache_branch_id: Which branch to use for caching (0=shallowest, higher=deeper)
        """
        self.model = model
        self.cache_interval = cache_interval
        self.cache_branch_id = cache_branch_id
        self.cached_features = {}
        self.step_counter = 0
        self.enabled = False
        
        # Store original forward method
        self.original_down_forward = {}
        self.original_middle_forward = None
        self.original_up_forward = {}
        
        # Keep track of execution state
        self.current_down_idx = 0
        self.is_up_phase = False
        self.current_up_idx = 0
        
    def enable(self):
        """Enable DeepCache by replacing forward methods with patched versions"""
        if self.enabled:
            return
            
        # Store references to the original methods
        self.original_middle_forward = self.model.middle_block.forward
        
        # Patch down blocks
        for i, block in enumerate(self.model.down_blocks):
            self.original_down_forward[i] = block.forward
            
            # Create closure to capture the block index
            def create_down_patch(idx):
                def patched_forward(hidden_states, temb, context=None):
                    self.current_down_idx = idx
                    
                    # Use cache for blocks after the cache branch when cached features are available
                    if (idx > self.cache_branch_id and 
                        self.step_counter % self.cache_interval != 0 and 
                        f'down_{idx}' in self.cached_features):
                        return self.cached_features[f'down_{idx}']
                    
                    # Call original method and cache the result if needed
                    result = self.original_down_forward[idx](hidden_states, temb, context)
                    
                    # Cache results if this is a caching step
                    if self.step_counter % self.cache_interval == 0:
                        self.cached_features[f'down_{idx}'] = result
                        
                    return result
                return patched_forward
                
            # Apply the patch
            block.forward = create_down_patch(i)
            
        # Patch middle block
        def patched_middle_forward(hidden_states, temb, context=None):
            # Use cache for middle block when cached features are available
            if (self.step_counter % self.cache_interval != 0 and 
                'middle' in self.cached_features):
                return self.cached_features['middle']
                
            # Call original method and cache the result if needed
            result = self.original_middle_forward(hidden_states, temb, context)
            
            # Cache results if this is a caching step
            if self.step_counter % self.cache_interval == 0:
                self.cached_features['middle'] = result
                
            return result
            
        # Apply the middle patch
        self.model.middle_block.forward = patched_middle_forward
        
        # Patch up blocks
        for i, block in enumerate(self.model.up_blocks):
            self.original_up_forward[i] = block.forward
            
            # Create closure to capture the block index
            def create_up_patch(idx):
                def patched_forward(hidden_states, res_hidden_states_list, temb, context=None):
                    self.is_up_phase = True
                    self.current_up_idx = idx
                    
                    # Use cache for blocks before the cache branch when cached features are available
                    up_idx_from_end = len(self.model.up_blocks) - 1 - idx
                    if (up_idx_from_end > self.cache_branch_id and 
                        self.step_counter % self.cache_interval != 0 and 
                        f'up_{idx}' in self.cached_features):
                        return self.cached_features[f'up_{idx}']
                    
                    # Call original method and cache the result if needed
                    result = self.original_up_forward[idx](hidden_states, res_hidden_states_list, temb, context)
                    
                    # Cache results if this is a caching step
                    if self.step_counter % self.cache_interval == 0:
                        self.cached_features[f'up_{idx}'] = result
                        
                    return result
                return patched_forward
                
            # Apply the patch
            block.forward = create_up_patch(i)
            
        self.enabled = True
        
    def disable(self):
        """Disable DeepCache by restoring original forward methods"""
        if not self.enabled:
            return
            
        # Restore original methods
        self.model.middle_block.forward = self.original_middle_forward
        
        for i, block in enumerate(self.model.down_blocks):
            block.forward = self.original_down_forward[i]
            
        for i, block in enumerate(self.model.up_blocks):
            block.forward = self.original_up_forward[i]
            
        self.enabled = False
        
    def pre_step(self):
        """Call before each denoising step"""
        self.step_counter += 1
        
    def clear_cache(self):
        """Clear all cached features"""
        self.cached_features = {}
        self.step_counter = 0

class MonaiAdaptiveDeepCache:
    """
    AdaCache-inspired implementation for MONAI UNet that supports caching across
    images within a batch, with an adaptive caching schedule based on similarity.
    """
    def __init__(self, model, cache_interval=5, cache_branch_id=0, similarity_threshold=0.85):
        """
        Initialize AdaptiveDeepCache for a MONAI diffusion model.
        
        Args:
            model: MONAI diffusion model
            cache_interval: Initial/default cache interval
            cache_branch_id: Which branch to use for caching (0=shallowest, higher=deeper)
            similarity_threshold: Threshold to determine when to recompute vs. reuse cache
        """
        self.model = model
        self.cache_interval = cache_interval
        self.cache_branch_id = cache_branch_id
        self.similarity_threshold = similarity_threshold
        
        # Cache storage
        self.cached_features = {}
        self.cached_features_by_img = {}
        self.step_counter = 0
        self.enabled = False
        
        # Batch tracking
        self.batch_size = None
        self.similarity_matrix = None
        
        # Adaptive caching
        self.distance_metrics = {}
        self.caching_schedule = {}
        self.codebook = {
            0.05: 12, 0.10: 8, 0.15: 6, 0.20: 4, 0.30: 2, 1.00: 1
        }
        
        # Store original forward methods
        self.original_down_forward = {}
        self.original_middle_forward = None
        self.original_up_forward = {}
        
        # Track execution state
        self.current_down_idx = 0
        self.is_up_phase = False
        self.current_up_idx = 0
        
    def setup_batch(self, batch_input):
        """
        Setup caching for a new batch of images.
        
        Args:
            batch_input: The current batch input tensor [B, C, H, W]
        """
        self.batch_size = batch_input.shape[0]
        self.cached_features_by_img = {}
        
        # Compute feature representation for each image in the batch
        with torch.no_grad():
            # Simple feature extraction
            img_repr = batch_input.mean(dim=(2, 3))[:, :min(16, batch_input.shape[1])]
            
            # Compute similarity matrix between all images in the batch
            img_repr_norm = img_repr / (torch.norm(img_repr, dim=1, keepdim=True) + 1e-8)
            self.similarity_matrix = torch.mm(img_repr_norm, img_repr_norm.transpose(0, 1))
            
        # Initialize adaptive caching parameters for each image
        self.distance_metrics = {i: {} for i in range(self.batch_size)}
        self.caching_schedule = {i: self.cache_interval for i in range(self.batch_size)}
            
    def compute_distance_metric(self, features_current, features_cached):
        """
        Calculate L1 distance between feature maps.
        
        Args:
            features_current: Current feature output (may be a tuple for down blocks)
            features_cached: Cached feature output (may be a tuple for down blocks)
            
        Returns:
            Distance metric (float)
        """
        with torch.no_grad():
            # Handle tuple output case (downsample blocks return (output, residual))
            if isinstance(features_current, tuple) and isinstance(features_cached, tuple):
                # Just compare the main output (first element of the tuple)
                main_current = features_current[0]
                main_cached = features_cached[0]
                return torch.abs(main_current - main_cached).mean().item()
            else:
                # Standard case - direct comparison
                return torch.abs(features_current - features_cached).mean().item()
    
    def update_caching_schedule(self, img_idx, layer_idx, current_features, cached_features):
        """
        Update the caching schedule based on feature distance.
        
        Args:
            img_idx: Index of the image in the batch
            layer_idx: Index of the layer
            current_features: Current feature output
            cached_features: Cached feature output
        """
        if cached_features is None:
            return
            
        # Compute distance between current and cached features
        distance = self.compute_distance_metric(current_features, cached_features)
        self.distance_metrics[img_idx][layer_idx] = distance
        
        # Get average distance across all tracked layers for this image
        if len(self.distance_metrics[img_idx]) > 0:
            avg_distance = sum(self.distance_metrics[img_idx].values()) / len(self.distance_metrics[img_idx])
            
            # Update caching schedule based on distance
            for threshold, rate in sorted(self.codebook.items()):
                if avg_distance <= threshold:
                    self.caching_schedule[img_idx] = rate
                    break
    
    def enable(self):
        """Enable AdaptiveDeepCache by replacing forward methods with patched versions"""
        if self.enabled:
            return
            
        # Store references to the original methods
        self.original_middle_forward = self.model.middle_block.forward
        
        # Patch down blocks
        for i, block in enumerate(self.model.down_blocks):
            self.original_down_forward[i] = block.forward
            
            # Create closure to capture the block index
            def create_down_patch(idx):
                def patched_forward(hidden_states, temb, context=None):
                    self.current_down_idx = idx
                    
                    # Skip caching logic for shallow blocks or non-batch processing
                    if idx <= self.cache_branch_id or self.batch_size is None or self.batch_size <= 1:
                        result = self.original_down_forward[idx](hidden_states, temb, context)
                        if self.step_counter % self.cache_interval == 0:
                            self.cached_features[f'down_{idx}'] = result
                        return result
                    
                    # For batch processing with multiple images
                    batch_chunks = []
                    res_lists = []  # Each element will be a list of residual tensors
                    
                    for img_idx in range(self.batch_size):
                        img_cache_key = f'down_{idx}_{img_idx}'
                        img_input = hidden_states[img_idx:img_idx+1]
                        img_temb = temb[img_idx:img_idx+1] if temb is not None else None
                        img_context = context[img_idx:img_idx+1] if context is not None else None
                        
                        # Check if we should use cache for this image based on its adaptive schedule
                        img_cache_interval = self.caching_schedule.get(img_idx, self.cache_interval)
                        use_cache = (self.step_counter % img_cache_interval != 0 and 
                                    img_cache_key in self.cached_features_by_img)
                        
                        if use_cache:
                            # Use cached features for this image
                            img_result = self.cached_features_by_img[img_cache_key]
                            # Split into output and residual components
                            img_output, img_residual = img_result
                        else:
                            # Compute new features for this image
                            img_output, img_residual = self.original_down_forward[idx](img_input, img_temb, img_context)
                            img_result = (img_output, img_residual)
                            
                            # Cache the result
                            self.cached_features_by_img[img_cache_key] = img_result
                            
                            # Update caching schedule for next time if we have previous results
                            if img_cache_key in self.cached_features_by_img:
                                self.update_caching_schedule(
                                    img_idx, idx, img_result, 
                                    self.cached_features_by_img[img_cache_key]
                                )
                        
                        # Collect results for later concatenation
                        batch_chunks.append(img_output)
                        res_lists.append(img_residual)  # This is already a list of tensors
                    
                    # Concatenate results back into a batch
                    output = torch.cat(batch_chunks, dim=0)
                    
                    # Handle residuals - they are lists of tensors
                    # We need to concatenate corresponding tensors across all images
                    combined_residuals = []
                    if len(res_lists) > 0 and isinstance(res_lists[0], list):
                        # Determine how many residual tensors we have per image
                        num_residuals = len(res_lists[0])
                        
                        # For each position in the residual list
                        for res_idx in range(num_residuals):
                            # Collect the residual tensor at this position from each image
                            res_at_idx = [res_list[res_idx] for res_list in res_lists]
                            # Concatenate these tensors
                            combined_res = torch.cat(res_at_idx, dim=0)
                            combined_residuals.append(combined_res)
                    else:
                        # If residuals are not lists, concatenate them directly
                        combined_residuals = torch.cat(res_lists, dim=0)
                    
                    result = (output, combined_residuals)
                    
                    # Also update global cache for backward compatibility
                    if self.step_counter % self.cache_interval == 0:
                        self.cached_features[f'down_{idx}'] = result
                        
                    return result
                return patched_forward
                
            # Apply the patch
            block.forward = create_down_patch(i)
            
        # Patch middle block
        def patched_middle_forward(hidden_states, temb, context=None):
            # Skip caching logic for non-batch processing
            if self.batch_size is None or self.batch_size <= 1:
                result = self.original_middle_forward(hidden_states, temb, context)
                if self.step_counter % self.cache_interval == 0:
                    self.cached_features['middle'] = result
                return result
            
            # For batch processing with multiple images
            batch_chunks = []
            
            for img_idx in range(self.batch_size):
                img_cache_key = f'middle_{img_idx}'
                img_input = hidden_states[img_idx:img_idx+1]
                img_temb = temb[img_idx:img_idx+1] if temb is not None else None
                img_context = context[img_idx:img_idx+1] if context is not None else None
                
                # Check if we should use cache based on adaptive schedule
                img_cache_interval = self.caching_schedule.get(img_idx, self.cache_interval)
                use_cache = (self.step_counter % img_cache_interval != 0 and 
                              img_cache_key in self.cached_features_by_img)
                
                if use_cache:
                    # Use cached features for this image
                    img_result = self.cached_features_by_img[img_cache_key]
                else:
                    # Compute new features for this image
                    img_result = self.original_middle_forward(img_input, img_temb, img_context)
                    
                    # Cache the result
                    self.cached_features_by_img[img_cache_key] = img_result
                    
                    # Update caching schedule for next time
                    if img_cache_key in self.cached_features_by_img:
                        self.update_caching_schedule(
                            img_idx, 0, img_result, 
                            self.cached_features_by_img[img_cache_key]
                        )
                
                batch_chunks.append(img_result)
            
            # Concatenate results back into a batch
            result = torch.cat(batch_chunks, dim=0)
            
            # Also update global cache for backward compatibility
            if self.step_counter % self.cache_interval == 0:
                self.cached_features['middle'] = result
                
            return result
            
        # Apply the middle patch
        self.model.middle_block.forward = patched_middle_forward
        
        # Patch up blocks
        for i, block in enumerate(self.model.up_blocks):
            self.original_up_forward[i] = block.forward
            
            def create_up_patch(idx):
                def patched_forward(hidden_states, res_hidden_states_list, temb, context=None):
                    self.is_up_phase = True
                    self.current_up_idx = idx
                    
                    # Skip caching logic for shallow blocks or non-batch processing
                    up_idx_from_end = len(self.model.up_blocks) - 1 - idx
                    if up_idx_from_end <= self.cache_branch_id or self.batch_size is None or self.batch_size <= 1:
                        result = self.original_up_forward[idx](hidden_states, res_hidden_states_list, temb, context)
                        if self.step_counter % self.cache_interval == 0:
                            self.cached_features[f'up_{idx}'] = result
                        return result
                    
                    # For batch processing with multiple images
                    batch_chunks = []
                    
                    for img_idx in range(self.batch_size):
                        img_cache_key = f'up_{idx}_{img_idx}'
                        img_input = hidden_states[img_idx:img_idx+1]
                        img_res_list = [res[img_idx:img_idx+1] for res in res_hidden_states_list]
                        img_temb = temb[img_idx:img_idx+1] if temb is not None else None
                        img_context = context[img_idx:img_idx+1] if context is not None else None
                        
                        # Check if we should use cache based on adaptive schedule
                        img_cache_interval = self.caching_schedule.get(img_idx, self.cache_interval)
                        use_cache = (self.step_counter % img_cache_interval != 0 and 
                                    img_cache_key in self.cached_features_by_img)
                        
                        if use_cache:
                            # Use cached features for this image
                            img_result = self.cached_features_by_img[img_cache_key]
                        else:
                            # Compute new features for this image
                            img_result = self.original_up_forward[idx](img_input, img_res_list, img_temb, img_context)
                            
                            # Cache the result
                            self.cached_features_by_img[img_cache_key] = img_result
                            
                            # Update caching schedule for next time
                            if img_cache_key in self.cached_features_by_img:
                                self.update_caching_schedule(
                                    img_idx, idx + 100, img_result,  # Use idx+100 to differentiate from down blocks
                                    self.cached_features_by_img[img_cache_key]
                                )
                        
                        batch_chunks.append(img_result)
                    
                    # Concatenate results back into a batch
                    result = torch.cat(batch_chunks, dim=0)
                    
                    # Also update global cache for backward compatibility
                    if self.step_counter % self.cache_interval == 0:
                        self.cached_features[f'up_{idx}'] = result
                        
                    return result
                return patched_forward
                
            # Apply the patch
            block.forward = create_up_patch(i)
            
        self.enabled = True
    
    def pre_step(self):
        """Call before each denoising step"""
        self.step_counter += 1
        
    def disable(self):
        """Disable DeepCache by restoring original forward methods"""
        if not self.enabled:
            return
            
        # Restore original methods
        self.model.middle_block.forward = self.original_middle_forward
        
        for i, block in enumerate(self.model.down_blocks):
            if i in self.original_down_forward:
                block.forward = self.original_down_forward[i]
            
        for i, block in enumerate(self.model.up_blocks):
            if i in self.original_up_forward:
                block.forward = self.original_up_forward[i]
            
        self.enabled = False
        
    def clear_cache(self):
        """Clear all caches"""
        self.cached_features = {}
        self.cached_features_by_img = {}
        self.distance_metrics = {}
        self.caching_schedule = {}
        self.similarity_matrix = None
        self.batch_size = None
        self.step_counter = 0

def adaptive_deepcache_inference(model, device, scheduler, t1w, inference_step=100, 
                                cache_interval=5, cache_branch_id=0, similarity_threshold=0.85,
                                progressive_inference=False, noise_level=0.7, use_half=True):
    """
    Run inference with Adaptive DeepCache optimization for faster generation with batch-aware caching.
    
    Args:
        model: The diffusion model
        device: Device to run inference on
        scheduler: Diffusion scheduler
        t1w: Input batch of images [B, C, H, W]
        cache_interval: Initial cache interval (will be adapted per image)
        cache_branch_id: Which branch to use for caching (0=shallowest, higher=deeper)
        similarity_threshold: Threshold to determine when to reuse cache between similar images
        progressive_inference: Whether to use progressive inference
        noise_level: Progressive inference noise level (0.0 to 1.0)
        use_half: Whether to use half precision (FP16)
    
    Returns:
        Generated batch of images
    """
    model.eval()
    
    # Initialize Adaptive DeepCache
    adaptive_cache = MonaiAdaptiveDeepCache(
        model, 
        cache_interval=cache_interval, 
        cache_branch_id=cache_branch_id,
        similarity_threshold=similarity_threshold
    )
    
    try:
        adaptive_cache.enable()
        
        with torch.no_grad():
            # Setup
            input_img = t1w
            
            # Initialize adaptive caching for this batch
            adaptive_cache.setup_batch(input_img)
            
            if use_half:
                input_img = input_img.half()
                
            scheduler.set_timesteps(num_inference_steps=inference_step)
            
            # Initial setup for progressive inference if enabled
            if progressive_inference:
                current_img = input_img.clone()
                noise_timestep = int(noise_level * len(scheduler.timesteps))
                noise_timestep = max(min(noise_timestep, len(scheduler.timesteps)-1), 0)
                noise = torch.randn_like(current_img, device=device)
                if use_half:
                    noise = noise.half()
                    
                t = scheduler.timesteps[noise_timestep]

                alpha_cumprod = scheduler.alphas_cumprod.to(device)
                if use_half:
                    alpha_cumprod = alpha_cumprod.half()
                    
                sqrt_alpha_t = alpha_cumprod[t] ** 0.5
                sqrt_one_minus_alpha_t = (1 - alpha_cumprod[t]) ** 0.5
                current_img = sqrt_alpha_t * current_img + sqrt_one_minus_alpha_t * noise
                
                starting_timestep_idx = noise_timestep
                timesteps = scheduler.timesteps[starting_timestep_idx:]
            else:
                noise = torch.randn_like(input_img).to(device)
                if use_half:
                    noise = noise.half()
                    
                current_img = noise  # Start from random noise
                timesteps = scheduler.timesteps
            
            # Main inference loop
            batch_size = input_img.shape[0]
            progress_bar = tqdm(timesteps, desc=f"Adaptive DeepCache Inferencing (Batch: {batch_size})")
            
            for t in progress_bar:
                # Prepare for this step
                adaptive_cache.pre_step()
                
                # Calculate average cache interval across all images in batch
                avg_cache_interval = sum(adaptive_cache.caching_schedule.values()) / max(len(adaptive_cache.caching_schedule), 1)
                
                # Update progress bar with cache usage info
                if adaptive_cache.step_counter % cache_interval == 0:
                    progress_bar.set_postfix({"mode": "full", "avg_interval": f"{avg_cache_interval:.1f}"})
                else:
                    progress_bar.set_postfix({"mode": "cached", "avg_interval": f"{avg_cache_interval:.1f}"})
                    
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half):
                    # Combine input and current noisy image
                    combined = torch.cat((input_img, current_img), dim=1)
                    
                    # Expand timesteps for batch processing
                    batch_timesteps = torch.Tensor((t,)).to(device).repeat(batch_size)
                    
                    # Run model with Adaptive DeepCache
                    model_output = model(combined, timesteps=batch_timesteps)
                    
                    # Update the noisy image
                    current_img, _ = scheduler.step(model_output, t, current_img)
                    
            return current_img
    finally:
        # Always restore the original model
        adaptive_cache.disable()

def load_model(checkpoint_path, device="cuda", use_xformers=True, use_half=True, torch_compile=True, channels_last=False):
    
    """Load the pretrained diffusion model with various acceleration options
    
    Args:
        checkpoint_path: Path to model checkpoint
        device: Target device ('cuda' or 'cpu')
        use_xformers: Use xFormers memory-efficient attention
        use_half: Use FP16 precision
        torch_compile: Compile model with PyTorch 2.0
        enable_quantization: Apply dynamic quantization to linear layers
        use_cuda_graph: Enable CUDA graph capture
        triton_optimize: Use Triton optimized kernels (if available)
        attention_slicing: Enable attention slicing for memory saving
        channels_last: Use channels-last memory format
    """
    # Initialize model
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,  # T1w + noise
        out_channels=1,  # TSE
        channels=(256, 256, 512),
        attention_levels=(False, False, True),
        num_res_blocks=2,
        num_head_channels=512,
        with_conditioning=False,
    ).to(device)
    
    # Load checkpoint with safety checks
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        # Handle weight mismatches gracefully
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # Memory format optimization
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    
    # Precision conversion
    if use_half:
        model = model.half()
    
    # Attention optimizations
    if use_xformers:
        try:
            replace_unet_attention_with_xformers(model)
        except ImportError:
            print("xFormers not available, falling back to default attention")
    
    # Torch compilation
    if torch_compile and hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Final device placement
    model.to(device)
    return model

def replace_unet_attention_with_xformers(unet):
    """Replace the attention mechanism in UNet with xformers for memory efficiency"""
    # Find all attention modules in the model
    for name, module in unet.named_modules():
        # Look for attention modules - adjust the class name based on MONAI's implementation
        if "SelfAttention" in module.__class__.__name__ or "CrossAttention" in module.__class__.__name__:
            # Replace the attention operation with xformers efficient attention
            module.set_use_memory_efficient_attention_xformers(True)
            print(f"Replaced attention mechanism in {name} with xformers")

def quantile_normalization(input_nii, lower_quantile=0.05, upper_quantile=0.95):
    """
    Normalizes the voxel values of a NIfTI volume based on quantiles.
    """
    # Load NIfTI file if provided as a path
    if isinstance(input_nii, str):
        img = nib.load(input_nii)
    else:
        img = input_nii

    # Get the image data as a numpy array
    data = img.get_fdata()
    data = np.nan_to_num(data, nan=0.0, posinf=None, neginf=None)
    
    # Flatten the data for quantile calculation
    data_flat = data.flatten()

    # Compute the specified quantiles
    lower = np.percentile(data_flat, lower_quantile * 100)
    upper = np.percentile(data_flat, upper_quantile * 100)

    # Apply the quantile-based normalization
    data_normalized = np.clip(data, lower, upper)  # Clip values to be within the quantile range
    data_normalized = (data_normalized - lower) / (upper - lower + 1e-3)  # Normalize to the [0, 1] range

    return data_normalized

def vanilla_inference(model, device, scheduler, t1w, progressive_inference=False, noise_level=0.7, use_half=True):
    """
    Standard inference without DeepCache optimization but with half precision.
    """
    model.eval()
    with torch.no_grad():
        # Setup
        input_img = t1w
        if use_half:
            input_img = input_img.half()
            
        scheduler.set_timesteps(num_inference_steps=1000)

        if progressive_inference:
            current_img = input_img.clone()
            noise_timestep = int(noise_level * len(scheduler.timesteps))
            noise_timestep = max(min(noise_timestep, len(scheduler.timesteps)-1), 0)
            noise = torch.randn_like(current_img, device=device)
            if use_half:
                noise = noise.half()
                
            t = scheduler.timesteps[noise_timestep]

            alpha_cumprod = scheduler.alphas_cumprod.to(device)
            if use_half:
                alpha_cumprod = alpha_cumprod.half()
                
            sqrt_alpha_t = alpha_cumprod[t] ** 0.5
            sqrt_one_minus_alpha_t = (1 - alpha_cumprod[t]) ** 0.5
            current_img = sqrt_alpha_t * current_img + sqrt_one_minus_alpha_t * noise
            
            starting_timestep_idx = noise_timestep
            timesteps = scheduler.timesteps[starting_timestep_idx:]
        else:
            noise = torch.randn_like(input_img).to(device)
            if use_half:
                noise = noise.half()
                
            current_img = noise  # for the TSE image, we start from random noise.
            timesteps = scheduler.timesteps

        progress_bar = tqdm(timesteps, desc="Vanilla Inferencing")

        for t in progress_bar:  # go through the noising process
            with torch.autocast(device_type="cuda",  dtype=torch.float16, enabled=use_half):
                combined = torch.cat((input_img, current_img), dim=1)
                model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                current_img, _ = scheduler.step(model_output, t, current_img)
                
                    
        return current_img

def patch_inference(model, device, scheduler, t1w, patch_size=128, base_overlap=32, intensity_scale=2.0, target_patches=None, progressive_inference=False, noise_level=0.7, use_half=True, use_deepcache=True, cache_interval=20, cache_branch_id=0):
    """
    Patch-based inference for large images using adaptive patch placement with optional DeepCache optimization.
    
    Args:
        model: The model to use for inference
        device: Computation device
        scheduler: Diffusion scheduler
        t1w: Input image tensor
        patch_size (int): Size of each patch (e.g., 128).
        base_overlap (int): Minimum overlap between patches (reduces seams).
        intensity_scale (float): How much to scale overlap based on information content
        target_patches (int): Target number of patches (None = minimum required for coverage)
        progressive_inference (bool): Whether to use progressive inference
        noise_level (float): Noise level for progressive inference
        use_half (bool): Whether to use half precision
        use_deepcache (bool): Whether to use DeepCache optimization
        cache_interval (int): How often to update the cache (N value), only used if use_deepcache=True
        cache_branch_id (int): Which branch to use for caching (0=shallowest, higher=deeper), only used if use_deepcache=True
    """
    import math
    from tqdm import tqdm
    
    model.eval()
    
    # Initialize DeepCache if enabled
    deepcache = None
    if use_deepcache:
        deepcache = MonaiDeepCache(model, cache_interval=cache_interval, cache_branch_id=cache_branch_id)
        deepcache.enable()
    
    try:
        # --- Adaptive Patch Extraction ---
        B, C, H, W = t1w.shape
        
        patch_num = (H//patch_size) ** 2

        # Extract single image from batch for patch planning
        # Assuming we process one image at a time (B=1)
        image = t1w[0, 0].cpu().numpy()  # Use first channel for information map
        
        # Get adaptive patch positions
        positions, overlap_map = adaptive_patch_locations(
            image, 
            patch_size=patch_size, 
            base_overlap=base_overlap, 
            intensity_scale=intensity_scale,
            target_patches=target_patches
        )
        plot_patch_heatmap(image, positions, overlap_map)
        # Extract patches based on adaptive positions
        patches = []
        for y, x in positions:
            # Ensure patch is within bounds
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            
            # Handle edge cases - make sure patches have consistent size
            if y_end - y < patch_size:
                y = max(0, y_end - patch_size)
            if x_end - x < patch_size:
                x = max(0, x_end - patch_size)
                
            patch = t1w[..., y:y+patch_size, x:x+patch_size]
            patches.append(patch)
        
        # --- Process Each Patch ---
        processed_patches = []
        for patch_idx, patch in enumerate(tqdm(patches, desc="Processing Patches")):
            with torch.no_grad():
                # Move patch to device & cast to half if needed
                patch = patch.to(device)
                if use_half:
                    patch = patch.half()
                
                # --- Inference Logic (Per Patch) ---
                if progressive_inference:
                    current_img = patch.clone()
                    noise_timestep = int(noise_level * len(scheduler.timesteps))
                    noise_timestep = max(min(noise_timestep, len(scheduler.timesteps)-1), 0)
                    noise = torch.randn_like(current_img, device=device)
                    if use_half:
                        noise = noise.half()
                    
                    t = scheduler.timesteps[noise_timestep]
                    alpha_cumprod = scheduler.alphas_cumprod.to(device)
                    if use_half:
                        alpha_cumprod = alpha_cumprod.half()
                    
                    sqrt_alpha_t = alpha_cumprod[t] ** 0.5
                    sqrt_one_minus_alpha_t = (1 - alpha_cumprod[t]) ** 0.5
                    current_img = sqrt_alpha_t * current_img + sqrt_one_minus_alpha_t * noise
                    timesteps = scheduler.timesteps[noise_timestep:]
                else:
                    noise = torch.randn_like(patch).to(device)
                    if use_half:
                        noise = noise.half()
                    current_img = noise
                    timesteps = scheduler.timesteps

                # --- Denoising Loop (with or without DeepCache) ---
                progress_desc = f"Patch {patch_idx+1}/{len(patches)}"
                progress_bar = tqdm(timesteps, desc=progress_desc, leave=False)
                
                for t in progress_bar:
                    # DeepCache pre-step preparation if enabled
                    if use_deepcache:
                        deepcache.pre_step()
                        
                        # Update progress bar with mode for DeepCache
                        if deepcache.step_counter % cache_interval == 0:
                            progress_bar.set_postfix({"mode": "full  "})
                        else:
                            progress_bar.set_postfix({"mode": "cached"})
                    
                    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half):
                        combined = torch.cat((patch, current_img), dim=1)
                        model_output = model(combined, timesteps=torch.Tensor((t,)).to(device))
                        current_img, _ = scheduler.step(model_output, t, current_img)
                
                processed_patches.append(current_img.cpu())
        
        # --- Reconstruct Full Image ---
        output = torch.zeros_like(t1w, device="cpu")
        count = torch.zeros_like(t1w, device="cpu")  # For averaging overlaps
        
        for (y, x), patch in zip(positions, processed_patches):
            # Ensure we're within bounds
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            
            # Handle edge cases
            if y_end - y < patch_size:
                y = max(0, y_end - patch_size)
            if x_end - x < patch_size:
                x = max(0, x_end - patch_size)
                
            output[..., y:y+patch_size, x:x+patch_size] += patch
            count[..., y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping regions
        non_zero_mask = (count > 0)
        output[non_zero_mask] /= count[non_zero_mask]
        
        return output.to(device)
    
    finally:
        # Always disable DeepCache if it was enabled
        if use_deepcache and deepcache is not None:
            deepcache.disable()

def compute_information_map(image):
    """Convert image to gradient magnitude (high = more info)."""
    from skimage.filters import sobel
    
    if isinstance(image, torch.Tensor):
        image = image.squeeze().cpu().numpy()  # Assume shape (H, W)
    grad_mag = sobel(image)  # Sobel edge detection
    return grad_mag / grad_mag.max()  # Normalize to [0, 1]

def adaptive_patch_locations(image, patch_size=128, base_overlap=32, intensity_scale=2.0, target_patches=None):
    """
    Generate adaptive patch locations with control over total patch count.
    
    Args:
        image: Input image
        patch_size: Size of each patch (assumed square)
        base_overlap: Minimum overlap between patches
        intensity_scale: How much to scale overlap based on information content
        target_patches: Target number of patches (None = minimum required for coverage)
    
    Returns: 
        positions (list): [(y1, x1), (y2, x2), ...] patch coordinates
        overlap_map (np.array): Heatmap of overlap density
    """
    import math
    import numpy as np
    
    info_map = compute_information_map(image)
    H, W = info_map.shape
    overlap_map = np.zeros((H, W))
    
    # Calculate minimum patches needed for coverage
    stride_max = patch_size  # No overlap in empty regions
    min_patches_h = math.ceil(H / stride_max)
    min_patches_w = math.ceil(W / stride_max)
    min_patches = min_patches_h * min_patches_w
    
    # Set actual target (use minimum if not specified)
    if target_patches is None or target_patches < min_patches:
        target_patches = min_patches
    
    # --- First ensure minimum coverage with evenly spaced patches ---
    positions = []
    if target_patches == min_patches:
        # Just use a regular grid with maximum stride
        for y in range(0, H, stride_max):
            for x in range(0, W, stride_max):
                positions.append((min(y, H - patch_size), min(x, W - patch_size)))
        
        # Update overlap map
        for y, x in positions:
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            overlap_map[y:y_end, x:x_end] += 1
    else:
        # --- Two-phase patch placement ---
        # Phase 1: Ensure coverage with minimum patches
        covered = np.zeros((H, W), dtype=bool)
        stride_min = patch_size - base_overlap
        
        # Start with high-information regions
        remaining_info = info_map.copy()
        while len(positions) < target_patches:
            if len(positions) < min_patches:
                # First ensure basic coverage - prioritize uncovered areas
                remaining_mask = ~covered
                if remaining_mask.sum() == 0:
                    # Everything covered, now focus on information density
                    remaining_mask = np.ones_like(covered)
                
                remaining = remaining_info * remaining_mask
                if remaining.sum() == 0:
                    # If no information in remaining areas, just use mask
                    remaining = remaining_mask.astype(float)
                
                y, x = np.unravel_index(remaining.argmax(), remaining.shape)
                
                # Adaptive stride based on local info
                local_info = info_map[
                    max(0, y-patch_size//2):min(H, y+patch_size//2), 
                    max(0, x-patch_size//2):min(W, x+patch_size//2)
                ].mean()
                
                stride = int(stride_max - (stride_max - stride_min) * local_info * intensity_scale)
                stride = max(stride_min, min(stride, stride_max))
                
                # Adjust coordinates to avoid going out of bounds
                y = min(max(0, y - patch_size//2), H - patch_size)
                x = min(max(0, x - patch_size//2), W - patch_size)
                
                # Add patch and mark covered area
                positions.append((y, x))
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                overlap_map[y:y_end, x:x_end] += 1
                covered[y:y_end, x:x_end] = True
                
                # Reduce information in this patch area to avoid concentration
                reduction_mask = np.zeros_like(remaining_info)
                reduction_mask[y:y_end, x:x_end] = 1
                remaining_info = remaining_info * (1 - reduction_mask * 0.8)
            else:
                # Phase 2: Add extra patches to reach target, prioritizing information density
                y, x = np.unravel_index(remaining_info.argmax(), remaining_info.shape)
                
                # Adjust coordinates to avoid going out of bounds
                y = min(max(0, y - patch_size//2), H - patch_size)
                x = min(max(0, x - patch_size//2), W - patch_size)
                
                positions.append((y, x))
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                overlap_map[y:y_end, x:x_end] += 1
                
                # Reduce information in this patch area to avoid concentration
                reduction_mask = np.zeros_like(remaining_info)
                reduction_mask[y:y_end, x:x_end] = 1
                remaining_info = remaining_info * (1 - reduction_mask * 0.8)
    
    return positions, overlap_map

def additional_adaptive_patch_locations(image, patch_size=128, base_overlap=32, intensity_scale=2.0, 
                            target_patches=None, min_info_threshold=0.1):
    """
    Generate adaptive patch locations with better focus on information-rich regions.
    
    Args:
        image: Input image
        patch_size: Size of each patch (assumed square)
        base_overlap: Minimum overlap between patches
        intensity_scale: How much to scale overlap based on information content
        target_patches: Target number of patches (None = minimum required for coverage)
        min_info_threshold: Minimum information content required for a patch to be placed
    
    Returns: 
        positions (list): [(y1, x1), (y2, x2), ...] patch coordinates
        overlap_map (np.array): Heatmap of overlap density
    """
    import math
    import numpy as np
    
    # Get information map and create a binary mask of regions worth covering
    info_map = compute_information_map(image)
    H, W = info_map.shape
    overlap_map = np.zeros((H, W))
    
    # Create a binary mask identifying areas with sufficient information content
    info_binary = info_map > min_info_threshold
    
    # Find connected regions in the binary mask
    from scipy import ndimage
    labeled_regions, num_regions = ndimage.label(info_binary)
    
    # Calculate region properties
    region_props = []
    for i in range(1, num_regions+1):
        region_mask = (labeled_regions == i)
        y_indices, x_indices = np.where(region_mask)
        if len(y_indices) > 0:  # Ensure region isn't empty
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
            size = np.sum(region_mask)
            region_props.append({
                'center': (center_y, center_x),
                'size': size,
                'mask': region_mask
            })
    
    # Sort regions by size (largest first)
    region_props.sort(key=lambda x: x['size'], reverse=True)
    
    # Calculate minimum patches needed for coverage of information regions
    # Instead of covering the whole image, we focus on covering the information-rich areas
    info_area = np.sum(info_binary)
    coverage_ratio = info_area / (H * W)
    
    # Adjust minimum patches based on information area
    stride_max = patch_size  # No overlap in empty regions
    min_patches_h = math.ceil(H * math.sqrt(coverage_ratio) / stride_max)
    min_patches_w = math.ceil(W * math.sqrt(coverage_ratio) / stride_max)
    min_patches = max(1, min_patches_h * min_patches_w)
    
    # Set actual target (use minimum if not specified)
    if target_patches is None or target_patches < min_patches:
        target_patches = min_patches
    
    # --- Patch placement strategy ---
    positions = []
    
    # Phase 1: Place patches at region centers to ensure coverage of all regions
    for region in region_props:
        center_y, center_x = region['center']
        
        # Adjust coordinates to avoid going out of bounds
        y = min(max(0, center_y - patch_size//2), H - patch_size)
        x = min(max(0, center_x - patch_size//2), W - patch_size)
        
        positions.append((y, x))
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        overlap_map[y:y_end, x:x_end] += 1
        
        # If we have enough patches, stop adding more
        if len(positions) >= target_patches:
            break
    
    # Phase 2: If we still need more patches, add them based on information content
    if len(positions) < target_patches:
        # Create a mask of areas already covered
        covered = np.zeros((H, W), dtype=bool)
        for y, x in positions:
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            covered[y:y_end, x:x_end] = True
        
        # Create a modified information map that prioritizes uncovered areas with information
        remaining_info = info_map.copy()
        remaining_info[covered] *= 0.2  # Reduce priority of already covered areas
        
        # Reduce priority in areas with low information content
        remaining_info[~info_binary] *= 0.1
        
        stride_min = patch_size - base_overlap
        
        while len(positions) < target_patches:
            # Find position with highest remaining information
            y, x = np.unravel_index(remaining_info.argmax(), remaining_info.shape)
            
            # Compute local information density
            local_region = info_map[
                max(0, y-patch_size//2):min(H, y+patch_size//2), 
                max(0, x-patch_size//2):min(W, x+patch_size//2)
            ]
            local_info = local_region.mean()
            
            # Skip if the local information is too low
            if local_info < min_info_threshold and len(positions) >= min_patches:
                # Zero out this area to avoid selecting it again
                y_start = max(0, y - patch_size//2)
                x_start = max(0, x - patch_size//2)
                y_end = min(H, y + patch_size//2)
                x_end = min(W, x + patch_size//2)
                remaining_info[y_start:y_end, x_start:x_end] = 0
                continue
            
            # Adaptive stride based on local info
            stride = int(stride_max - (stride_max - stride_min) * local_info * intensity_scale)
            stride = max(stride_min, min(stride, stride_max))
            
            # Adjust coordinates to avoid going out of bounds
            y = min(max(0, y - patch_size//2), H - patch_size)
            x = min(max(0, x - patch_size//2), W - patch_size)
            
            # Add patch
            positions.append((y, x))
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            overlap_map[y:y_end, x:x_end] += 1
            
            # Reduce information in this patch area to avoid concentration
            reduction_mask = np.zeros_like(remaining_info)
            reduction_mask[y:y_end, x:x_end] = 1
            remaining_info = remaining_info * (1 - reduction_mask * 0.8)
    
    return positions, overlap_map

def deepcache_inference(model, device, scheduler, t1w, inference_step=100, cache_interval=5, cache_branch_id=0, 
                        progressive_inference=False, noise_level=0.7, use_half=True):
    """
    Run inference with DeepCache optimization for faster generation with half precision.
    
    Args:
        model: The diffusion model
        device: Device to run inference on
        scheduler: Diffusion scheduler
        t1w: Input image
        cache_interval: How often to update the cache (N value)
        cache_branch_id: Which branch to use for caching (0=shallowest, higher=deeper)
        progressive_inference: Whether to use progressive inference
        noise_level: Progressive inference noise level (0.0 to 1.0)
        use_half: Whether to use half precision (FP16)
    
    Returns:
        Generated image
    """
    model.eval()
    
    # Initialize DeepCache
    deepcache = MonaiDeepCache(model, cache_interval=cache_interval, cache_branch_id=cache_branch_id)
    deepcache.enable()
    
    try:
        with torch.no_grad():
            # Setup
            torch.cuda.empty_cache()

            input_img = t1w
            if use_half:
                input_img = input_img.half()
                
            scheduler.set_timesteps(num_inference_steps=inference_step)
            
            # Initial setup for progressive inference if enabled
            if progressive_inference:
                current_img = input_img.clone()
                noise_timestep = int(noise_level * len(scheduler.timesteps))
                noise_timestep = max(min(noise_timestep, len(scheduler.timesteps)-1), 0)
                noise = torch.randn_like(current_img, device=device)
                if use_half:
                    noise = noise.half()
                    
                t = scheduler.timesteps[noise_timestep]

                alpha_cumprod = scheduler.alphas_cumprod.to(device)
                if use_half:
                    alpha_cumprod = alpha_cumprod.half()
                    
                sqrt_alpha_t = alpha_cumprod[t] ** 0.5
                sqrt_one_minus_alpha_t = (1 - alpha_cumprod[t]) ** 0.5
                current_img = sqrt_alpha_t * current_img + sqrt_one_minus_alpha_t * noise
                
                starting_timestep_idx = noise_timestep
                timesteps = scheduler.timesteps[starting_timestep_idx:]
            else:
                noise = torch.randn_like(input_img).to(device)
                if use_half:
                    noise = noise.half()
                    
                current_img = noise  # for the TSE image, we start from random noise
                timesteps = scheduler.timesteps
            
            # Main inference loop
            progress_bar = tqdm(timesteps, desc="DeepCache Inferencing")
            
            for t in progress_bar:
                # Prepare for this step
                deepcache.pre_step()
                
                # Update progress bar with mode
                if deepcache.step_counter % cache_interval == 0:
                    progress_bar.set_postfix({"mode": "full  "})
                else:
                    progress_bar.set_postfix({"mode": "cached"})
                    
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half):
                    # Combine input and current noisy image
                    combined = torch.cat((input_img, current_img), dim=1)
                    
                    # Run model with DeepCache
                    model_output = model(combined, timesteps=torch.Tensor((t,)).to(device))
                    
                    # Update the noisy image
                    current_img, _ = scheduler.step(model_output, t, current_img)
                        
            return current_img
    finally:
        # Always restore the original model
        deepcache.disable()

def non_uniform_deepcache_inference(model, device, scheduler, t1w, 
                                   center_timestep=400, power=1.2, cache_branch_id=0,
                                   progressive_inference=False, noise_level=0.7):
    """
    DeepCache with non-uniform intervals as described in the paper.
    This samples more frequently around critical timesteps.
    
    Args:
        model: The diffusion model
        device: Device to run inference on
        scheduler: Diffusion scheduler
        t1w: Input image
        center_timestep: Timestep to center the non-uniform sampling around
        power: Power for the quadratic increase in intervals
        cache_branch_id: Which branch to use for caching
        progressive_inference: Whether to use progressive inference
        noise_level: Progressive inference noise level (0.0 to 1.0)
    
    Returns:
        Generated image
    """
    model.eval()
    
    # Initialize DeepCache
    deepcache = MonaiDeepCache(model, cache_interval=1, cache_branch_id=cache_branch_id)
    deepcache.enable()
    
    try:
        with torch.no_grad():
            scheduler.set_timesteps(num_inference_steps=1000)
            num_inference_steps = len(scheduler.timesteps)
            
            # Determine non-uniform caching steps
            # Sample more densely around center_timestep
            steps = np.arange(num_inference_steps)
            distances = np.abs(steps - center_timestep)
            # Quadratic increase in interval size as we move away from center
            intervals = np.maximum(1, np.power(distances / 10, power)).astype(int)
            
            # Create list of steps where we'll do full inference
            full_inference_steps = []
            current_step = 0
            while current_step < num_inference_steps:
                full_inference_steps.append(current_step)
                current_step += intervals[current_step]
            
            # Setup
            input_img = t1w
            
            # Initial setup for progressive inference if enabled
            if progressive_inference:
                current_img = input_img.clone()
                noise_timestep = int(noise_level * num_inference_steps)
                noise_timestep = max(min(noise_timestep, num_inference_steps-1), 0)
                noise = torch.randn_like(current_img, device=device)
                t = scheduler.timesteps[noise_timestep]

                alpha_cumprod = scheduler.alphas_cumprod.to(device)
                sqrt_alpha_t = alpha_cumprod[t] ** 0.5
                sqrt_one_minus_alpha_t = (1 - alpha_cumprod[t]) ** 0.5
                current_img = sqrt_alpha_t * current_img + sqrt_one_minus_alpha_t * noise
                
                starting_timestep_idx = noise_timestep
                timesteps = scheduler.timesteps[starting_timestep_idx:]
            else:
                noise = torch.randn_like(input_img).to(device)
                current_img = noise
                timesteps = scheduler.timesteps
            
            # Main inference loop
            progress_bar = tqdm(enumerate(timesteps), total=len(timesteps), desc="Non-Uniform DeepCache")
            
            for step_idx, t in progress_bar:
                # Determine if this is a step for full inference
                is_caching_step = step_idx in full_inference_steps
                deepcache.step_counter = 0 if is_caching_step else 1  # Force cache usage for non-caching steps
                
                # Update progress bar with mode
                if is_caching_step:
                    progress_bar.set_postfix({"mode": "full"})
                else:
                    progress_bar.set_postfix({"mode": "cached"})
                
                with autocast("cuda", enabled=True):
                    combined = torch.cat((input_img, current_img), dim=1)
                    
                    # Run the model with DeepCache
                    model_output = model(combined, timesteps=torch.Tensor((t,)).to(device))
                    
                    # Update the noisy image
                    current_img, _ = scheduler.step(model_output, t, current_img)
                        
            return current_img
    finally:
        # Always restore the original model
        deepcache.disable()

def main():
    import time
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # checkpoint_path = "/ix3/tibrahim/jil202/cfg_gen/src/training/checkpoints/320/0.5/anatomy_scheduler_False_lpips_1_320_ssim_8.46.pt" # 
    checkpoint_path = "/ix3/tibrahim/jil202/cfg_gen/src/training/checkpoints/320/0.5/patch_data_aug_lpips_1_320_ssim_8.14.pt"
    # checkpoint_path = "/ix3/tibrahim/jil202/cfg_gen/src/training/checkpoints/320/0.35/baseline_1.57mm_320_ssim_7.17.pt"
    image_path = '/ix3/tibrahim/jil202/cfg_gen/qc_image_nii/denoised_mp2rage/IBRA-BIO1/PRT170058/converted/MP2RAGE_UNI_Images/md_denoised_PRT170058_20180912172034.nii.gz'
    # image_path = 'undersampled.nii.gz'

    # DeepCache parameters to test
    cache_interval = 2    # N value - higher means more speedup but potentially lower quality
    cache_branch_id = 0   # Branch ID - 0=shallowest (fastest), higher=deeper (better quality)
    patch_size = 128
    image_size = 384
    target_patches = 32
    patch_num = (image_size//patch_size) ** 2
    additional_patches = target_patches - patch_num
    base_overlap = 32
    batch_size = 1
    inference_step = 100
    # Performance optimization settings
    use_xformers = True   # Enable xformers for memory-efficient attention
    use_half = True  # Enable half precision (FP16)
    use_progressive = False
    use_model_compile = True
    channels_last = False
        
    # Load model with optimizations
    print("Loading model with xformers and half precision...")
    model = load_model(checkpoint_path, device="cuda", use_xformers=use_xformers, use_half=use_half, torch_compile=use_model_compile, channels_last=channels_last)
    
    # Load and preprocess example input (single slice for benchmarking)
    input_nii = nib.load(image_path)
    data = quantile_normalization(input_nii, lower_quantile=0.01, upper_quantile=0.99)
    data = crop_mri_volume(torch.tensor(data)).unsqueeze(0).to(device).squeeze()[:,:,150:200]
    final_outputs_dimensions = []

    for dimension in ['axial']: #'sagittal', 'coronal'
        dataset = VolumeDataset(data, dimension)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        # Initialize scheduler
        # scheduler = DDPMScheduler(num_train_timesteps=1000)
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=inference_step)
        # Run benchmarks with optimizations
        print("\n--- Running inference benchmarks with xformers and half precision ---")
        cache_outputs = []
        final_outputs = []
        orig_inputs = []
        input_slices = []
        
        for idx, batch_data in enumerate(tqdm(dataloader)):
            # Select a single slice for testing
            input_slice = batch_data[0]
            orig_slice = batch_data[1]
            print(f"---------{dimension} | {idx}/{len(dataloader)}--------")
            for _ in range(1):
                gpu_memory_tracker()
                cache_output = patch_inference(model, device, scheduler, input_slice, patch_size=patch_size, base_overlap=32, intensity_scale=2.0, target_patches=18, cache_interval=cache_interval, cache_branch_id=cache_branch_id)
                # cache_output = deepcache_inference(model, device, scheduler, input_slice, inference_step=inference_step, cache_interval=cache_interval, cache_branch_id=cache_branch_id, use_half=use_half)
                cache_outputs.append(cache_output.detach().cpu())
                gpu_memory_tracker()
            cache_output = torch.clamp(torch.nan_to_num(torch.stack(cache_outputs).mean(0)), 0, 1)
            cache_outputs = []
            
            final_outputs.extend(cache_output)
            orig_inputs.extend(orig_slice.detach().cpu().numpy())
            input_slices.extend(torch.rot90(input_slice.detach().cpu(), k=-1, dims=(2, 3)))

        if dimension == 'coronal':
            final_outputs = torch.rot90(torch.stack(final_outputs), k=-1, dims=(2, 3)).squeeze().detach().cpu().permute(1,0,-1).float().numpy()
            
        elif dimension == 'sagittal':
            final_outputs = torch.rot90(torch.stack(final_outputs), k=-1, dims=(2, 3)).squeeze().detach().cpu().float().numpy()
            
        elif dimension == 'axial':
            final_outputs = torch.rot90(torch.stack(final_outputs), k=-1, dims=(2, 3)).squeeze().detach().cpu().permute(1,-1,0).float().numpy()

        final_outputs_dimensions.append(final_outputs)

    

    final_outputs = np.stack(final_outputs_dimensions).mean(0)
    
    input_slices = torch.tensor(np.squeeze(np.stack(input_slices))).permute(1,-1,0).numpy()

    cache_metrics = evaluate_image_quality(final_outputs.transpose(-1,0,1)[:,None,...], data.cpu().numpy().transpose(-1,0,1)[:,None,...])

    pdb.set_trace()
    data = tio.transforms.CropOrPad(input_nii.shape)(data.unsqueeze(0).cpu().numpy()).squeeze()
    final_outputs = tio.transforms.CropOrPad(input_nii.shape)(final_outputs[None]).squeeze()
    input_slices = tio.transforms.CropOrPad(input_nii.shape)(input_slices[None]).squeeze()

    
    nib.save(nib.Nifti1Image(np.round(final_outputs * 1000), affine=input_nii.affine), 'cache_output.nii.gz')
    nib.save(nib.Nifti1Image(data.cpu().numpy(), affine=input_nii.affine), 'cache_input.nii.gz')
    nib.save(nib.Nifti1Image(input_slices, affine=input_nii.affine), 'lowres_1.57mm.nii.gz')
    
if __name__ == "__main__":
    main()