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
from metrics import evaluate_image_quality, calculate_lpips, calculate_ssim
import xformers.ops
from torch.cuda.amp import GradScaler, autocast
import torchvision
import torch.nn.functional as F
from skimage.filters import sobel
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union
import time
from itertools import chain
import argparse

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
    sleep_interval: Optional[float] = None) -> Optional[dict]:
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
    
    def __init__(self, volume_array, dimension, scale_factor, transform=None):
        """
        Args:
            volume_array (numpy.ndarray): 3D volume data of shape (D, H, W)
            transform (callable, optional): Optional transform to apply to samples
        """
        self.volume = volume_array
        self.transform = transform
        self.dimension = dimension
        self.scale_factor = scale_factor

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
        slice_tensor = downsample_upsample(slice_tensor.unsqueeze(0), scale_factor=self.scale_factor, mode='bilinear')
        
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

def parselate(image, patch_size, overlap, patch_num=None, positions=None, overlap_map=None, cache_depth=1):
    """
    Parselate an image into patches with adaptive patch location selection,
    ensuring complete coverage of the image. If patch_num is None, computes the
    minimum number of patches needed to cover the entire image.
    
    Args:
        image: torch.Tensor of shape (B, C, H, W) or (C, H, W) or (H, W)
        patch_size: Size of each patch (assumed square)
        overlap: Base overlap between patches
        patch_num: Target number of patches per image (None for auto-compute)
        positions: Optional pre-computed patch positions
        overlap_map: Optional pre-computed overlap map
        cache_depth: Number of consecutive batch items to share the same patch locations
                    (must divide batch size evenly)
    
    Returns:
        patches: Tensor of patches with shape (B, patch_number, C, patch_size, patch_size)
        positions: List of patch positions with length B (expanded from cache groups)
        overlap_map: List of overlap maps with length B (expanded from cache groups)
    """
    import torch
    import numpy as np
    
    # Handle different input shapes
    if len(image.shape) == 2:  # (H, W)
        H, W = image.shape
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        B, C = 1, 1
    elif len(image.shape) == 3:  # (C, H, W)
        C, H, W = image.shape
        image = image.unsqueeze(0)  # (1, C, H, W)
        B = 1
    elif len(image.shape) == 4:  # (B, C, H, W)
        B, C, H, W = image.shape
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    # Validate cache_depth
    if cache_depth > B:
        cache_depth = B
    if B % cache_depth != 0:
        raise ValueError(f"Batch size ({B}) must be divisible by cache_depth ({cache_depth})")
    
    # Initialize with proper dimensions for reshaping later
    batch_patches = []
    all_positions = []  # This will be the compressed version (length B/cache_depth)
    all_overlap_maps = []  # This will be the compressed version (length B/cache_depth)
    
    # Process each cache group
    for group_idx in range(0, B, cache_depth):
        # Compute positions based on the first image in the group
        if C > 1:
            info_image = image[group_idx].mean(dim=0).detach().cpu().numpy()
        else:
            info_image = image[group_idx, 0].detach().cpu().numpy()
        
        # Compute positions if not provided
        if positions is None or overlap_map is None:
            if patch_num is None:
                # Auto-compute minimum number of patches needed for full coverage
                group_positions = create_covering_grid(H, W, patch_size, overlap)
                group_overlap_map = np.ones((H, W))  # Default overlap map
            else:
                # First get adaptive positions
                group_positions, group_overlap_map = adaptive_patch_locations(
                    info_image, 
                    patch_size=patch_size, 
                    base_overlap=overlap, 
                    intensity_scale=2.0, 
                    target_patches=patch_num
                )
                
                # Then ensure full coverage with a grid fallback
                grid_positions = create_covering_grid(H, W, patch_size, overlap)
                
                # Combine adaptive and grid positions, removing duplicates
                all_pos = group_positions + grid_positions
                unique_pos = []
                seen = set()
                for pos in all_pos:
                    if pos not in seen:
                        seen.add(pos)
                        unique_pos.append(pos)
                group_positions = unique_pos
        else:
            group_positions, group_overlap_map = positions, overlap_map
        
        num_patches = len(group_positions)
        
        # Store the positions and overlap map for this cache group
        all_positions.append(group_positions)
        all_overlap_maps.append(group_overlap_map)
        
        # Extract patches for all images in this cache group using the same positions
        for b in range(group_idx, min(group_idx + cache_depth, B)):
            # For each batch item, create a list to hold all its patches
            image_patches = []
            
            for y, x in group_positions:
                # Ensure patch is within bounds
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                
                # Handle edge cases - make sure patches have consistent size
                if y_end - y < patch_size:
                    y = max(0, y_end - patch_size)
                if x_end - x < patch_size:
                    x = max(0, x_end - patch_size)
                    
                patch = image[b, :, y:y+patch_size, x:x+patch_size]
                image_patches.append(patch)
            
            # Stack all patches for this image into a tensor of shape (num_patches, C, patch_size, patch_size)
            image_patches_tensor = torch.stack(image_patches)
            batch_patches.append(image_patches_tensor)
    
    # Stack all batch patches into a tensor of shape (B, num_patches, C, patch_size, patch_size)
    patches_tensor = torch.stack(batch_patches)
    
    # --- Expand positions and overlap maps to match batch size ---
    expanded_positions = []
    expanded_overlap_maps = []
    
    for group_idx in range(0, B, cache_depth):
        cache_group_idx = group_idx // cache_depth
        group_positions = all_positions[cache_group_idx]
        group_overlap_map = all_overlap_maps[cache_group_idx]
        
        # Repeat the positions and overlap map for each image in this cache group
        for b in range(group_idx, min(group_idx + cache_depth, B)):
            expanded_positions.append(group_positions)
            expanded_overlap_maps.append(group_overlap_map)
    
    return patches_tensor, expanded_positions, expanded_overlap_maps

def create_covering_grid(H, W, patch_size, overlap):
    """
    Create a grid of patch positions that ensures complete coverage of the image.
    Returns the minimum number of patch positions needed to cover the entire image.
    """
    stride = max(1, patch_size - overlap)
    
    # Calculate number of patches needed in each dimension
    num_y = int(np.ceil((H - patch_size) / stride)) + 1 if H > patch_size else 1
    num_x = int(np.ceil((W - patch_size) / stride)) + 1 if W > patch_size else 1
    
    # Adjust stride to ensure perfect coverage
    if num_y > 1:
        actual_stride_y = (H - patch_size) / (num_y - 1)
    else:
        actual_stride_y = 0
    
    if num_x > 1:
        actual_stride_x = (W - patch_size) / (num_x - 1)
    else:
        actual_stride_x = 0
    
    # Generate grid positions
    positions = []
    for i in range(num_y):
        for j in range(num_x):
            y = min(int(round(i * actual_stride_y)), H - patch_size)
            x = min(int(round(j * actual_stride_x)), W - patch_size)
            positions.append((y, x))
    
    return positions

def parselate_with_lpips(image, patch_size, overlap, patch_num, ssim_threshold=0.85, positions=None, overlap_map=None, cache_depth=2, device='cuda'):
    """
    Parselate an image into patches with adaptive patch location selection and divide them into
    full inference and progressive inference based on LPIPS scores.
    
    Args:
        image: torch.Tensor of shape (B, C, H, W) or (C, H, W) or (H, W)
        patch_size: Size of each patch (assumed square)
        overlap: Base overlap between patches
        patch_num: Target number of patches per image
        ssim_threshold: Threshold for SSIM score to determine inference method
        positions: Optional pre-computed patch positions
        overlap_map: Optional pre-computed overlap map
        cache_depth: Number of consecutive batch items to share the same patch locations
                    (must divide batch size evenly)
        device: Device to use for SSIM calculation ('cpu')
        
    Returns:
        full_inference_patches: Tensor of patches for full inference
        progressive_inference_patches: Tensor of patches for progressive inference
        full_inference_positions: PositionInfo object containing positions and batch indices
        progressive_inference_positions: PositionInfo object containing positions and batch indices
        similar_patch_indices: List where similar_patch_indices[i] is the index in full_inference_patches
                              that is similar to progressive_inference_patches[i]
    """
    import torch
    from itertools import chain
    
    # First, use the original parselate function to get patches, positions, and overlap maps
    patches_tensor, expanded_positions, expanded_overlap_maps = parselate(
        image, patch_size, overlap, patch_num, positions, overlap_map, cache_depth
    )
    
    # Get dimensions
    B, num_patches, C, H, W = patches_tensor.shape
    
    # Validate batch size and cache depth
    if cache_depth > B:
        cache_depth = B

    if B % cache_depth != 0:
        raise ValueError(f"Batch size ({B}) must be divisible by cache_depth ({cache_depth})")
    
    # Ensure cache_depth is at least 2 (needed for LPIPS comparison)
    if cache_depth < 2:
        raise ValueError(f"cache_depth must be at least 2 for LPIPS comparison, got {cache_depth}")
    
    # Initialize lists for full and progressive inference
    full_inference_patches = []
    progressive_inference_patches = []
    full_inference_positions = []
    progressive_inference_positions = []
    full_inference_batch_indices = []
    progressive_inference_batch_indices = []
    similar_patch_indices = []
    
    # We'll need to track all full inference patches and their original positions
    full_patches_tracking = []
    full_positions_tracking = []
    full_batch_indices_tracking = []
    
    # Process each cache group
    num_groups = B // cache_depth
    
    for group_idx in range(num_groups):
        base_idx = group_idx * cache_depth
        
        # Always add the first patch in each group to full inference
        full_inference_patches.append(patches_tensor[base_idx])
        full_inference_positions.append(expanded_positions[base_idx])
        
        # Track these patches in our tracking lists
        full_patches_tracking.extend(patches_tensor[base_idx])
        full_positions_tracking.extend(expanded_positions[base_idx])
        full_batch_indices_tracking.extend([base_idx] * len(expanded_positions[base_idx]))
        
        # Track batch indices for all patches in the reference image
        full_inference_batch_indices.extend([base_idx] * len(expanded_positions[base_idx]))
        
        # For each remaining patch in the group, calculate LPIPS with the reference patch
        # and assign to full or progressive inference based on the threshold
        for offset in range(1, cache_depth):
            compare_idx = base_idx + offset
            
            # Skip if we've reached the end of the batch
            if compare_idx >= B:
                continue
            
            # Calculate LPIPS between reference patch and current patch
            lpips_score = calculate_ssim(patches_tensor[base_idx], patches_tensor[compare_idx], return_mean=False)
            # lpips_score = calculate_lpips(patches_tensor[base_idx], patches_tensor[compare_idx], return_mean=False,  device=device)
            
            # Ensure lpips_score is a PyTorch tensor
            if not isinstance(lpips_score, torch.Tensor):
                lpips_score = torch.tensor(lpips_score, device=device)
            
            # Create masks for high and low similarity
            high_dissimilarity_mask = lpips_score < ssim_threshold
            low_dissimilarity_mask = lpips_score >= ssim_threshold
            
            # Convert to boolean mask if needed
            if high_dissimilarity_mask.dtype != torch.bool:
                high_dissimilarity_mask = high_dissimilarity_mask.bool()
            if low_dissimilarity_mask.dtype != torch.bool:
                low_dissimilarity_mask = low_dissimilarity_mask.bool()
                
            # Add patches with high dissimilarity to full inference
            if torch.any(high_dissimilarity_mask):
                # Extract patches that exceed the threshold
                high_patches = patches_tensor[compare_idx][high_dissimilarity_mask.cpu()]
                
                # Only append if there are any patches
                if len(high_patches) > 0:
                    full_inference_patches.append(high_patches)
                    
                    # Convert mask to numpy for indexing lists
                    high_mask_np = high_dissimilarity_mask.cpu().numpy()
                    
                    # Also extract corresponding positions
                    high_positions = [pos for i, pos in enumerate(expanded_positions[compare_idx]) 
                                     if i < len(high_mask_np) and high_mask_np[i]]
                    
                    full_inference_positions.append(high_positions)
                    
                    # Track these patches in our tracking lists
                    full_patches_tracking.extend(high_patches)
                    full_positions_tracking.extend(high_positions)
                    full_batch_indices_tracking.extend([compare_idx] * len(high_positions))
                    
                    # Track batch indices for high dissimilarity patches
                    full_inference_batch_indices.extend([compare_idx] * len(high_positions))
            
            # Add patches with low dissimilarity to progressive inference
            if torch.any(low_dissimilarity_mask):
                # Extract patches that are below the threshold
                low_patches = patches_tensor[compare_idx][low_dissimilarity_mask.cpu()]
                
                # Only append if there are any patches
                if len(low_patches) > 0:
                    progressive_inference_patches.append(low_patches)
                    
                    # Convert mask to numpy for indexing lists
                    low_mask_np = low_dissimilarity_mask.cpu().numpy()
                    
                    # Also extract corresponding positions
                    low_positions = [pos for i, pos in enumerate(expanded_positions[compare_idx]) 
                                    if i < len(low_mask_np) and low_mask_np[i]]
                    
                    progressive_inference_positions.append(low_positions)
                    
                    # Track batch indices for low dissimilarity patches
                    progressive_inference_batch_indices.extend([compare_idx] * len(low_positions))
                    
                    # For each low similarity patch, find the corresponding similar patch in full_inference_patches
                    # These are the patches from the base_idx that have low LPIPS score
                    for i in range(len(low_mask_np)):
                        if low_mask_np[i]:
                            # Find the corresponding patch in the base_idx patches
                            # We need to find which patch in full_patches_tracking matches this
                            # We can use the position information to match them
                            current_position = expanded_positions[compare_idx][i]
                            
                            # Find the patch in base_idx with the same position
                            for j, pos in enumerate(expanded_positions[base_idx]):
                                if pos == current_position:
                                    # Now find where this base_idx patch is in full_patches_tracking
                                    for k, (track_pos, track_batch) in enumerate(zip(full_positions_tracking, full_batch_indices_tracking)):
                                        if track_pos == pos and track_batch == base_idx:
                                            similar_patch_indices.append(k)
                                            break
                                    break
    
    # Stack patches if there are any
    if full_inference_patches:
        full_inference_patches = torch.cat(full_inference_patches, dim=0)
    else:
        full_inference_patches = torch.empty((0, C, H, W), device=patches_tensor.device)
        
    if progressive_inference_patches:
        progressive_inference_patches = torch.cat(progressive_inference_patches, dim=0)
    else:
        progressive_inference_patches = torch.empty((0, C, H, W), device=patches_tensor.device)
    
    # Flatten position lists
    full_inference_positions_flat = list(chain(*full_inference_positions))
    progressive_inference_positions_flat = list(chain(*progressive_inference_positions))
    
    # Create PositionInfo objects to hold positions and batch indices
    class PositionInfo:
        def __init__(self, positions, batch_indices):
            self.positions = positions
            self.batch_indices = batch_indices
    
    full_info = PositionInfo(full_inference_positions_flat, full_inference_batch_indices)
    progressive_info = PositionInfo(progressive_inference_positions_flat, progressive_inference_batch_indices)
    
    return full_inference_patches, progressive_inference_patches, full_info, progressive_info, similar_patch_indices

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

def patch_inference(model, device, scheduler, patches, positions_info, original_shape,
                 progressive_inference=False, noise_level=0.7, use_half=True,
                 use_deepcache=True, cache_interval=20, cache_branch_id=0, patch_ref=None):
    """
    Patch-based inference for large images using pre-computed patches with optional DeepCache optimization.
    Processes all patches in parallel and reconstructs them by batch index.
    
    Args:
        model: The model to use for inference
        device: Computation device
        scheduler: Diffusion scheduler
        patches: Pre-computed patches tensor with shape (num_total_patches, C, patch_size, patch_size)
        positions_info: PositionInfo object containing positions and batch indices
        original_shape: Original shape of the input image (B, C, H, W)
        progressive_inference (bool): Whether to use progressive inference
        noise_level (float): Noise level for progressive inference
        use_half (bool): Whether to use half precision
        use_deepcache (bool): Whether to use DeepCache optimization
        cache_interval (int): How often to update the cache (N value), only used if use_deepcache=True
        cache_branch_id (int): Which branch to use for caching (0=shallowest, higher=deeper), only used if use_deepcache=True
    """
    import torch
    from tqdm import tqdm
    
    model.eval()
    
    # Initialize DeepCache if enabled
    deepcache = None
    if use_deepcache:
        deepcache = MonaiDeepCache(model, cache_interval=cache_interval, cache_branch_id=cache_branch_id)
        deepcache.enable()
    
    try:
        # Get dimensions
        num_total_patches, C, patch_size, patch_size_h = patches.shape
        B, _, H, W = original_shape
        
        # Move patches to device & cast to half if needed
        device_patches = patches.to(device)
        if use_half:
            device_patches = device_patches.half()
        
        # --- Parallel Inference Logic ---
        if progressive_inference:
            current_img = patch_ref.clone()
            noise_timestep = int(noise_level * len(scheduler.timesteps))
            noise_timestep = max(min(noise_timestep, len(scheduler.timesteps)-1), 0)
            
            # Generate noise for all patches at once
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
            noise = torch.randn_like(device_patches, device=device)
            if use_half:
                noise = noise.half()
            current_img = noise
            timesteps = scheduler.timesteps
        
        # --- Denoising Loop ---
        with torch.no_grad():
            progress_bar = tqdm(timesteps, desc=f"Processing {num_total_patches} patches")
            
            for t in progress_bar:
                # DeepCache pre-step preparation if enabled
                if use_deepcache:
                    deepcache.pre_step()
                    
                    # Update progress bar with mode for DeepCache
                    if deepcache.step_counter % cache_interval == 0:
                        progress_bar.set_postfix({"mode": "full  "})
                    else:
                        progress_bar.set_postfix({"mode": "cached"})
                
                # Process all patches at once
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_half):
                    timestep_batch = torch.full((num_total_patches,), t, device=device, dtype=torch.long)
                    
                    # Prepare input by combining original patches and current denoised images
                    try:
                        combined_input = torch.cat([device_patches, current_img], dim=1)
                    except:
                        pdb.set_trace()
                    
                    # Forward pass through the model with all patches at once
                    model_output = model(combined_input, timesteps=timestep_batch)
                    
                    # Apply scheduler step to all patches at once
                    current_img, _ = scheduler.step(model_output, t, current_img)
        
        # Move processed patches back to CPU
        processed_patches = current_img.cpu()

        # --- Reconstruct Full Images ---
        final_output = torch.zeros((B, C, H, W), device="cpu")
        count = torch.zeros_like(final_output, device="cpu")
        
        # Get the extracted batch indices from the positions_info
        batch_indices = positions_info.batch_indices
        positions = positions_info.positions
        
        # Place each patch in its position in the appropriate batch output
        for idx, (y, x) in enumerate(positions):
            if idx >= num_total_patches:
                break
                
            patch = processed_patches[idx]
            batch_idx = batch_indices[idx]
            
            # Ensure we're within bounds
            y_end = min(y + patch_size, H)
            x_end = min(x + patch_size, W)
            
            # Handle edge cases
            if y_end - y < patch_size:
                y = max(0, y_end - patch_size)
            if x_end - x < patch_size:
                x = max(0, x_end - patch_size)
                
            final_output[batch_idx, :, y:y+patch_size, x:x+patch_size] += patch
            count[batch_idx, :, y:y+patch_size, x:x+patch_size] += 1
        
        # Average overlapping regions
        non_zero_mask = (count > 0)
        final_output[non_zero_mask] /= count[non_zero_mask]
        
        return final_output.to(device), current_img
    
    finally:
        # Always disable DeepCache if it was enabled
        if use_deepcache and deepcache is not None:
            deepcache.disable()

def inference_patch_idx(full_inference_positions, progressive_inference_positions, inference_batch_id, cache_depth=2):
    """
    Find matching position indices between full inference and progressive inference for a specific batch.
    
    Args:
        full_inference_positions: PositionInfo object with positions and batch_indices attributes
        progressive_inference_positions: PositionInfo object with positions and batch_indices attributes
        inference_batch_id: Base batch ID to use for full inference
        cache_depth: Depth of the cache (default: 2)
    
    Returns:
        tuple: (matching_indices, original_indices) where:
            - matching_indices: Local indices in the batch_positions array
            - original_indices: Global indices in the full positions array
    """
    import numpy as np
    
    # Convert to numpy arrays
    full_positions = np.array(full_inference_positions.positions)
    full_batch_indices = np.array(full_inference_positions.batch_indices)
    
    prog_positions = np.array(progressive_inference_positions.positions)
    prog_batch_indices = np.array(progressive_inference_positions.batch_indices)
    
    # Filter positions for the specified batches
    base_batch_mask = (full_batch_indices == inference_batch_id)
    comp_batch_mask = (prog_batch_indices == inference_batch_id + cache_depth - 1)
    
    # Get positions for the specified batches
    base_batch_positions = full_positions[base_batch_mask]
    comp_batch_positions = prog_positions[comp_batch_mask]
    
    # Get original indices
    base_original_indices = np.where(base_batch_mask)[0]
    
    # Initialize mask for matches
    matches = np.zeros(len(base_batch_positions), dtype=bool)
    
    # For each position in the comparison batch, find matches in the base batch
    for pos in comp_batch_positions:
        current_matches = np.all(base_batch_positions == pos, axis=1)
        matches = matches | current_matches
    
    # Get the indices of matches
    matching_indices = np.where(matches)[0]
    
    # Map local indices to original indices
    original_indices = base_original_indices[matching_indices]
    
    return matching_indices, original_indices

def inference_patch_idx_all(full_inference_positions, progressive_inference_positions, cache_depth=2):
    """
    Find all matching position indices across all batches.
    
    Args:
        full_inference_positions: PositionInfo object with positions and batch_indices attributes
        progressive_inference_positions: PositionInfo object with positions and batch_indices attributes
        cache_depth: Depth of the cache (default: 2)
    
    Returns:
        dict: A dictionary containing:
            - 'local_indices': List of local indices in their respective batch arrays
            - 'global_indices': List of global indices in the full positions array
            - 'batch_mapping': Dictionary mapping batch ID to indices in the returned lists
    """
    import numpy as np
    
    # Get unique batch IDs from full inference
    full_batch_indices = np.array(full_inference_positions.batch_indices)
    unique_batches = np.unique(full_batch_indices)
    
    # Calculate number of cache groups
    max_batch = max(unique_batches)
    cache_groups = (max_batch + 1) // cache_depth
    
    # Initialize result containers
    local_indices = []
    global_indices = []
    batch_mapping = {}
    
    # Process each cache group
    for group_idx in range(cache_groups):
        base_batch_id = group_idx * cache_depth
        
        # Skip if this batch doesn't exist in our data
        if base_batch_id not in unique_batches:
            continue
            
        # Track start index for this batch in our result lists
        batch_mapping[base_batch_id] = len(local_indices)
        
        # Get matching indices for this batch
        batch_local, batch_global = inference_patch_idx(
            full_inference_positions, 
            progressive_inference_positions, 
            base_batch_id, 
            cache_depth
        )
        
        # Add to our result lists
        local_indices.extend(batch_local)
        global_indices.extend(batch_global)
    
    return {
        'local_indices': local_indices,
        'global_indices': global_indices,
        'batch_mapping': batch_mapping
    }

def combine_inference_results(full_results, progressive_results, full_inference_positions, progressive_inference_positions, patch_size):
    """
    Combine full inference and progressive inference results into a single output with improved edge handling.
    
    Args:
        full_results: Tensor with full inference results, shape (B, C, H, W)
        progressive_results: Tensor with progressive inference results, shape (B, C, H, W)
        full_inference_positions: PositionInfo object for full inference
        progressive_inference_positions: PositionInfo object for progressive inference
        patch_size: Size of patches used for inference
        
    Returns:
        combined_results: Tensor with combined results, shape (B, C, H, W)
    """
    import torch
    import numpy as np
    
    # Get batch dimensions
    B, C, H, W = full_results.shape
    
    # Initialize tracking tensors for both methods
    full_mask = torch.zeros((B, 1, H, W), device=full_results.device)
    prog_mask = torch.zeros((B, 1, H, W), device=progressive_results.device)
    
    # Initialize the result tensor with zeros
    combined_results = torch.zeros_like(full_results)
    
    # Initialize weight accumulation tensors
    full_weights = torch.zeros((B, 1, H, W), device=full_results.device)
    prog_weights = torch.zeros((B, 1, H, W), device=progressive_results.device)
    
    # Generate window function once - use Hann window for smoother blending
    window_1d = torch.hann_window(patch_size, periodic=False, device=full_results.device)
    # Create 2D window by outer product
    window_2d = window_1d.unsqueeze(1) * window_1d.unsqueeze(0)
    
    # Fill in results from full inference
    for idx, ((y, x), batch_idx) in enumerate(zip(full_inference_positions.positions, full_inference_positions.batch_indices)):
        # Calculate effective patch coordinates with proper boundary handling
        y_start = y
        x_start = x
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        
        # Calculate patch dimensions
        patch_h = y_end - y_start
        patch_w = x_end - x_start
        
        # Get the corresponding section of the pre-computed window
        weight_grid = window_2d[:patch_h, :patch_w]
        
        # Update the full inference region
        h_slice = slice(y_start, y_end)
        w_slice = slice(x_start, x_end)
        
        # Apply weighted contribution
        full_weights[batch_idx, 0, h_slice, w_slice] += weight_grid
        
        # For each channel, apply the weighted contribution
        for c in range(C):
            patch_data = full_results[batch_idx, c, h_slice, w_slice]
            combined_results[batch_idx, c, h_slice, w_slice] += patch_data * weight_grid
        
        # Mark this region as filled by full inference
        full_mask[batch_idx, 0, h_slice, w_slice] = 1
    
    # Fill in results from progressive inference
    for idx, ((y, x), batch_idx) in enumerate(zip(progressive_inference_positions.positions, progressive_inference_positions.batch_indices)):
        # Calculate effective patch coordinates with proper boundary handling
        y_start = y
        x_start = x
        y_end = min(y + patch_size, H)
        x_end = min(x + patch_size, W)
        
        # Calculate patch dimensions
        patch_h = y_end - y_start
        patch_w = x_end - x_start
        
        # Get the corresponding section of the pre-computed window
        weight_grid = window_2d[:patch_h, :patch_w]
        
        # Update the progressive inference region
        h_slice = slice(y_start, y_end)
        w_slice = slice(x_start, x_end)
        
        # Apply a gradually decreasing weight for progressive results where full results exist
        # This helps create smoother transitions between the two result types
        overlap_factor = torch.ones((patch_h, patch_w), device=progressive_results.device)
        if torch.any(full_mask[batch_idx, 0, h_slice, w_slice] > 0):
            # Create a smooth transition in overlapping regions
            overlap_factor = torch.clamp(1.0 - full_mask[batch_idx, 0, h_slice, w_slice] * 0.8, 0.2, 1.0)
        
        effective_weight = weight_grid * overlap_factor
        
        # Apply weighted contribution
        prog_weights[batch_idx, 0, h_slice, w_slice] += effective_weight
        
        # For each channel, apply the weighted contribution
        for c in range(C):
            patch_data = progressive_results[batch_idx, c, h_slice, w_slice]
            combined_results[batch_idx, c, h_slice, w_slice] += patch_data * effective_weight
    
    # Normalize by accumulated weights to get weighted average, with epsilon to avoid division by zero
    epsilon = 1e-6
    total_weights = full_weights + prog_weights + epsilon
    
    # Normalize results by the total weights
    for b in range(B):
        for c in range(C):
            combined_results[b, c] /= total_weights[b, 0]
    
    # Apply edge feathering to reduce artifacts
    # Create feathering masks for image boundaries
    feather_size = min(16, patch_size // 4)  # Size of feathering region
    
    # Create horizontal feathering masks
    left_feather = torch.linspace(0.5, 1.0, feather_size, device=full_results.device)
    right_feather = torch.linspace(1.0, 0.5, feather_size, device=full_results.device)
    
    # Create vertical feathering masks
    top_feather = torch.linspace(0.5, 1.0, feather_size, device=full_results.device)
    bottom_feather = torch.linspace(1.0, 0.5, feather_size, device=full_results.device)
    
    # Apply feathering to the edges
    for b in range(B):
        # Left edge
        combined_results[b, :, :, :feather_size] *= left_feather.view(1, 1, -1)
        # Right edge
        combined_results[b, :, :, -feather_size:] *= right_feather.view(1, 1, -1)
        # Top edge
        combined_results[b, :, :feather_size, :] *= top_feather.view(1, -1, 1)
        # Bottom edge
        combined_results[b, :, -feather_size:, :] *= bottom_feather.view(1, -1, 1)
    
    return combined_results

def parse_arguments():
    """
    Parse and validate all input arguments for the main function.
    Returns a namespace object containing all parameters.
    """
    parser = argparse.ArgumentParser(description="MRI Denoising with Diffusion Models")
    
    # Model configuration
    parser.add_argument('--checkpoint_path', type=str, 
                       default="/ix3/tibrahim/jil202/cfg_gen/src/training/checkpoints/320/0.5/patch_data_aug_lpips_1_320_ssim_8.14.pt",
                       help="Path to the pretrained model checkpoint")
    parser.add_argument('--image_path', type=str,
                       default='/ix3/tibrahim/jil202/cfg_gen/qc_image_nii/denoised_mp2rage/IBRA-BIO1/PRT170058/converted/MP2RAGE_UNI_Images/md_denoised_PRT170058_20180912172034.nii.gz', #md_12898_20230628104454.nii.gz, /ix3/tibrahim/jil202/cfg_gen/qc_image_nii/denoised_mp2rage/IBRA-BIO1/PRT170058/converted/MP2RAGE_UNI_Images/md_denoised_PRT170058_20180912172034.nii.gz
                       help="Path to the input NIfTI image")
    
    # DeepCache parameters
    parser.add_argument('--cache_interval', type=int, default=1,
                       help="Cache interval value (higher means more speedup but potentially lower quality)")
    parser.add_argument('--cache_branch_id', type=int, default=0,
                       help="Branch ID (0=shallowest/fastest, higher=deeper/better quality)")
    parser.add_argument('--cache_depth', type=int, default=4,
                       help="Cache depth for progressive inference")
    parser.add_argument('--ssim_threshold', type=float, default=0.99,
                       help="ssim threshold for patch selection")
    
    # Patch parameters
    parser.add_argument('--patch_size', type=int, default=128,
                       help="Size of patches for processing")
    parser.add_argument('--target_patches', type=int, default=None,
                       help="Target number of patches")
    parser.add_argument('--base_overlap', type=int, default=32,
                       help="Base overlap between patches")
    parser.add_argument('--scale_factor', type=float, default=0.5,
                       help="scale_factor")
    
    # Scheduler parameters
    parser.add_argument('--ddim', action='store_true', default=False,
                       help="Use DDIM scheduler instead of DDPMScheduler")
    parser.add_argument('--ddpm', action='store_true', default=False,
                       help="Use DDPM scheduler instead of DDIMScheduler")
    parser.add_argument('--inference_step', type=int, default=1000,
                       help="Number of inference steps")
    
    # Processing parameters
    parser.add_argument('--batch_size', type=int, default=8,
                       help="Batch size for processing")
    
    parser.add_argument('--dimensions', type=str, nargs='+', 
                       default=['axial', 'sagittal', 'coronal'],
                       help="Dimensions to process (axial, sagittal, coronal)")
    
    # Performance optimization flags
    parser.add_argument('--use_xformers', action='store_true', default=True,
                       help="Enable xformers for memory-efficient attention")
    parser.add_argument('--no_xformers', dest='use_xformers', action='store_false',
                       help="Disable xformers")
    parser.add_argument('--use_half', action='store_true', default=True,
                       help="Enable half precision (FP16)")
    parser.add_argument('--no_half', dest='use_half', action='store_false',
                       help="Disable half precision")
    parser.add_argument('--use_model_compile', action='store_true', default=True,
                       help="Enable model compilation (torch.compile)")
    parser.add_argument('--channels_last', action='store_true', default=False,
                       help="Use channels-last memory format")
    
    # Output options
    parser.add_argument('--baseline', action='store_true', default=False,)
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help="Directory to save output files")
    parser.add_argument('--save', action='store_true', default=False,
                       help="Directory to save output files")
    
    args = parser.parse_args()
      
    if args.cache_interval < 1:
        raise ValueError("cache_interval must be at least 1")
    
    if not 0 <= args.ssim_threshold <= 1:
        raise ValueError("ssim_threshold must be between 0 and 1")
    
    if args.ddim and args.inference_step == 1000:
        raise ValueError("DDIM requires fewer than 1000 inference steps")
    
    return args

def main():
    import time
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_arguments()
    checkpoint_path = args.checkpoint_path 
    image_path = args.image_path

    cache_interval = args.cache_interval    # N value - higher means more speedup but potentially lower quality
    cache_branch_id = args.cache_branch_id   # Branch ID - 0=shallowest (fastest), higher=deeper (better quality)
    cache_depth = args.cache_depth           # Cache depth for progressive inference
    ssim_threshold = args.ssim_threshold
    patch_size = args.patch_size
    target_patches = args.target_patches
    base_overlap = args.base_overlap
    batch_size = args.batch_size
    inference_step = args.inference_step
    # Performance optimization settings
    use_xformers = args.use_xformers   # Enable xformers for memory-efficient attention
    use_half = args.use_half  # Enable half precision (FP16)
    use_model_compile = args.use_model_compile  # Enable model compilation (torch.compile)
    channels_last = args.channels_last
        
    # Load model with optimizations
    print("Loading model with xformers and half precision...")
    model = load_model(checkpoint_path, device="cuda", use_xformers=use_xformers, use_half=use_half, torch_compile=use_model_compile, channels_last=channels_last)
    if args.ddpm:
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        inference_step = 1000
    elif args.ddim:
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(num_inference_steps=inference_step)
    # Load and preprocess example input (single slice for benchmarking)
    input_nii = nib.load(image_path)
    data = quantile_normalization(input_nii, lower_quantile=0.01, upper_quantile=0.99)
    data = crop_mri_volume(torch.tensor(data)).unsqueeze(0).to(device).squeeze()
    final_outputs_dimensions = []

    start_time = time.time()
    for dimension in args.dimensions: 
        dataset = VolumeDataset(data, dimension, scale_factor=args.scale_factor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
        # Run benchmarks with optimizations
        print("\n--- Running inference benchmarks with xformers and half precision ---")
        cache_outputs = []
        final_outputs = []
        orig_inputs = []
        input_slices = []
        
        for idx, batch_data in enumerate(tqdm(dataloader)):
            input_slice = batch_data[0]
            orig_slice = batch_data[1]
            print(f"---------{dimension} | {idx}/{len(dataloader)}--------")
            for _ in range(2):
                if args.baseline:
                    gpu_memory_tracker()
                    cache_output = deepcache_inference(model, device, scheduler, input_slice, progressive_inference=True, noise_level=0.5, inference_step=inference_step, cache_interval=cache_interval, cache_branch_id=cache_branch_id, use_half=use_half)
                    cache_outputs.append(cache_output.detach().cpu())
                    gpu_memory_tracker()
                else:
                    gpu_memory_tracker()
                    full_inference_patches, progressive_inference_patches, full_inference_positions, progressive_inference_positions, patch_ids = parselate_with_lpips(input_slice, patch_size, overlap=base_overlap, patch_num=target_patches, ssim_threshold=ssim_threshold, positions=None, overlap_map=None, cache_depth=cache_depth, device='cuda')      
                    print(f'{idx} | Progressive patches: {len(progressive_inference_patches)}, Full patches: {len(full_inference_patches)}, Patch ref: {len(patch_ids)}')
                    full_results, patch_ref = patch_inference(model, device, scheduler, full_inference_patches, full_inference_positions, cache_interval=cache_interval, cache_branch_id=cache_branch_id, progressive_inference=True, noise_level=0.5, original_shape=input_slice.shape, patch_ref=full_inference_patches.half())
                    assert len(patch_ref[patch_ids]) == len(progressive_inference_patches)
                    progressive_results, _ = patch_inference(model, device, scheduler, progressive_inference_patches, progressive_inference_positions, cache_interval=cache_interval, progressive_inference=True, noise_level=0.8, original_shape=input_slice.shape, patch_ref=patch_ref[patch_ids])
                    ensemble_results = combine_inference_results(full_results, progressive_results, full_inference_positions, progressive_inference_positions, patch_size)                
                    # cache_output = deepcache_inference(model, device, scheduler, input_slice, inference_step=inference_step, cache_interval=cache_interval, cache_branch_id=cache_branch_id, use_half=use_half)
                    # full_inference_patches, progressive_inference_patches, full_inference_positions, progressive_inference_positions = parselate_with_lpips(input_slice, patch_size, overlap=32, patch_num=target_patches, lpips_threshold=0.01, positions=None, overlap_map=None, cache_depth=2, device='cuda')
                    # full_results, patch_ref = patch_inference(model, device, scheduler, full_inference_patches, full_inference_positions, cache_interval=cache_interval, progressive_inference=False, original_shape=input_slice.shape)
                    # evaluate_image_quality(orig_slice.cpu().numpy(), ensemble_results.cpu().numpy())
                    cache_outputs.append(ensemble_results.detach().cpu())
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

    elapsed_time = time.time() - start_time

    final_outputs = np.stack(final_outputs_dimensions).mean(0)
    
    input_slices = torch.tensor(np.squeeze(np.stack(input_slices))).permute(1,-1,0).numpy()
    cache_metrics = evaluate_image_quality(final_outputs.transpose(-1,0,1)[:,None,...], data.cpu().numpy().transpose(-1,0,1)[:,None,...])
    print(f'Time passed: {elapsed_time:.2f} seconds')
    print(f'SSIM: {args.ssim_threshold}, cache_depth: {args.cache_depth}, inference_step: {args.inference_step}')
    print(cache_metrics)

    if args.save:
        data = tio.transforms.CropOrPad(input_nii.shape)(final_outputs[None]).squeeze()
        nib.save(nib.Nifti1Image(data, affine=input_nii.affine), f'cached_output.nii.gz')
    # final_outputs = tio.transforms.CropOrPad(input_nii.shape)(final_outputs[None]).squeeze()
    # input_slices = tio.transforms.CropOrPad(input_nii.shape)(input_slices[None]).squeeze()

    
    # nib.save(nib.Nifti1Image(np.round(final_outputs * 1000), affine=input_nii.affine), './3dFWHMx/cache_output.nii.gz')
    # nib.save(nib.Nifti1Image(data, affine=input_nii.affine), 'cache_input.nii.gz')
    # nib.save(nib.Nifti1Image(input_slices, affine=input_nii.affine), 'lowres_1.57mm.nii.gz')
    
if __name__ == "__main__":
    main()