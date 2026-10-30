import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler
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

def load_model(checkpoint_path, device="cuda", use_xformers=True, use_half=True):
    """Load the pretrained diffusion model with xformers and half precision support"""
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,  # T1w + noise
        out_channels=1,  # TSE
        channels=(256, 256, 512),
        attention_levels=(False, False, True),  # Attention only in last level
        num_res_blocks=2,
        num_head_channels=512,
        with_conditioning=False,
    ).to(device)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Enable xformers efficient attention if requested
    if use_xformers:
        replace_unet_attention_with_xformers(model)
    
    # Convert to half precision if requested
    if use_half:
        model = model.half()
    
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

def deepcache_inference(model, device, scheduler, t1w, cache_interval=5, cache_branch_id=0, 
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
            input_img = t1w
            if use_half:
                input_img = input_img.half()
                
            scheduler.set_timesteps(num_inference_steps=1000)
            
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
    checkpoint_path = "/ix3/tibrahim/jil202/cfg_gen/src/training/checkpoints/320/0.5/anatomy_scheduler_False_lpips_1_320_ssim_8.46.pt" # 
    
    # DeepCache parameters to test
    cache_interval = 20    # N value - higher means more speedup but potentially lower quality
    cache_branch_id = 0   # Branch ID - 0=shallowest (fastest), higher=deeper (better quality)
    
    # Performance optimization settings
    use_xformers = True   # Enable xformers for memory-efficient attention
    use_half = True  # Enable half precision (FP16)
    
    # Load model with optimizations
    print("Loading model with xformers and half precision...")
    model = load_model(checkpoint_path, device, use_xformers=use_xformers, use_half=use_half)
    model.eval()
    
    # Load and preprocess example input (single slice for benchmarking)
    input_nii = nib.load('/ix3/tibrahim/jil202/cfg_gen/qc_image_nii/denoised_mp2rage/IBRA-BIO1/PRT170058/converted/MP2RAGE_UNI_Images/md_denoised_PRT170058_20180912172034.nii.gz')
    input_nii = tio.transforms.CropOrPad((384,384,384))(input_nii)
    data = quantile_normalization(input_nii, lower_quantile=0.01, upper_quantile=0.99)
    data = torch.tensor(data).to(device)
    
    # Select a single slice for testing
    test_slice = torch.rot90(data[120,:,:], 1).unsqueeze(0).unsqueeze(0).float().to(device)
    input_slice = downsample_upsample(test_slice, scale_factor=0.5, mode='bilinear')
    
    # Initialize scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Run benchmarks with optimizations
    print("\n--- Running inference benchmarks with xformers and half precision ---")
    
    # Vanilla inference with optimizations
    start_time = time.time()
    vanilla_output = vanilla_inference(model, device, scheduler, input_slice, 
                                       progressive_inference=True, noise_level=0.7, use_half=use_half)
    vanilla_time = time.time() - start_time
    print(f"Vanilla inference time: {vanilla_time:.2f} seconds")
    
    # DeepCache inference with optimizations
    start_time = time.time()
    deepcache_outputs = []
    for _ in range(5):
        cache_output = deepcache_inference(model, device, scheduler, input_slice, 
                                        cache_interval=cache_interval, cache_branch_id=cache_branch_id,
                                        use_half=use_half, progressive_inference=True, noise_level=0.7)
        deepcache_outputs.append(cache_output)
    cache_time = time.time() - start_time
    print(f"DeepCache inference time: {cache_time:.2f} seconds")
    print(f"Speedup: {vanilla_time / cache_time:.2f}x")

    cache_output = torch.stack(deepcache_outputs).squeeze().mean(0)
    # Compare image quality

    # ssim_input, psnr, lpips = evaluate_image_quality(torch.clamp(input_slice.squeeze(), 0, 1).detach().cpu().numpy(), test_slice.squeeze().detach().cpu().numpy())
    vanilla_metrics = evaluate_image_quality(torch.clamp(vanilla_output.squeeze(), 0, 1).detach().cpu().numpy(), test_slice.squeeze().detach().cpu().numpy())
    ssim_vanilla, psnr_vanilla, lpips_vanilla = vanilla_metrics['SSIM'], vanilla_metrics['PSNR'], vanilla_metrics['LPIPS']
    cached_metrics = evaluate_image_quality(torch.clamp(cache_output.squeeze(), 0, 1).detach().cpu().numpy(), test_slice.squeeze().detach().cpu().numpy())
    ssim_cached, psnr_cached, lpips_cached = cached_metrics['SSIM'], cached_metrics['PSNR'], cached_metrics['LPIPS']
    print(f"Quality comparison - Vanilla: PSNR: {psnr_vanilla:.2f} dB, SSIM: {ssim_vanilla:.4f}, LPIPS: {lpips_vanilla:.4f}")
    torchvision.utils.save_image(torch.hstack((torch.clamp(input_slice.squeeze(), 0, 1).detach().cpu(), 
                                               torch.clamp(vanilla_output.squeeze(), 0, 1).detach().cpu(), 
                                               test_slice.squeeze().detach().cpu())), f'vanilla_inference_use_xformers_{use_xformers}_half_precision_{use_half}.png')
    
    print(f"Quality comparison - Cached: PSNR: {psnr_cached:.2f} dB, SSIM: {ssim_cached:.4f}, LPIPS: {lpips_cached:.4f}")
    torchvision.utils.save_image(torch.hstack((torch.clamp(input_slice.squeeze(), 0, 1).detach().cpu(), 
                                               torch.clamp(cache_output.squeeze(), 0, 1).detach().cpu(), 
                                               test_slice.squeeze().detach().cpu())), f'Cached_inference_use_xformers_{use_xformers}_half_precision_{use_half}.png')

if __name__ == "__main__":
    main()