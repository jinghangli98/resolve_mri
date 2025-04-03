import os
import time
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

def load_model(checkpoint_path, device="cuda"):
    """Load the pretrained diffusion model"""
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
    
    return model

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

def vanilla_inference(model, device, scheduler, t1w, progressive_inference=False, noise_level=0.7):
    """
    Standard inference without DeepCache optimization.
    """
    model.eval()
    
    with torch.no_grad():
        # Setup
        input_img = t1w
        scheduler.set_timesteps(num_inference_steps=1000)

        if progressive_inference:
            current_img = input_img.clone()
            noise_timestep = int(noise_level * len(scheduler.timesteps))
            noise_timestep = max(min(noise_timestep, len(scheduler.timesteps)-1), 0)
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
            current_img = noise  # for the TSE image, we start from random noise.
            timesteps = scheduler.timesteps

        progress_bar = tqdm(timesteps, desc="Vanilla Inferencing")

        for t in progress_bar:  # go through the noising process
            with autocast("cuda", enabled=False):
                combined = torch.cat((input_img, current_img), dim=1)
                model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                current_img, _ = scheduler.step(model_output, t, current_img)
                    
        return current_img

def deepcache_inference(model, device, scheduler, t1w, cache_interval=5, cache_branch_id=0, progressive_inference=False, noise_level=0.7):
    """
    Run inference with DeepCache optimization for faster generation.
    
    Args:
        model: The diffusion model
        device: Device to run inference on
        scheduler: Diffusion scheduler
        t1w: Input image
        cache_interval: How often to update the cache (N value)
        cache_branch_id: Which branch to use for caching (0=shallowest, higher=deeper)
        progressive_inference: Whether to use progressive inference
        noise_level: Progressive inference noise level (0.0 to 1.0)
    
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
            scheduler.set_timesteps(num_inference_steps=1000)
            
            # Initial setup for progressive inference if enabled
            if progressive_inference:
                current_img = input_img.clone()
                noise_timestep = int(noise_level * len(scheduler.timesteps))
                noise_timestep = max(min(noise_timestep, len(scheduler.timesteps)-1), 0)
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
                current_img = noise  # for the TSE image, we start from random noise
                timesteps = scheduler.timesteps
            
            # Main inference loop
            progress_bar = tqdm(timesteps, desc="DeepCache Inferencing")
            
            for t in progress_bar:
                # Prepare for this step
                deepcache.pre_step()
                
                # Update progress bar with mode
                if deepcache.step_counter % cache_interval == 0:
                    progress_bar.set_postfix({"mode": "full"})
                else:
                    progress_bar.set_postfix({"mode": "cached"})
                    
                with autocast("cuda", enabled=False):
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
                
                with autocast("cuda", enabled=False):
                    combined = torch.cat((input_img, current_img), dim=1)
                    
                    # Run the model with DeepCache
                    model_output = model(combined, timesteps=torch.Tensor((t,)).to(device))
                    
                    # Update the noisy image
                    current_img, _ = scheduler.step(model_output, t, current_img)
                        
            return current_img
    finally:
        # Always restore the original model
        deepcache.disable()

def benchmark_inference(model, device, scheduler, t1w, 
                        cache_intervals=[1, 2, 5, 10, 20], 
                        branch_ids=[0, 1, 2]):
    """
    Benchmark the inference speed and quality with different configurations.
    
    Args:
        model: The diffusion model
        device: Device to run inference on
        scheduler: Diffusion scheduler
        t1w: Input image
        cache_intervals: List of cache intervals to test
        branch_ids: List of branch IDs to test
        
    Returns:
        dict: Results including timing and outputs for each method
    """
    results = {
        'vanilla': {'time': None, 'output': None},
    }
    
    for interval in cache_intervals:
        for branch_id in branch_ids:
            config_name = f'deepcache_N{interval}_B{branch_id}'
            results[config_name] = {'time': None, 'output': None}
    
    # Test vanilla inference
    print("Running vanilla inference...")
    start_time = time.time()
    vanilla_output = vanilla_inference(model, device, scheduler, t1w)
    vanilla_time = time.time() - start_time
    results['vanilla']['time'] = vanilla_time
    results['vanilla']['output'] = vanilla_output
    print(f"Vanilla inference time: {vanilla_time:.2f}s")
    
    # Test DeepCache with different configurations
    for interval in cache_intervals:
        for branch_id in branch_ids:
            config_name = f'deepcache_N{interval}_B{branch_id}'
            print(f"Running DeepCache with interval={interval}, branch_id={branch_id}...")
            
            start_time = time.time()
            cache_output = non_uniform_deepcache_inference(
                model, device, scheduler, t1w, 
                cache_interval=interval, 
                cache_branch_id=branch_id,
            )
            cache_time = time.time() - start_time
            speedup = vanilla_time / cache_time
            
            results[config_name]['time'] = cache_time
            results[config_name]['output'] = cache_output
            
            print(f"DeepCache (N={interval}, B={branch_id}): {cache_time:.2f}s, Speedup: {speedup:.2f}×")
    
    return results

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/ix3/tibrahim/jil202/cfg_gen/src/training/checkpoints/320/t1w_to_tse_model_320_ssim_8.36.pt"
    
    # DeepCache parameters to test
    cache_interval = 5    # N value - higher means more speedup but potentially lower quality
    cache_branch_id = 0   # Branch ID - 0=shallowest (fastest), higher=deeper (better quality)
    
    # Load model
    print("Loading model...")
    model = load_model(checkpoint_path, device)
    model.eval()
    
    # Load and preprocess example input (single slice for benchmarking)
    input_nii = nib.load('/ix3/tibrahim/jil202/cfg_gen/qc_image_nii/denoised_mp2rage/IBRA-BIO1/PRT170058/converted/MP2RAGE_UNI_Images/md_denoised_PRT170058_20180912172034.nii.gz')
    input_nii = tio.transforms.Resample((1,1,1))(input_nii)
    input_nii = tio.transforms.Resample((0.55,0.55,0.55))(input_nii)
    input_nii = tio.transforms.CropOrPad((300,360,384))(input_nii)
    data = quantile_normalization(input_nii, lower_quantile=0.01, upper_quantile=0.99)
    data = torch.tensor(data).to(device)
    
    # Select a single slice for testing
    test_slice = torch.rot90(data[:,:,190], 1).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Initialize scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # Run benchmark with more configurations
    print("\nRunning comprehensive benchmark...")
    benchmark_results = benchmark_inference(
        model, device, scheduler, test_slice, 
        cache_intervals=[20],
        branch_ids=[0]  # Test with first and second branch
    )
    
    # Save visual comparison of results
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 4, 1)
    plt.imshow(test_slice.squeeze().cpu(), cmap='gray')
    plt.title("Input")
    
    plt.subplot(3, 4, 2)
    plt.imshow(benchmark_results['vanilla']['output'].squeeze().cpu(), cmap='gray')
    plt.title(f"Vanilla ({benchmark_results['vanilla']['time']:.2f}s)")
    
    # Plot branch 0 results
    row, col = 0, 0
    for interval in [20]:
        config_name = f'deepcache_N{interval}_B0'
        if config_name in benchmark_results:
            row += 1
            plt.subplot(3, 4, 4 + row)
            plt.imshow(benchmark_results[config_name]['output'].squeeze().cpu(), cmap='gray')
            result_time = benchmark_results[config_name]['time']
            speedup = benchmark_results['vanilla']['time'] / result_time
            plt.title(f"B0, N={interval}\n({result_time:.2f}s, {speedup:.2f}×)")
    
    # Plot branch 1 results
    row, col = 0, 0
    for interval in [20]:
        config_name = f'deepcache_N{interval}_B1'
        if config_name in benchmark_results:
            row += 1
            plt.subplot(3, 4, 8 + row)
            plt.imshow(benchmark_results[config_name]['output'].squeeze().cpu(), cmap='gray')
            result_time = benchmark_results[config_name]['time']
            speedup = benchmark_results['vanilla']['time'] / result_time
            plt.title(f"B1, N={interval}\n({result_time:.2f}s, {speedup:.2f}×)")
    
    plt.tight_layout()
    plt.savefig("deepcache_benchmark_comparison.png")
    plt.close()
    
    # Print benchmark summary
    print("\nBenchmark Summary:")
    print(f"Vanilla: {benchmark_results['vanilla']['time']:.2f}s")
    
    # Print results sorted by speedup
    results_list = []
    for config, metrics in benchmark_results.items():
        if config != 'vanilla':
            speedup = benchmark_results['vanilla']['time'] / metrics['time']
            results_list.append((config, metrics['time'], speedup))
    
    # Sort by speedup (descending)
    results_list.sort(key=lambda x: x[2], reverse=True)
    
    print("\nConfigurations ranked by speedup:")
    for config, time, speedup in results_list:
        print(f"{config}: {time:.2f}s, Speedup: {speedup:.2f}×")
    
    print("\nDeepCache benchmark completed!")

if __name__ == "__main__":
    main()