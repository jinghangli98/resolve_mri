from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
import pdb

def calculate_lpips(generated, target, return_mean=True, device='cpu'):
    """
    Calculate LPIPS between generated and target images
    
    Args:
        generated: Tensor or array of shape (B, C, H, W) or (H, W)
        target: Tensor or array of shape (B, C, H, W) or (H, W)
        
    Returns:
        Mean LPIPS value across the batch
    """
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    
    # Convert to tensor if not already
    if not isinstance(generated, torch.Tensor):
        generated = torch.tensor(generated)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    
    # Add missing dimensions if needed
    if generated.dim() == 2:  # (H, W) -> (1, 1, H, W)
        generated = generated.unsqueeze(0).unsqueeze(0)
    elif generated.dim() == 3:  # (C, H, W) -> (1, C, H, W)
        generated = generated.unsqueeze(0)
        
    if target.dim() == 2:  # (H, W) -> (1, 1, H, W)
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 3:  # (C, H, W) -> (1, C, H, W)
        target = target.unsqueeze(0)
    
    # Make sure we have float tensors
    generated = generated.float()
    target = target.float()
    
    # Handle channel dimension - LPIPS expects 3 channels
    if generated.shape[1] == 1:
        generated = generated.repeat(1, 3, 1, 1)
    if target.shape[1] == 1:
        target = target.repeat(1, 3, 1, 1)
    
    # Calculate LPIPS
    batch_size = generated.shape[0]
    lpips_values = []
    
    # Process batch one by one to handle large batches
    for i in range(batch_size):
        lpips_value = loss_fn(generated[i:i+1], target[i:i+1]).item()
        lpips_values.append(lpips_value)
    
    if return_mean:
        return np.mean(lpips_values)
    
    return lpips_values

def calculate_ssim(generated, target, return_mean=True):
    """
    Calculate SSIM between generated and target images
    
    Args:
        generated: Tensor or array of shape (B, C, H, W) or (H, W)
        target: Tensor or array of shape (B, C, H, W) or (H, W)
        
    Returns:
        Mean SSIM value across the batch
    """
    # Convert to numpy if tensors
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Handle different input dimensions
    if generated.ndim == 2 and target.ndim == 2:
        # Single 2D image, just calculate directly
        data_range = target.max() - target.min()
        return ssim(generated, target, data_range=data_range)
    
    # For 4D tensors (B, C, H, W)
    batch_size = generated.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        for c in range(generated.shape[1]):
            gen_slice = generated[i, c]
            target_slice = target[i, c]
            data_range = target_slice.max() - target_slice.min()
            # Avoid division by zero
            if data_range == 0:
                data_range = 1
            ssim_val = ssim(gen_slice, target_slice, data_range=data_range)
            ssim_values.append(ssim_val)
    
    if return_mean:
        return np.mean(ssim_values)
    else:
        return ssim_values

def calculate_psnr(generated, target):
    """
    Calculate PSNR between generated and target images
    
    Args:
        generated: Tensor or array of shape (B, C, H, W) or (H, W)
        target: Tensor or array of shape (B, C, H, W) or (H, W)
        
    Returns:
        Mean PSNR value across the batch
    """
    # Convert to numpy if tensors
    if isinstance(generated, torch.Tensor):
        generated = generated.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # Handle different input dimensions
    if generated.ndim == 2 and target.ndim == 2:
        # Single 2D image, just calculate directly
        data_range = target.max() - target.min()
        return psnr(target, generated, data_range=data_range)
    
    # For 4D tensors (B, C, H, W)
    batch_size = generated.shape[0]
    psnr_values = []
    
    for i in range(batch_size):
        for c in range(generated.shape[1]):
            gen_slice = generated[i, c]
            target_slice = target[i, c]
            data_range = target_slice.max() - target_slice.min()
            # Avoid division by zero
            if data_range == 0:
                data_range = 1
            psnr_val = psnr(target_slice, gen_slice, data_range=data_range)
            psnr_values.append(psnr_val)
    
    return np.mean(psnr_values)

def evaluate_image_quality(generated, target, verbose=True):
    """
    Evaluate image quality metrics between generated and target images
    
    Args:
        generated: Tensor or array of shape (B, C, H, W) or (H, W)
        target: Tensor or array of shape (B, C, H, W) or (H, W)
        verbose: Whether to print results
        
    Returns:
        Dictionary with quality metrics
    """
    ssim_value = calculate_ssim(generated, target)
    psnr_value = calculate_psnr(generated, target)
    lpips_value = calculate_lpips(generated, target)
    
    if verbose:
        print(f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}, LPIPS: {lpips_value:.4f}")
    
    return {
        "SSIM": ssim_value,
        "PSNR": psnr_value,
        "LPIPS": lpips_value,
    }