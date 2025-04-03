from skimage.metrics import structural_similarity as ssim
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
import torch
import pdb

def calculate_lpips(generated, target):
    loss_fn = lpips.LPIPS(net='vgg')  # or 'vgg'
    generated_tensor = torch.tensor(generated).unsqueeze(0).unsqueeze(0).float()
    target_tensor = torch.tensor(target).unsqueeze(0).unsqueeze(0).float()
    return loss_fn(generated_tensor, target_tensor).item()

def calculate_ssim(generated, target):
    return ssim(generated, target, data_range=target.max() - target.min())


def calculate_psnr(generated, target):
    return psnr(target, generated, data_range=target.max() - target.min())

def evaluate_image_quality(generated, target):
    ssim_value = calculate_ssim(generated, target)
    psnr_value = calculate_psnr(generated, target)
    lpips_value = calculate_lpips(generated, target)
    
    print(f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.4f}, LPIPS: {lpips_value:.4f}")
    return {
        "SSIM": ssim_value,
        "PSNR": psnr_value,
        "LPIPS": lpips_value,
    }