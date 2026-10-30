import torch
import numpy as np
import torch
import torchvision.transforms as T
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from metrics import evaluate_image_quality
from PIL import Image
import torch.distributed as dist
import wandb
import os
import torch.nn.functional as F

def norm(img):
    """Normalize the image to 0-255 range."""
    img = img.float()  # Ensure we're working with float tensor
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).byte()

def downsample_upsample(tensor, scale_factor=0.75, mode='bilinear'):
    """
    Downsamples a tensor by a scale factor and then upsamples it back to the original size.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape [B, C, H, W]
        scale_factor (float): Scale factor for downsampling (0 < scale_factor < 1)
        mode (str): Interpolation mode ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear')
                   Default is 'bilinear' which works well for most images
    
    Returns:
        torch.Tensor: Tensor that has been downsampled and upsampled back to original size
    """
    # Check that scale factor is valid
    if not (0 < scale_factor < 1):
        raise ValueError("Scale factor must be between 0 and 1")
    
    # Get original size
    batch_size, channels, height, width = tensor.shape
    
    # Calculate the new size for downsampling
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)
    
    # Downsample
    downsampled = F.interpolate(
        tensor, 
        size=(new_height, new_width), 
        mode=mode, 
        align_corners=False if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
    )
    
    # Upsample back to original size
    upsampled = F.interpolate(
        downsampled, 
        size=(height, width), 
        mode=mode,
        align_corners=False if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None
    )
    
    return upsampled
