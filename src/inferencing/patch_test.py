from anatomy_cache import adaptive_patch_locations, crop_mri_volume, quantile_normalization, plot_patch_heatmap
import nibabel as nib
import torch
import pdb
from lpips import lpips
import numpy as np
import matplotlib.pyplot as plt

def plot_image_pairs(tensor1, tensor2, titles=None, figsize=(15, 20), cmap='gray'):
    """
    Plot two sets of images side by side, iterating through all slices.
    
    Args:
        tensor1: Tensor of shape [N, C, H, W]
        tensor2: Tensor of shape [N, C, H, W]
        titles: Tuple of (left_title, right_title) for column headers
        figsize: Figure size (width, height)
        cmap: Colormap to use (default: 'gray' for medical images)
    """
    # Ensure tensors are the same shape
    assert tensor1.shape == tensor2.shape, f"Tensor shapes don't match: {tensor1.shape} vs {tensor2.shape}"
    
    # Get number of slices
    num_slices = tensor1.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(num_slices, 2, figsize=figsize)
    
    # Default titles if not provided
    if titles is None:
        titles = ('Image 1', 'Image 2')
    
    # Add column titles
    fig.suptitle('Image Comparison', fontsize=16)
    axes[0, 0].set_title(titles[0], fontsize=14)
    axes[0, 1].set_title(titles[1], fontsize=14)
    
    # Plot each slice
    for i in range(num_slices):
        # Convert tensors to numpy arrays
        img1 = tensor1[i].squeeze().cpu().numpy()
        img2 = tensor2[i].squeeze().cpu().numpy()
        
        # Normalize if needed
        if img1.max() > 1.0:
            img1 = img1 / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0
            
        # Plot images
        axes[i, 0].imshow(img1, cmap=cmap)
        axes[i, 1].imshow(img2, cmap=cmap)
        
        # Add slice number
        axes[i, 0].set_ylabel(f'Slice {i+1}', fontsize=12)
        
        # Remove ticks
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Make room for suptitle
    plt.savefig('image_comparison.png', dpi=300)
    
def calculate_lpips(generated, target):
    """
    Calculate LPIPS between generated and target images
    
    Args:
        generated: Tensor or array of shape (B, C, H, W) or (H, W)
        target: Tensor or array of shape (B, C, H, W) or (H, W)
        
    Returns:
        Mean LPIPS value across the batch
    """
    loss_fn = lpips.LPIPS(net='vgg')
    
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
    
    return lpips_values

def parselate(image, patch_size, overlap, patch_num, positions=None, overlap_map=None):
    """
    image: torch.Tensor of shape (B, C, H, W)
    """
    # Extract patches based on adaptive positions
    if len(image.shape) != 4 :
        raise ValueError("Input image must have at least 3 dimensions (B, C, H, W)")

    B, C, H, W = image.shape
    if positions is None or overlap_map is None:

        positions, overlap_map = adaptive_patch_locations(image, patch_size=patch_size, base_overlap=overlap, intensity_scale=2.0, target_patches=patch_num)

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
            
        patch = image[..., y:y+patch_size, x:x+patch_size]
        patches.append(patch)
    
    return patches, positions, overlap_map

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_path = '/ix3/tibrahim/jil202/cfg_gen/qc_image_nii/denoised_mp2rage/IBRA-BIO1/PRT170058/converted/MP2RAGE_UNI_Images/md_denoised_PRT170058_20180912172034.nii.gz'
input_nii = nib.load(image_path)
data = quantile_normalization(input_nii, lower_quantile=0.01, upper_quantile=0.99)
data = crop_mri_volume(torch.tensor(data)).unsqueeze(0).to(device).squeeze()

patch_size = 128
base_overlap = 32
intensity_scale = 2.0
patch_num = 16

patches_150, positions_150, overlap_map_150 = parselate(torch.rot90(data[:,:,150]).unsqueeze(0).unsqueeze(0), patch_size, base_overlap, patch_num)
patches_155, positions_155, overlap_map_155 = parselate(torch.rot90(data[:,:,151]).unsqueeze(0).unsqueeze(0), patch_size, base_overlap, patch_num, positions=positions_150, overlap_map = overlap_map_150)
scores = calculate_lpips(torch.stack(patches_150).squeeze(1).cpu(), torch.stack(patches_155).squeeze(1).cpu())
pdb.set_trace()

tensor1 = torch.stack(patches_150).squeeze(1).cpu()[np.array(scores) < 0.1]
tensor2 = torch.stack(patches_155).squeeze(1).cpu()[np.array(scores) < 0.1]
plot_image_pairs(tensor1, tensor2, titles=None, figsize=(15, 20), cmap='gray')

plot_patch_heatmap(torch.rot90(data[:,:,150], 1).cpu(), positions_150, overlap_map_150)

torch.stack(patches_150)
torch.stack(patches_155)