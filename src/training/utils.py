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

def visualize_and_save(epoch, model, device, scheduler, t1w, tse, args, local_rank=0):
    """
    Run visualization and evaluation across all GPUs, then gather results.
    """
    model.eval()
    
    # Determine which indices this rank should process
    total_eval_images = args.eval_num
    indices_per_rank = total_eval_images // args.world_size
    # Handle remainder by assigning extra indices to early ranks
    remainder = total_eval_images % args.world_size
    start_idx = local_rank * indices_per_rank + min(local_rank, remainder)
    end_idx = start_idx + indices_per_rank + (1 if local_rank < remainder else 0)
    
    # Each rank processes its assigned indices
    all_metrics = []
    generated_images = []
    
    with torch.no_grad():
        for ind in range(start_idx, end_idx):
            # Ensure the index is valid for the input tensors
            if ind >= len(t1w):
                continue
                
            input_img = t1w[ind:ind+1]
            target_img = tse[ind:ind+1]
            scheduler.set_timesteps(num_inference_steps=1000)

            if args.progressive_inference:
                current_img = input_img.clone()
                noise_timestep = int(args.noise_level * len(scheduler.timesteps))
                noise_timestep = max(min(noise_timestep, len(scheduler.timesteps)-1), 0)
                noise = torch.randn_like(current_img, device=device)
                t = scheduler.timesteps[noise_timestep]

                alpha_cumprod = scheduler.alphas_cumprod.to(device)
                sqrt_alpha_t = alpha_cumprod[t] ** 0.5
                sqrt_one_minus_alpha_t = (1 - alpha_cumprod[t]) ** 0.5
                current_img = sqrt_alpha_t * current_img + sqrt_one_minus_alpha_t * noise
                
                starting_timestep_idx = noise_timestep
                timesteps = scheduler.timesteps[starting_timestep_idx:]
                
                # Only show progress bar on rank 0
                if local_rank == 0:
                    progress_bar = tqdm(timesteps, desc=f"Generating TSE Starting with Existing TSE (Epoch {epoch+1}, Img {ind})")
                else:
                    progress_bar = timesteps
                    
                for t in progress_bar:
                    combined = torch.cat((input_img, current_img), dim=1)
                    model_output = model(combined, timesteps=torch.Tensor((t,)).to(device))
                    current_img, _ = scheduler.step(model_output, t, current_img)
            else:
                noise = torch.randn_like(input_img).to(device)
                current_img = noise  # for the TSE image, we start from random noise.
                combined = torch.cat((input_img, noise), dim=1)

                # Only show progress bar on rank 0 and for the first image
                if local_rank == 0 and ind == start_idx:
                    progress_bar = tqdm(scheduler.timesteps, desc=f"Generating TSE (Epoch {epoch+1}, Rank {local_rank})")
                else:
                    progress_bar = scheduler.timesteps

                for t in progress_bar:  # go through the noising process
                    with autocast(enabled=False):
                        # Unwrap DDP model for inference if needed
                        if isinstance(model, DDP):
                            model_output = model.module(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                        else:
                            model_output = model(combined, timesteps=torch.Tensor((t,)).to(current_img.device))
                            
                        current_img, _ = scheduler.step(model_output, t, current_img)
                        combined = torch.cat((input_img, current_img), dim=1)
                        
            # Compute metrics for this sample
            metrics = evaluate_image_quality(current_img.cpu().squeeze().numpy(), target_img.cpu().squeeze().numpy())
            all_metrics.append(metrics)
            
            # Only save visualizations on rank 0 or if requested
            if local_rank == 0 or args.save_all_ranks:
                vis = torch.hstack((norm(input_img.squeeze().cpu()), 
                                    norm(current_img.squeeze().cpu()), 
                                    norm(target_img.squeeze().cpu())))
                vis = vis.numpy().astype(np.uint8)
                vis = Image.fromarray(vis)
                
                # Save the visualization to disk
                vis_path = f"visualization_results/{args.scale_factor}/epoch_{epoch+1}_rank_{local_rank}_idx_{ind}.png"
                vis.save(vis_path)
                
                # Store the generated image for potential wandb logging
                generated_images.append((ind, vis))
    
    # Now gather metrics from all ranks
    if dist.is_initialized():
        # Create a list to store metrics from all ranks
        gathered_metrics = [None for _ in range(args.world_size)]
        
        # First, serialize the metrics to a binary format
        metrics_tensor = torch.tensor([
            [metric['SSIM'] for metric in all_metrics],
            [metric['PSNR'] for metric in all_metrics],
            [metric['LPIPS'] for metric in all_metrics]
        ], device=device)
        
        # Create a list of tensors to gather into
        output_tensors = [torch.zeros_like(metrics_tensor) for _ in range(args.world_size)]
        
        # Gather metrics from all processes
        dist.all_gather(output_tensors, metrics_tensor)
        
        # Process the gathered metrics
        all_ssim = []
        all_psnr = []
        all_lpips = []
        
        for tensor in output_tensors:
            # Extract metrics from each rank
            ssim_values = tensor[0].cpu().numpy()
            psnr_values = tensor[1].cpu().numpy()
            lpips_values = tensor[2].cpu().numpy()
            
            # Add non-zero values to our aggregated lists
            all_ssim.extend([v for v in ssim_values if v > 0])
            all_psnr.extend([v for v in psnr_values if v > 0])
            all_lpips.extend([v for v in lpips_values if v > 0])
        
        # Compute global averages
        avg_ssim = np.mean(all_ssim) if all_ssim else 0
        avg_psnr = np.mean(all_psnr) if all_psnr else 0
        avg_lpips = np.mean(all_lpips) if all_lpips else 0
    else:
        # If not distributed, just compute the average from this process
        avg_ssim = np.mean([metric['SSIM'] for metric in all_metrics]) if all_metrics else 0
        avg_psnr = np.mean([metric['PSNR'] for metric in all_metrics]) if all_metrics else 0
        avg_lpips = np.mean([metric['LPIPS'] for metric in all_metrics]) if all_metrics else 0
    
    # Only log to wandb from rank 0
    if args.log and local_rank == 0 and generated_images:
        # Use the first image for logging (or you could log multiple)
        wandb.log({
            "Generated Image": wandb.Image(generated_images[0][1]),
            "SSIM": avg_ssim,
            "PSNR": avg_psnr,
            "LPIPS": avg_lpips,
            **vars(args)
        }, step=epoch)

    return avg_ssim, avg_psnr, avg_lpips

def progressive_patch_shuffle(image, num_patches, points=None):
    """
    Apply progressive patch shuffling to an image.
    
    Args:
        image (torch.Tensor): Input image tensor of shape [B, C, H, W]
        num_patches (int or tuple): Number of patches along each dimension (e.g., 2 for 2x2, or (2,2))
        points (torch.Tensor, optional): Point coordinates to transform, shape [B, N, 2]
        
    Returns:
        torch.Tensor: Shuffled image
        torch.Tensor: Transformed points (if points is not None)
    """
    B, C, H, W = image.shape
    
    # Handle different input formats for num_patches
    if isinstance(num_patches, int):
        h_patches, w_patches = num_patches, num_patches
    else:
        h_patches, w_patches = num_patches
    
    # Calculate patch size
    patch_h, patch_w = H // h_patches, W // w_patches
    
    # Reshape image into patches
    patches = image.view(B, C, h_patches, patch_h, w_patches, patch_w)
    patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches = patches.view(B, h_patches * w_patches, C, patch_h, patch_w)
    
    # Create random permutation indices for each batch
    indices = torch.stack([torch.randperm(h_patches * w_patches) for _ in range(B)], dim=0).to(image.device)
    
    # Shuffle patches
    shuffled_patches = torch.stack([patches[b, indices[b]] for b in range(B)], dim=0)
    
    # Reshape back to image
    shuffled_patches = shuffled_patches.view(B, h_patches, w_patches, C, patch_h, patch_w)
    shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled_image = shuffled_patches.view(B, C, H, W)
    
    # Transform point coordinates if provided
    transformed_points = None
    if points is not None:
        transformed_points = points.clone()
        for b in range(B):
            for p in range(points.shape[1]):
                if torch.all(points[b, p] == 0):  # Skip if point is [0, 0] (no point)
                    continue
                
                # Calculate which patch the point belongs to
                x, y = points[b, p]
                patch_idx_h, patch_idx_w = int(y // patch_h), int(x // patch_w)
                patch_idx = patch_idx_h * w_patches + patch_idx_w
                
                # Find where this patch went in the shuffle
                new_patch_idx = int((indices[b] == patch_idx).nonzero(as_tuple=True)[0])
                new_patch_idx_h, new_patch_idx_w = new_patch_idx // w_patches, new_patch_idx % w_patches
                
                # Calculate relative position within patch
                rel_y, rel_x = y % patch_h, x % patch_w
                
                # Calculate new absolute position
                new_y = new_patch_idx_h * patch_h + rel_y
                new_x = new_patch_idx_w * patch_w + rel_x
                
                transformed_points[b, p, 0] = new_x
                transformed_points[b, p, 1] = new_y
    
    return shuffled_image, transformed_points

def shuffle_images_identically(image1, image2, num_patches):
    """
    Shuffle two images with identical patch permutation.
    
    Args:
        image1 (torch.Tensor): First image tensor of shape [B, C, H, W]
        image2 (torch.Tensor): Second image tensor of shape [B, C, H, W]
        num_patches (int): Number of patches along each dimension
        
    Returns:
        tuple: Two shuffled image tensors
    """
    B, C1, H, W = image1.shape
    _, C2, _, _ = image2.shape
    
    # Handle different input formats for num_patches
    if isinstance(num_patches, int):
        h_patches, w_patches = num_patches, num_patches
    else:
        h_patches, w_patches = num_patches
    
    # Calculate patch size
    patch_h, patch_w = H // h_patches, W // w_patches
    
    # Create SAME random permutation indices for each batch
    # Note: We create once and reuse for both images
    indices = torch.stack([torch.randperm(h_patches * w_patches) for _ in range(B)], dim=0).to(image1.device)
    
    # First image
    patches1 = image1.view(B, C1, h_patches, patch_h, w_patches, patch_w)
    patches1 = patches1.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches1 = patches1.view(B, h_patches * w_patches, C1, patch_h, patch_w)
    
    # Shuffle patches for first image using indices
    shuffled_patches1 = torch.stack([patches1[b, indices[b]] for b in range(B)], dim=0)
    
    # Reshape back to image
    shuffled_patches1 = shuffled_patches1.view(B, h_patches, w_patches, C1, patch_h, patch_w)
    shuffled_patches1 = shuffled_patches1.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled_image1 = shuffled_patches1.view(B, C1, H, W)
    
    # Second image - same exact process but REUSE same indices
    patches2 = image2.view(B, C2, h_patches, patch_h, w_patches, patch_w)
    patches2 = patches2.permute(0, 2, 4, 1, 3, 5).contiguous()
    patches2 = patches2.view(B, h_patches * w_patches, C2, patch_h, patch_w)
    
    # Shuffle patches for second image using SAME indices
    shuffled_patches2 = torch.stack([patches2[b, indices[b]] for b in range(B)], dim=0)
    
    # Reshape back to image
    shuffled_patches2 = shuffled_patches2.view(B, h_patches, w_patches, C2, patch_h, patch_w)
    shuffled_patches2 = shuffled_patches2.permute(0, 3, 1, 4, 2, 5).contiguous()
    shuffled_image2 = shuffled_patches2.view(B, C2, H, W)
    
    return shuffled_image1, shuffled_image2