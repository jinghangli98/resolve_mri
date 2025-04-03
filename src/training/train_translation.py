import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from monai.inferers import DiffusionInferer
from monai.networks.nets import DiffusionModelUNet
from monai.networks.schedulers import DDPMScheduler, DDIMScheduler
from dataset import getloader
import pdb
from pynvml.smi import nvidia_smi
from PIL import Image
from metrics import evaluate_image_quality
import argparse
import wandb
import torchvision
from utils import shuffle_images_identically, visualize_and_save, downsample_upsample

# Add imports for DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Image translation diffusion model training")

    # Add arguments
    parser.add_argument("--lr", type=float, default=3e-6,
                        help="Learning rate for the optimizer (default: 1e-5).")
    parser.add_argument("--crop_size", type=int, default=320,
                        help="Size to crop images to (default: 384).")
    parser.add_argument("--resize_size", type=int, default=320,
                        help="Size to resize images to (default: 384).")
    parser.add_argument("--data_path", type=str, default='/ix3/tibrahim/jil202/cfg_gen/qc_image_png/denoised_mp2rage',
                        help="Path to the dataset directory.")
    parser.add_argument("--inf_path", type=str, default='/ix3/tibrahim/jil202/cfg_gen/qc_image_png/denoised_mp2rage_inf',
                        help="Path to the inference dataset directory.")
    parser.add_argument("--progressive", action="store_true",
                        help="Use progressive patch shuffling for training.")
    parser.add_argument("--save_model", action="store_true",
                        help="Flag to save diffusion model checkpoints.")
    parser.add_argument("--progressive_inference", action="store_true",
                        help="Use progressive inference starting from existing images.")
    parser.add_argument("--noise_level", type=float, default=0.7,
                        help="Progressive inference noise level, 0.0 to 1.0 (0.0 means starting from pure noise, 1.0 means starting from the original image).")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="Batch size for training (default: 12).")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Maximum number of training epochs (default: 100).")
    parser.add_argument("--sample", type=int, default=100,
                        help="Number of samples to use from the dataset (default: 100).")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="Interval for validation (default: 1).")
    parser.add_argument("--log", type=bool, default=False,
                        help="Enable wandb logging metrics.")
    parser.add_argument("--eval_num", type=int, default=1,
                        help="Number of images to evaluate per GPU (default: 1).")
    parser.add_argument("--checkpoint_path", type=str, default="/ix3/tibrahim/jil202/cfg_gen/src/training/checkpoints/320/t1w_to_tse_model_320_ssim_8.36.pt",
                        help="Path to save or load the model checkpoint (default: ./t1w_to_tse_model_448x448_DDPM_100_overfit.pt).")
    parser.add_argument("--scale_factor", type=float, default=0.5,
                        help="Scale factor for downsampling and upsampling the TSE image (default: 0.5).")
    
    # Add DDP arguments
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training with DDP.")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="Local rank for distributed training.")
    parser.add_argument("--world_size", type=int, default=1,
                        help="Number of processes/GPUs for distributed training.")
    parser.add_argument("--dist_url", type=str, default="env://",
                        help="URL used to set up distributed training.")
    parser.add_argument("--dist_backend", type=str, default="nccl",
                        help="Distributed backend to use (nccl, gloo, etc.).")
    parser.add_argument("--master_addr", type=str, default="localhost",
                        help="Master address for distributed training.")
    parser.add_argument("--master_port", type=str, default="12355",
                        help="Master port for distributed training.")
    
    # New arguments for distributed evaluation
    parser.add_argument("--save_all_ranks", action="store_true",
                        help="Save visualization results from all ranks, not just rank 0.")
    parser.add_argument("--verbose_eval", action="store_true",
                        help="Print evaluation metrics from all ranks, not just rank 0.")

    # Parse arguments
    args = parser.parse_args()
    return args

def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")

def norm(img):
    """Normalize the image to 0-255 range."""
    img = img.float()  # Ensure we're working with float tensor
    img = (img - img.min()) / (img.max() - img.min())
    return (img * 255).byte()


def setup_distributed(rank, world_size, master_addr, master_port, backend='nccl'):
    """Initialize distributed training"""
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize process group
    dist_url = f'tcp://{master_addr}:{master_port}'
    print(f"Initializing process group rank {rank}/{world_size} with URL: {dist_url}")
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )
    
    print(f"Process group initialized: rank {rank}/{world_size}")
    return rank

def cleanup_distributed():
    """Clean up distributed training resources"""
    if dist.is_initialized():
        print("Destroying process group")
        dist.destroy_process_group()

def train(local_rank, args):
    """Main training function that handles distributed training"""
    # Ensure distributed setup if needed
    if args.distributed:
        # Setup distributed process group
        if "LOCAL_RANK" in os.environ:
            # Running with torchrun
            args.local_rank = int(os.environ["LOCAL_RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            rank = args.local_rank
        else:
            # Running with manual spawn
            rank = local_rank
            
        setup_distributed(
            rank=rank,
            world_size=args.world_size,
            master_addr=args.master_addr,
            master_port=args.master_port,
            backend=args.dist_backend
        )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Single GPU training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0
        args.world_size = 1
    
    # Initialize wandb only on main process
    if args.log and local_rank == 0:
        wandb.init(project="T1SuperResolution", config=vars(args)) 
    
    # Create output directories on main process
    if local_rank == 0:
        os.makedirs("visualization_results", exist_ok=True)
        os.makedirs(f"checkpoints/{args.resize_size}", exist_ok=True)
        
    # Setup data loaders with distributed samplers if using DDP
    if args.distributed:
        train_loader, val_loader = getloader(
            batch_size=args.batch_size,
            data_root=args.data_path,
            crop_size=args.crop_size,
            size=args.resize_size,
            sample=args.sample,
            type='img',
            distributed=True,
            rank=local_rank,
            world_size=args.world_size
        )

        inf_loader, _ = getloader(
            batch_size=16,
            data_root=args.inf_path,
            crop_size=args.crop_size,
            size=args.resize_size,
            sample=100,
            type='img',
            distributed=True,
            rank=local_rank,
            world_size=args.world_size,
            train_shuffle=False
        )
    else:
        train_loader, val_loader = getloader(
            batch_size=args.batch_size,
            data_root=args.data_path,
            crop_size=args.crop_size,
            size=args.resize_size,
            sample=args.sample
        )
        inf_loader, _ = getloader(
            batch_size=16, 
            data_root=args.inf_path, 
            crop_size=args.crop_size, 
            size=args.resize_size, 
            sample=100, 
            train_shuffle=False
        )
    
    if local_rank == 0:
        print(f"Data loaders created, train: {len(train_loader)}, val: {len(val_loader)}")
    
    # Create model
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=2,  # T1w + noise
        out_channels=1,  # TSE
        channels=(256, 256, 512),
        attention_levels=(False, False, True),
        num_res_blocks=2,
        num_head_channels=512,
        with_conditioning=False,
    )
    
    # Load checkpoint if available
    if os.path.exists(args.checkpoint_path):
        if local_rank == 0:
            print(f"Loading checkpoint from {args.checkpoint_path}")
        # Map model to be loaded to specified single gpu
        state_dict = torch.load(args.checkpoint_path, map_location=f'cuda:{local_rank}', weights_only=True)
        model.load_state_dict(state_dict)
    
    model.to(device)
    
    # Wrap model with DDP
    if args.distributed:
        if local_rank == 0:
            print("Wrapping model with DDP")
        # Make sure all processes have loaded the model before wrapping with DDP
        if dist.is_initialized():
            dist.barrier()
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    inferer = DiffusionInferer(scheduler)

    # Training setup
    epoch_loss_list = []
    val_epoch_loss_list = []

    scaler = GradScaler()
    total_start = time.time()

    best_ssim_psnr = 0
    
    if local_rank == 0:
        print("Starting training loop")
        
    for epoch in range(args.max_epochs):
        # Set epoch for distributed sampler
        if args.distributed:
            for loader in [train_loader, val_loader]:
                if hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(epoch)
            
        model.train()
        epoch_loss = 0
        if local_rank == 0:
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        else:
            progress_bar = enumerate(train_loader)

        for _step, (t1w, tse) in progress_bar:
            t1w, tse = t1w.to(device).unsqueeze(1), tse.to(device).unsqueeze(1) #t1 is the low resolution image
            t1w = downsample_upsample(t1w, scale_factor=args.scale_factor, mode='bilinear')
            if args.progressive:
                if epoch >= 0 and epoch < 40:
                    t1w, tse = shuffle_images_identically(t1w, tse, num_patches=2)
                elif epoch >= 40 and epoch < 60:
                    t1w, tse = shuffle_images_identically(t1w, tse, num_patches=3)
                elif epoch >= 60:
                    t1w, tse = shuffle_images_identically(t1w, tse, num_patches=4)

            # Debug plots only on rank 0
            if local_rank == 0 and _step == 0 and epoch % 10 == 0:
                plt.imshow(t1w.cpu()[0].squeeze().numpy(), cmap='gray')
                plt.savefig('t1w.png')
                plt.close()
                plt.imshow(tse.cpu()[0].squeeze().numpy(), cmap='gray')
                plt.savefig('t2w.png')
                plt.close()

            optimizer.zero_grad(set_to_none=True)
            timesteps = torch.randint(0, 1000, (t1w.shape[0],)).to(device)

            with autocast(enabled=True):
                noise = torch.randn_like(tse).to(device)
                noisy_tse = scheduler.add_noise(original_samples=tse, noise=noise, timesteps=timesteps)
                combined = torch.cat((t1w, noisy_tse), dim=1)
                prediction = model(x=combined, timesteps=timesteps)
                loss = F.mse_loss(prediction.float(), noise.float())
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            
            if local_rank == 0 and isinstance(progress_bar, tqdm):
                progress_bar.set_postfix({"loss": epoch_loss / (_step + 1)})

        # Synchronize loss across all processes
        if args.distributed:
            epoch_loss_tensor = torch.tensor(epoch_loss, device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            epoch_loss = epoch_loss_tensor.item() / args.world_size
            
        epoch_loss /= len(train_loader)
        epoch_loss_list.append(epoch_loss)
        
        if local_rank == 0:
            print(f"Epoch {epoch+1}/{args.max_epochs}, Training loss: {epoch_loss:.4f}")
            print_gpu_memory_report()

        if (epoch + 1) % args.val_interval == 0:
            model.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                pass
            
            # Synchronize validation loss
            if args.distributed:
                val_loss_tensor = torch.tensor(val_epoch_loss, device=device)
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                val_epoch_loss = val_loss_tensor.item() / args.world_size
                
            val_epoch_loss /= len(val_loader)
            val_epoch_loss_list.append(val_epoch_loss)
            
            if local_rank == 0:
                print(f"Epoch {epoch+1}/{args.max_epochs}, Validation loss: {val_epoch_loss:.4f}")

            # Fetch evaluation data - we need to make sure all ranks have the same
            # examples for proper distributed evaluation
            eval_t1w = []
            eval_tse = []
            
            # Get first batch for evaluation
            for t1w_eval, tse_eval in inf_loader:
                eval_t1w = t1w_eval.to(device).unsqueeze(1) #t1 is the low resolution image
                eval_tse = tse_eval.to(device).unsqueeze(1)
                
                eval_t1w = downsample_upsample(eval_t1w, scale_factor=args.scale_factor, mode='bilinear')
                break
            
            # Make sure every rank has data to evaluate
            if args.distributed:
                # Ensure all processes are in sync before evaluation
                dist.barrier()
            
            # Now all ranks participate in evaluation
            ssim, psnr, lpips = visualize_and_save(
                epoch=epoch, 
                model=model, 
                device=device, 
                scheduler=scheduler, 
                t1w=eval_t1w, 
                tse=eval_tse, 
                args=args, 
                local_rank=local_rank
            )
            
            ssim_psnr = 0.7* ssim + 0.3 * psnr
            # Log from all ranks if verbose logging is enabled
            if args.distributed and args.verbose_eval:
                print(f"Rank {local_rank}: Evaluated metrics - SSIM: {ssim:.4f}, PSNR: {psnr:.4f}, LPIPS: {lpips:.4f}")
            
            # Save best model on main process
            if local_rank == 0 and ssim_psnr > best_ssim_psnr:
                best_ssim_psnr = ssim_psnr
                best_ssim = ssim
                checkpoint_path = f"./checkpoints/{args.resize_size}/{args.scale_factor}/t1w_to_tse_model_{args.resize_size}_ssim_{best_ssim_psnr:.2f}.pt"
                
                # Save the model state dict (unwrap DDP model if needed)
                if args.save_model:
                    if args.distributed:
                        torch.save(model.module.state_dict(), checkpoint_path)
                    else:
                        torch.save(model.state_dict(), checkpoint_path)
                    
                print(f"Saved best model with SSIM: {best_ssim:.4f}, PSNR: {psnr:.4f}, LPIPS: {lpips:.4f}")

    # Total training time
    total_time = time.time() - total_start
    if local_rank == 0:
        print(f"Total training time: {total_time:.2f} seconds")
        
    # Wait for all processes to finish before cleaning up
    if args.distributed:
        dist.barrier()
        
    # Clean up distributed resources
    if args.distributed:
        cleanup_distributed()

def main():
    args = parse_args()
    
    if args.distributed:
        # Set environment variables for the main process
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        
        # If running with torchrun or torch.distributed.launch
        if "LOCAL_RANK" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            
            # Start training directly
            print(f"Using torchrun with LOCAL_RANK={args.local_rank}, WORLD_SIZE={args.world_size}")
            train(args.local_rank, args)
        else:
            # Manual process spawning
            print(f"Spawning {args.world_size} processes for distributed training")
            mp.set_start_method('spawn', force=True)
            mp.spawn(train, nprocs=args.world_size, args=(args,), join=True)
    else:
        # Single GPU training
        train(0, args)


if __name__ == "__main__":
    main()