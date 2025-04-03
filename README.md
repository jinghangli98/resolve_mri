# Resolve-MRI

A deep learning framework for high-resolution MRI super-resolution and image translation.

## Overview

Resolve-MRI is a comprehensive toolkit for processing and enhancing MRI images using state-of-the-art diffusion models. The project implements methods for transforming low-resolution MRI scans into high-quality, detailed images suitable for clinical and research applications.

Key features:

- **Super-resolution**: Transform low-resolution MRI scans to higher resolution images
- **Image translation**: Convert between different MRI contrasts (e.g., T1w to T2w-like images)
- **Efficient inference**: Implements DeepCache optimization for faster diffusion model inference
- **Preprocessing tools**: Utilities for preparing and standardizing MRI data

## Project Structure

```
resolve_mri/
├── src/
│   ├── preprocessing/
│   │   └── reslice_mp2rage.py  # Tools for processing MP2RAGE scans
│   ├── training/
│   │   ├── dataset.py          # Data loading and processing
│   │   ├── train_translation.py # Training script for diffusion models
│   │   └── utils.py            # Training utility functions
│   └── inferencing/
│       ├── deep_cache_inf.py   # Optimized inference with DeepCache
│       ├── metrics.py          # Image quality evaluation metrics
│       └── utils.py            # Inference utility functions
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 1.9+

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/resolve_mri.git
cd resolve_mri

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preprocessing

Convert NIFTI MP2RAGE images to PNG slices for training:

```bash
python src/preprocessing/reslice_mp2rage.py
```

### Model Training

Train a diffusion model for T1w to TSE image translation:

```bash
python src/training/train_translation.py \
  --data_path /path/to/training/data \
  --batch_size 12 \
  --resize_size 320 \
  --crop_size 320 \
  --max_epochs 100 \
  --save_model
```

#### Training options

- `--progressive`: Enable progressive patch shuffling for training
- `--scale_factor`: Scale factor for downsampling and upsampling (default: 0.5)
- `--distributed`: Enable distributed training with DDP
- See `train_translation.py` for additional options

### Inference

Run efficient inference with DeepCache optimization:

```bash
python src/inferencing/deep_cache_inf.py
```

## DeepCache Optimization

The project implements DeepCache for accelerated diffusion model inference, providing:

- Faster inference with minimal quality loss
- Non-uniform caching strategies
- Configurable cache intervals and branch selection
- Support for progressive inference

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## Acknowledgements

[Add acknowledgements here]
