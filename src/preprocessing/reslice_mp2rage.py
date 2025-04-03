import os
import nibabel as nib
import numpy as np
from PIL import Image
from pathlib import Path
import pdb
import glob
from tqdm import tqdm


def normalize_and_save_nii_to_png(nii_path, output_dir, lower_percentile=1, upper_percentile=99):
    """
    Load a NIfTI volume, normalize it using percentile-based windowing to handle
    hyperintense vessels in T1w images, and save as PNG slices.
    
    Args:
        nii_path: Path to the NIfTI file
        output_dir: Directory to save PNG slices
        lower_percentile: Lower percentile to exclude dark outliers
        upper_percentile: Upper percentile to exclude bright outliers
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the NIfTI file
    nii_img = nib.load(nii_path)
    volume_data = nii_img.get_fdata()
    
    # Replace NaN and Inf values
    volume_data = np.nan_to_num(volume_data, nan=0.0, posinf=None, neginf=None)
    
    # Calculate percentile values for the entire volume
    vol_min = np.percentile(volume_data, lower_percentile)
    vol_max = np.percentile(volume_data, upper_percentile)
    
    print(f"Volume percentiles: {lower_percentile}% = {vol_min}, {upper_percentile}% = {vol_max}")
    
    # Check if the volume is flat or nearly flat
    if np.isclose(vol_min, vol_max) or vol_max - vol_min < 1e-6:
        print("Warning: Volume has almost uniform intensity!")
    
    # Get file basename for naming slices
    base_name = Path(nii_path).stem
    
    # Process each orientation (axial, sagittal, coronal)
    orientations = ['axial', 'sagittal', 'coronal']
    
    for orientation in orientations:
        # Create subdirectory for this orientation
        orient_dir = os.path.join(output_dir, orientation)
        os.makedirs(orient_dir, exist_ok=True)
        
        # Determine which dimension to iterate over based on orientation
        if orientation == 'axial':
            slices = range(volume_data.shape[2])
            get_slice = lambda i: volume_data[:, :, i]
        elif orientation == 'sagittal':
            slices = range(volume_data.shape[0])
            get_slice = lambda i: volume_data[i, :, :]
        else:  # coronal
            slices = range(volume_data.shape[1])
            get_slice = lambda i: volume_data[:, i, :]
        
        # Process each slice
        for i in tqdm(slices, desc=f"Saving {orientation} slices"):
            # Get the slice data
            slice_data = get_slice(i)
            
            # Skip empty slices (all zeros or NaNs)
            if np.all(np.isclose(slice_data, 0)) or np.all(np.isnan(slice_data)):
                continue
            
            # Clip values to the percentile range then normalize
            slice_data = np.clip(slice_data, vol_min, vol_max)
            normalized = ((slice_data - vol_min) / (vol_max - vol_min) * 255.0).astype(np.uint8)
            
            # Create PIL image and save as PNG
            # Transpose to ensure correct orientation in the saved image
            if orientation == 'axial':
                img = Image.fromarray(np.rot90(normalized,1))
            else:
                img = Image.fromarray(np.rot90(normalized,1))
            
            # Save the image
            base_name = base_name.split('.')[0]
            img.save(os.path.join(orient_dir, f"{base_name}_{i:04d}.png"))
    
    print(f"Saved all slices to {output_dir}")
    return output_dir

if __name__ == "__main__":
    # Set your directories here
    nii_base_dir = "/ix3/tibrahim/jil202/cfg_gen/qc_image_nii/denoised_mp2rage/mp2rage"
    png_base_dir = "/ix3/tibrahim/jil202/cfg_gen/qc_image_png/denoised_mp2rage/"
    
    # Process all files
    nii_files = glob.glob(os.path.join(nii_base_dir, "md*.nii.gz"))
    for nii_file in nii_files:
        print(f"Processing {nii_file}")
        normalize_and_save_nii_to_png(nii_file, png_base_dir, lower_percentile=1, upper_percentile=99)
