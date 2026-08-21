import os
import sys
import glob
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from spa3_vlm.data.utils import load_and_preprocess_images

def generate_calibration_dataset(
    dataset_dir: str,
    output_dir: str,
    num_samples: int = 500,
    target_size: int = 224
):
    print(f"Scanning {dataset_dir} for images...")
    
    # Find all images (jpg, png, webp)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "**", ext), recursive=True))
        
    image_paths = sorted(list(set(image_paths)))
    
    if not image_paths:
        print(f"Error: No images found in {dataset_dir}")
        return
        
    print(f"Found {len(image_paths)} images. Selecting the first {num_samples} for calibration...")
    image_paths = image_paths[:num_samples]
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating calibration dataset in {output_dir}...")
    success_count = 0
    
    for i, img_path in enumerate(tqdm(image_paths)):
        try:
            # load_and_preprocess_images expects a list, returns (1, C, H, W)
            tensor = load_and_preprocess_images([img_path], mode="crop", target_size=target_size)
            
            # Add time dimension: (1, 3, H, W) -> (1, 1, 3, H, W)
            tensor = tensor.unsqueeze(1)
            
            # Convert to numpy float32
            np_data = tensor.numpy().astype(np.float32)
            
            # Save as .npy
            out_filename = f"{success_count:06d}.npy"
            out_path = os.path.join(output_dir, out_filename)
            np.save(out_path, np_data)
            
            success_count += 1
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")
            
    print(f"Successfully generated {success_count} calibration samples in {output_dir}")

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    dataset_dir = os.path.join(project_root, "dataset")
    output_dir = os.path.join(project_root, "spa3r_edge", "calibration")
    
    generate_calibration_dataset(dataset_dir, output_dir, num_samples=500, target_size=224)
