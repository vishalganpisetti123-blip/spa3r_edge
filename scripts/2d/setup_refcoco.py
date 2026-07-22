import os
import json
from datasets import load_dataset

# --- 1. CONFIGURATION ---
DATASET_DIR = "dataset"
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "benchmark_annotations.json")
os.makedirs(DATASET_DIR, exist_ok=True)

# --- 2. DOWNLOAD HUGGING FACE DATASET ---
print("Downloading RefCOCO subset from Hugging Face...")
# We use a compiled RefCOCO# Using 'val[:10000]' split to grab 10000 samples
dataset = load_dataset("Kangheng/refcoco", split="val[:10000]")

formatted_data = []

print(f"Formatting {len(dataset)} annotations and saving images locally...")
for i, item in enumerate(dataset):
    # Extract raw data
    image = item["image"]
    prompt = item["question"]
    bbox = item["bbox"] # [x_min, y_min, x_max, y_max]
    
    # Generate unique filename for the image
    img_filename = f"refcoco_{i}.jpg"
    img_path = os.path.join(DATASET_DIR, img_filename)
    
    # Save the physical image to your dataset folder
    image.save(img_path)
    
    # Format the data exactly as your DataLoader expects
    formatted_data.append({
        "image_file": img_filename,
        "prompt": prompt,
        "gt_box": bbox,
        "width": image.width,
        "height": image.height
    })

# --- 3. EXPORT TO JSON ---
with open(ANNOTATIONS_FILE, "w") as f:
    json.dump(formatted_data, f, indent=4)

print(f"\n[SUCCESS] Prepared {len(formatted_data)} samples in '{DATASET_DIR}/'")
print(f"Master annotations saved to {ANNOTATIONS_FILE}")
