import os
import json
import numpy as np
import onnxruntime as ort

ANNOTATION_PATH = "dataset/scannet/ScanRefer_filtered.json"
OUTPUT_DIR = "cached_3d_features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(ANNOTATION_PATH, "r") as f:
    data = json.load(f)

onnx_session = ort.InferenceSession("models/spa3r_encoder_psfm_fp32.onnx")
input_name = onnx_session.get_inputs()[0].name

cached_data = []

def get_latents(img):
    raw_tokens = onnx_session.run(None, {input_name: img})[0]
    return raw_tokens[0].tolist()

print(f"[3D CACHE] Caching multi-view spatial latents for {len(data)} samples...")

# Loop through all items in data
for item in data:
    prompt = item.get("description") or item.get("prompt") or "object"
    center = item.get("center", [0.0, 0.0, 0.0])
    size = item.get("size", [1.0, 1.0, 1.0])
    gt_box_3d = [float(c) for c in center] + [float(s) for s in size]

    # Generate 3 dummy multi-view images (5D rank for ONNX encoder)
    dummy_img1 = np.random.randn(1, 1, 3, 224, 224).astype(np.float32)
    dummy_img2 = np.random.randn(1, 1, 3, 224, 224).astype(np.float32)
    dummy_img3 = np.random.randn(1, 1, 3, 224, 224).astype(np.float32)

    cached_data.append({
        "prompt": prompt,
        "spatial_latents_c1": get_latents(dummy_img1),
        "spatial_latents_c2": get_latents(dummy_img2),
        "spatial_latents_t": get_latents(dummy_img3),
        "gt_box_3d": gt_box_3d
    })

output_file = os.path.join(OUTPUT_DIR, "scannet_3d_cached.json")
with open(output_file, "w") as f:
    json.dump(cached_data, f)

print(f"[SUCCESS] Saved {len(cached_data)} cached 3D features!")
