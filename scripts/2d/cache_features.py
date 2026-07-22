import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import onnxruntime as ort

# --- CONFIGURATION ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16
CACHE_DIR = "cached_features"
ANNOTATIONS_FILE = "dataset/benchmark_annotations.json"
ONNX_MODEL_PATH = "models/spa3r_encoder_psfm_fp32.onnx" # Note: updating this path to the actual ONNX model in our project

os.makedirs(CACHE_DIR, exist_ok=True)

# --- LOAD MODELS ---
print("[CACHE] Loading Qwen2-VL VLM...")
model_id = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=dtype, device_map=device
)
vlm_model.eval()

print("[CACHE] Loading ONNX Edge Encoder...")
onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = onnx_session.get_inputs()[0].name

with open(ANNOTATIONS_FILE, "r") as f:
    dataset = json.load(f)

print(f"[CACHE] Pre-computing features for {len(dataset)} samples...")

for i, item in enumerate(tqdm(dataset)):
    cache_path = os.path.join(CACHE_DIR, f"sample_{i}.pt")
    if os.path.exists(cache_path):
        continue

    # 1. Process Image for ONNX
    img_path = os.path.join("dataset", item["image_file"])
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224))
    img_tensor = np.expand_dims(np.expand_dims(img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0, axis=0), axis=0)

    # 2. Extract ONNX Edge Tokens
    raw_tokens = onnx_session.run(None, {input_name: img_tensor})[0]
    spatial_latents = raw_tokens[0] # [196, 3]

    # 3. Process Prompt for Qwen2-VL
    messages = [
        {"role": "system", "content": "You are a 3D spatial AI."},
        {"role": "user", "content": item["prompt"]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(device)

    # 4. Extract Frozen VLM Hidden States
    with torch.no_grad():
        outputs = vlm_model.model(**inputs, output_hidden_states=True)
        vlm_features = outputs.hidden_states[-1].squeeze(0).to(torch.float32).cpu()
        input_ids_cpu = inputs.input_ids.squeeze(0).cpu()
        
    # Free VLM memory to prevent MPS memory leak slowdowns
    del outputs
    del inputs
    if device == "mps":
        torch.mps.empty_cache()

    # Save lightweight tensors to disk
    torch.save({
        "spatial_latents": torch.tensor(spatial_latents, dtype=torch.float32),
        "vlm_features": vlm_features,
        "input_ids": input_ids_cpu
    }, cache_path)

print(f"\n[CACHE SUCCESS] All features cached to '{CACHE_DIR}/'!")
