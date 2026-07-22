import json
import ast
import os
import time
import requests
import numpy as np
import cv2
import onnxruntime as ort

# --- 1. CONFIGURATION ---
DATASET_DIR = "dataset"
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "benchmark_annotations.json")
SERVER_URL = "http://127.0.0.1:8080/predict"
ONNX_MODEL_PATH = "models/spa3r_encoder_psfm_fp32.onnx"

# --- 2. MATH & EXTRACTION HELPERS ---
def calculate_iou(pred_box, gt_box):
    xA, yA = max(pred_box[0], gt_box[0]), max(pred_box[1], gt_box[1])
    xB, yB = min(pred_box[2], gt_box[2]), min(pred_box[3], gt_box[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    return interArea / float(boxAArea + boxBArea - interArea)


# --- 3. PIPELINE EXECUTION ---
def run_benchmark():
    print(f"[BENCHMARK] Loading annotations from {ANNOTATIONS_FILE}...")
    with open(ANNOTATIONS_FILE, 'r') as f:
        dataset = json.load(f)[:20] # Evaluate on 20 samples for quick benchmarking
        
    print(f"[BENCHMARK] Initializing Edge Encoder: {ONNX_MODEL_PATH}")
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    total_iou = 0.0
    latencies = []
    
    for i, item in enumerate(dataset):
        img_path = os.path.join(DATASET_DIR, item["image_file"])
        print(f"\n[{i+1}/{len(dataset)}] Evaluating: {item['prompt']}")
        
        # A. Edge Preprocessing
        img = cv2.imread(img_path)
        img_resized = cv2.resize(img, (224, 224))
        img_tensor = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0) # Add Batch dimension
        img_tensor = np.expand_dims(img_tensor, axis=0) # Add N_views dimension for PSFM
        
        # B. Edge Inference
        raw_tokens = session.run(None, {input_name: img_tensor})[0]
        
        # C. Cloud Request
        payload = {"edge_tokens": raw_tokens.tolist(), "prompt": item["prompt"]}
        start_time = time.time()
        
        try:
            response = requests.post(SERVER_URL, json=payload)
            response.raise_for_status()
            latency = (time.time() - start_time) * 1000
            latencies.append(latency)
            
            # D. Scoring
            pred_norm = response.json().get("pred_norm", [0, 0, 0, 0])
            
            # Sanitize dimensions and GT box
            width = int(item.get("width", 224))
            height = int(item.get("height", 224))
            
            raw_gt = item.get("gt_box")
            if isinstance(raw_gt, str):
                try:
                    raw_gt = ast.literal_eval(raw_gt)
                except Exception:
                    raw_gt = [0.0, 0.0, 0.0, 0.0]
            
            if isinstance(raw_gt, (list, tuple)):
                gt_box = [float(x) for x in raw_gt]
            else:
                gt_box = [0.0, 0.0, 0.0, 0.0]
            
            pred_box = [
                int(pred_norm[0] * width),
                int(pred_norm[1] * height),
                int(pred_norm[2] * width),
                int(pred_norm[3] * height)
            ]
            
            iou = calculate_iou(pred_box, gt_box)
            total_iou += iou
            
            print(f"  -> Latency : {latency:.2f} ms")
            print(f"  -> Pred Box: {pred_box} | GT Box: {item['gt_box']}")
            print(f"  -> IoU     : {iou:.4f}")
            
        except Exception as e:
            print(f"  -> [ERROR] Failed to process payload: {e}")

    # --- 4. RESULTS ---
    print("\n" + "="*50)
    print("      ACADEMIC BENCHMARK RESULTS")
    print("="*50)
    print(f"Total Images Evaluated : {len(dataset)}")
    print(f"Mean IoU (mIoU)        : {total_iou / len(dataset):.4f}")
    print(f"Average Cloud Latency  : {np.mean(latencies):.2f} ms" if latencies else "N/A")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()
