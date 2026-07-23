import os
import time
import json
import requests
import numpy as np
from tqdm import tqdm

def calculate_3d_iou(box1, box2):
    """
    Calculate 3D Intersection over Union (IoU) for two axis-aligned bounding boxes.
    Boxes are in format [x, y, z, l, w, h] where x,y,z is the center.
    """
    # Convert from [x, y, z, l, w, h] to [x_min, y_min, z_min, x_max, y_max, z_max]
    b1_min = [box1[0] - box1[3]/2, box1[1] - box1[4]/2, box1[2] - box1[5]/2]
    b1_max = [box1[0] + box1[3]/2, box1[1] + box1[4]/2, box1[2] + box1[5]/2]
    
    b2_min = [box2[0] - box2[3]/2, box2[1] - box2[4]/2, box2[2] - box2[5]/2]
    b2_max = [box2[0] + box2[3]/2, box2[1] + box2[4]/2, box2[2] + box2[5]/2]
    
    # Calculate intersection coordinates
    inter_min = np.maximum(b1_min, b2_min)
    inter_max = np.minimum(b1_max, b2_max)
    
    # Calculate intersection volume
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = inter_dims[0] * inter_dims[1] * inter_dims[2]
    
    # Calculate union volume
    b1_vol = box1[3] * box1[4] * box1[5]
    b2_vol = box2[3] * box2[4] * box2[5]
    union_vol = b1_vol + b2_vol - inter_vol
    
    if union_vol <= 0:
        return 0.0
    return inter_vol / union_vol

def run_benchmark(num_samples=50):
    print("Loading Edge Dataset Cache...")
    with open("cached_3d_features/scannet_3d_cached_FULL.json", "r") as f:
        data = json.load(f)
        
    samples_to_test = data[:num_samples] if len(data) > num_samples else data
    total_samples = len(samples_to_test)
    
    print(f"Starting benchmark on {total_samples} samples...")
    
    ious = []
    latencies = []
    
    # Counter for fallback failures (402 errors from Qwen API)
    fallback_count = 0
    
    for i, sample in enumerate(tqdm(samples_to_test, desc="Benchmarking Edge-Cloud API")):
        gt_box = sample['gt_box_cam']
        prompt = sample['prompt']
        
        # 1. Edge Encoding (Simulated by extracting from cache)
        # Latents are already encoded by CNN/ONNX model
        c1 = np.array(sample['spatial_latents_c1'], dtype=np.float32)
        c2 = np.array(sample['spatial_latents_c2'], dtype=np.float32)
        latents = np.concatenate([c1, c2], axis=-1) # [196, 6]
        
        data_payload = {
            "prompt": prompt,
            "dtype": "float32",
            "shape": "196,6"
        }
        
        binary_payload = latents.tobytes()
        files = {
            "tensor_bytes": ("latents.bin", binary_payload, "application/octet-stream")
        }
        
        # 2. Cloud Transmission & Processing
        try:
            start_time = time.time()
            response = requests.post("http://localhost:8000/predict_3d_spatial/", data=data_payload, files=files)
            response.raise_for_status()
            result = response.json()
            latency = (time.time() - start_time) * 1000 # ms
            
            latencies.append(latency)
            
            # 3. Accuracy Evaluation
            pred_box = result['spa3r_adapter_box']
            iou = calculate_3d_iou(pred_box, gt_box)
            ious.append(iou)
            
            # Track if Qwen API failed and triggered fallback
            if "Qwen API failed" in result.get('qwen_spatial_response', ''):
                fallback_count += 1
                
        except Exception as e:
            print(f"\n[ERROR] Sample {i} failed: {e}")
            continue

    if not ious:
        print("\n[ERROR] No successful samples to benchmark. Is the server running?")
        return

    # Aggregate Metrics
    mean_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    ious = np.array(ious)
    miou = np.mean(ious)
    acc_25 = np.mean(ious >= 0.25) * 100
    acc_50 = np.mean(ious >= 0.50) * 100

    print("\n" + "="*50)
    print(" BENCHMARKING REPORT: EDGE-CLOUD PIPELINE ")
    print("="*50)
    print(f"Total Samples Tested : {len(ious)}")
    print(f"Binary Payload Size  : {len(binary_payload) / 1024:.2f} KB (Per Request)")
    print(f"API Fallbacks        : {fallback_count} / {len(ious)} (due to API Key Billing)")
    print("-" * 50)
    print(f"Mean Roundtrip Latency : {mean_latency:.2f} ms")
    print(f"95th Percentile Latency: {p95_latency:.2f} ms")
    print("-" * 50)
    print(f"Mean 3D IoU          : {miou:.4f}")
    print(f"Accuracy @ IoU=0.25  : {acc_25:.1f}%")
    print(f"Accuracy @ IoU=0.50  : {acc_50:.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_benchmark(num_samples=100)
