import numpy as np
import cv2
import requests
import json
import time

# --- 1. CORE MATH: INTERSECTION OVER UNION (IoU) ---
def calculate_iou(pred_box, gt_box):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.
    Boxes are in format: [x_min, y_min, x_max, y_max]
    """
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    boxBArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# --- 2. PIPELINE: HEATMAP TO BOUNDING BOX ---
def extract_bounding_box_from_heatmap(heatmap_14x14, orig_width, orig_height, percentile=85):
    """
    Extracts bounding box for the highest activation cluster while suppressing
    border/interpolation artifacts at the edges of the image canvas.
    """
    heatmap_resized = cv2.resize(np.array(heatmap_14x14), (orig_width, orig_height), interpolation=cv2.INTER_CUBIC)
    
    # 1. Suppress canvas edge boundary artifacts (zero out outer 2% border)
    border_x = int(orig_width * 0.02)
    border_y = int(orig_height * 0.02)
    
    mask_core = np.zeros_like(heatmap_resized)
    mask_core[border_y:-border_y, border_x:-border_x] = heatmap_resized[border_y:-border_y, border_x:-border_x]
    
    # 2. Isolate top spatial activation regions (top 15% brightest core pixels)
    thresh_val = np.percentile(mask_core, percentile)
    binary_mask = (mask_core >= thresh_val).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return [0, 0, 0, 0]

    # 3. Select the contour with the highest spatial activation density
    best_contour = None
    max_density = -1.0
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 15 and h > 15: # Ignore minor noise specs
            contour_mask = np.zeros_like(binary_mask)
            cv2.drawContours(contour_mask, [c], -1, 255, -1)
            mean_val = cv2.mean(mask_core, mask=contour_mask)[0]
            if mean_val > max_density:
                max_density = mean_val
                best_contour = c
                
    if best_contour is None:
        return [0, 0, 0, 0]
        
    bx, by, bw, bh = cv2.boundingRect(best_contour)
    return [int(bx), int(by), int(bx + bw), int(by + bh)]

# --- 3. AUTOMATED BENCHMARK LOOP ---
def run_refcoco_benchmark():
    print("[BENCHMARK] Initializing Spa3R-Qwen2-VL Evaluation Loop...")
    
    # MOCK DATASET: Replace this with your actual RefCOCO / VSI-Bench JSON loader
    # Format: {"image_path": str, "prompt": str, "gt_box": [x1, y1, x2, y2], "img_size": (w, h)}
    mock_dataset = [
        {
            "image_path": "data/gym.webp",
            "prompt": "Isolate the foreground dumbbells on the rug.",
            "gt_box": [100, 750, 300, 850], # Example ground truth coordinates
            "img_size": (1024, 1024)
        }
    ]
    
    server_url = 'http://localhost:8080/predict'
    total_iou = 0.0
    latencies = []

    for i, item in enumerate(mock_dataset):
        print(f"\nEvaluating Sample {i+1}/{len(mock_dataset)}: '{item['prompt']}'")
        
        # 1. Edge Inference (Simulated here: replace with your actual ONNX session run)
        # edge_tokens = onnx_session.run(None, {input_name: preprocess(item["image_path"])})[0]
        dummy_edge_tokens = np.random.rand(3, 14, 14).tolist() 
        
        # 2. Cloud Transmission & VLM Alignment
        start_time = time.time()
        payload = {"edge_tokens": dummy_edge_tokens, "prompt": item["prompt"]}
        
        try:
            response = requests.post(server_url, json=payload)
            response.raise_for_status()
            cloud_latency = (time.time() - start_time) * 1000
            latencies.append(cloud_latency)
            
            # 3. Process Result
            spatial_field = response.json()["spatial_field"]
            channel_0 = spatial_field[0] # Target Activation Map
            
            # 4. Score the output
            w, h = item["img_size"]
            pred_box = extract_bounding_box_from_heatmap(channel_0, w, h)
            iou_score = calculate_iou(pred_box, item["gt_box"])
            total_iou += iou_score
            
            print(f" -> VLM Latency : {cloud_latency:.2f} ms")
            print(f" -> Pred Box    : {pred_box}")
            print(f" -> Ground Truth: {item['gt_box']}")
            print(f" -> IoU Score   : {iou_score:.4f}")

        except requests.exceptions.RequestException as e:
            print(f" -> [ERROR] Network failure: {e}")
            continue

    # --- FINAL REPORT ---
    mean_iou = total_iou / len(mock_dataset)
    mean_latency = np.mean(latencies) if latencies else 0
    
    print("\n==================================================")
    print("           SPATIAL BENCHMARK RESULTS")
    print("==================================================")
    print(f"Mean IoU (mIoU)        : {mean_iou:.4f}")
    print(f"Average Cloud Latency  : {mean_latency:.2f} ms")
    print("==================================================")

if __name__ == "__main__":
    run_refcoco_benchmark()
