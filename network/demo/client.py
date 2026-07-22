import requests
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import onnxruntime as ort
from torchvision import transforms
from PIL import Image

SERVER_URL = "http://localhost:8080/predict"
ONNX_MODEL_PATH = "models/spa3r_encoder_int8.onnx"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def run_onnx_edge_inference(image_path, text_prompt):
    print(f"\n[EDGE ENGINE] Initializing Optimized Quantized Pipeline: {ONNX_MODEL_PATH}")
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"[ERROR] Quantized model file '{ONNX_MODEL_PATH}' not found in root workspace.")
        return

    try:
        raw_file_size = os.path.getsize(image_path)
        img = Image.open(image_path).convert('RGB')
        img_t = transform(img).unsqueeze(0).unsqueeze(0).numpy() # Shape: [1, 1, 3, 224, 224]
        query_slots = np.zeros((1, 256, 16), dtype=np.float32)
        
        # --- Local Quantized ONNX Execution ---
        print("[EDGE ENGINE] Loading INT8 graph and processing patch features...")
        session = ort.InferenceSession(ONNX_MODEL_PATH)
        
        # Run local forward inference pass on the Mac hardware
        onnx_start = time.time()
        onnx_outputs = session.run(None, {'context_views': img_t, 'query_slots': query_slots})
        onnx_latency = (time.time() - onnx_start) * 1000
        
        # Extract the structured patch layers (mapping out features down to [3, 14, 14])
        raw_features = onnx_outputs[0]
        
        # The model outputs a flattened 16x16 grid (256 length).
        # We reshape it, crop to 14x14, and expand to 3 channels for the server.
        flat_grid = raw_features.flatten()
        spatial_grid = flat_grid.reshape(16, 16)
        cropped_grid = spatial_grid[:14, :14]
        edge_tokens = np.stack([cropped_grid, cropped_grid, cropped_grid], axis=0)

        # --- LOCAL GEOMETRIC MAPPING (OFFLINE VISUALIZATION) ---
        # The ONNX model outputs the 196 (14x14) x 3-channel geometry tokens.
        # We calculate the feature magnitude (norm) to create a local geometric complexity map.
        print(f"\n[EDGE ENGINE] Local Offline Visualization Loop:")
        
        # Calculate feature magnitude at each 14x14 grid location: sqrt(c0^2 + c1^2 + c2^2)
        depth_proxy = np.sqrt(np.sum(edge_tokens**2, axis=0)) # Shape [14, 14]

        # Apply min-max normalization to fit values 0.0 to 1.0 cleanly
        norm_depth = (depth_proxy - depth_proxy.min()) / (depth_proxy.max() - depth_proxy.min() + 1e-8)

        # Plot and save the local depth/saliency map without hitting the cloud
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 6))
        plt.imshow(norm_depth, cmap='inferno', interpolation='nearest', vmin=0, vmax=1)
        plt.colorbar(label="Geometric Saliency")
        plt.title(f"Edge Saliency (Generated Local M-series Silicon)")
        plt.axis('off')
        plt.savefig('edge_depth_map.png', bbox_inches='tight')
        plt.close()
        print(f"[EDGE ENGINE] Saved local geometric depth proxy to: edge_depth_map.png")
        # --------------------------------------------------------

        payload_data = {
            "edge_tokens": edge_tokens.tolist(),
            "prompt": text_prompt
        }
        
        json_payload_string = json.dumps(payload_data)
        network_payload_size = len(json_payload_string.encode('utf-8'))
        
        print(f"[EDGE ENGINE] Local inference complete ({onnx_latency:.2f} ms). Streaming spatial array layout...")
        
        # Network Round-trip tracking
        net_start = time.time()
        response = requests.post(SERVER_URL, json=payload_data)
        net_latency = (time.time() - net_start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            spatial_field = np.array(result.get("spatial_field"), dtype=np.float32) # Shape: [2, 14, 14]
            
            # --- TELEMETRY PERFORMANCE REPORT ---
            compression_ratio = raw_file_size / network_payload_size
            print("\n" + "="*50)
            print("         SPA3R ONNX CORE TELEMETRY REPORT        ")
            print("="*50)
            print(f" Local INT8 Engine Latency  : {onnx_latency:.2f} ms")
            print(f" Network Transmission Delay : {net_latency:.2f} ms")
            print(f" Source Image Disk Weight   : {raw_file_size:,} bytes")
            print(f" Transmitted Vector Burden  : {network_payload_size:,} bytes")
            print(f" Optimization Scaling factor: {compression_ratio:.2f}x compressed")
            print("="*50 + "\n")
            
            # Multi-channel 3-panel visualization output render block
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(img.resize((224, 224)))
            axes[0].set_title("Edge Sensor View")
            axes[0].axis('off')
            
            im1 = axes[1].imshow(spatial_field[0], cmap='inferno', vmin=0, vmax=1)
            axes[1].set_title("Channel 0: High-Contrast Edges")
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            im2 = axes[2].imshow(spatial_field[1], cmap='inferno', vmin=0, vmax=1)
            axes[2].set_title("Channel 1: Flat Uniform Surfaces")
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.suptitle(f"Multi-Channel Spatial Attention Query: '{text_prompt}'", fontsize=11)
            
            output_filename = "data/spatial_visualization.png"
            plt.savefig(output_filename, bbox_inches='tight')
            print(f"[SUCCESS] Multi-Channel telemetry maps rendered to: {output_filename}")
            
        else:
            print(f"\n[SERVER ERROR] Status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"\n[EDGE CRITICAL ERROR] Pipeline execution aborted: {e}")

if __name__ == "__main__":
    # 1. Point the edge encoder to the new gym image
    image_path = "data/gym.webp"

    # 2. Set the dynamic Qwen target query
    prompt = "Isolate the central treadmill console and the foreground dumbbells."
    
    run_onnx_edge_inference(image_path, prompt)