import requests
import json
import numpy as np
import time

def test_cloud_api():
    print("Loading a test sample from the Edge Device cache...")
    # Simulate the edge device by loading a pre-extracted spatial latent
    with open("cached_3d_features/scannet_3d_cached_FULL.json", "r") as f:
        data = json.load(f)
        
    if not data:
        print("Error: No cached data found!")
        return
        
    # Pick a random sample
    sample = data[0]
    
    # The edge device concatenates the two context latents
    c1 = np.array(sample['spatial_latents_c1'], dtype=np.float32) # [196, 3]
    c2 = np.array(sample['spatial_latents_c2'], dtype=np.float32) # [196, 3]
    latents = np.concatenate([c1, c2], axis=-1) # [196, 6]
    
    # Send a complex prompt to test the Qwen reasoning & Spatial injection
    prompt = "I'm looking for the cabinet. Is it big enough to fit a large TV inside?"
    
    # Prepare the payload for multipart/form-data
    data_payload = {
        "prompt": prompt,
        "dtype": "float32",
        "shape": "196,6"
    }
    
    # Prepare the binary file
    files = {
        "tensor_bytes": ("latents.bin", latents.tobytes(), "application/octet-stream")
    }
    
    print(f"\n--- SENDING REQUEST TO CLOUD VLM ---")
    print(f"User Prompt: '{prompt}'")
    print(f"Spatial Tensor Shape: {latents.shape}")
    print(f"Binary Payload Size: {len(latents.tobytes()) / 1024:.2f} KB")
    
    # Send the request to the Cloud API
    start_time = time.time()
    try:
        response = requests.post("http://localhost:8000/predict_3d_spatial/", data=data_payload, files=files)
        response.raise_for_status()
        
        result = response.json()
        end_time = time.time()
        
        print("\n--- CLOUD RESPONSE RECEIVED ---")
        print(f"Status: {result['status']}")
        print(f"Original Prompt: '{result['original_prompt']}'")
        print(f"Parsed Intent (Stage A): '{result.get('parsed_intent', 'N/A')}'")
        
        box = result['spa3r_adapter_box']
        print(f"Spa3R Predicted Center: [x={box[0]}, y={box[1]}, z={box[2]}]")
        print(f"Spa3R Predicted Size: [l={box[3]}m, w={box[4]}m, h={box[5]}m]")
        
        print(f"\n--- QWEN SPATIAL REASONING (Stage C & D) ---")
        print(result.get('qwen_spatial_response', 'N/A'))
        
        print(f"\nTotal Round-Trip Latency: {result.get('total_latency_ms', 0)} ms")
        
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Connection failed. Is the cloud_vlm_api.py server running on port 8000?")
    except requests.exceptions.HTTPError as e:
        print(f"\n[ERROR] HTTP Request failed: {e}")
        if response.content:
            print(f"Response Content: {response.content.decode()}")
    except Exception as e:
        print(f"\n[ERROR] Request failed: {e}")

if __name__ == "__main__":
    test_cloud_api()
