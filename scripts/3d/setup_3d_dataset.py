import os
import json

def main():
    print("[SETUP] Creating 3D dataset structure for ScanNet...")
    dataset_dir = "dataset/scannet"
    os.makedirs(dataset_dir, exist_ok=True)
    
    file_path = os.path.join(dataset_dir, "ScanRefer_filtered.json")
    
    # Check if the real dataset was already downloaded
    if os.path.exists(file_path):
        # The file might be our 14-byte 404 error
        if os.path.getsize(file_path) > 1000:
            print(f"[SETUP] {file_path} already exists and looks valid. Skipping mock generation.")
            return
            
    print(f"[SETUP] The official ScanRefer_filtered.json requires an academic agreement or download from the author's Google Drive.")
    print(f"[SETUP] To unblock testing, generating a 1000-sample mock {file_path} with proper 3D bounding boxes...")
    
    mock_data = []
    import random
    for i in range(1000):
        mock_data.append({
            "scene_id": f"scene{i:04d}_00",
            "object_id": str(i % 5),
            "description": f"synthetic object {i} in room",
            "center": [
                random.uniform(-5.0, 5.0),
                random.uniform(-5.0, 5.0),
                random.uniform(0.0, 3.0)
            ],
            "size": [
                random.uniform(0.1, 2.0),
                random.uniform(0.1, 2.0),
                random.uniform(0.1, 2.0)
            ]
        })
    
    with open(file_path, "w") as f:
        json.dump(mock_data, f, indent=2)
        
    print(f"[SUCCESS] Wrote {len(mock_data)} mock samples to {file_path}")
    print(f"[INFO] You can now run `python cache_3d_features.py` to cache the 3D features.")

if __name__ == "__main__":
    main()
