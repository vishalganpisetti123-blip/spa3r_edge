import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import open3d as o3d

class ScanReferDataset(Dataset):
    def __init__(self, scanrefer_json, scannet_dir, processor, num_points=40000):
        # Load the ScanRefer annotations
        with open(scanrefer_json, 'r') as f:
            self.data = json.load(f)
            
        self.scannet_dir = scannet_dir
        self.processor = processor
        self.num_points = num_points

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        scene_id = item["scene_id"]
        prompt = item["description"]
        
        # 1. Load the 3D Point Cloud (.ply)
        # ScanNet scenes are stored as sceneXXXX_YY_vh_clean_2.ply
        ply_path = os.path.join(self.scannet_dir, scene_id, f"{scene_id}_vh_clean_2.ply")
        if not os.path.exists(ply_path):
            raise FileNotFoundError(f"Missing point cloud: {ply_path}")
            
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # Extract physical XYZ coordinates and RGB color data
        points = np.asarray(pcd.points) 
        colors = np.asarray(pcd.colors) 
        
        # Concatenate into a single point cloud array: [N_points, 6]
        point_cloud = np.concatenate([points, colors], axis=1).astype(np.float32)
        
        # 2. Point Cloud Subsampling
        # GPUs require fixed-size tensors for batching. We randomly sample 40,000 points.
        if len(point_cloud) > self.num_points:
            indices = np.random.choice(len(point_cloud), self.num_points, replace=False)
            point_cloud = point_cloud[indices]
        else:
            padding = np.zeros((self.num_points - len(point_cloud), 6), dtype=np.float32)
            point_cloud = np.vstack((point_cloud, padding))
            
        # 3. Extract 3D Ground Truth Bounding Box 
        # Output format: [x, y, z, length, width, height]
        # (Assuming the JSON stores these under 'center' and 'size' keys)
        center = np.array(item.get("center", [0,0,0]), dtype=np.float32) 
        size = np.array(item.get("size", [1,1,1]), dtype=np.float32)     
        target_3d_box = np.concatenate([center, size])
        
        # 4. Tokenize the Language Prompt for Qwen2-VL
        messages = [
            {"role": "system", "content": "You are a 3D spatial AI. Predict the 3D bounding box [x, y, z, l, w, h]."},
            {"role": "user", "content": prompt}
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        return {
            "spatial_latent": torch.tensor(point_cloud), # This replaces the 2D ONNX edge tokens
            "target_box": torch.tensor(target_3d_box),
            "text": text
        }

def custom_collate_3d(batch, processor):
    """Dynamic padding for text sequences alongside 3D tensor batching."""
    spatial_latents = torch.stack([item["spatial_latent"] for item in batch])
    target_boxes = torch.stack([item["target_box"] for item in batch])
    texts = [item["text"] for item in batch]
    
    # processor needs to be available in the module scope or passed in
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    
    return inputs, spatial_latents, target_boxes
