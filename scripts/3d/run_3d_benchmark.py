import os
import json
import numpy as np
import torch
import torch.nn as nn

ALLOWED_CLASSES = [
    "table", "chair", "sofa", "couch", "bed", "window", 
    "door", "doorframe", "desk", "cabinet", "monitor", 
    "bookshelf", "kitchen counter", "tv", "refrigerator",
    "coffee table"
]

def encode_prompt(prompt):
    prompt = prompt.lower()
    cls_idx = 0
    for i, c in enumerate(ALLOWED_CLASSES):
        if c in prompt:
            cls_idx = i
            break
            
    mod_idx = 3 # none
    if "left" in prompt: mod_idx = 0
    elif "right" in prompt: mod_idx = 1
    elif "middle" in prompt: mod_idx = 2
    
    vec = np.zeros(20, dtype=np.float32)
    vec[cls_idx] = 1.0
    vec[16 + mod_idx] = 1.0
    return vec

# 2. Spa3R Adapter with Dual Heads
class Spa3RAdapter(nn.Module):
    def __init__(self, in_dim=6, hidden_dim=512, text_dim=20):
        super().__init__()
        # 1. 2D CNN Spatial Backbone
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2), # 14x14 -> 7x7
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2), # 7x7 -> 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out_dim = 256 * 4 * 4 # 4096
        
        # 2. MLP with Text Fusion
        self.mlp = nn.Sequential(
            nn.Linear(cnn_out_dim + text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

        # 3D Box Regression Heads
        self.center_head = nn.Linear(hidden_dim, 3) # [x, y, z]
        self.size_head = nn.Sequential(
            nn.Linear(hidden_dim, 3),             # [dx, dy, dz]
            nn.Softplus()                         # Strictly positive sizes!
        )
        # PSFM Feature Reconstruction Head (L_PSFM)
        self.psfm_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 196 * 3)
        )

    def forward(self, latents, text_vec):
        B = latents.size(0)
        x = latents.permute(0, 2, 1).contiguous().view(B, 6, 14, 14) # [B, 6, 14, 14]
        
        feat = self.cnn_backbone(x)
        fused = torch.cat([feat, text_vec], dim=-1)
        
        out = self.mlp(fused)
        
        centers = self.center_head(out)
        sizes = self.size_head(out) + 0.05
        pred_box = torch.cat([centers, sizes], dim=-1)
        
        pred_target_feats = self.psfm_head(out).contiguous().view(-1, 196, 3)
        return pred_box, pred_target_feats

def calculate_3d_iou(box1, box2):
    """ Computes 3D Axis-Aligned Bounding Box (AABB) IoU. """
    b1_min = np.array(box1[:3]) - np.array(box1[3:]) / 2.0
    b1_max = np.array(box1[:3]) + np.array(box1[3:]) / 2.0
    b2_min = np.array(box2[:3]) - np.array(box2[3:]) / 2.0
    b2_max = np.array(box2[:3]) + np.array(box2[3:]) / 2.0

    inter_min = np.maximum(b1_min, b2_min)
    inter_max = np.minimum(b1_max, b2_max)
    inter_dims = np.maximum(0.0, inter_max - inter_min)
    
    inter_vol = np.prod(inter_dims)
    vol1 = np.prod(box1[3:])
    vol2 = np.prod(box2[3:])
    
    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0.0

import os

def load_axis_alignment(meta_file_path):
    if not os.path.exists(meta_file_path):
        return np.eye(4)
        
    with open(meta_file_path, 'r') as f:
        for line in f:
            if line.startswith('axisAlignment'):
                vals = [float(x) for x in line.strip().split('=')[1].split()]
                return np.array(vals).reshape(4, 4)
    return np.eye(4)

def pixel_to_world_aligned(u, v, depth, intrinsic_matrix, pose_matrix, align_matrix):
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    # 1. 2D -> 3D Camera Coordinates
    x_c = (u - cx) * depth / fx
    y_c = (v - cy) * depth / fy
    z_c = depth
    p_cam = np.array([x_c, y_c, z_c, 1.0])
    
    # 2. Camera -> Unaligned World Space
    p_world_unaligned = pose_matrix @ p_cam
    
    # 3. Unaligned World -> Aligned Bounding Box Space
    p_world_aligned = align_matrix @ p_world_unaligned
    
    return p_world_aligned[:3]

with open("cached_3d_features/scannet_3d_cached_FULL.json", "r") as f:
    samples = json.load(f)

# intrinsic_matrix = np.loadtxt("dataset/scannet/scans/scene0000_00/intrinsic/intrinsic_color.txt")
# POSE_DIR = "dataset/scannet/scans/scene0000_00/pose/"
# align_matrix = load_axis_alignment("dataset/scannet/scans/scene0000_00/scene0000_00.txt")

model = Spa3RAdapter()
model.load_state_dict(torch.load("models/spa3r_adapter_3d_weights.pth"))
model.eval()

ious = []
print("="*60)
print(" 3D SCANNET SPATIAL GROUNDING BENCHMARK (DUAL-LOSS)")
print("="*60)

for i, item in enumerate(samples):
    latents1 = torch.tensor(item["spatial_latents_c1"], dtype=torch.float32)
    latents2 = torch.tensor(item["spatial_latents_c2"], dtype=torch.float32)
    latents = torch.cat([latents1, latents2], dim=-1).unsqueeze(0)
    
    prompt_vec = torch.tensor(encode_prompt(item["prompt"]), dtype=torch.float32).unsqueeze(0)
    
    gt_box = item["gt_box_3d"]
    frame_id = item["target_frame_id"]
    scene_id = item["scene_id"]
    
    scene_dir = os.path.join("dataset/scannet/scans", scene_id)
    pose_matrix = np.loadtxt(os.path.join(scene_dir, "pose", f"{frame_id}.txt"))
    intrinsic_matrix = np.loadtxt(os.path.join(scene_dir, "intrinsic", "intrinsic_color.txt"))
    align_matrix = load_axis_alignment(os.path.join(scene_dir, f"{scene_id}.txt"))

    with torch.no_grad():
        pred_box, _ = model(latents, prompt_vec)
        pred_box = pred_box.squeeze().tolist()
        
    # Un-normalize
    u = pred_box[0] * 1296.0
    v = pred_box[1] * 968.0
    d = pred_box[2] * 10.0
    
    world_centers = pixel_to_world_aligned(u, v, d, intrinsic_matrix, pose_matrix, align_matrix)
    pred_world_box = list(world_centers) + pred_box[3:]

    iou = calculate_3d_iou(pred_world_box, gt_box)
    ious.append(iou)

    print(f"[{i+1:02d}/{len(samples)}] Prompt: {item['prompt'][:35]}")
    print(f"   -> Pred 3D Box (World): {[round(float(x), 2) for x in pred_world_box]}")
    print(f"   -> GT 3D Box (World):   {[round(x,2) for x in gt_box]}")
    print(f"   -> 3D IoU:      {iou:.4f}\n")

m_iou = sum(ious) / len(ious)
acc_25 = sum(1 for x in ious if x >= 0.25) / len(ious) * 100
acc_50 = sum(1 for x in ious if x >= 0.50) / len(ious) * 100

print("="*60)
print(f" FINAL 3D BENCHMARK RESULTS")
print(f" Mean 3D IoU (mIoU): {m_iou:.4f} ({m_iou*100:.2f}%)")
print(f" Accuracy @ 0.25 3D:  {acc_25:.2f}%")
print(f" Accuracy @ 0.50 3D:  {acc_50:.2f}%")
print("="*60)
