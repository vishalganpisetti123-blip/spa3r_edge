import os
import json
import numpy as np
import torch
import torch.nn as nn

# 2. Spa3R Adapter with Dual Heads
class Spa3RAdapter(nn.Module):
    def __init__(self, in_dim=3, hidden_dim=256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(196 * in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
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
            nn.Linear(512, 196 * in_dim)
        )

    def forward(self, x):
        feat = self.backbone(x)
        centers = self.center_head(feat)
        sizes = self.size_head(feat) + 0.05
        pred_box = torch.cat([centers, sizes], dim=-1)
        
        # Reconstruct target spatial field
        pred_target_feats = self.psfm_head(feat).view(-1, 196, 3)
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

# Load dataset & weights
with open("cached_3d_features/scannet_3d_cached.json", "r") as f:
    samples = json.load(f)[:20]

model = Spa3RAdapter()
model.load_state_dict(torch.load("models/spa3r_adapter_3d_weights.pth"))
model.eval()

ious = []
print("="*60)
print(" 3D SCANNET SPATIAL GROUNDING BENCHMARK (DUAL-LOSS)")
print("="*60)

for i, item in enumerate(samples):
    latents = torch.tensor(item["spatial_latents_c1"], dtype=torch.float32).unsqueeze(0)
    
    gt_box = item["gt_box_3d"]

    with torch.no_grad():
        pred_box, _ = model(latents)
        pred_box = pred_box.squeeze().tolist()

    iou = calculate_3d_iou(pred_box, gt_box)
    ious.append(iou)

    print(f"[{i+1:02d}/20] Prompt: {item['prompt'][:35]}")
    print(f"   -> Pred 3D Box: {[round(x,2) for x in pred_box]}")
    print(f"   -> GT 3D Box:   {[round(x,2) for x in gt_box]}")
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
