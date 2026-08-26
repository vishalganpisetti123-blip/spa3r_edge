import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3

import random
import numpy as np

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

def jitter_box(gt_box, noise_scale=0.05):
    if random.random() > 0.5:
        noise = np.random.uniform(-noise_scale, noise_scale, size=6)
        return gt_box + noise
    return gt_box

# 1. Dataset Loader with Context & Target Features
class PSFM3DDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        latents1 = torch.tensor(item["spatial_latents_c1"], dtype=torch.float32)
        latents2 = torch.tensor(item["spatial_latents_c2"], dtype=torch.float32)
        latents = torch.cat([latents1, latents2], dim=-1) # [196, 6]
        
        prompt_vec = torch.tensor(encode_prompt(item["prompt"]), dtype=torch.float32)
        
        gt_box_arr = np.array(item["gt_box_cam"], dtype=np.float32)
        gt_box_arr = jitter_box(gt_box_arr)
        gt_box = torch.tensor(gt_box_arr, dtype=torch.float32)       # [6]
        # Target feature representation for PSFM reconstruction
        target_feats = torch.tensor(item["spatial_latents_t"], dtype=torch.float32)
        return latents, gt_box, target_feats, prompt_vec

# 2. Spa3R Adapter with Dual Heads (CNN + Text Fusion)
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

if __name__ == "__main__":
    dataset = PSFM3DDataset("cached_3d_features/scannet_3d_cached_FULL.json")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = Spa3RAdapter().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    box_loss_fn = nn.SmoothL1Loss(beta=0.1)
    psfm_loss_fn = nn.MSELoss()

    print("[3D TRAIN] Starting Dual-Loss PSFM Training Loop with CNN + Text Conditioning...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss, total_box_l, total_psfm_l = 0.0, 0.0, 0.0
        
        for latents, gt_boxes, target_feats, prompt_vec in dataloader:
            latents = latents.to(DEVICE)
            gt_boxes = gt_boxes.to(DEVICE)
            target_feats = target_feats.to(DEVICE)
            prompt_vec = prompt_vec.to(DEVICE)

            optimizer.zero_grad()
            
            pred_boxes, pred_target_feats = model(latents, prompt_vec)
            
            l_center = box_loss_fn(pred_boxes[:, :3].contiguous(), gt_boxes[:, :3].contiguous())
            l_size = box_loss_fn(pred_boxes[:, 3:].contiguous(), gt_boxes[:, 3:].contiguous())
            l_box = 5.0 * l_center + l_size
            
            l_psfm = psfm_loss_fn(pred_target_feats, target_feats)
            
            loss = l_box + 0.5 * l_psfm
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_box_l += l_box.item()
            total_psfm_l += l_psfm.item()

        scheduler.step()

        avg_loss = total_loss / len(dataloader)
        avg_box = total_box_l / len(dataloader)
        avg_psfm = total_psfm_l / len(dataloader)
        
        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} | Total Loss: {avg_loss:.4f} (Box: {avg_box:.4f}, PSFM: {avg_psfm:.4f})")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/spa3r_adapter_3d_weights.pth")
    print("[SUCCESS] Saved dual-loss Spa3R adapter weights to 'models/spa3r_adapter_3d_weights.pth'!")
