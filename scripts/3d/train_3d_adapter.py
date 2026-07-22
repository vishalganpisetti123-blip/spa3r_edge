import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3

# 1. Dataset Loader with Context & Target Features
class PSFM3DDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        latents = torch.tensor(item["spatial_latents_c1"], dtype=torch.float32) # [196, 3]
        gt_box = torch.tensor(item["gt_box_3d"], dtype=torch.float32)       # [6]
        # Target feature representation for PSFM reconstruction
        target_feats = torch.tensor(item["spatial_latents_t"], dtype=torch.float32)
        return latents, gt_box, target_feats

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

# Initialize
dataset = PSFM3DDataset("cached_3d_features/scannet_3d_cached.json")
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Spa3RAdapter().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

box_loss_fn = nn.SmoothL1Loss(beta=0.1)
psfm_loss_fn = nn.MSELoss()

print("[3D TRAIN] Starting Dual-Loss PSFM Training Loop...")

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss, total_box_l, total_psfm_l = 0.0, 0.0, 0.0
    
    for latents, gt_boxes, target_feats in dataloader:
        latents = latents.to(DEVICE)
        gt_boxes = gt_boxes.to(DEVICE)
        target_feats = target_feats.to(DEVICE)

        optimizer.zero_grad()
        
        pred_boxes, pred_target_feats = model(latents)
        
        # Compute losses
        l_box = box_loss_fn(pred_boxes, gt_boxes)
        l_psfm = psfm_loss_fn(pred_target_feats, target_feats)
        
        loss = l_box + 0.5 * l_psfm
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_box_l += l_box.item()
        total_psfm_l += l_psfm.item()

    avg_loss = total_loss / len(dataloader)
    avg_box = total_box_l / len(dataloader)
    avg_psfm = total_psfm_l / len(dataloader)
    
    print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS:02d} | Total Loss: {avg_loss:.4f} (Box: {avg_box:.4f}, PSFM: {avg_psfm:.4f})")

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/spa3r_adapter_3d_weights.pth")
print("[SUCCESS] Saved dual-loss Spa3R adapter weights to 'models/spa3r_adapter_3d_weights.pth'!")
