import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from adapter import ResidualCrossAttentionAdapter, SpatialRegressionHead
import json
import re
import ast

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.bfloat16

print(f"[TRAIN SETUP] Hardware Target: {device.upper()} | Precision: {dtype}")

# --- CACHED DATASET ---
class CachedRefCOCODataset(Dataset):
    def __init__(self, cache_dir="cached_features", annotation_file="dataset/benchmark_annotations.json"):
        self.cache_dir = cache_dir
        self.files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith(".pt")]
        with open(annotation_file, "r") as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        data = torch.load(file_path)
        
        # Extract sample ID K from "sample_K.pt"
        filename = os.path.basename(file_path)
        match = re.search(r"sample_(\d+)\.pt", filename)
        if match:
            k = int(match.group(1))
            ann = self.annotations[k]
            raw_gt = ann.get("gt_box", [0, 0, 0, 0])
            
            if isinstance(raw_gt, str):
                try:
                    raw_gt = ast.literal_eval(raw_gt)
                except Exception:
                    raw_gt = [0.0, 0.0, 0.0, 0.0]
                    
            if isinstance(raw_gt, (list, tuple)):
                gt_box = [float(x) for x in raw_gt]
            else:
                gt_box = [0.0, 0.0, 0.0, 0.0]
                
            img_w = float(ann.get("width", 1))
            img_h = float(ann.get("height", 1))
            
            # Normalize to [0, 1]
            gt_box_norm = [
                gt_box[0] / img_w,
                gt_box[1] / img_h,
                gt_box[2] / img_w,
                gt_box[3] / img_h
            ]
        else:
            gt_box_norm = [0.0, 0.0, 0.0, 0.0]
            
        return data["spatial_latents"], data["vlm_features"], data["input_ids"], torch.tensor(gt_box_norm, dtype=torch.float32)

def cached_collate_fn(batch):
    spatial_latents = torch.stack([item[0] for item in batch])
    vlm_features = torch.nn.utils.rnn.pad_sequence([item[1] for item in batch], batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True, padding_value=0)
    gt_boxes_norm = torch.stack([item[3] for item in batch])
    return spatial_latents, vlm_features, targets, gt_boxes_norm

# --- INITIALIZE ADAPTER & HEAD ---
spatial_adapter = ResidualCrossAttentionAdapter(vlm_dim=1536, spa_dim=3).to(device).to(torch.float32)
spatial_adapter.train()

regression_head = SpatialRegressionHead(vlm_dim=1536, out_dim=4).to(device).to(torch.float32)
regression_head.train()

optimizer = AdamW(list(spatial_adapter.parameters()) + list(regression_head.parameters()), lr=3e-5, weight_decay=1e-2)
loss_fn = torch.nn.SmoothL1Loss(beta=1.0)

train_dataset = CachedRefCOCODataset()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=cached_collate_fn)

# --- LIGHTNING-FAST TRAINING LOOP ---
epochs = 25
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
print(f"\n[TRAIN CORE] Training Adapter on {len(train_dataset)} cached samples...")

for epoch in range(epochs):
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for spatial_latents, vlm_features, targets, gt_boxes_norm in progress_bar:
        optimizer.zero_grad()
        
        spatial_latents = spatial_latents.to(device)
        vlm_features = vlm_features.to(device)
        gt_boxes_norm = gt_boxes_norm.to(device)
        
        # Forward pass through adapter
        fused_embeddings = spatial_adapter(vlm_features, spatial_latents)[0].to(dtype)
        
        # Pool the embeddings (e.g. use the final token)
        pooled_embeddings = fused_embeddings[:, -1, :].to(torch.float32)
        
        # Forward pass through regression head
        pred_boxes_norm = regression_head(pooled_embeddings)
        
        # Coordinate Loss
        loss = loss_fn(pred_boxes_norm, gt_boxes_norm) 
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
    scheduler.step()
    print(f"Epoch {epoch+1} Completed | Average Loss: {epoch_loss / len(train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

# Save updated weights
os.makedirs("models", exist_ok=True)
torch.save(spatial_adapter.state_dict(), "models/spa3r_adapter_weights.pth")
torch.save(regression_head.state_dict(), "models/spa3r_head_weights.pth")
print("\n[TRAIN SUCCESS] Saved trained adapter and head weights to 'models/'")
