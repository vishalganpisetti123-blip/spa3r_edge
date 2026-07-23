import torch
import numpy as np
import os
import argparse
from train_3d_adapter import Spa3RAdapter, PSFM3DDataset
import torchvision.utils as vutils
import torch.nn.functional as F
from torchvision.utils import draw_bounding_boxes

def visualize(custom_prompt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = Spa3RAdapter().to(device)
    weights_path = "models/spa3r_adapter_3d_weights.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Loaded weights.")
    model.eval()

    # Load dataset
    import json
    with open("cached_3d_features/scannet_3d_cached_FULL.json", "r") as f:
        raw_data = json.load(f)
        
    dataset = PSFM3DDataset("cached_3d_features/scannet_3d_cached_FULL.json")
    
    # Filter indices based on prompt if provided
    indices = []
    if custom_prompt:
        for i, item in enumerate(raw_data):
            if custom_prompt.lower() in item['prompt'].lower():
                indices.append(i)
        if not indices:
            print(f"No samples found for prompt: {custom_prompt}")
            indices = np.random.choice(len(dataset), 4, replace=False).tolist()
        else:
            np.random.shuffle(indices)
            indices = indices[:4]
    else:
        indices = np.random.choice(len(dataset), 4, replace=False).tolist()
    
    img_list = []
    
    for i, idx in enumerate(indices):
        latents_cpu, gt_box, target_feats_cpu, prompt_vec_cpu = dataset[idx]
        actual_prompt = raw_data[idx]['prompt']
        print(f"[{i+1}/4] Prompt: {actual_prompt}")
        
        # Prepare input
        latents = latents_cpu.unsqueeze(0).to(device)
        prompt_vec = prompt_vec_cpu.unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_boxes, pred_target_feats = model(latents, prompt_vec)
        
        # Extract individual views for visualization
        c1 = latents_cpu[:, :3].view(14, 14, 3).permute(2, 0, 1) # [3, 14, 14]
        c2 = latents_cpu[:, 3:].view(14, 14, 3).permute(2, 0, 1) # [3, 14, 14]
        
        # Target (GT) features from c3 (3 channels, 14x14)
        gt_feat = target_feats_cpu.view(14, 14, 3).permute(2, 0, 1) # [3, 14, 14]
        
        # Predicted target features
        pred_feat = pred_target_feats.squeeze(0).view(14, 14, 3).permute(2, 0, 1).cpu() # [3, 14, 14]
        
        # Normalize to 0-1 for visualization
        def norm(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 1e-6:
                return (x - x_min) / (x_max - x_min)
            return x
            
        # Upsample to make them look nice like the paper (e.g. 224x224)
        def upsample(x):
            return F.interpolate(x.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
            
        c1_up = upsample(norm(c1))
        c2_up = upsample(norm(c2))
        gt_up = upsample(norm(gt_feat))
        pred_up = upsample(norm(pred_feat))
        
        # Draw bounding boxes!
        # Convert float images (0-1) to uint8 (0-255) for bounding box drawing
        def to_uint8(x):
            return (x * 255).to(torch.uint8)
            
        gt_up_u8 = to_uint8(gt_up)
        pred_up_u8 = to_uint8(pred_up)
        
        # Calculate 2D Box coordinates on the 224x224 grid
        H, W = 224, 224
        box_size = 40 # Fixed 2D size for visualization
        
        # GT Box (Green)
        gt_u, gt_v = gt_box[0].item(), gt_box[1].item()
        gx, gy = int(gt_u * W), int(gt_v * H)
        gx_min, gx_max = max(0, gx - box_size//2), min(W-1, gx + box_size//2)
        gy_min, gy_max = max(0, gy - box_size//2), min(H-1, gy + box_size//2)
        gt_box_2d = torch.tensor([[gx_min, gy_min, gx_max, gy_max]], dtype=torch.float)
        gt_up_u8 = draw_bounding_boxes(gt_up_u8, gt_box_2d, colors="green", width=3)
        
        # Pred Box (Red)
        pred_u, pred_v = pred_boxes[0, 0].item(), pred_boxes[0, 1].item()
        px, py = int(pred_u * W), int(pred_v * H)
        px_min, px_max = max(0, px - box_size//2), min(W-1, px + box_size//2)
        py_min, py_max = max(0, py - box_size//2), min(H-1, py + box_size//2)
        pred_box_2d = torch.tensor([[px_min, py_min, px_max, py_max]], dtype=torch.float)
        pred_up_u8 = draw_bounding_boxes(pred_up_u8, pred_box_2d, colors="red", width=3)
        
        # Convert back to float 0-1 for saving
        gt_up = gt_up_u8.float() / 255.0
        pred_up = pred_up_u8.float() / 255.0
        
        # Stack them side-by-side: [C1, C2, GT Target, Pred Target]
        row = torch.cat([c1_up, c2_up, gt_up, pred_up], dim=2) # Concat along width
        img_list.append(row)
        
    # Combine all samples vertically
    final_grid = torch.cat(img_list, dim=1) # Concat along height
    
    os.makedirs('outputs', exist_ok=True)
    vutils.save_image(final_grid, "outputs/feature_visualization.png")
    print("Saved visualization to outputs/feature_visualization.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to filter by")
    args = parser.parse_args()
    visualize(args.prompt)
