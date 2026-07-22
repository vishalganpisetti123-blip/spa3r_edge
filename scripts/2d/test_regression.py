import torch
from adapter import ResidualCrossAttentionAdapter, SpatialRegressionHead

device = "cpu"

vlm_features = torch.randn(2, 50, 1536, device=device) # Batch: 2, Seq: 50, Dim: 1536
spatial_latents = torch.randn(2, 40000, 6, device=device) # Batch: 2, Points: 40k, Dim: 6

adapter = ResidualCrossAttentionAdapter(vlm_dim=1536, spa_dim=6).to(device)
regression = SpatialRegressionHead(vlm_dim=1536, out_dim=6).to(device)

fused_embeddings, attn = adapter(vlm_features, spatial_latents)
final_token = fused_embeddings[:, -1, :]
pred_boxes = regression(final_token)

print(f"Fused Embeddings Shape: {fused_embeddings.shape}")
print(f"Pred Boxes Shape: {pred_boxes.shape}")
print("SUCCESS: 3D Regression Pipeline Dry-Run complete.")
