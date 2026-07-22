import torch
import torch.nn as nn

class ResidualCrossAttentionAdapter(nn.Module):
    def __init__(self, vlm_dim=1536, spa_dim=3, num_heads=8):
        super().__init__()
        self.vlm_dim = vlm_dim
        
        # Project spatial tokens (dim=3) up to VLM embedding dimension (dim=1536)
        self.dropout = nn.Dropout(p=0.1)
        self.spatial_proj = nn.Linear(spa_dim, vlm_dim)
        
        # Cross-Attention: VLM queries the Spatial Context
        self.cross_attn = nn.MultiheadAttention(embed_dim=vlm_dim, num_heads=num_heads, batch_first=True)
        
        # Zero-initialized MLP to ensure stable training at initialization
        self.mlp_gate = nn.Sequential(
            nn.Linear(vlm_dim, vlm_dim),
            nn.GELU(),
            nn.Linear(vlm_dim, vlm_dim)
        )
        
        # Initialize the final layer with small normal weights so gradients flow immediately
        nn.init.normal_(self.mlp_gate[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp_gate[-1].bias)
        
        self.layer_norm = nn.LayerNorm(vlm_dim)

    def forward(self, vlm_features, spatial_latent):
        """
        vlm_features: Native 2D visual embeddings from Qwen [Batch, Seq_VLM, 1536]
        spatial_latent: The 3D geometry tokens from the Edge [Batch, Seq_Spa, 3]
        """
        # Project edge spatial tokens into the VLM's dimensional space
        spatial_latent = self.dropout(spatial_latent)
        spatial_context = self.spatial_proj(spatial_latent)
        
        # Cross-Attention (Query: VLM, Key/Value: Spatial)
        attn_output, attn_weights = self.cross_attn(
            query=vlm_features, 
            key=spatial_context, 
            value=spatial_context
        )
        
        # Pass through zero-initialized MLP
        gated_output = self.mlp_gate(attn_output)
        
        # Residual Connection: Add spatial awareness back into the original VLM features
        fused_features = self.layer_norm(vlm_features + gated_output)
        
        return fused_features, attn_weights

class SpatialRegressionHead(nn.Module):
    def __init__(self, vlm_dim=1536, out_dim=6):
        super().__init__()
        # Projects the high-dimensional language features down to a 6-parameter continuous coordinate
        self.regression = nn.Sequential(
            nn.Linear(vlm_dim, 512),
            nn.GELU(),
            nn.Linear(512, out_dim)
        )
        
    def forward(self, vlm_features):
        # We assume vlm_features is the hidden state of the final token [Batch, 1536]
        return self.regression(vlm_features)
