"""
Split export wrappers for Spa3R.

Splits the monolithic Spa3R model into two ONNX-exportable modules:

1. AggregatorWrapper  — VGGT backbone (909M params, runs on CPU via ONNX Runtime)
   Input:  images (1, 1, 3, 224, 224)
   Output: visual_tokens (1, N, 2048)  where N = (H/14) * (W/14) = 256

2. EncoderWrapper     — Spa3R encoder (43M params, target for Hailo-8L NPU)
   Input:  visual_tokens (1, N, 2048)
   Output: latents (1, 256, 768)

Together they produce the same output as the original Spa3RWrapper.
"""

import torch
import torch.nn as nn
from .patches import OnnxRotaryPositionEmbedding2D


class ExportLayerNorm(nn.Module):
    """Decomposed LayerNorm for ONNX export to ensure primitive ops are used instead
    of the fused LayerNormalization node, which trips up some edge compilers."""
    def __init__(self, ln):
        super().__init__()
        self.weight = ln.weight
        self.bias = ln.bias
        self.eps = ln.eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class ExportRMSNorm(nn.Module):
    """Decomposed RMSNorm for ONNX export to ensure primitive ops are used instead
    of a fused node that trips up edge compilers."""
    def __init__(self, rm):
        super().__init__()
        self.weight = rm.weight
        self.eps = rm.eps if getattr(rm, 'eps', None) is not None else 1e-6

    def forward(self, x):
        var = (x ** 2).mean(dim=-1, keepdim=True)
        x = x / torch.sqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x


def patch_layernorm(module):
    """Recursively replaces nn.LayerNorm and nn.RMSNorm with Export variants."""
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, ExportLayerNorm(child))
        elif hasattr(nn, 'RMSNorm') and isinstance(child, nn.RMSNorm):
            setattr(module, name, ExportRMSNorm(child))
        else:
            patch_layernorm(child)




class AggregatorWrapper(nn.Module):
    """Wraps the VGGT aggregator + projection for ONNX export.

    This runs on CPU (via ONNX Runtime) and produces visual tokens
    that are then fed to the EncoderWrapper on the Hailo NPU.
    """

    def __init__(self, model, height=224, width=224):
        super().__init__()
        self.aggregator = model.aggregator
        self.projection = model.projection
        self.patch_size = model.patch_size

        # Bake positional embeddings (same as original wrapper)
        self._bake_pos_embeddings(height, width)

    def _bake_pos_embeddings(self, h, w):
        for module in self.aggregator.modules():
            if not (hasattr(module, 'pos_embed') and
                    hasattr(module, 'interpolate_pos_encoding')):
                continue
            patch_size = module.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]
            w_patches = w // patch_size
            h_patches = h // patch_size
            npatch = w_patches * h_patches
            dim = module.pos_embed.shape[-1]
            dummy_x = torch.zeros(1, npatch + 1, dim,
                                  dtype=module.pos_embed.dtype,
                                  device=module.pos_embed.device)
            with torch.no_grad():
                new_pos_embed = module.interpolate_pos_encoding(dummy_x, w, h)
            module.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, images):
        """
        Args:
            images: (B, V, 3, H, W)
        Returns:
            f_ctx: (B, V*N, 2048) — raw visual tokens from VGGT aggregator
        """
        with torch.no_grad():
            aggr_tokens, patch_start_idx = self.aggregator(images)
            f_ctx = aggr_tokens[-1][:, :, patch_start_idx:]
            # f_ctx shape: (B, V, N, 2048) -> flatten views
            return f_ctx.flatten(1, 2)  # (B, V*N, 2048)


class EncoderWrapper(nn.Module):
    """Wraps the Spa3R encoder transformer for ONNX export.

    This is the target for Hailo-8L compilation.
    43M parameters, ~162 MB FP32, ~40 MB INT8.
    """

    def __init__(self, model, height=224, width=224):
        super().__init__()
        self.projection = model.projection
        self.encoder = model.encoder
        self.query_embed = model.query_embed
        self.num_queries = model.num_queries
        self.patch_size = model.patch_size

        # Patch RoPE for ONNX compatibility
        from spa3r.models.layers import RotaryPositionEmbedding2D
        from spa3r.models.layers.attention import Attention
        from .patches import attention_forward_onnx
        import types
        for module in self.encoder.modules():
            if hasattr(module, 'rope') and module.rope is not None:
                # We need to maintain the same frequency scale as the original RoPE
                module.rope = OnnxRotaryPositionEmbedding2D(frequency=module.rope.base_frequency)
            if isinstance(module, Attention):
                module.fused_attn = False
                module.forward = types.MethodType(attention_forward_onnx, module)

        # Patch LayerNorm to decompose it for edge compiler compatibility
        patch_layernorm(self.encoder)

        # Pre-compute position encoding for the fixed resolution
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        n = h_patches * w_patches  # number of visual tokens per view

        # Build position grid (same logic as PositionGetter)
        positions_h = torch.arange(h_patches).unsqueeze(1).expand(h_patches, w_patches).reshape(-1)
        positions_w = torch.arange(w_patches).unsqueeze(0).expand(h_patches, w_patches).reshape(-1)
        positions = torch.stack([positions_h, positions_w], dim=-1).float()
        # Shape: (N, 2) — stored as buffer
        self.register_buffer('visual_positions', positions)
        self.n_visual = n

    def forward(self, features_2048):
        """
        Args:
            features_2048: (B, N, 2048) raw VGGT features from AggregatorWrapper

        Returns:
            latents: (B, 256, 768) spatial latent tokens
        """
        bs = features_2048.shape[0]

        # Projection: (B, N, 2048) -> (B, N, 768)
        visual_tokens = self.projection(features_2048)

        # Query embeddings
        query = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)  # (B, 256, 768)
        query_pos = torch.zeros(bs, self.num_queries, 2,
                                device=features_2048.device,
                                dtype=features_2048.dtype) - 1.0

        # Visual positions — expand for batch
        vis_pos = self.visual_positions.unsqueeze(0).expand(bs, -1, -1)  # (B, N, 2)

        # Concatenate query + visual tokens
        encoder_input = torch.cat([query, visual_tokens], dim=1)  # (B, 256+N, 768)
        encoder_pos = torch.cat([query_pos, vis_pos], dim=1)      # (B, 256+N, 2)

        # Run encoder
        enc_out = self.encoder(encoder_input, pos=encoder_pos)
        latents = enc_out[:, :self.num_queries]  # (B, 256, 768)

        return latents
