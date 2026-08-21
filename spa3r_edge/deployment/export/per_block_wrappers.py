import torch
import torch.nn as nn
from .patches import OnnxRotaryPositionEmbedding2D, attention_forward_onnx
from .split_wrappers import patch_layernorm
import types

class StemWrapper(nn.Module):
    """Wraps the initial projection and token prep for the encoder."""
    def __init__(self, model, height=224, width=224):
        super().__init__()
        self.projection = model.projection
        self.query_embed = model.query_embed
        self.num_queries = model.num_queries
        self.patch_size = model.patch_size
        self.register_tokens = model.encoder.register_tokens
        
        # Pre-compute position encoding for the fixed resolution
        h_patches = height // self.patch_size
        w_patches = width // self.patch_size
        n = h_patches * w_patches
        
        positions_h = torch.arange(h_patches).unsqueeze(1).expand(h_patches, w_patches).reshape(-1)
        positions_w = torch.arange(w_patches).unsqueeze(0).expand(h_patches, w_patches).reshape(-1)
        positions = torch.stack([positions_h, positions_w], dim=-1).float()
        
        # We also need query_pos for the RoPE
        query_pos = torch.zeros(1, self.num_queries, 2, dtype=torch.float32) - 1.0
        
        # Let's bake all of encoder_pos exactly as the transformer sees it.
        vis_pos = positions.unsqueeze(0)
        enc_pos_base = torch.cat([query_pos, vis_pos], dim=1)
        
        enc_pos_base = enc_pos_base + 1
        pos_special = torch.zeros(1, self.register_tokens.size(1), 2, dtype=torch.float32)
        final_pos = torch.cat([pos_special, enc_pos_base], dim=1)
        
        self.register_buffer('final_pos', final_pos)
        self.n_visual = n

    def forward(self, features_2048):
        """
        Args:
            features_2048: (B, N, 2048)
        Returns:
            encoder_input: (B, 516, 768)
        """
        bs = features_2048.shape[0]
        
        # Projection: (B, N, 2048) -> (B, N, 768)
        visual_tokens = self.projection(features_2048)
        
        # Query embeddings
        query = self.query_embed.weight.unsqueeze(0).expand(bs, -1, -1)  # (B, 256, 768)
        
        # Concatenate query + visual tokens
        encoder_input = torch.cat([query, visual_tokens], dim=1)  # (B, 256+N, 768)
        
        # Prepend register tokens
        if self.register_tokens is not None:
            encoder_input = torch.cat((self.register_tokens.expand(bs, -1, -1), encoder_input), dim=1)
            
        return encoder_input


class BlockWrapper(nn.Module):
    """Wraps a single transformer block for ONNX export."""
    def __init__(self, block, final_pos):
        super().__init__()
        self.block = block
        
        # Patch RoPE for ONNX compatibility
        from spa3r.models.layers.attention import Attention
        for module in self.block.modules():
            if hasattr(module, 'rope') and module.rope is not None:
                module.rope = OnnxRotaryPositionEmbedding2D(frequency=module.rope.base_frequency)
            if isinstance(module, Attention):
                module.fused_attn = False
                module.forward = types.MethodType(attention_forward_onnx, module)
                
        # Patch LayerNorm to decompose it for edge compiler compatibility
        patch_layernorm(self.block)
        
        # Register the position buffer
        self.register_buffer('pos', final_pos)

    def forward(self, x):
        """
        Args:
            x: (B, 516, 768)
        Returns:
            out: (B, 516, 768)
        """
        bs = x.shape[0]
        pos = self.pos.expand(bs, -1, -1)
        out = self.block(x, pos=pos)
        return out


class HeadWrapper(nn.Module):
    """Wraps the final norm and slicing of the encoder."""
    def __init__(self, encoder, num_queries):
        super().__init__()
        self.norm = encoder.norm
        self.num_register_tokens = encoder.num_register_tokens
        self.num_queries = num_queries
        
        patch_layernorm(self)

    def forward(self, x):
        """
        Args:
            x: (B, 516, 768)
        Returns:
            latents: (B, 256, 768)
        """
        x_norm = self.norm(x)
        x_norm_sliced = x_norm[:, self.num_register_tokens:]
        latents = x_norm_sliced[:, :self.num_queries]
        return latents
