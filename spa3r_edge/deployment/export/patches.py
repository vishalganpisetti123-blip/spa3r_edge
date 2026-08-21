import torch

class CartesianProdPatch:
    """
    Context manager to patch torch.cartesian_prod for ONNX export.
    The original torch.cartesian_prod uses operations that ONNX tracing
    fails on, so we replace it with a statically traceable equivalent.
    """
    def __init__(self):
        self.original = torch.cartesian_prod

    def __enter__(self):
        torch.cartesian_prod = self.patched_cartesian_prod
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cartesian_prod = self.original

    @staticmethod
    def patched_cartesian_prod(*tensors):
        """
        A traceable version of cartesian_prod.
        Only supports exactly 2 tensors for now (which is what Spa3R needs).
        """
        if len(tensors) != 2:
            raise NotImplementedError("Patched cartesian_prod only supports exactly 2 tensors.")
        
        t1, t2 = tensors
        n1 = t1.size(0)
        n2 = t2.size(0)
        
        t1_exp = t1.unsqueeze(1).expand(n1, n2)
        t2_exp = t2.unsqueeze(0).expand(n1, n2)
        
        t1_flat = t1_exp.reshape(-1, 1)
        t2_flat = t2_exp.reshape(-1, 1)
        
        return torch.cat([t1_flat, t2_flat], dim=1)

class OnnxRotaryPositionEmbedding2D(torch.nn.Module):
    """
    ONNX-safe version of RoPE.
    It completely removes dynamic frequency generation from the exported graph
    by caching the cos/sin components as static buffers on the first forward pass.
    """
    def __init__(self, frequency=100.0):
        super().__init__()
        self.base_frequency = frequency

    def forward(self, tokens, positions):
        if not hasattr(self, 'cos_comp'):
            # We precompute this on the first forward pass (which we'll trigger before export)
            feature_dim = tokens.size(-1) // 2
            
            exponents = torch.arange(0, feature_dim, 2, device=tokens.device).float() / feature_dim
            inv_freq = 1.0 / (self.base_frequency ** exponents)
            
            # Use positions to compute angles statically
            angles = torch.einsum("...i,j->...ij", positions.float(), inv_freq)
            angles = torch.cat((angles, angles), dim=-1)
            
            cos_comp = angles.cos().to(tokens.dtype)
            sin_comp = angles.sin().to(tokens.dtype)
            
            # Shape: (B, 1, N, D)
            cos_comp = cos_comp.unsqueeze(1)
            sin_comp = sin_comp.unsqueeze(1)
            
            self.register_buffer("cos_comp", cos_comp)
            self.register_buffer("sin_comp", sin_comp)

        v_feat, h_feat = tokens.chunk(2, dim=-1)
        
        def rotate(x):
            d = x.shape[-1]
            x1, x2 = x[..., :d//2], x[..., d//2:]
            return torch.cat((-x2, x1), dim=-1)
            
        v_rotated = (v_feat * self.cos_comp[..., 0, :]) + (rotate(v_feat) * self.sin_comp[..., 0, :])
        h_rotated = (h_feat * self.cos_comp[..., 1, :]) + (rotate(h_feat) * self.sin_comp[..., 1, :])
        
        out = torch.cat((v_rotated, h_rotated), dim=-1)
        
        return out

def attention_forward_onnx(self, x: torch.Tensor, attn_mask: torch.Tensor = None, pos: torch.Tensor = None) -> torch.Tensor:
    import torch.nn.functional as F
    B, N, C = x.shape
    qkv = self.qkv(x) # (B, N, C * 3)
    
    # Split directly into Q, K, V avoiding 5D reshapes
    q, k, v = torch.chunk(qkv, 3, dim=-1)
    
    q = q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    
    q, k = self.q_norm(q), self.k_norm(k)

    if self.rope is not None:
        q = self.rope(q, pos)
        k = self.rope(k, pos)

    # fused_attn should be False for ONNX export
    if self.fused_attn:
        x_attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_attn = attn @ v

    x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
    x_attn = self.norm(x_attn)
    x_attn = self.proj(x_attn)
    x_attn = self.proj_drop(x_attn)
    return x_attn
