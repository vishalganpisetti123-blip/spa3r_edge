import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder
from spa3r_edge.deployment.export.wrapper import Spa3RWrapper

def get_cosine_sim(a, b):
    a_flat = a.flatten()
    b_flat = b.flatten()
    if np.all(a_flat == 0) and np.all(b_flat == 0):
        return 1.0
    return np.dot(a_flat, b_flat) / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-8)

def print_diff(name, pt_tensor, onnx_tensor):
    max_err = float(np.max(np.abs(pt_tensor - onnx_tensor)))
    mean_err = float(np.mean(np.abs(pt_tensor - onnx_tensor)))
    cos_sim = float(get_cosine_sim(pt_tensor, onnx_tensor))
    print(f"{name:15s} | Max: {max_err:.6f} | Mean: {mean_err:.6f} | CosSim: {cos_sim:.6f}")
    return max_err

class OnnxRoPE(torch.nn.Module):
    """
    ONNX-safe version of RoPE.
    It intercepts tokens and positions, and returns all intermediates for comparison.
    """
    def __init__(self, base_freq=100.0):
        super().__init__()
        self.base_freq = base_freq

    def forward(self, tokens, positions):
        # We rewrite RoPE to be fully trace-friendly, no int() or max() breaking graph
        feature_dim = tokens.size(-1) // 2
        
        exponents = torch.arange(0, feature_dim, 2, device=tokens.device).float() / feature_dim
        inv_freq = 1.0 / (self.base_freq ** exponents)
        
        # ONNX safe gather/embedding using einsum directly on float positions
        angles = torch.einsum("...i,j->...ij", positions.float(), inv_freq)
        angles = torch.cat((angles, angles), dim=-1)
        
        cos_comp = angles.cos().to(tokens.dtype)
        sin_comp = angles.sin().to(tokens.dtype)
        
        # Original adds a None dimension to match tokens: (B, 1, N, D)
        cos_comp = cos_comp.unsqueeze(1)
        sin_comp = sin_comp.unsqueeze(1)
        
        v_feat, h_feat = tokens.chunk(2, dim=-1)
        
        def rotate(x):
            d = x.shape[-1]
            x1, x2 = x[..., :d//2], x[..., d//2:]
            return torch.cat((-x2, x1), dim=-1)
            
        v_rotated = (v_feat * cos_comp[..., 0, :]) + (rotate(v_feat) * sin_comp[..., 0, :])
        h_rotated = (h_feat * cos_comp[..., 1, :]) + (rotate(h_feat) * sin_comp[..., 1, :])
        
        out = torch.cat((v_rotated, h_rotated), dim=-1)
        
        return cos_comp, sin_comp, v_rotated, h_rotated, out


class DebugRoPEWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        for name, mod in model.named_modules():
            if "encoder.blocks.0.attn.rope" in name:
                self.orig_rope = mod
                break
                
        self.onnx_rope = OnnxRoPE()

    def forward(self, tokens, positions):
        # 1. Run Original
        orig_out = self.orig_rope(tokens, positions)
        
        feature_dim = tokens.size(-1) // 2
        max_pos = int(positions.max()) + 1
        orig_cos, orig_sin = self.orig_rope._compute_frequency_components(feature_dim, max_pos, tokens.device, tokens.dtype)
        
        # Handle the -1 indexing exactly as PyTorch does when running eager
        # To avoid IndexError if we just use embedding natively (if they clamp it, but they don't clamp it, so maybe they do?)
        # Let's just use F.embedding
        try:
            orig_cos_gathered = torch.nn.functional.embedding(positions, orig_cos)[:, None, :, :, :]
            orig_sin_gathered = torch.nn.functional.embedding(positions, orig_sin)[:, None, :, :, :]
        except IndexError:
            # If PyTorch throws, it means -1 isn't supported here either, so we clamp for comparison
            clamped_pos = positions.clamp(min=0)
            orig_cos_gathered = torch.nn.functional.embedding(clamped_pos, orig_cos)[:, None, :, :, :]
            orig_sin_gathered = torch.nn.functional.embedding(clamped_pos, orig_sin)[:, None, :, :, :]

        # 2. Run ONNX-safe version
        onnx_cos, onnx_sin, onnx_v, onnx_h, onnx_out = self.onnx_rope(tokens, positions)
        
        return {
            "positions": positions,
            "orig_cos": orig_cos_gathered,
            "orig_sin": orig_sin_gathered,
            "orig_out": orig_out,
            "onnx_cos": onnx_cos,
            "onnx_sin": onnx_sin,
            "onnx_out": onnx_out
        }


def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "latents"))
    ckpt_path = os.path.abspath(os.path.join(out_dir, "..", "..", "spa3r_weights.ckpt"))
    
    print("[rope_compare] Loading model...")
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path=ckpt_path)
    model = pt_backend.model
    model.eval()

    bs, n, dim = 1, 1025, 1024
    tokens = torch.randn(bs, 1, n, dim)
    positions = torch.zeros(bs, n, 2, dtype=torch.long)
    # The Transformer adds 1 to pos before it hits RoPE, so -1 becomes 0
    positions[:, 0, :] = 0
    
    positions[:, 1:, 0] = torch.arange(16).repeat_interleave(64).unsqueeze(0)
    positions[:, 1:, 1] = torch.arange(64).repeat(16).unsqueeze(0)

    debug_wrapper = DebugRoPEWrapper(model)
    
    print("[rope_compare] Running PyTorch evaluation...")
    with torch.no_grad():
        res = debug_wrapper(tokens, positions)
        
    print("\n--- RoPE Implementation Comparison (PyTorch -> PyTorch) ---")
    
    orig_cos = res["orig_cos"].squeeze(1)
    
    print_diff("positions", res["positions"].numpy(), res["positions"].numpy())
    
    # We compare the expanded sine/cosine
    print_diff("cos_comp", orig_cos.numpy(), res["onnx_cos"].numpy())
    print_diff("sin_comp", res["orig_sin"].squeeze(1).numpy(), res["onnx_sin"].numpy())
    print_diff("rotated_out", res["orig_out"].numpy(), res["onnx_out"].numpy())

if __name__ == "__main__":
    main()
