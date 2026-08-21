"""
Activation Statistics Generator

Profiles the intermediate activations for each major Transformer block
and outputs min, max, mean, and std. Used for INT8 Calibration.
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from deployment.export.wrapper import Spa3RWrapper
from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder

def generate_activation_report():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "latents"))
    os.makedirs(out_dir, exist_ok=True)
    
    report_path = os.path.join(out_dir, "activation_report.json")
    
    print("[activation_report] Loading PyTorch Model...")
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "spa3r_weights.ckpt"))
    
    # We use the raw model wrapped in Spa3RWrapper so it matches exactly what gets exported
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path=ckpt_path)
    raw_model = pt_backend.model
    model = Spa3RWrapper(raw_model, height=224, width=224)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 1, 3, 224, 224)
    ref_input_path = os.path.join(out_dir, "reference_input.npy")
    if os.path.exists(ref_input_path):
        dummy_input = torch.from_numpy(np.load(ref_input_path))
        
    activation_stats = []
    
    def hook_fn(name):
        def fn(module, inp, out):
            # If output is a tuple, take the first tensor (common in transformers)
            if isinstance(out, tuple):
                t = out[0]
            else:
                t = out
                
            if isinstance(t, torch.Tensor) and t.dtype == torch.float32:
                activation_stats.append({
                    "layer": name,
                    "min": round(float(t.min()), 4),
                    "max": round(float(t.max()), 4),
                    "mean": round(float(t.mean()), 4),
                    "std": round(float(t.std()), 4)
                })
        return fn

    # Register hooks on likely transformer blocks
    hooks = []
    for name, module in model.named_modules():
        # Look for typical transformer block names
        if name.endswith("blocks") or "block" in name.lower() or "layer" in name.lower():
            # Don't hook standard layers, hook custom blocks
            if type(module).__name__ not in ["Linear", "LayerNorm", "Dropout", "Conv2d", "ReLU", "GELU"]:
                hooks.append(module.register_forward_hook(hook_fn(name)))
                
    print("[activation_report] Running forward pass to collect stats...")
    with torch.no_grad():
        model(dummy_input)
        
    for h in hooks:
        h.remove()
        
    # Sort by layer name
    activation_stats.sort(key=lambda x: x["layer"])
    
    with open(report_path, "w") as f:
        json.dump(activation_stats, f, indent=2)
        
    print(f"[activation_report] ✓ Saved {len(activation_stats)} layer stats to {report_path}")

if __name__ == "__main__":
    generate_activation_report()
