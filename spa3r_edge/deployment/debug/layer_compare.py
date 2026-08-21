"""
Layer-by-layer comparison between PyTorch and ONNX for Spa3R.

This script exposes intermediate layers (patch_embed, blocks) as outputs,
exports a debug ONNX model, runs both backends, and generates a diff report.
"""
import os
import sys
import json

import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder
from spa3r_edge.deployment.export.wrapper import Spa3RWrapper
from spa3r_edge.deployment.export.patches import CartesianProdPatch
from spa3r_edge.deployment.export.symbolic_registry import register_all


class DebugSpa3RWrapper(Spa3RWrapper):
    def __init__(self, model, height=224, width=224):
        super().__init__(model, height=height, width=width)
        self.intermediate_tensors = {}
        self.hook_names = []

        def get_hook(name):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                if isinstance(out, torch.Tensor):
                    self.intermediate_tensors[name] = out
            return hook

        # Find modules to hook
        for name, module in self.model.named_modules():
            if name.endswith("patch_embed"):
                self.hook_names.append(name)
                module.register_forward_hook(get_hook(name))
            elif ("block" in name.lower() or "layer" in name.lower() or "projection" in name.lower() or "position" in name.lower() or "norm" in name.lower()) and type(module).__name__ not in ["Dropout", "Conv2d", "ReLU", "GELU", "ModuleList", "Sequential"]:
                self.hook_names.append(name)
                module.register_forward_hook(get_hook(name))
                        
        print(f"[DebugWrapper] Registered {len(self.hook_names)} hooks.")

    def forward(self, images):
        self.intermediate_tensors.clear()
        
        # Call original wrapper forward
        latents = super().forward(images)
        
        # Build tuple of outputs, latents first
        out = [latents]
        
        # Filter hook_names to only those that were actually called
        called_hooks = [name for name in self.hook_names if name in self.intermediate_tensors]
        
        # Store for the export step so we know what output_names to use
        self.active_hooks = called_hooks
        
        for name in called_hooks:
            out.append(self.intermediate_tensors[name])
            
        return tuple(out)


def run_debug_comparison():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "latents"))
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt_path = os.path.abspath(os.path.join(out_dir, "..", "..", "spa3r_weights.ckpt"))
    
    print("[layer_compare] Loading PyTorch model...")
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path=ckpt_path)
    model = pt_backend.model
    model.eval()

    h, w = 224, 224
    wrapper = DebugSpa3RWrapper(model, height=h, width=w)
    wrapper.eval()

    register_all()

    torch.manual_seed(42)
    b, v, c = 1, 1, 3
    dummy_input = torch.randn(b, v, c, h, w)

    print("[layer_compare] Running PyTorch forward pass...")
    with torch.no_grad():
        pt_outputs = wrapper(dummy_input)
        
    pt_latents = pt_outputs[0].cpu().numpy()
    pt_intermediates = {name: out.cpu().numpy() for name, out in zip(wrapper.active_hooks, pt_outputs[1:])}

    debug_onnx_path = os.path.join(out_dir, "spa3r_encoder_debug.onnx")
    
    output_names = ["latents"] + wrapper.active_hooks
    
    print(f"[layer_compare] Exporting debug ONNX model with {len(output_names)} outputs...")
    with CartesianProdPatch():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            debug_onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            dynamo=False,
            input_names=["images"],
            output_names=output_names,
        )

    print("[layer_compare] Running ONNX Runtime inference...")
    import onnxruntime as ort
    sess = ort.InferenceSession(debug_onnx_path, providers=["CPUExecutionProvider"])
    
    onnx_inputs = {"images": dummy_input.numpy()}
    onnx_outputs_raw = sess.run(output_names, onnx_inputs)
    
    onnx_latents = onnx_outputs_raw[0]
    onnx_intermediates = {name: out for name, out in zip(output_names[1:], onnx_outputs_raw[1:])}

    print("[layer_compare] Computing layer-by-layer diffs...")
    diffs = {}
    
    for name in output_names[1:]:
        pt_val = pt_intermediates[name]
        onnx_val = onnx_intermediates[name]
        
        if pt_val.shape != onnx_val.shape:
            print(f"  Shape mismatch at {name}! PT: {pt_val.shape} vs ONNX: {onnx_val.shape}")
            if pt_val.size == onnx_val.size:
                max_err = float(np.max(np.abs(pt_val.flatten() - onnx_val.flatten())))
            else:
                max_err = -1.0
        else:
            max_err = float(np.max(np.abs(pt_val - onnx_val)))
            
        diffs[name] = max_err
        print(f"  {name:50s} : Max Error = {max_err:.7f}")
        
    latents_err = float(np.max(np.abs(pt_latents - onnx_latents)))
    diffs["latents"] = latents_err
    print(f"  {'latents':50s} : Max Error = {latents_err:.7f}")

    json_path = os.path.join(out_dir, "layer_diff.json")
    with open(json_path, "w") as f:
        json.dump(diffs, f, indent=2)
        
    csv_path = os.path.join(out_dir, "layer_diff.csv")
    with open(csv_path, "w") as f:
        f.write("layer,max_error\n")
        for k, v in diffs.items():
            f.write(f"{k},{v}\n")
            
    print(f"\n[layer_compare] ✓ Saved diff reports to latents/layer_diff.json and latents/layer_diff.csv")


if __name__ == "__main__":
    run_debug_comparison()
