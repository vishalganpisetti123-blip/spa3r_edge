"""
ONNX export entry point for the Spa3R encoder.

Responsibility: PyTorch → ONNX. Nothing else.

Usage:
    cd spa3r_edge
    python -m deployment.export.export
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder

from .wrapper import Spa3RWrapper
from .patches import CartesianProdPatch
from .symbolic_registry import register_all


def load_config(config_path="deployment/configs/export.yaml"):
    """Load export configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["export"]


def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return "unknown"


def export(config=None, checkpoint_path="../spa3r_weights.ckpt"):
    """Export the Spa3R encoder to ONNX.

    Also saves reference I/O (reference_input.npy, reference_output.npy)
    and a manifest.json alongside the ONNX file. These are the canonical baseline 
    for verify.py, ensuring reproducibility and traceability.

    Args:
        config: Export config dict. If None, loaded from default YAML.
        checkpoint_path: Path to the Spa3R checkpoint.

    Returns:
        Tuple[str, np.ndarray, np.ndarray] — (onnx_path, ref_input, ref_output)
    """
    if config is None:
        config = load_config()

    h, w = config["height"], config["width"]
    opset = config["opset_version"]
    output_path = config["output_path"]
    out_dir = os.path.dirname(output_path) or "."

    print(f"[export] Loading PyTorch model from {checkpoint_path}...")
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path=checkpoint_path)
    model = pt_backend.model
    model.eval()

    print(f"[export] Wrapping model for export (resolution: {h}x{w})...")
    wrapper = Spa3RWrapper(model, height=h, width=w)
    wrapper.eval()

    # Register custom ONNX symbolics
    register_all()

    # Deterministic dummy input (used for both tracing and reference output)
    torch.manual_seed(42)
    b = config["batch_size"]
    v = config["num_views"]
    c = config["channels"]
    dummy_input = torch.randn(b, v, c, h, w)

    # ── Save reference output BEFORE export (same model, same input) ──
    print("[export] Computing reference output (canonical PyTorch baseline)...")
    with torch.no_grad():
        ref_output = wrapper(dummy_input).cpu().numpy()

    os.makedirs(out_dir, exist_ok=True)
    ref_input_path = os.path.join(out_dir, "reference_input.npy")
    ref_output_path = os.path.join(out_dir, "reference_output.npy")
    manifest_path = os.path.join(out_dir, "manifest.json")
    
    np.save(ref_input_path, dummy_input.numpy())
    np.save(ref_output_path, ref_output)
    
    manifest = {
        "model": "Spa3R Edge",
        "checkpoint": checkpoint_path,
        "git_commit": get_git_commit(),
        "input_shape": list(dummy_input.shape),
        "output_shape": list(ref_output.shape),
        "dtype": str(dummy_input.dtype),
        "opset": opset,
        "export_time": datetime.now(timezone.utc).isoformat()
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[export]   Reference input:  {ref_input_path}  {list(dummy_input.shape)}")
    print(f"[export]   Reference output: {ref_output_path}  {list(ref_output.shape)}")
    print(f"[export]   Manifest:         {manifest_path}")

    # ── ONNX export ────────────────────────────────────────────────
    print(f"[export] Exporting to ONNX (opset {opset}, shape: {list(dummy_input.shape)})...")
    start = time.perf_counter()

    with CartesianProdPatch():
        torch.onnx.export(
            wrapper,
            (dummy_input,),
            output_path,
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
            input_names=["images"],
            output_names=["latents"],
        )

    elapsed = time.perf_counter() - start
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print(f"[export] ✓ Exported to {output_path}")
    print(f"[export]   Time:   {elapsed:.1f}s")
    print(f"[export]   Size:   {file_size_mb:.1f} MB")
    print(f"[export]   Opset:  {opset}")

    return output_path, dummy_input.numpy(), ref_output


def main():
    onnx_path, ref_input, ref_output = export()

if __name__ == "__main__":
    main()
