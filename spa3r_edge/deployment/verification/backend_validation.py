import os
import sys
import json
import numpy as np
import torch
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../submodules/vggt")))

from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder
from spa3r_edge.edge.encoder.backends import PyTorchBackend, ONNXBackend, HailoBackend

def compute_max_error(a, b):
    return float(np.max(np.abs(a - b)))

def validate_backends(calibration_dir, onnx_path, checkpoint_path, output_json):
    print("Initializing PyTorch Backend...")
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path=checkpoint_path)
    
    print(f"Initializing ONNX Backend from {onnx_path}...")
    onnx_backend = ONNXBackend(onnx_path)
    
    # Optional: Hailo backend will fail since it's a stub, but we include it for structure
    hailo_backend = None
    
    # Find a calibration sample
    calib_files = sorted(list(Path(calibration_dir).glob("*.npy")))
    if not calib_files:
        print("No calibration files found! Using random tensor.")
        test_input = np.random.randn(1, 1, 3, 224, 224).astype(np.float32)
    else:
        print(f"Using calibration sample: {calib_files[0]}")
        test_input = np.load(calib_files[0])
    
    input_tensor = torch.from_numpy(test_input)
    
    print("Running PyTorch Backend...")
    t0 = time.time()
    pt_res = pt_backend.encode(input_tensor)
    pt_time = time.time() - t0
    
    print("Running ONNX Backend...")
    t0 = time.time()
    onnx_res = onnx_backend.encode(input_tensor)
    onnx_time = time.time() - t0
    
    pt_latents = pt_res["latents"]
    onnx_latents = onnx_res["latents"]
    
    pt_vs_onnx_err = compute_max_error(pt_latents, onnx_latents)
    print(f"PyTorch vs ONNX Max Error: {pt_vs_onnx_err:.7f}")
    
    results = {
        "pytorch_vs_onnx": {
            "max_error": pt_vs_onnx_err,
            "pytorch_time_sec": pt_time,
            "onnx_time_sec": onnx_time
        },
        "onnx_vs_hailo": {
            "max_error": "Pending compilation"
        },
        "pytorch_vs_hailo": {
            "max_error": "Pending compilation"
        }
    }
    
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"Validation results saved to {output_json}")
    return results

if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    calibration_dir = os.path.join(project_root, "spa3r_edge", "calibration")
    onnx_path = os.path.join(project_root, "spa3r_edge", "latents", "spa3r_encoder.onnx")
    checkpoint_path = os.path.join(project_root, "spa3r_weights.ckpt")
    output_json = os.path.join(project_root, "spa3r_edge", "reports", "backend_validation.json")
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # We use test.onnx if spa3r_encoder.onnx doesn't exist yet, but let's assume we exported it
    if not os.path.exists(onnx_path) and os.path.exists(os.path.join(project_root, "test.onnx")):
        onnx_path = os.path.join(project_root, "test.onnx")
        
    validate_backends(calibration_dir, onnx_path, checkpoint_path, output_json)
