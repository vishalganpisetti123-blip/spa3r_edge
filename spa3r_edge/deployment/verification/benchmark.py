"""
ONNX Runtime Benchmark

Benchmarks the canonical PyTorch model against the ONNX Runtime for CPU/FP32.
Outputs benchmark.json with speedup metrics.
"""
import os
import sys
import time
import json
import torch
import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from deployment.export.wrapper import Spa3RWrapper
from spa3r_edge.edge.encoder.backends import ONNXBackend

def load_config(config_path="deployment/configs/export.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["export"]

def benchmark():
    config = load_config()
    onnx_path = config["output_path"]
    
    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    if os.path.exists(simplified_path):
        onnx_path = simplified_path
        
    out_dir = os.path.dirname(onnx_path) or "."
    
    # Load input
    ref_input_path = os.path.join(out_dir, "reference_input.npy")
    if not os.path.exists(ref_input_path):
        print(f"❌ Missing {ref_input_path}")
        return
        
    input_tensor = torch.from_numpy(np.load(ref_input_path))
    
    print("[benchmark] Loading PyTorch Model...")
    from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder
    ckpt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "spa3r_weights.ckpt"))
    
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path=ckpt_path)
    raw_model = pt_backend.model
    pt_model = Spa3RWrapper(raw_model, height=224, width=224)
    pt_model.eval()
    
    print(f"[benchmark] Loading ONNX Model ({onnx_path})...")
    onnx_backend = ONNXBackend(onnx_path)
    
    # Warmup
    print("[benchmark] Warming up...")
    for _ in range(3):
        with torch.no_grad():
            pt_model(input_tensor)
        onnx_backend.encode(input_tensor)
        
    # Benchmarking
    iterations = 10
    print(f"[benchmark] Running {iterations} iterations...")
    
    # PyTorch
    start_pt = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            pt_model(input_tensor)
    pt_ms = ((time.perf_counter() - start_pt) / iterations) * 1000
    
    # ONNX
    start_onnx = time.perf_counter()
    for _ in range(iterations):
        onnx_backend.encode(input_tensor)
    onnx_ms = ((time.perf_counter() - start_onnx) / iterations) * 1000
    
    speedup = pt_ms / onnx_ms
    
    report = {
        "pytorch_ms": round(pt_ms, 2),
        "onnx_ms": round(onnx_ms, 2),
        "speedup": round(speedup, 2)
    }
    
    report_path = os.path.join(out_dir, "benchmark.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print("\nBenchmark Results:")
    print("--------------------------------------------------")
    print(f" PyTorch (CPU) : {report['pytorch_ms']:.2f} ms")
    print(f" ONNX (CPU)    : {report['onnx_ms']:.2f} ms")
    print(f" Speedup       : {report['speedup']:.2f}x")
    print("--------------------------------------------------")
    print(f"✓ Saved {report_path}\n")

if __name__ == "__main__":
    benchmark()
