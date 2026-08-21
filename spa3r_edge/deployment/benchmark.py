"""
Performance benchmark: PyTorch vs ONNX Runtime latency.

Usage:
    cd spa3r_edge
    python -m deployment.benchmark
"""

import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder
from spa3r_edge.edge.encoder.backends import PyTorchBackend, ONNXBackend


def measure_latency(backend, input_tensor, num_runs=50, warmup=10):
    """Measure inference latency over multiple runs.

    Returns dict with avg/p95/max times in milliseconds.
    """
    for _ in range(warmup):
        backend.encode(input_tensor)

    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        backend.encode(input_tensor)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    times = np.array(times)
    return {
        "avg": float(np.mean(times)),
        "p95": float(np.percentile(times, 95)),
        "max": float(np.max(times)),
    }


def main():
    with open("deployment/configs/export.yaml", "r") as f:
        config = yaml.safe_load(f)["export"]

    b = config["batch_size"]
    v = config["num_views"]
    c = config["channels"]
    h, w = config["height"], config["width"]
    dummy_input = torch.randn(b, v, c, h, w)

    # --- PyTorch CPU ---
    print("[benchmark] Loading PyTorch CPU backend...")
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path="../spa3r_weights.ckpt")

    # --- ONNX ---
    onnx_path = config["output_path"]
    onnx_backend = None
    if os.path.exists(onnx_path):
        print("[benchmark] Loading ONNX backend...")
        onnx_backend = ONNXBackend(onnx_path)
    else:
        print(f"[benchmark] ONNX model not found at {onnx_path}, skipping ONNX benchmark.")

    # --- GPU ---
    has_gpu = torch.cuda.is_available()

    print("\n[benchmark] Starting benchmarks...\n")

    print("[benchmark] Benchmarking PyTorch CPU...")
    pt_cpu_stats = measure_latency(pt_backend, dummy_input)

    pt_gpu_stats = None
    if has_gpu:
        print("[benchmark] Benchmarking PyTorch GPU...")
        gpu_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path="../spa3r_weights.ckpt")
        gpu_input = dummy_input.cuda()
        pt_gpu_stats = measure_latency(gpu_backend, gpu_input)

    onnx_stats = None
    if onnx_backend is not None:
        print("[benchmark] Benchmarking ONNX Runtime (CPU)...")
        onnx_stats = measure_latency(onnx_backend, dummy_input)

    # --- Results ---
    print("\n" + "=" * 45)
    print("  BENCHMARK RESULTS (ms)")
    print("=" * 45)

    print(f"  PyTorch CPU:")
    print(f"    Avg: {pt_cpu_stats['avg']:.2f}")
    print(f"    P95: {pt_cpu_stats['p95']:.2f}")
    print(f"    Max: {pt_cpu_stats['max']:.2f}")

    if pt_gpu_stats:
        print(f"\n  PyTorch GPU:")
        print(f"    Avg: {pt_gpu_stats['avg']:.2f}")
        print(f"    P95: {pt_gpu_stats['p95']:.2f}")
        print(f"    Max: {pt_gpu_stats['max']:.2f}")

    if onnx_stats:
        print(f"\n  ONNX Runtime (CPU):")
        print(f"    Avg: {onnx_stats['avg']:.2f}")
        print(f"    P95: {onnx_stats['p95']:.2f}")
        print(f"    Max: {onnx_stats['max']:.2f}")

        speedup = pt_cpu_stats['avg'] / onnx_stats['avg'] if onnx_stats['avg'] > 0 else 0
        print(f"\n  ONNX speedup vs PT CPU: {speedup:.2f}x")

    print("=" * 45)


if __name__ == "__main__":
    main()
