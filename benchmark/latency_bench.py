import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run_matrix_benchmark():
    print("=================================================================")
    print("  EDGE-SPA3R CORE BENCHMARKING SUITE (SIMULATED CONSTRAINTS)   ")
    print("=================================================================\n")
    print("Disclaimer: Simulated edge constraints (CPU-affinity-limited local process),")
    print("not measured on physical Raspberry Pi 5 hardware.\n")

    process = psutil.Process(os.getpid())
    try:
        available_cores = process.cpu_affinity()
    except AttributeError:
        available_cores = list(range(os.cpu_count() or 1))
    target_cores = available_cores[:4] if len(available_cores) >= 4 else available_cores
    if hasattr(process, "cpu_affinity"):
        process.cpu_affinity(target_cores)

    fp32_path = os.environ.get("FP32_MODEL", "models/spa3r_encoder_fp32.onnx")
    int8_path = os.environ.get("INT8_MODEL", "models/spa3r_encoder_int8.onnx")

    if not os.path.exists(fp32_path):
        raise FileNotFoundError(f"Missing FP32 model: {fp32_path}")
    if not os.path.exists(int8_path):
        raise FileNotFoundError(f"Missing INT8 model: {int8_path}")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4

    print("[BENCHMARK] Initializing ONNX Runtime execution engines...")
    session_fp32 = ort.InferenceSession(fp32_path, sess_opts, providers=["CPUExecutionProvider"])
    session_int8 = ort.InferenceSession(int8_path, sess_opts, providers=["CPUExecutionProvider"])

    view_sweep = [4, 8, 12]
    iterations = 10

    print(f"\nRunning {iterations} matrix iterations per evaluation profile:\n")
    print(f"{'Format':<8} | {'Views':<6} | {'Avg Latency (ms)':<18} | {'Peak RAM Δ (MB)':<16}")
    print("-" * 58)

    for views in view_sweep:
        mock_views = np.random.randn(1, views, 3, 224, 224).astype(np.float32)
        mock_queries = np.random.randn(1, 256, 16).astype(np.float32)

        ram_start = process.memory_info().rss / (1024 * 1024)
        time_records = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = session_fp32.run(["spatial_latents_z"], {"context_views": mock_views, "query_slots": mock_queries})
            time_records.append((time.perf_counter() - t0) * 1000)
        ram_end = process.memory_info().rss / (1024 * 1024)
        print(f"{'FP32':<8} | {views:<6} | {np.mean(time_records):<18.2f} | {ram_end - ram_start:<16.2f}")

        ram_start = process.memory_info().rss / (1024 * 1024)
        time_records = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = session_int8.run(["spatial_latents_z"], {"context_views": mock_views, "query_slots": mock_queries})
            time_records.append((time.perf_counter() - t0) * 1000)
        ram_end = process.memory_info().rss / (1024 * 1024)
        print(f"{'INT8':<8} | {views:<6} | {np.mean(time_records):<18.2f} | {ram_end - ram_start:<16.2f}")
        print("-" * 58)


if __name__ == "__main__":
    run_matrix_benchmark()
