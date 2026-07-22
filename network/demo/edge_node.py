import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.serialize import serialize_tensor
from network.socket_utils import create_client_socket


def enforce_edge_hardware_constraints():
    """Restrict core affinity to mimic a Raspberry Pi 5-style quad-core setup when supported."""
    process = psutil.Process(os.getpid())
    try:
        available_cores = process.cpu_affinity()
    except AttributeError:
        available_cores = list(range(os.cpu_count() or 1))
    target_cores = available_cores[:4] if len(available_cores) >= 4 else available_cores
    if hasattr(process, "cpu_affinity"):
        process.cpu_affinity(target_cores)
    print(f"[EDGE SIMULATION] Core affinity locked to: {target_cores}")
    return process


def main():
    process = enforce_edge_hardware_constraints()

    onnx_model_path = "models/spa3r_encoder_int8.onnx"
    cloud_host = "localhost"
    cloud_port = 8080

    if not os.path.exists(onnx_model_path):
        raise FileNotFoundError(
            f"Missing quantized weights file: {onnx_model_path}. Run quantize_encoder.py first."
        )

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        onnx_model_path,
        sess_options=sess_options,
        providers=["CPUExecutionProvider"],
    )

    mock_views = np.random.randn(1, 4, 3, 224, 224).astype(np.float32)
    mock_queries = np.random.randn(1, 256, 16).astype(np.float32)

    print("[EDGE SIMULATION] Processing multi-view visual features on CPU...")
    start_memory = process.memory_info().rss / (1024 * 1024)
    start_time = time.time()

    onnx_outputs = session.run(
        ["spatial_latents_z"],
        {"context_views": mock_views, "query_slots": mock_queries},
    )

    latency_ms = (time.time() - start_time) * 1000
    end_memory = process.memory_info().rss / (1024 * 1024)
    spatial_latent_z = onnx_outputs[0]
    print(
        f" -> Local Inference Complete. Latency: {latency_ms:.2f} ms | Allocation Δ: {end_memory - start_memory:.2f} MB"
    )

    try:
        print(f"[EDGE SIMULATION] Establishing network bridge link to {cloud_host}:{cloud_port}...")
        client_net_node = create_client_socket(cloud_host, cloud_port)
        serialized_payload = serialize_tensor(spatial_latent_z)
        payload_bytes_size = len(serialized_payload)

        tx_start = time.time()
        client_net_node.sendall(serialized_payload)
        tx_latency_ms = (time.time() - tx_start) * 1000

        print(
            f" -> Transmission Complete. Payload Size: {payload_bytes_size} bytes | TX Time: {tx_latency_ms:.2f} ms"
        )
        client_net_node.close()
    except ConnectionRefusedError:
        print("[ERROR] Cloud server connection refused. Ensure cloud_server.py is running first on port 8080.")


if __name__ == "__main__":
    main()
