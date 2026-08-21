"""
ONNX graph simplifier.

Responsibility: ONNX → simplified ONNX.
Runs onnx-simplifier to fold constants, eliminate dead nodes, and
canonicalize the graph. This is a prerequisite for quantization and
Hailo compilation.

Usage:
    cd spa3r_edge
    python -m deployment.optimization.simplify
"""

import os
import sys
import time

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))


def load_config(config_path="deployment/configs/export.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["export"]


def simplify(input_path=None, output_path=None, config=None) -> str:
    """Simplify an ONNX model using onnx-simplifier.

    Args:
        input_path: Path to input ONNX file. If None, uses config.
        output_path: Path for simplified output. If None, appends '_simplified'.
        config: Export config dict.

    Returns:
        Path to the simplified ONNX model.

    Raises:
        ImportError: If onnxsim is not installed.
        FileNotFoundError: If the input ONNX does not exist.
    """
    try:
        import onnx
        import onnxsim
    except ImportError as e:
        raise ImportError(
            f"Required package not found: {e}. "
            "Install with: pip install onnxsim"
        ) from e

    if config is None:
        config = load_config()

    if input_path is None:
        input_path = config["output_path"]

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"ONNX model not found: {input_path}. Run export first.")

    if output_path is None:
        output_path = input_path.replace(".onnx", "_simplified.onnx")

    input_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print(f"[simplify] Input:  {input_path}  ({input_size_mb:.1f} MB)")
    print(f"[simplify] Output: {output_path}")

    print("[simplify] Loading ONNX model...")
    model = onnx.load(input_path)

    print("[simplify] Running onnxsim...")
    start = time.perf_counter()
    simplified_model, check = onnxsim.simplify(model)
    elapsed = time.perf_counter() - start

    if not check:
        print("[simplify] ⚠️  onnxsim check failed — saving anyway.")
    else:
        print(f"[simplify] ✓ Simplification check passed.")

    onnx.save(simplified_model, output_path)
    output_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    reduction_pct = (1 - output_size_mb / input_size_mb) * 100
    print(f"[simplify] ✓ Saved to {output_path}")
    print(f"[simplify]   Time:      {elapsed:.1f}s")
    print(f"[simplify]   Input:     {input_size_mb:.1f} MB")
    print(f"[simplify]   Output:    {output_size_mb:.1f} MB")
    print(f"[simplify]   Reduction: {reduction_pct:.1f}%")

    return output_path


def main():
    simplify()


if __name__ == "__main__":
    main()
