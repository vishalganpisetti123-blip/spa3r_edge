"""
INT8 Post-Training Quantization (PTQ) for the Spa3R encoder.

Responsibility: ONNX → INT8-calibrated ONNX.

⚠️  STUB — not yet implemented.
Requires a calibration dataset (real images) and either:
  - onnxruntime.quantization for static PTQ
  - Hailo Model Zoo calibration pipeline

This module will be filled in during Sprint 4 (Hailo compilation).

Usage (future):
    cd spa3r_edge
    python -m deployment.compile.quantize --calib-dir /path/to/images
"""

import sys


def quantize(
    onnx_path: str,
    output_path: str = None,
    calib_data_dir: str = None,
    num_calib_images: int = 100,
) -> str:
    """Quantize an ONNX model to INT8 using static PTQ.

    Args:
        onnx_path: Path to the (simplified) ONNX model.
        output_path: Path for the INT8 ONNX output.
        calib_data_dir: Directory of calibration images.
        num_calib_images: Number of images to use for calibration.

    Returns:
        Path to the quantized ONNX model.
    """
    raise NotImplementedError(
        "INT8 quantization not yet implemented.\n"
        "Prerequisites:\n"
        "  1. Hailo Dataflow Compiler installed (hailo_sdk_client)\n"
        "  2. Calibration dataset (real images)\n"
        "  3. Simplified ONNX from deployment.optimization.simplify\n"
        "  4. Operator report shows 0 unsupported ops\n"
        "\nRun the quality gate first:\n"
        "  python -m deployment.verification.verify"
    )


def main():
    print("[quantize] ⚠️  INT8 quantization stub — not yet implemented.")
    print("[quantize]    Run the quality gate first:")
    print("[quantize]      python -m deployment.verification.verify")
    sys.exit(0)


if __name__ == "__main__":
    main()
