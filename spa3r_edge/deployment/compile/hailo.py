"""
Hailo HEF compilation for the Spa3R encoder.

Responsibility: INT8-calibrated ONNX → HEF (Hailo Executable Format).

⚠️  STUB — not yet implemented.
Requires:
  - Hailo Dataflow Compiler (hailo_sdk_client) installed on compile machine
  - Valid INT8 ONNX from deployment.compile.quantize
  - Hailo-8 target specification

The HEF is the final artifact deployed to the Raspberry Pi 5 + Hailo-8 HAT.

Usage (future):
    cd spa3r_edge
    python -m deployment.compile.hailo --int8-onnx latents/spa3r_int8.onnx
"""

import sys


def compile_hef(
    int8_onnx_path: str,
    output_path: str = "latents/spa3r_encoder.hef",
    target: str = "hailo8",
    optimization_level: int = 2,
) -> str:
    """Compile INT8 ONNX to Hailo HEF.

    Args:
        int8_onnx_path: Path to the INT8-calibrated ONNX model.
        output_path: Path for the .hef output.
        target: Hailo target chip (hailo8 or hailo8l).
        optimization_level: Hailo compiler optimization level (0-4).

    Returns:
        Path to the compiled HEF file.
    """
    raise NotImplementedError(
        "Hailo HEF compilation not yet implemented.\n"
        "Prerequisites:\n"
        "  1. Hailo Dataflow Compiler (pip install hailo_sdk_client)\n"
        "  2. INT8 ONNX from deployment.compile.quantize\n"
        "  3. Hailo-8 hardware target connected or emulated\n"
        "\nPipeline:\n"
        "  python -m deployment.verification.verify    # quality gate\n"
        "  python -m deployment.optimization.simplify  # graph optimization\n"
        "  python -m deployment.compile.quantize       # INT8 calibration\n"
        "  python -m deployment.compile.hailo          # HEF compilation\n"
    )


def main():
    print("[hailo] ⚠️  HEF compilation stub — not yet implemented.")
    print("[hailo]    Full pipeline when ready:")
    print("[hailo]      python -m deployment.verification.verify")
    print("[hailo]      python -m deployment.optimization.simplify")
    print("[hailo]      python -m deployment.compile.quantize")
    print("[hailo]      python -m deployment.compile.hailo")
    sys.exit(0)


if __name__ == "__main__":
    main()
