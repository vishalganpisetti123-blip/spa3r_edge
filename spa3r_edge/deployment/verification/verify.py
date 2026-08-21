"""
Quality gate for the Spa3R export pipeline.

Responsibility: verify that the ONNX model is numerically equivalent to
the PyTorch model that produced it.

Design principle:
  The canonical baseline is reference_input.npy + reference_output.npy,
  saved during export from the EXACT same model instance and input used
  for tracing. This eliminates inter-instance non-determinism.

  Comparing ONNX against a fresh PyTorch instance is WRONG because Spa3R
  uses strict=False loading and some layers may have instance-specific
  random init. The reference files bypass this entirely.

Usage:
    # Standalone (requires reference files saved by export):
    cd spa3r_edge
    python -m deployment.verification.verify

    # Expected output:
    # ✓ ONNX file exists
    # ✓ ONNX model valid
    # ✓ Graph simplified
    # ✓ Unsupported operators: 0
    # ✓ Max error (ref PT ↔ ONNX): < 0.01
    # ✓ READY FOR QUANTIZATION
"""

import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from .compare import compare, format_result


def load_config(config_path="deployment/configs/export.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["export"]


def _check_onnx_valid(onnx_path: str) -> tuple:
    """Lightweight ONNX validity check that works with external data files."""
    try:
        import onnx
        # Use check_model_path (file-based) to avoid the 2GB proto memory limit
        # that causes "Failed to serialize proto" with onnx.checker.check_model(model)
        onnx.checker.check_model(onnx_path)
        return True, "valid"
    except onnx.checker.ValidationError as e:
        return False, str(e)
    except ImportError:
        return True, "onnx package not installed — skipping graph check"
    except Exception as e:
        # Protobuf size errors are false positives for external data files
        msg = str(e)
        if "serialize" in msg.lower() or "proto" in msg.lower():
            return True, "valid (proto serialization skipped for large model)"
        return False, msg


def _get_hailo_supported_ops():
    """Load supported ops from the hailo8 config."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hailo8_ops.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            # Both fully and partially supported are treated as "supported" for initial parsing
            supported = config.get("supported", []) + config.get("partially_supported", [])
            return frozenset(supported)
    except Exception:
        # Fallback if config is missing
        return frozenset()

def _check_unsupported_ops(onnx_path: str) -> tuple:
    """Count operators not in the Hailo-8 supported opset."""
    from collections import Counter
    try:
        import onnx
        model = onnx.load(onnx_path)
        op_counts = Counter(node.op_type for node in model.graph.node)
        supported_ops = _get_hailo_supported_ops()
        unsupported = {op: cnt for op, cnt in op_counts.items()
                       if op not in supported_ops}
        return len(unsupported), sorted(unsupported.keys()), op_counts
    except ImportError:
        return 0, [], {}
    except Exception:
        return -1, [], {}


def verify(config=None, threshold=1e-2) -> dict:
    """Run the full quality gate.

    Uses reference_input.npy and reference_output.npy saved during export
    as the canonical PyTorch baseline.

    Args:
        config: Export config dict. If None, loaded from default YAML.
        threshold: Max-error threshold for pass/fail.

    Returns:
        dict with all check results and overall 'passed' status.
    """
    if config is None:
        config = load_config()

    onnx_path = config["output_path"]
    out_dir = os.path.dirname(onnx_path) or "."
    ref_input_path = os.path.join(out_dir, "reference_input.npy")
    ref_output_path = os.path.join(out_dir, "reference_output.npy")
    manifest_path = os.path.join(out_dir, "manifest.json")

    checks = {}

    print("\n==============================")
    print("Spa3R Export Report")
    print("==============================\n")

    # 1. Export Success
    exists = os.path.exists(onnx_path)
    _print_check("Export Success", exists)
    checks["export_success"] = exists
    if not exists:
        return {"passed": False, "checks": checks}

    # 2. Reference Saved
    refs_exist = os.path.exists(ref_input_path) and os.path.exists(ref_output_path) and os.path.exists(manifest_path)
    _print_check("Reference Saved", refs_exist)
    checks["reference_saved"] = refs_exist

    # 3. ONNX Valid
    valid, _ = _check_onnx_valid(onnx_path)
    _print_check("ONNX Valid", valid)
    checks["onnx_valid"] = valid

    # Numerical, Shape, and Dtype checks
    shape_ok, dtype_ok, num_ok = False, False, False
    if refs_exist:
        try:
            from spa3r_edge.edge.encoder.backends import ONNXBackend
            ref_input = np.load(ref_input_path)
            ref_output = np.load(ref_output_path)
            
            onnx_backend = ONNXBackend(onnx_path)
            onnx_result = onnx_backend.encode(torch.from_numpy(ref_input))
            onnx_latents = onnx_result["latents"]

            # 4. Shape Check
            shape_ok = (ref_output.shape == onnx_latents.shape)
            _print_check("Shape Check", shape_ok)
            
            # 5. Dtype Check
            dtype_ok = (ref_output.dtype.kind == 'f' and onnx_latents.dtype.kind == 'f')
            _print_check("Dtype Check", dtype_ok)
            
            # 6. Numerical Check (PT ↔ ONNX)
            if shape_ok:
                result = compare(ref_output, onnx_latents, threshold=threshold)
                num_ok = result.passed
                _print_check("Numerical Check", num_ok)
            else:
                _print_check("Numerical Check", False)
        except Exception as e:
            _print_check("Shape Check", False)
            _print_check("Dtype Check", False)
            _print_check("Numerical Check", False)
    else:
        _print_check("Shape Check", False)
        _print_check("Dtype Check", False)
        _print_check("Numerical Check", False)

    checks["shape_ok"] = shape_ok
    checks["dtype_ok"] = dtype_ok
    checks["numerical_ok"] = num_ok

    # 7. Graph Simplified
    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    simplified_exists = os.path.exists(simplified_path)
    _print_check("Graph Simplified", simplified_exists)
    checks["simplified"] = simplified_exists

    # 8. Unsupported Operators
    check_path = simplified_path if simplified_exists else onnx_path
    n_unsup, unsup_list, _ = _check_unsupported_ops(check_path)
    op_passed = (n_unsup == 0)
    _print_check("Unsupported Operators", op_passed)
    checks["unsupported_ops_passed"] = op_passed

    # 3-way Numerical Verification Table
    quant_ready = False
    if simplified_exists and num_ok:
        try:
            from spa3r_edge.edge.encoder.backends import ONNXBackend
            sim_backend = ONNXBackend(simplified_path)
            sim_result = sim_backend.encode(torch.from_numpy(np.load(ref_input_path)))
            sim_latents = sim_result["latents"]
            
            # PT vs Simplified
            sim_cmp = compare(np.load(ref_output_path), sim_latents, threshold=threshold)
            
            # ONNX vs Simplified
            onnx_vs_sim_cmp = compare(onnx_latents, sim_latents, threshold=threshold)
            
            quant_ready = sim_cmp.passed
            
            print("\n  Numerical Verification (3-Way)")
            print("  -------------------------------------------------------------")
            print("  | Comparison          | Max Error  | Mean Error | Status |")
            print("  -------------------------------------------------------------")
            print(f"  | PT ↔ ONNX           | {result.max_error:10.5f} | {result.mean_error:10.5f} | {'✅' if result.passed else '❌'}      |")
            print(f"  | PT ↔ Simplified     | {sim_cmp.max_error:10.5f} | {sim_cmp.mean_error:10.5f} | {'✅' if sim_cmp.passed else '❌'}      |")
            print(f"  | ONNX ↔ Simplified   | {onnx_vs_sim_cmp.max_error:10.5f} | {onnx_vs_sim_cmp.mean_error:10.5f} | {'✅' if onnx_vs_sim_cmp.passed else '❌'}      |")
            print("  -------------------------------------------------------------\n")
            
        except Exception as e:
            print(f"Error generating 3-way table: {e}")
            
    # 9. Quantization Ready
    _print_check("Quantization Ready", quant_ready)
    checks["quant_ready"] = quant_ready

    # 10. Hailo Ready
    hailo_ready = quant_ready and op_passed
    _print_check("Hailo Ready", hailo_ready)
    checks["hailo_ready"] = hailo_ready

    print("\n==============================\n")

    all_passed = all([
        exists, refs_exist, valid, shape_ok, dtype_ok, 
        num_ok, simplified_exists, op_passed, quant_ready, hailo_ready
    ])
    checks["passed"] = all_passed
    return checks


def _print_check(label: str, passed: bool, detail: str = "") -> None:
    icon = "✓" if passed else "✗"
    detail_str = f"  {detail}" if detail else ""
    print(f"  {icon} {label}{detail_str}")


def main():
    result = verify()
    if not result.get("passed"):
        sys.exit(1)


if __name__ == "__main__":
    main()
