"""
ONNX operator inventory report.

Responsibility: scan an ONNX graph and report all operator types, their
frequencies, and whether each is supported by Hailo-8.

Usage:
    cd spa3r_edge
    python -m deployment.optimization.operator_report
"""

import os
import sys
from collections import Counter

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

def get_hailo_supported_ops():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hailo8_ops.yaml")
    try:
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return frozenset(config.get("supported", []) + config.get("partially_supported", []))
    except Exception as e:
        print(f"Warning: could not load {config_path}: {e}")
        return frozenset()


def load_config(config_path="deployment/configs/export.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["export"]


def operator_report(onnx_path=None, config=None) -> dict:
    """Generate an operator inventory report for an ONNX model.

    Args:
        onnx_path: Path to the ONNX file. If None, uses config output_path.
        config: Export config dict.

    Returns:
        Dict with 'all_ops', 'unsupported_ops', 'supported_count',
        'unsupported_count', 'hailo_ready'.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("Install onnx: pip install onnx")

    if config is None:
        config = load_config()

    if onnx_path is None:
        # Prefer simplified model if it exists
        base = config["output_path"]
        simplified = base.replace(".onnx", "_simplified.onnx")
        onnx_path = simplified if os.path.exists(simplified) else base

    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = onnx.load(onnx_path)
    op_counts = Counter(node.op_type for node in model.graph.node)

    supported_ops = get_hailo_supported_ops()
    supported = {op: cnt for op, cnt in op_counts.items() if op in supported_ops}
    unsupported = {op: cnt for op, cnt in op_counts.items() if op not in supported_ops}
    hailo_ready = len(unsupported) == 0

    result = {
        "onnx_path": onnx_path,
        "all_ops": dict(op_counts),
        "supported_ops": supported,
        "unsupported_ops": unsupported,
        "supported_count": sum(supported.values()),
        "unsupported_count": sum(unsupported.values()),
        "unique_unsupported": len(unsupported),
        "hailo_ready": hailo_ready,
    }

    # Print report
    print("\n" + "=" * 55)
    print("  OPERATOR REPORT")
    print("=" * 55)
    print(f"  Model: {onnx_path}")
    print(f"  Total nodes: {sum(op_counts.values())}")
    print(f"  Unique op types: {len(op_counts)}")
    print()

    print("  Supported ops (Hailo-8):")
    for op, cnt in sorted(supported.items()):
        print(f"    ✓  {op:<30} x{cnt}")

    if unsupported:
        print()
        print("  ⚠️  Unsupported ops (require custom layer or graph surgery):")
        for op, cnt in sorted(unsupported.items()):
            print(f"    ✗  {op:<30} x{cnt}")
    else:
        print()
        print("  ✓ All operators are Hailo-8 supported.")

    print()
    verdict = "✅ HAILO-READY" if hailo_ready else f"❌ {len(unsupported)} UNSUPPORTED OP TYPE(S)"
    print(f"  Verdict: {verdict}")
    print("=" * 55 + "\n")

    return result


def main():
    operator_report()


if __name__ == "__main__":
    main()
