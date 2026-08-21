"""
Custom ONNX symbolic handlers for operators unsupported in standard opsets.

This is the ONLY file that touches torch.onnx internals. All custom op
registrations live here so they are easy to audit and maintain.
"""

import torch
from torch.onnx import register_custom_op_symbolic


def _rms_norm_symbolic(g, input, normalized_shape, weight, eps):
    """Custom ONNX symbolic for aten::rms_norm (not in opset 18).

    Implements: output = input / sqrt(mean(input^2) + eps) * weight
    """
    import torch.onnx.symbolic_helper as sym_help

    # Determine reduction axes from normalized_shape
    if isinstance(normalized_shape, (list, tuple)):
        axes = list(range(-len(normalized_shape), 0))
    else:
        try:
            val = sym_help._parse_arg(normalized_shape, "is")
            axes = list(range(-len(val), 0))
        except Exception:
            axes = [-1]

    two = g.op("Constant", value_t=torch.tensor(2.0, dtype=torch.float32))
    x_squared = g.op("Pow", input, two)

    # Opset 18: ReduceMean takes axes as an input tensor
    axes_tensor = g.op("Constant", value_t=torch.tensor(axes, dtype=torch.int64))
    mean_x2 = g.op("ReduceMean", x_squared, axes_tensor, keepdims_i=1)

    # Parse eps robustly
    try:
        if isinstance(eps, float):
            eps_val = eps
        elif isinstance(eps, torch.Tensor):
            eps_val = eps.item()
        else:
            eps_val = sym_help._parse_arg(eps, "f")
    except Exception:
        eps_val = float(torch.finfo(torch.float32).eps)

    if eps_val is None:
        eps_val = float(torch.finfo(torch.float32).eps)

    eps_tensor = g.op("Constant", value_t=torch.tensor(eps_val, dtype=torch.float32))
    added = g.op("Add", mean_x2, eps_tensor)
    denom = g.op("Sqrt", added)
    normed = g.op("Div", input, denom)

    if weight is not None and not sym_help._is_none(weight):
        return g.op("Mul", normed, weight)
    return normed


_registered = False


def register_all():
    """Register all custom ONNX symbolics. Safe to call multiple times."""
    global _registered
    if _registered:
        return

    register_custom_op_symbolic("aten::rms_norm", _rms_norm_symbolic, 18)
    _registered = True
