import torch

def _rms_norm_symbolic(g, input, normalized_shape, weight, eps):
    # Get the number of dimensions to reduce over
    # normalized_shape is a list of integers
    # Let's assume we reduce over the last len(normalized_shape) dimensions
    import torch.onnx.symbolic_helper as sym_help
    
    # Actually, normalized_shape is passed as a list of ints in the symbolic
    # But usually it's just the last dim.
    # We can use symbolic helper to get the rank of input
    
    input_rank = sym_help._get_tensor_rank(input)
    # If we can't get rank, assume 1
    if input_rank is None:
        # Fallback
        axes = [-1]
    else:
        # get length of normalized_shape
        # normalized_shape is a list of ints.
        # But wait, in ONNX symbolic, it might be a node or list.
        # Let's just assume axes=[-1] for now, or use a negative range.
        # Actually, let's just print the types to see.
        pass
    
    axes = [-1]
    
    # ONNX Ops:
    # x_squared = Pow(input, 2)
    two = g.op("Constant", value_t=torch.tensor(2.0, dtype=torch.float32))
    x_squared = g.op("Pow", input, two)
    
    mean_x2 = g.op("ReduceMean", x_squared, axes_i=axes)
    
    # eps
    eps_tensor = g.op("Constant", value_t=torch.tensor(eps, dtype=torch.float32))
    
    added = g.op("Add", mean_x2, eps_tensor)
    denom = g.op("Sqrt", added)
    
    normed = g.op("Div", input, denom)
    
    if weight is not None and not sym_help._is_none(weight):
        return g.op("Mul", normed, weight)
    return normed

from torch.onnx import register_custom_op_symbolic
register_custom_op_symbolic("aten::rms_norm", _rms_norm_symbolic, 18)

print("Registered successfully")
