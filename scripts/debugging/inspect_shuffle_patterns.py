import onnx
from onnx import shape_inference
from collections import defaultdict

m = onnx.load("spa3r_edge/latents/bisect/spa3r_encoder_1_blocks.onnx")
m = shape_inference.infer_shapes(m)

node_by_output = {}
successors = defaultdict(list)
node_by_name = {}

for node in m.graph.node:
    node_by_name[node.name] = node
    for out in node.output:
        node_by_output[out] = node.name
    for inp in node.input:
        successors[inp].append(node.name)

# Helper to get tensor shape
def get_shape(tensor_name):
    for vi in m.graph.value_info:
        if vi.name == tensor_name:
            if vi.type.tensor_type.HasField("shape"):
                return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
    for vi in m.graph.output:
        if vi.name == tensor_name:
            if vi.type.tensor_type.HasField("shape"):
                return [dim.dim_value for dim in vi.type.tensor_type.shape.dim]
    return "Unknown"

# Helper to format node info
def get_node_info(node):
    if not node:
        return "None"
    
    op = node.op_type
    name = node.name
    
    perm = "None"
    if op == "Transpose":
        for attr in node.attribute:
            if attr.name == "perm":
                perm = str(list(attr.ints))
                
    in_shape = get_shape(node.input[0]) if len(node.input) > 0 else "Unknown"
    out_shape = get_shape(node.output[0]) if len(node.output) > 0 else "Unknown"
    
    return f"{op} '{name}' (perm: {perm}) | Shape: {in_shape} -> {out_shape}"

print("--- ANALYZING FEATURE SHUFFLE PATTERNS ---")

# Look for patterns
for node in m.graph.node:
    name = node.name
    op = node.op_type
    
    if op in ["Transpose", "Reshape", "Concat"]:
        # Check successors
        for out in node.output:
            for succ_name in successors[out]:
                succ_node = node_by_name.get(succ_name)
                if not succ_node:
                    continue
                    
                succ_op = succ_node.op_type
                
                # Check 2-node patterns
                pattern = f"{op} -> {succ_op}"
                if pattern in ["Transpose -> Reshape", "Reshape -> Transpose", 
                              "Transpose -> Concat", "Concat -> Transpose"]:
                    print(f"\nPattern found: {pattern}")
                    print(f"  Node 1: {get_node_info(node)}")
                    print(f"  Node 2: {get_node_info(succ_node)}")
                    
                    # Check for 3-node patterns (Transpose -> Reshape -> Transpose)
                    if pattern == "Transpose -> Reshape":
                        for succ_out in succ_node.output:
                            for succ2_name in successors[succ_out]:
                                succ2_node = node_by_name.get(succ2_name)
                                if succ2_node and succ2_node.op_type == "Transpose":
                                    print(f"  *** Extended Pattern: Transpose -> Reshape -> Transpose ***")
                                    print(f"  Node 3: {get_node_info(succ2_node)}")
                                    
