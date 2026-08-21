import onnx
from collections import defaultdict

m = onnx.load("spa3r_edge/latents/bisect/spa3r_encoder_1_blocks.onnx")

# Build maps for predecessors and successors
node_by_output = {}
successors = defaultdict(list)

for node in m.graph.node:
    for out in node.output:
        node_by_output[out] = node.name
    for inp in node.input:
        successors[inp].append(node.name)

# Collect all nodes of interest
transpose_nodes = []
reshape_nodes = []
flatten_nodes = []
split_nodes = []
concat_nodes = []

for node in m.graph.node:
    if node.op_type == "Transpose":
        transpose_nodes.append(node)
    elif node.op_type == "Reshape":
        reshape_nodes.append(node)
    elif node.op_type == "Flatten":
        flatten_nodes.append(node)
    elif node.op_type == "Split":
        split_nodes.append(node)
    elif node.op_type == "Concat":
        concat_nodes.append(node)

print(f"Found {len(transpose_nodes)} Transpose nodes.")
print(f"Found {len(reshape_nodes)} Reshape nodes.")
print(f"Found {len(flatten_nodes)} Flatten nodes.")
print(f"Found {len(split_nodes)} Split nodes.")
print(f"Found {len(concat_nodes)} Concat nodes.")

print("\n--- TRANSPOSE NODES ANALYSIS ---")
for node in transpose_nodes:
    print(f"Node: {node.name}")
    # Extract permutation
    perm = None
    for attr in node.attribute:
        if attr.name == "perm":
            perm = list(attr.ints)
    print(f"  Permutation: {perm}")
    
    # Predecessors
    preds = []
    for inp in node.input:
        if inp in node_by_output:
            preds.append(node_by_output[inp])
        else:
            preds.append(f"Input/Initializer: {inp}")
    print(f"  Predecessors: {preds}")
    
    # Successors
    succs = []
    for out in node.output:
        if out in successors:
            succs.extend(successors[out])
    print(f"  Successors: {succs}")
    print("-")

