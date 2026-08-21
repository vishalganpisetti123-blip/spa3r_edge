import onnx
import sys

def analyze_onnx(model_path):
    print(f"Loading {model_path}...")
    try:
        model = onnx.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\n--- Opset Version ---")
    try:
        opset = model.opset_import[0].version
        print(f"Opset: {opset}")
    except IndexError:
        print("Opset: Unknown")

    print("\n--- Operator Counts ---")
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1

    for k, v in sorted(ops.items()):
        print(f"{k}: {v}")

    print("\n--- Normalization Operators ---")
    norm_count = 0
    for node in model.graph.node:
        if "Norm" in node.op_type:
            print(f"Name: {node.name}, Op_Type: {node.op_type}")
            norm_count += 1
    if norm_count == 0:
        print("No operators containing 'Norm' found.")

if __name__ == "__main__":
    analyze_onnx(sys.argv[1] if len(sys.argv) > 1 else "spa3r_edge/latents/spa3r_encoder_hailo.onnx")
