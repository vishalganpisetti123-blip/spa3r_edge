import onnx

m = onnx.load("spa3r_edge/latents/bisect/spa3r_encoder_1_blocks.onnx")

print("Before Shape Inference:")
for node in m.graph.node:
    if node.name == "/encoder/blocks.0/attn/Mul":
        print(node)

print("\nRunning Shape Inference...")
m = onnx.shape_inference.infer_shapes(m)
onnx.save(m, "spa3r_edge/latents/bisect/spa3r_encoder_1_blocks_inferred.onnx")

print("\nAfter Shape Inference:")
m2 = onnx.load("spa3r_edge/latents/bisect/spa3r_encoder_1_blocks_inferred.onnx")
for node in m2.graph.node:
    if node.name == "/encoder/blocks.0/attn/Mul":
        print(node)
