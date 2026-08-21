import onnx

m = onnx.load("spa3r_edge/latents/bisect/spa3r_encoder_1_blocks.onnx")

for node in m.graph.node:
    for out in node.output:
        if out == "/encoder/blocks.0/attn/Sqrt_1_output_0":
            print("Producer Node:", node.name)
            print("Op:", node.op_type)
            print("Inputs:", node.input)
            print("Outputs:", node.output)
            print("---")
            
