import onnx
model = onnx.load("latents/spa3r_encoder.onnx")

for node in model.graph.node:
    if node.op_type == "ReduceMean":
        print("Found ReduceMean:", node.name)
        print("Inputs:", node.input)
        for init in model.graph.initializer:
            if init.name in node.input:
                from onnx import numpy_helper
                print("Initializer:", init.name, numpy_helper.to_array(init))
        break
