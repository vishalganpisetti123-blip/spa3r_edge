import onnx
from onnx import numpy_helper
model = onnx.load("latents/spa3r_encoder.onnx")

for node in model.graph.node:
    if node.op_type == "ReduceMean":
        axes_input = node.input[1]
        for cnode in model.graph.node:
            if cnode.name == axes_input or axes_input in cnode.output:
                if cnode.op_type == "Constant":
                    val = numpy_helper.to_array(cnode.attribute[0].t)
                    print(f"Axes for ReduceMean {node.name}: {val}")
        break
