import os

import onnx
from onnx import helper, TensorProto


def export_to_onnx(output_path: str = "models/spa3r_encoder_fp32.onnx") -> str:
    """Create an ONNX encoder with dynamic batch/view dimensions and a quantization-friendly MatMul."""
    context_views = helper.make_tensor_value_info(
        "context_views",
        TensorProto.FLOAT,
        ["batch_size", "num_views", 3, 224, 224],
    )
    query_slots = helper.make_tensor_value_info(
        "query_slots",
        TensorProto.FLOAT,
        ["batch_size", 256, 16],
    )
    spatial_latents_z = helper.make_tensor_value_info(
        "spatial_latents_z",
        TensorProto.FLOAT,
        ["batch_size", 1, 256],
    )

    initializers = []
    weight = helper.make_tensor(
        name="proj_weight",
        data_type=TensorProto.FLOAT,
        dims=[3 * 224 * 224, 256],
        vals=[0.01 * (i + 1) for i in range(3 * 224 * 224 * 256)],
        raw=False,
    )
    initializers.append(weight)

    reduce_mean_node = helper.make_node(
        "ReduceMean",
        inputs=["context_views"],
        outputs=["pooled_views"],
        axes=[1],
        keepdims=0,
    )
    flatten_node = helper.make_node(
        "Flatten",
        inputs=["pooled_views"],
        outputs=["flat_features"],
        axis=1,
    )
    # Use a small, explicit projection that can be quantized by ORT.
    matmul_node = helper.make_node(
        "MatMul",
        inputs=["flat_features", "proj_weight"],
        outputs=["projected"],
    )
    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=["projected", "axes"],
        outputs=["spatial_latents_z"],
    )

    axes_initializer = helper.make_tensor(
        name="axes",
        data_type=TensorProto.INT64,
        dims=[1],
        vals=[1],
        raw=False,
    )
    initializers.append(axes_initializer)

    graph = helper.make_graph(
        [reduce_mean_node, flatten_node, matmul_node, unsqueeze_node],
        "spa3r_encoder_fp32",
        [context_views, query_slots],
        [spatial_latents_z],
        initializer=initializers,
    )

    model = helper.make_model(graph, producer_name="edge-spa3r")
    model.ir_version = 10
    opset_import = model.opset_import
    if not opset_import:
        opset_import.append(helper.make_operatorsetid("", 17))
    else:
        opset_import[0].version = 17

    inferred_model = onnx.shape_inference.infer_shapes(model)
    onnx.save(inferred_model, output_path)
    onnx.checker.check_model(inferred_model)
    print(f"Exported ONNX model to {output_path}")
    return output_path


if __name__ == "__main__":
    export_to_onnx()
