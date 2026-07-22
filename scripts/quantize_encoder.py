import os
import sys

import onnx
import onnxruntime as ort


def quantize_encoder(input_path: str = "models/spa3r_encoder_fp32.onnx", output_path: str = "models/spa3r_encoder_int8.onnx") -> str:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"FP32 ONNX model not found: {input_path}")

    # ONNX Runtime provides dynamic quantization support via the quantize_dynamic helper.
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_model = quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    print(f"Quantized model written to {output_path}")
    return output_path


if __name__ == "__main__":
    quantize_encoder()
