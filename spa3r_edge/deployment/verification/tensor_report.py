"""
Tensor Report Generator

Analyzes the ONNX graph to estimate memory consumption for 
intermediate tensors. This is useful for checking SRAM limits.
"""
import os
import sys
import json
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "export.yaml"))
    with open(config_path, "r") as f:
        return yaml.safe_load(f)["export"]

def _parse_shape(value_info):
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            shape.append(dim.dim_param)
        else:
            shape.append(None)
    return shape

def generate_tensor_report():
    import onnx
    config = load_config()
    onnx_path = config["output_path"]
    
    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    if os.path.exists(simplified_path):
        onnx_path = simplified_path
        
    out_dir = os.path.dirname(onnx_path) or "."
    report_path = os.path.join(out_dir, "tensor_report.json")
    
    print(f"[tensor_report] Analyzing {onnx_path}...")
    model = onnx.load(onnx_path, load_external_data=False)
    
    # We will use ONNX shape inference to get intermediate tensor shapes
    from onnx import shape_inference
    try:
        inferred_model = shape_inference.infer_shapes(model)
        value_infos = inferred_model.graph.value_info
    except Exception as e:
        print(f"Shape inference failed: {e}")
        value_infos = []

    report = []
    
    def process_tensor(info, category):
        shape = _parse_shape(info)
        # Assuming FP32 (4 bytes per element) for estimation
        try:
            num_elements = 1
            for dim in shape:
                if isinstance(dim, int):
                    num_elements *= dim
                else:
                    # Dynamic shape, default to 1 for estimation
                    num_elements *= 1
            bytes_size = num_elements * 4
        except Exception:
            bytes_size = 0
            
        report.append({
            "tensor": info.name,
            "category": category,
            "shape": shape,
            "bytes": bytes_size,
            "kilobytes": bytes_size / 1024.0,
            "megabytes": bytes_size / (1024.0 * 1024.0)
        })

    for info in model.graph.input:
        process_tensor(info, "input")
        
    for info in model.graph.output:
        process_tensor(info, "output")
        
    for info in value_infos:
        process_tensor(info, "intermediate")

    # Sort by size descending
    report.sort(key=lambda x: x["bytes"], reverse=True)
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"[tensor_report] ✓ Saved report to {report_path}")

if __name__ == "__main__":
    generate_tensor_report()
