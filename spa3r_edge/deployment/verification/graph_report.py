"""
Graph Report Generator

Extracts structural statistics from the ONNX graph and saves them to a JSON file.
This allows the edge runtime to perform a fast sanity check before loading the HEF.
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

def _get_hailo_supported_ops():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "hailo8_ops.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            supported = config.get("supported", []) + config.get("partially_supported", [])
            return frozenset(supported)
    except Exception:
        return frozenset()

def generate_report():
    import onnx
    from onnx import numpy_helper
    
    config = load_config()
    onnx_path = config["output_path"]
    
    # Prefer simplified graph if available
    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    if os.path.exists(simplified_path):
        onnx_path = simplified_path
        
    out_dir = os.path.dirname(onnx_path) or "."
    report_path = os.path.join(out_dir, "graph_report.json")
    
    print(f"[graph_report] Analyzing {onnx_path}...")
    model = onnx.load(onnx_path, load_external_data=False)
    graph = model.graph
    
    # Node and Parameter Count
    node_count = len(graph.node)
    
    param_count = 0
    for init in graph.initializer:
        try:
            tensor = numpy_helper.to_array(init)
            param_count += tensor.size
        except Exception:
            pass # external data without loading it
            
    # Shapes
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
        
    input_shapes = {i.name: _parse_shape(i) for i in graph.input}
    output_shapes = {o.name: _parse_shape(o) for o in graph.output}
    
    # Detect dynamic shapes
    has_dynamic = False
    for shapes in list(input_shapes.values()) + list(output_shapes.values()):
        if any(isinstance(d, str) or d is None for d in shapes):
            has_dynamic = True
            break
            
    # Detect external data
    has_external = False
    for init in graph.initializer:
        if init.HasField("data_location") and init.data_location == onnx.TensorProto.EXTERNAL:
            has_external = True
            break
            
    # Unsupported ops
    from collections import Counter
    op_counts = Counter(node.op_type for node in graph.node)
    supported_ops = _get_hailo_supported_ops()
    unsupported = [op for op in op_counts.keys() if op not in supported_ops]
    
    # Opset
    opset = model.opset_import[0].version if len(model.opset_import) > 0 else "unknown"

    report = {
        "nodes": node_count,
        "parameters": param_count,
        "input_shape": input_shapes.get("images", input_shapes[list(input_shapes.keys())[0]]),
        "output_shape": output_shapes.get("latents", output_shapes[list(output_shapes.keys())[0]]),
        "opset": opset,
        "dynamic": has_dynamic,
        "external_data": has_external,
        "unsupported_ops": unsupported
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"[graph_report] ✓ Saved report to {report_path}")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    generate_report()
