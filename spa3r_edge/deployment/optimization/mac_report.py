"""
MAC/FLOPs Profiler

Analyzes the ONNX graph and computes estimated MACs per layer, 
along with memory and parameter counts.
Outputs mac_report.json to see which blocks consume the most compute.
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

def generate_mac_report():
    import onnx
    from onnx import shape_inference
    
    config = load_config()
    onnx_path = config["output_path"]
    simplified_path = onnx_path.replace(".onnx", "_simplified.onnx")
    if os.path.exists(simplified_path):
        onnx_path = simplified_path
        
    out_dir = os.path.dirname(onnx_path) or "."
    report_path = os.path.join(out_dir, "mac_report.json")
    
    print(f"[mac_report] Analyzing {onnx_path}...")
    model = onnx.load(onnx_path, load_external_data=False)
    try:
        model = shape_inference.infer_shapes(model)
    except:
        pass
        
    graph = model.graph
    
    # Map tensor names to shapes
    shapes = {}
    for info in list(graph.input) + list(graph.value_info) + list(graph.output):
        shapes[info.name] = _parse_shape(info)
        
    report = []
    total_macs = 0
    total_params = 0
    
    for node in graph.node:
        macs = 0
        params = 0
        if node.op_type == "MatMul" or node.op_type == "Gemm":
            if node.input[0] in shapes and node.input[1] in shapes:
                A = shapes[node.input[0]]
                B = shapes[node.input[1]]
                if A and B and all(isinstance(x, int) for x in A) and all(isinstance(x, int) for x in B):
                    # Simple heuristic for M x K * K x N
                    if len(A) >= 2 and len(B) >= 2:
                        M = np.prod(A[:-1]) if len(A) > 2 else A[0]
                        K = A[-1]
                        N = B[-1]
                        macs = int(M * K * N)
        elif node.op_type == "Conv":
            if node.input[0] in shapes and node.input[1] in shapes:
                X = shapes[node.input[0]]
                W = shapes[node.input[1]]
                if X and W and all(isinstance(x, int) for x in X) and all(isinstance(x, int) for x in W):
                    # X: N, C, H, W. W: M, C/group, kH, kW
                    if len(X) == 4 and len(W) == 4:
                        N, C, H, W_dim = X
                        M, C_in, kH, kW = W
                        macs = int(N * M * H * W_dim * kH * kW * C_in)
                        params = int(M * C_in * kH * kW)
                        
        if macs > 0 or params > 0:
            report.append({
                "layer": node.name or node.op_type,
                "type": node.op_type,
                "macs": macs,
                "parameters": params
            })
            total_macs += macs
            total_params += params
            
    # Calculate percentages
    if total_macs > 0:
        for r in report:
            r["compute_percentage"] = round((r["macs"] / total_macs) * 100, 2)
            
    report.sort(key=lambda x: x["macs"], reverse=True)
    
    summary = {
        "total_macs": total_macs,
        "total_parameters": total_params,
        "top_layers": report[:20]
    }
    
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"[mac_report] ✓ Saved MAC report to {report_path}")
    print(f"Total MACs: {total_macs:,} | Total Params mapped: {total_params:,}")

if __name__ == "__main__":
    import numpy as np
    generate_mac_report()
