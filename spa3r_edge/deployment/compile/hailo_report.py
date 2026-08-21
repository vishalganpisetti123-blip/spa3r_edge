"""
Hailo Precompile Report Generator

Analyzes the ONNX graph and the graph_report to produce a Markdown 
document detailing compilation readiness, memory estimates, and 
candidate split points for the Hailo DSP compiler.
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

def generate_precompile_report():
    config = load_config()
    onnx_path = config["output_path"]
    out_dir = os.path.dirname(onnx_path) or "."
    
    graph_report_path = os.path.join(out_dir, "graph_report.json")
    report_out_path = os.path.join(out_dir, "precompile_report.md")
    
    print("[hailo_report] Generating precompile report...")
    
    if not os.path.exists(graph_report_path):
        print(f"❌ Cannot find {graph_report_path}. Run graph_report.py first.")
        sys.exit(1)
        
    with open(graph_report_path, "r") as f:
        graph_stats = json.load(f)

    # Readiness Checks
    fixed_shape = not graph_stats.get("dynamic", True)
    no_unsupported = len(graph_stats.get("unsupported_ops", [])) == 0
    int8_ready = fixed_shape and no_unsupported
    
    # Simple Heuristics for sizes
    input_shape = graph_stats.get("input_shape", [1,1,3,224,224])
    output_shape = graph_stats.get("output_shape", [1,256,768])
    
    # 256 tokens * 768 dim * 4 bytes (FP32) = ~786 KB per intermediate activation
    activations_size_kb = (256 * 768 * 4) / 1024
    
    # Generate Markdown
    md = [
        "# Hailo-8 Pre-Compilation Report",
        "",
        "## Readiness Checks",
        f"- **Fixed Input Shape**: {'✅ Yes' if fixed_shape else '❌ No (Dynamic shapes detected)'}",
        f"- **Unsupported Ops**: {'✅ None' if no_unsupported else '❌ Found ' + str(len(graph_stats.get('unsupported_ops', [])))}",
        f"- **INT8 Ready**: {'✅ Yes' if int8_ready else '❌ No'}",
        "",
        "## Resource Estimates",
        f"- **Parameters**: {graph_stats.get('parameters', 0):,}",
        f"- **Input Size (FP32)**: {int((224*224*3*4)/1024)} KB",
        f"- **Output Size (FP32)**: {int(activations_size_kb)} KB",
        f"- **Intermediate Activation Size**: ~{int(activations_size_kb)} KB per transformer block",
        "",
        "## Candidate Split Points",
        "If the full model exceeds the Hailo-8 SRAM limits during compilation, consider splitting the graph at these boundaries.",
        "",
        "### Candidate A: Split after VGGT Aggregator",
        "- **Context**: Before the first Cross-Attention layer. Runs VGGT natively on Hailo.",
        "- **Pros**: Keeps dense vision processing on DSP.",
        "- **Cons**: Forces CPU to run heavy transformer cross-attention.",
        "- **Estimated Latency Penalty**: +15ms (Host-DSP context switch)",
        "- **Estimated Bandwidth**: ~786 KB transfer to CPU",
        "- **Estimated CPU Load**: High (Transformer blocks on CPU)",
        "",
        "### Candidate B: Split after Encoder Block 2",
        "- **Context**: Middle of the Spa3R Transformer.",
        "- **Pros**: Offloads half the transformer to DSP, balancing SRAM.",
        "- **Cons**: Activations must leave DSP and return, or stay on CPU for the rest.",
        "- **Estimated Latency Penalty**: +15ms",
        "- **Estimated Bandwidth**: ~786 KB transfer",
        "- **Estimated CPU Load**: Medium",
        "",
        "### Candidate C: Full Model on Hailo",
        "- **Context**: No split. 100% execution on Hailo-8.",
        "- **Pros**: Zero context switching latency. Maximum power efficiency.",
        "- **Cons**: May exceed 26MB SRAM limit, causing compiler to spill to host memory automatically.",
        "- **Estimated Latency Penalty**: 0ms",
        "- **Estimated Bandwidth**: 0 KB intermediate transfer",
        "- **Estimated CPU Load**: Low",
        ""
    ]
    
    with open(report_out_path, "w") as f:
        f.write("\n".join(md))
        
    print(f"[hailo_report] ✓ Saved report to {report_out_path}")

if __name__ == "__main__":
    generate_precompile_report()
