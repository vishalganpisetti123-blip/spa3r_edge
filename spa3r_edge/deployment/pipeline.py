"""
Edge Deployment Toolkit Pipeline
Orchestrates the entire export, verification, optimization, and reporting pipeline.

Usage:
    python -m deployment.pipeline
"""

import sys
import subprocess
import time

def run_stage(name, cmd):
    print(f"\n{'='*60}")
    print(f"🚀 STAGE: {name}")
    print(f"{'='*60}")
    start = time.perf_counter()
    
    result = subprocess.run(cmd, shell=True, text=True)
    
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        print(f"\n❌ Stage '{name}' FAILED after {elapsed:.1f}s")
        sys.exit(result.returncode)
    else:
        print(f"\n✅ Stage '{name}' COMPLETED in {elapsed:.1f}s")


def main():
    print("Starting Edge Deployment Toolkit Pipeline...\n")
    
    run_stage("Export (PyTorch -> ONNX)", f"{sys.executable} -m deployment.export.export")
    
    run_stage("Simplify Graph (onnxsim)", f"{sys.executable} -m deployment.optimization.simplify")
    
    run_stage("Verification & Quality Gate", f"{sys.executable} -m deployment.verification.verify")
    
    run_stage("Graph Report Generation", f"{sys.executable} -m deployment.verification.graph_report")
    
    run_stage("Hailo Precompile Report Generation", f"{sys.executable} -m deployment.compile.hailo_report")
    
    print("\n" + "="*60)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY 🎉")
    print("The model is fully verified and ready for INT8 quantization.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
