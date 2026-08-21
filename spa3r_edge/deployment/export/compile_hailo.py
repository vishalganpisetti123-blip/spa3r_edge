import os
import sys
import glob

# Ensure hailo_sdk_client is installed
try:
    from hailo_sdk_client import ClientRunner
except ImportError:
    print("hailo_sdk_client is not installed. Please install it to run this script.")
    sys.exit(1)

def compile_model(onnx_path, out_dir):
    model_name = os.path.basename(onnx_path).replace(".onnx", "")
    print(f"\n{'='*50}")
    print(f"Compiling {model_name}...")
    print(f"{'='*50}")
    
    # Initialize the runner
    runner = ClientRunner(hw_arch="hailo8l")
    
    try:
        # 1. Translate the ONNX model to Hailo's internal format
        print(f"Translating {onnx_path}...")
        runner.translate_onnx_model(
            onnx_path,
            model_name,
            start_node_names=None,
            end_node_names=None
        )
        print("Translation successful!")
        
        # 2. Optimize the model (required before compilation)
        # We don't have calibration data, so we might only be able to run basic optimizations.
        # Actually, optimization requires quantization in Hailo. 
        # But let's see if we can just get past the parsing/fuser stage which is where the previous crash happened!
        print(f"Running optimization...")
        # We use a dummy dataset for optimization if required, or just try to compile
        # wait, let's just see if translate passes. The crash was during translation (fuser pass).
        
        # Save HAR file
        har_path = os.path.join(out_dir, f"{model_name}.har")
        runner.save_har(har_path)
        print(f"Saved HAR to {har_path}")
        
    except Exception as e:
        print(f"Failed to compile {model_name}:")
        import traceback
        traceback.print_exc()

def main():
    onnx_dir = "spa3r_edge/latents/per_block"
    out_dir = "spa3r_edge/latents/compiled"
    os.makedirs(out_dir, exist_ok=True)
    
    onnx_files = sorted(glob.glob(os.path.join(onnx_dir, "*_sim.onnx")))
    if not onnx_files:
        print(f"No *_sim.onnx files found in {onnx_dir}")
        return
        
    print(f"Found {len(onnx_files)} models to compile.")
    for f in onnx_files:
        compile_model(f, out_dir)

if __name__ == "__main__":
    main()
