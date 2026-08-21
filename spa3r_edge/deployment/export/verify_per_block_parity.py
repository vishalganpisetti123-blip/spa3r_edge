import os
import sys
import torch
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../submodules/vggt")))

from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder
from spa3r_edge.deployment.export.per_block_wrappers import StemWrapper, BlockWrapper, HeadWrapper
from spa3r_edge.deployment.export.split_wrappers import EncoderWrapper

def main():
    print("Loading original PyTorch model...")
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path=None)
    model = pt_backend.model.eval()
    
    orig_encoder = EncoderWrapper(model, height=224, width=224).eval()
    
    # Generate random input
    torch.manual_seed(42)
    dummy_input = torch.randn(1, 256, 2048)
    
    print("\nRunning original PyTorch encoder...")
    with torch.no_grad():
        expected_out = orig_encoder(dummy_input).numpy()
        
    print("\nExporting ONNX models on the fly...")
    onnx_dir = os.path.abspath("spa3r_edge/latents/per_block_test")
    os.makedirs(onnx_dir, exist_ok=True)
    
    # Export Stem
    stem = StemWrapper(model, height=224, width=224).eval()
    stem_path = os.path.join(onnx_dir, "stem.onnx")
    torch.onnx.export(stem, dummy_input, stem_path, opset_version=16, input_names=["input"], output_names=["output"])
    
    # Export Blocks
    block_paths = []
    final_pos = stem.final_pos
    dummy_block_input = torch.randn(1, 516, 768)
    for i in range(model.encoder.n_blocks):
        block = BlockWrapper(model.encoder.blocks[i], final_pos).eval()
        # Initialize RoPE properly
        with torch.no_grad(): block(dummy_block_input)
        path = os.path.join(onnx_dir, f"block_{i}.onnx")
        torch.onnx.export(block, dummy_block_input, path, opset_version=16, input_names=["input"], output_names=["output"])
        block_paths.append(path)
        
    # Export Head
    head = HeadWrapper(model.encoder, model.num_queries).eval()
    head_path = os.path.join(onnx_dir, "head.onnx")
    torch.onnx.export(head, dummy_block_input, head_path, opset_version=16, input_names=["input"], output_names=["output"])
    
    print("\nRunning chained ONNX sessions...")
    providers = ['CPUExecutionProvider']
    
    sess_stem = ort.InferenceSession(stem_path, providers=providers)
    sess_blocks = [ort.InferenceSession(p, providers=providers) for p in block_paths]
    sess_head = ort.InferenceSession(head_path, providers=providers)
    
    # Run Stem
    x = sess_stem.run(None, {sess_stem.get_inputs()[0].name: dummy_input.numpy()})[0]
    # Run Blocks
    for i in range(6):
        x = sess_blocks[i].run(None, {sess_blocks[i].get_inputs()[0].name: x})[0]
    # Run Head
    actual_out = sess_head.run(None, {sess_head.get_inputs()[0].name: x})[0]
    
    print("\nComparing results...")
    max_diff = np.max(np.abs(expected_out - actual_out))
    mean_diff = np.mean(np.abs(expected_out - actual_out))
    
    print(f"Max absolute difference: {max_diff:.8f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")
    
    if max_diff < 1e-4:
        print("\n✅ SUCCESS: Chained ONNX execution matches original PyTorch encoder perfectly!")
    else:
        print("\n❌ ERROR: Significant differences detected between ONNX and PyTorch outputs!")

if __name__ == "__main__":
    main()
