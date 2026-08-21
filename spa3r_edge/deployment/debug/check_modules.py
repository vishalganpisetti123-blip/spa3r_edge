import os, sys, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "latents"))
    ckpt_path = os.path.abspath(os.path.join(out_dir, "..", "..", "spa3r_weights.ckpt"))
    
    pt_backend = Spa3REncoder.build_pytorch_backend(checkpoint_path=ckpt_path)
    model = pt_backend.model
    
    for name, module in model.named_modules():
        if "encoder.blocks.0.attn" in name:
            print(f"{name}: {type(module).__name__}")

if __name__ == "__main__":
    main()
