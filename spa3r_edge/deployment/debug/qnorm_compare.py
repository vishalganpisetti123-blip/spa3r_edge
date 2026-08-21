import torch
import torch.nn as nn
import numpy as np
import os
import onnxruntime as ort

class QNormModel(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.q_norm = nn.LayerNorm(head_dim, eps=1e-6)
        
    def forward(self, q):
        return self.q_norm(q)

def main():
    head_dim = 1024 // 12  # Wait, Spa3R embed_dim=1024 or 768? 
    # Whatever, let's just use 85 or 64. DINOv3 ViT-L has 1024 embed_dim, 16 heads -> 64.
    head_dim = 64
    model = QNormModel(head_dim).eval()
    
    # 4D tensor (B, num_heads, N, head_dim)
    # We will use exactly what's seen: (1, 16, 1025, 64)
    q = torch.randn(1, 16, 1025, head_dim) * 0.01  # small variance
    
    with torch.no_grad():
        pt_out = model(q)
        
    onnx_path = "latents/test_qnorm.onnx"
    torch.onnx.export(
        model,
        (q,),
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["q"],
        output_names=["out"]
    )
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(["out"], {"q": q.numpy()})[0]
    
    diff = np.abs(pt_out.numpy() - onnx_out)
    print(f"Q_Norm Max Error: {np.max(diff):.7f}")

if __name__ == "__main__":
    main()
