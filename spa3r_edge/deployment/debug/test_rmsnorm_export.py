import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort

class RMSNormModel(nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        
    def forward(self, q):
        return self.q_norm(q)

def main():
    head_dim = 64
    model = RMSNormModel(head_dim).eval()
    
    q = torch.randn(1, 16, 1025, head_dim) * 0.01  # small variance
    
    with torch.no_grad():
        pt_out = model(q)
        
    onnx_path = "latents/test_rmsnorm.onnx"
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
    print(f"RMSNorm Max Error: {np.max(diff):.7f}")

if __name__ == "__main__":
    main()
