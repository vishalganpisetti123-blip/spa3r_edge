# 🚀 Debug Sprint Concluded: Numerical Divergence Resolved!

I have excellent news: **The PyTorch-to-ONNX numerical divergence has been completely eliminated.** 

The max error across the entire model (`latents`) has dropped from **`1.23`** down to **`0.0000353`** (which is well within standard `float32` numerical tolerance thresholds). 

---

## 🔍 The Root Causes

As we traced layer-by-layer, the problem manifested across two entirely different components. Both bugs interacted during the tracing phase to silently corrupt the mathematical graph.

### 1. The RoPE Tracing Bug (Positions -> Constants)
In the original `spa3r` RoPE implementation (`spa3r/models/layers/rope.py`), the model dynamically computed the maximum position index for embedding lookups:
```python
max_position = int(positions.max()) + 1
```
During `torch.onnx.export` (which uses TorchScript tracing), the dynamic PyTorch tensor `positions.max()` was coerced into a static Python integer. This caused ONNX to bake in a constant graph that failed to generalize, breaking the positional encoding entirely.

**The Fix:** 
Without touching the original `rope.py` (respecting the read-only invariant), I created an `OnnxRotaryPositionEmbedding2D` in `patches.py`. This alternative implementation computes the Rotary Positional Embeddings using pure `torch.einsum`, eliminating `F.embedding` and integer-based bounds entirely. We injected this replacement cleanly via the `Spa3RWrapper` during export.

### 2. The Silent Killer: RMSNorm `eps` Parsing
After fixing RoPE, a massive explosion in error (from `0.00002` to `1.23`) was isolated precisely at `encoder.blocks.0.attn.q_norm`. 

`q_norm` uses `RMSNorm`. When tracing `RMSNorm`, the `spa3r_edge` repository used a custom symbolic override in `symbolic_registry.py` because the authors assumed `aten::rms_norm` was unsupported in ONNX opset 18. 
However, their custom ONNX symbolic had a critical parsing bug:
```python
    if eps_val is None:
        eps_val = 1e-6
```
In PyTorch 2.4, `nn.RMSNorm` defaults `eps` to `None`, which PyTorch internally resolves to the machine epsilon: `torch.finfo(torch.float32).eps` (`~1.19e-07`). 
Because the ONNX symbolic fell back to `1e-6`, it created a tiny but devastating denominator mismatch (`1e-6` vs `1.19e-7`). When applied to query/key tensors (`q` and `k`) that had extremely low variance, this small epsilon difference caused the normalized outputs to explode by a factor of 1000x.

**The Fix:**
I patched `symbolic_registry.py` to correctly resolve `eps=None` to `torch.finfo(torch.float32).eps`, matching PyTorch 2.4's native behavior exactly.

---

## ✅ Current Status

| Stage                  | Status     |
| ---------------------- | ---------- |
| Export                 | ✅ PASS     |
| Graph Valid            | ✅ PASS     |
| Simplification         | ⚠️ Skipped |
| Numerical Verification | ✅ PASS     |
| Quantization           | ⛔ STOP     |
| Hailo Compile          | ⛔ STOP     |

## ⏭️ Next Steps

With a numerically verified, valid, and accurate ONNX graph, we are now unblocked to proceed to **Stage 4: Quantization** and **Stage 5: Hailo Compilation**. 

Please let me know if you would like to proceed with the remaining deployment pipeline!
