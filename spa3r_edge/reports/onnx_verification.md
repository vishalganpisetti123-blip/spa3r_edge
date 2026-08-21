# ONNX Verification Report

**Model**: Spa3R Edge Encoder
**Status**: `READY FOR QUANTIZATION`

## Overview
This report details the numerical verification between the native PyTorch implementation and the exported ONNX graph. The evaluation confirms that the exported graph achieves mathematical equivalence with the PyTorch model within the limits of standard `float32` noise.

## Tensor Shapes
- **Input Shape**: `(1, 1, 3, 224, 224)`
- **Output Shape** (Latents): `(1, 256, 768)`

## Numerical Parity Metrics
- **Max Error**: `3.53e-05` (0.0000353)
- **Unsupported Ops**: `0`

## Known Bugs Resolved
During the export process, the following critical bugs causing graph divergence or failure were successfully mitigated:

1. **RoPE Embedding (Dynamic Tensor to Constant)**
   - **Status**: ✅ Fixed
   - **Resolution**: Replaced `RotaryPositionEmbedding2D` with an ONNX-safe custom implementation (`OnnxRotaryPositionEmbedding2D`) that utilizes pure `torch.einsum`, avoiding `F.embedding` and Python integer coercions that previously resulted in static baked-in graphs.

2. **RMSNorm `eps` Parsing (Numerical Explosion)**
   - **Status**: ✅ Fixed
   - **Resolution**: Identified a bug in the custom ONNX symbolic handler `_rms_norm_symbolic` which defaulted to `1e-6` when `eps=None`. The fallback was modified to use `torch.finfo(torch.float32).eps` (`~1.19e-07`), restoring identical mathematical parity with the `q_norm` and `k_norm` PyTorch standard behavior.

## Conclusion
The ONNX export is now considered **deployment-grade** and verified for further conversion. We are clear to proceed with graph quantization using a representative calibration dataset via the Hailo Dataflow Compiler.
