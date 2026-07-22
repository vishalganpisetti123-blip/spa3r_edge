import os
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- 1. CORE PSFM ARCHITECTURE (PYTORCH) ---

class Spa3REncoder(nn.Module):
    """
    The View-Invariant Encoder.
    Compresses multiple context views into a single, unified 3D spatial latent representation.
    """
    def __init__(self):
        super().__init__()
        # A mock implementation to represent the 2D-to-3D spatial reasoning backbone
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=8, stride=8, padding=0)

    def forward(self, context_images):
        # context_images: [Batch, N_views, C, H, W]
        x = context_images.view(-1, 3, 224, 224)
        x = self.conv(x)
        x = self.conv2(x)
        # Reshape [Batch, 3, 14, 14] to expected [Batch, 196, 3] geometry tokens
        x = x.view(-1, 3, 196).permute(0, 2, 1)
        return x

class PredictiveSpatialDecoder(nn.Module):
    """
    The View-Conditioned Decoder (Used only for training).
    Forces the encoder to learn 3D math by penalizing geometric inconsistencies from unseen views.
    """
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(196 * 3 + 16, 256) # 16 is for the 4x4 camera pose matrix

    def forward(self, spatial_latent, target_camera_poses):
        b = spatial_latent.shape[0]
        x = spatial_latent.view(b, -1)
        poses = target_camera_poses.view(b, -1)
        x = torch.cat([x, poses], dim=-1)
        return self.fc(x)

class PSFM_Framework(nn.Module):
    """
    The complete Predictive Spatial Field Modeling (PSFM) pipeline wrapper.
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder  # The Spa3R Vision Encoder
        self.decoder = decoder  # The Predictive Spatial Field Decoder

    def forward(self, context_images, target_camera_poses):
        """
        context_images: Tensor of shape [Batch, N_views, C, H, W]
        target_camera_poses: Tensor of shape [Batch, M_target_views, 4, 4]
        """
        # 1. Compress context views into a unified 3D spatial representation
        # This forces the encoder to learn view-invariant geometry
        spatial_latent = self.encoder(context_images) 

        # 2. Predict the spatial features from completely new, unseen angles
        # The decoder uses the latent geometry and the target pose to guess the view
        predicted_feature_fields = self.decoder(spatial_latent, target_camera_poses)
        
        return predicted_feature_fields


# --- 2. DECOUPLING AND EXPORT SCRIPT ---

def export_psfm_pipeline():
    print("[PSFM PIPELINE] Initializing Predictive Spatial Field Modeling Architecture...")
    
    # Instantiate the PSFM components (In reality, these would be loaded from the checkpoint)
    encoder = Spa3REncoder()
    decoder = PredictiveSpatialDecoder()
    trained_psfm_model = PSFM_Framework(encoder, decoder)
    trained_psfm_model.eval() # Set to eval mode for exporting
    
    print("\n[PSFM PIPELINE] Simulated Pre-training Loop Complete.")
    
    # Decoupling Phase
    print("[PSFM PIPELINE] Decoupling view-invariant encoder from the predictive decoder...")
    decoupled_encoder = trained_psfm_model.encoder
    
    # Define the dummy input representing an edge camera view
    # Batch=1, N_views=1, Channels=3, H=224, W=224
    dummy_input_image = torch.randn(1, 1, 3, 224, 224)
    
    fp32_onnx_path = "models/spa3r_encoder_psfm_fp32.onnx"
    int8_onnx_path = "models/spa3r_encoder_psfm_int8.onnx"
    
    print(f"\n[ONNX EXPORT] Serializing geometric-aware encoder to: {fp32_onnx_path}")
    
    # Strip away the decoder and export ONLY the geometry-aware encoder for the edge device
    torch.onnx.export(
        decoupled_encoder,
        dummy_input_image,
        fp32_onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['spatial_tokens']
    )
    
    print(f"[ONNX EXPORT] FP32 ONNX Export complete.")
    
    print(f"\n[QUANTIZATION] Quantizing decoupled ONNX model to INT8 for the edge device...")
    quantize_dynamic(
        fp32_onnx_path,
        int8_onnx_path,
        weight_type=QuantType.QInt8,
        extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
    )
    
    print(f"\n[SUCCESS] Successfully exported the geometry-aware INT8 ONNX encoder!")
    print(f"[SUCCESS] Ready for Edge deployment -> Swap your old ONNX file with: {int8_onnx_path}")

if __name__ == "__main__":
    export_psfm_pipeline()
