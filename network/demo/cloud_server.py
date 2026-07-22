import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from scipy.ndimage import gaussian_filter
from adapter import ResidualCrossAttentionAdapter, SpatialRegressionHead

app = Flask(__name__)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[SERVER INITIALIZATION] Booting Qwen2-VL-2B Spatial Engine on: {device.upper()}")

model_id = "Qwen/Qwen2-VL-2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
vlm_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.bfloat16, 
    device_map=device
)
vlm_model.eval()

# Initialize the Adapter and load trained weights
spatial_adapter = ResidualCrossAttentionAdapter(vlm_dim=1536, spa_dim=3).to(device)
regression_head = SpatialRegressionHead(vlm_dim=1536, out_dim=4).to(device)
try:
    spatial_adapter.load_state_dict(torch.load("models/spa3r_adapter_weights.pth", map_location=device, weights_only=True))
    regression_head.load_state_dict(torch.load("models/spa3r_head_weights.pth", map_location=device, weights_only=True))
    print("[SERVER CORE] Loaded fine-tuned adapter and head weights successfully.")
except FileNotFoundError:
    print("[SERVER CORE] Warning: weights not found. Using untrained models.")
spatial_adapter.eval()
regression_head.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        edge_tokens_raw = np.array(data.get("edge_tokens"), dtype=np.float32) # Expected Shape from PSFM: [1, 196, 3]
        prompt_text = data.get("prompt", "")
        
        print(f"\n[SERVER CORE] Qwen2-VL processing spatial query: '{prompt_text}'")
        
        # Directly use the pre-flattened tokens from the edge PSFM encoder
        # If shape is [1, 196, 3], we extract the batch element -> [196, 3]
        if edge_tokens_raw.ndim == 3 and edge_tokens_raw.shape[0] == 1:
            vision_tensor = torch.tensor(edge_tokens_raw[0], dtype=torch.float32).to(device)
        else:
            vision_tensor = torch.tensor(edge_tokens_raw, dtype=torch.float32).to(device)
        
        # 1. Format prompt for Qwen2-VL
        messages = [
            {"role": "system", "content": "You are a spatial reasoning AI analyzing visual geometry tokens."},
            {"role": "user", "content": prompt_text}
        ]
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text_prompt], return_tensors="pt", padding=True).to(device)

        # 2. Extract Qwen semantic text features
        with torch.no_grad():
            outputs = vlm_model.model(**inputs, output_hidden_states=True)
            vlm_features = outputs.hidden_states[-1].to(torch.float32) # Shape: [1, Seq_len, 1536]
            
        # 3. Spa3-VLM Fusion via Cross-Attention
        spatial_latent = vision_tensor.unsqueeze(0) # Shape: [1, 196, 3]
        
        with torch.no_grad():
            fused_embeddings, _ = spatial_adapter(vlm_features, spatial_latent)
            pooled_embeddings = fused_embeddings[:, -1, :].to(torch.float32)
            pred_boxes_norm = regression_head(pooled_embeddings)[0].cpu().numpy().tolist()
            
        # Memory Cleanup for MPS!
        del outputs
        del inputs
        del vlm_features
        if device == "mps":
            torch.mps.empty_cache()

        return jsonify({
            "status": "success",
            "pred_norm": pred_boxes_norm
        }), 200

    except Exception as e:
        print(f"[SERVER CRITICAL ERROR] Pipeline failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
