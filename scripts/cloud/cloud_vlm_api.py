import os
import time
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
import sys
import requests

# Import the local adapter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "3d")))
from train_3d_adapter import Spa3RAdapter, encode_prompt, ALLOWED_CLASSES

# 1. Initialize Open-Source OpenAI SDK client configured for Qwen API
from openai import OpenAI
API_KEY = os.getenv("QWEN_API_KEY", "NaJMdzYnOT2X3YEamwvYEtPEjhv0pZEf")
client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1" # Alibaba Cloud Model Studio Endpoint
)

app = FastAPI(title="Spa3R-VLM Hybrid API Receiver")

# 2. Local Spa3R Adapter Module wrapper
class LocalSpa3RAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Loading Spa3RAdapter on {self.device}...")
        self.adapter = Spa3RAdapter().to(self.device)
        weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models/spa3r_adapter_3d_weights.pth"))
        if os.path.exists(weights_path):
            self.adapter.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print(f"WARNING: No weights found at {weights_path}")
        self.adapter.eval()
        
    def extract_object_class(self, prompt: str):
        # Stage A: Fast intent parsing using the prompt
        system_msg = f"Extract the core object name and its spatial modifier (left, right, middle) from the user's prompt. You MUST pick the object ONLY from this exact list: {ALLOWED_CLASSES}. Reply ONLY with the chosen object name and modifier (e.g. 'cabinet on the right'). If you can't find a match, just say 'object in room'."
        try:
            response = client.chat.completions.create(
                model="qwen-plus", # fast language model
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip().lower()
        except Exception as e:
            print(f"Object extraction failed: {e}")
            return prompt # Fallback

    def predict_bounding_box(self, spatial_tensor: torch.Tensor, reasoned_prompt: str):
        # Predicts continuous 3D coordinates [x, y, z, l, w, h] from spatial latents Z_spa
        prompt_vec = encode_prompt(reasoned_prompt)
        prompt_tensor = torch.tensor(prompt_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        spatial_tensor = spatial_tensor.unsqueeze(0).to(self.device) # [1, 196, 6]
        
        with torch.no_grad():
            pred_boxes, _ = self.adapter(spatial_tensor, prompt_tensor)
            raw_box = pred_boxes[0].cpu().numpy().tolist()
            return [round(val, 2) for val in raw_box]

spa3r_adapter = LocalSpa3RAdapter()

@app.post("/predict_3d_spatial/")
async def predict_3d_spatial(
    prompt: str = Form(...),
    dtype: str = Form("float32"),
    shape: str = Form(...),  # e.g., "196,6"
    tensor_bytes: UploadFile = File(...)
):
    start_time = time.time()
    try:
        # Step A: Parse binary payload into PyTorch tensor
        content = await tensor_bytes.read()
        parsed_shape = [int(dim) for dim in shape.split(",")]
        np_dtype = np.float32 if dtype == "float32" else np.float16
        
        array = np.frombuffer(content, dtype=np_dtype).copy().reshape(parsed_shape)
        spatial_tensor = torch.from_numpy(array)
        
        # Stage B: Extract Object Class + Pass spatial tokens through local Spa3R adapter
        reasoned_prompt = spa3r_adapter.extract_object_class(prompt)
        
        adapter_start = time.time()
        estimated_box = spa3r_adapter.predict_bounding_box(spatial_tensor, reasoned_prompt)
        t_adapter_ms = round((time.time() - adapter_start) * 1000, 2)
        
        # Step C: Inject adapter spatial coordinates into the Qwen text prompt
        spatial_enhanced_prompt = (
            f"You are a 3D spatial visual assistant. "
            f"The edge spatial field encoder has detected candidate object bounds for '{reasoned_prompt}' at coordinate space "
            f"[x={estimated_box[0]}, y={estimated_box[1]}, z={estimated_box[2]}, "
            f"length={estimated_box[3]}m, width={estimated_box[4]}m, height={estimated_box[5]}m].\n"
            f"User Question: '{prompt}'.\n"
            f"Synthesize the spatial coordinates and answer the question accurately, referencing the object's physical size and position."
        )

        # Step D: Call Qwen API with spatially conditioned prompt
        qwen_start = time.time()
        try:
            response = client.chat.completions.create(
                model="qwen-plus",  # Or preferred Qwen endpoint
                messages=[
                    {"role": "system", "content": "You are a spatially aware AI agent."},
                    {"role": "user", "content": spatial_enhanced_prompt}
                ],
                temperature=0.2,
                max_tokens=150
            )
            qwen_reasoning = response.choices[0].message.content
        except Exception as e:
            print(f"Qwen synthesis failed: {e}")
            qwen_reasoning = "Qwen API failed (check billing/key). Falling back to raw coordinates: " + str(estimated_box)
            
        t_qwen_ms = round((time.time() - qwen_start) * 1000, 2)
        t_server_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "status": "success",
            "original_prompt": prompt,
            "parsed_intent": reasoned_prompt,
            "spa3r_adapter_box": estimated_box,
            "qwen_spatial_response": qwen_reasoning,
            "t_adapter_ms": t_adapter_ms,
            "t_qwen_ms": t_qwen_ms,
            "t_server_ms": t_server_ms
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integration Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
