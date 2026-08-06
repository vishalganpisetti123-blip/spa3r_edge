import os
import sys
import time
import numpy as np
import torch

# Add root directory to path to import spa3r
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from spa3r import Spa3R
from .base_encoder import BaseEncoder

class Spa3REncoder(BaseEncoder):
    def __init__(self, checkpoint_path=None):
        self.model = Spa3R(embed_dim=768, num_queries=256)
        
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
                
            state_dict_cleaned = {
                k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')
            }
            if not state_dict_cleaned:
                state_dict_cleaned = state_dict
                
            self.model.load_state_dict(state_dict_cleaned, strict=False)
            
        self.model.eval()

    def encode(self, image):
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).float()
        else:
            image_tensor = image
            
        if image_tensor.ndim == 3:
            # (C, H, W) -> (1, 1, C, H, W) for batch and view dimensions
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        elif image_tensor.ndim == 4:
            # (V, C, H, W) or (B, C, H, W) -> assume (1, V, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)
        image_tensor = image_tensor.to(device)
        
        inputs_dict = {"images": image_tensor}
        
        with torch.no_grad():
            if device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.amp.autocast("cuda", dtype=dtype):
                    features = self.model(inputs_dict, mode="predict")
            else:
                features = self.model(inputs_dict, mode="predict")
                
        latents_np = features.cpu().numpy()
        
        return {
            "latents": latents_np,
            "shape": latents_np.shape,
            "dtype": str(latents_np.dtype),
            "timestamp": time.time()
        }
