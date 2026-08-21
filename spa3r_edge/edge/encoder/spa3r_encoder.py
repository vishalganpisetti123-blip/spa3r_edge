import os
import sys
import time
import numpy as np
import torch

from spa3r import Spa3R
from .base_encoder import BaseEncoder
from .backends import PyTorchBackend

class Spa3REncoder(BaseEncoder):
    def __init__(self, backend=None):
        self.backend = backend

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
            
        return self.backend.encode(image_tensor)
        
    @staticmethod
    def build_pytorch_backend(checkpoint_path=None):
        model = Spa3R(embed_dim=768, num_queries=256)
        if checkpoint_path is not None:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
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
                
            model.load_state_dict(state_dict_cleaned, strict=False)
            
        model.eval()
        return PyTorchBackend(model)
