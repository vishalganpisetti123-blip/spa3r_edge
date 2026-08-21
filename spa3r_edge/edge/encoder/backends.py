from abc import ABC, abstractmethod
import time
import numpy as np

class BaseBackend(ABC):
    @abstractmethod
    def encode(self, image_tensor):
        pass

class PyTorchBackend(BaseBackend):
    def __init__(self, model):
        self.model = model
        
    def encode(self, image_tensor):
        import torch
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

class ONNXBackend(BaseBackend):
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
    def encode(self, image_tensor):
        # ONNX runtime expects numpy arrays
        if hasattr(image_tensor, "cpu"):
            image_np = image_tensor.cpu().numpy()
        else:
            image_np = image_tensor
            
        input_name = self.session.get_inputs()[0].name
        
        outputs = self.session.run(None, {input_name: image_np})
        latents_np = outputs[0]
        
        return {
            "latents": latents_np,
            "shape": latents_np.shape,
            "dtype": str(latents_np.dtype),
            "timestamp": time.time()
        }

class HailoBackend(BaseBackend):
    def __init__(self, hef_path):
        self.hef_path = hef_path
        # Will be implemented using HailoRT
        
    def encode(self, image_tensor):
        raise NotImplementedError("HailoRT inference is not yet implemented.")
