import time
from .base_dataset import BaseDataset

class ImageFolderDataset(BaseDataset):
    def __init__(self):
        # Implementation to follow later
        pass

    def next(self):
        # Return a dummy sample for now to test the pipeline
        import numpy as np
        return {
            "image": np.zeros((3, 224, 224)),
            "frame_id": 0,
            "timestamp": time.time()
        }
