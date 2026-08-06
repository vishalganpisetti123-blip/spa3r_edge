from dataclasses import dataclass
from typing import Any, Dict
import numpy as np

@dataclass
class ScenePacket:
    frame_id: int
    timestamp: float
    image: np.ndarray | None
    latents: np.ndarray
    metadata: Dict[str, Any]
