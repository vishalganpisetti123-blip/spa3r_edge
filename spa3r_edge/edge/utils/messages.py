from dataclasses import dataclass
from typing import Any, Dict
from pathlib import Path
import numpy as np

@dataclass
class ScenePacket:
    frame_id: int
    timestamp: float
    image_path: Path | None
    image: np.ndarray | None
    latents: np.ndarray | None
    metadata: Dict[str, Any]
