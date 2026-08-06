import os
import sys

# Make sure we can import spa3r from the root level
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from edge.input.image_folder import ImageFolderDataset
from edge.encoder.spa3r_encoder import Spa3REncoder
from edge.memory.memory import SceneMemory
from edge.utils.messages import ScenePacket
import numpy as np

def main():
    dataset = ImageFolderDataset()
    # checkpoint_path parameter can be adjusted. Assuming None for now since we don't have the weights
    encoder = Spa3REncoder(checkpoint_path="../spa3r_weights.ckpt")
    memory = SceneMemory()

    # Ensure latents directory exists
    os.makedirs("latents", exist_ok=True)

    frame_count = 0
    while True:
        frame_count += 1
        sample = dataset.next()
        
        # Create ScenePacket from dataset output
        packet = ScenePacket(
            frame_id=sample["frame_id"],
            timestamp=sample["timestamp"],
            image_path=None,
            image=sample["image"],
            latents=None,
            metadata={}
        )
        
        print(f"Frame {frame_count}")
        print("↓")
        print("Encoder")
        
        # Process through encoder
        encoder_output = encoder.encode(packet.image)
        packet.latents = encoder_output["latents"]
        
        print("↓")
        print("Latent")
        print(f"  Shape: {encoder_output['shape']}")
        print(f"  Dtype: {encoder_output['dtype']}")
        print(f"  Min:   {np.min(packet.latents)}")
        print(f"  Max:   {np.max(packet.latents)}")
        
        # Save latents
        save_path = f"latents/frame_{frame_count:06d}.npy"
        np.save(save_path, packet.latents)
        
        print("↓")
        print("Saved")
        print(f"({save_path})\n")
        
        # Add to memory
        memory.add(
            frame=packet.image,
            latent=packet.latents,
            metadata=packet.metadata
        )
        
        # For Sprint 2 deliverable, produce 3 frames then exit
        if frame_count >= 3:
            break

if __name__ == "__main__":
    main()
