from edge.input.image_folder import ImageFolderDataset
from edge.encoder.spa3r_encoder import Spa3REncoder
from edge.memory.memory import SceneMemory
from edge.utils.messages import ScenePacket

def main():
    dataset = ImageFolderDataset()
    encoder = Spa3REncoder()
    memory = SceneMemory()

    while True:
        sample = dataset.next()
        
        # Create ScenePacket from dataset output
        packet = ScenePacket(
            frame_id=sample["frame_id"],
            timestamp=sample["timestamp"],
            image=sample["image"],
            latents=None,
            metadata={}
        )
        
        # Process through encoder
        packet.latents = encoder.encode(packet.image)
        
        # Add to memory
        memory.add(
            frame=packet.image,
            latent=packet.latents,
            metadata=packet.metadata
        )
        
        # We can add routing/decision logic later
        
        break # Breaking just so it doesn't spin infinitely during our initial testing

if __name__ == "__main__":
    main()
