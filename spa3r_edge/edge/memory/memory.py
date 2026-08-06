class SceneMemory:
    def __init__(self):
        self.frames = []

    def add(self, frame, latent, metadata):
        self.frames.append({
            "frame": frame,
            "latent": latent,
            "metadata": metadata
        })

    def recent(self, n):
        return self.frames[-n:]
