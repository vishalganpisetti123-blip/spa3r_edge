import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms.v2 import functional as TF
from .patches import CartesianProdPatch, OnnxRotaryPositionEmbedding2D


class Spa3RWrapper(nn.Module):
    """
    A pure PyTorch nn.Module wrapper around the Spa3R model.
    It prepares the model for ONNX export by:
    1. Baking in the dynamic positional encodings (which trace poorly)
    2. Extracting only the encoder forward pass
    """

    def __init__(self, model, height=224, width=224):
        super().__init__()
        self.model = model
        self.patch_size = model.patch_size
        self.dino_input_shape = (224, 224)

        # Precompute the pos_embed at the expected fixed resolution
        print(f"[Wrapper] Precomputing pos_embed for {height}x{width}")
        self._bake_pos_embeddings(height, width)

    def _bake_pos_embeddings(self, h, w):
        """Pre-compute interpolated positional embeddings for target resolution.

        Finds all modules with a pos_embed parameter and an
        interpolate_pos_encoding method (i.e. DINOv2/VGGT ViT backbones),
        calls the interpolation once, and replaces the parameter with
        the pre-computed result.

        After this, the tracing path in prepare_tokens_with_masks will hit
        the early-return branch (npatch == N) and skip F.interpolate entirely.
        """
        # Patch RotaryPositionEmbedding2D everywhere it appears (like the Encoder)
        from spa3r.models.layers import RotaryPositionEmbedding2D
        for name, module in self.model.named_modules():
            if hasattr(module, 'rope') and isinstance(module.rope, RotaryPositionEmbedding2D):
                module.rope = OnnxRotaryPositionEmbedding2D(frequency=module.rope.base_frequency)

        for module in self.model.modules():
            if not (hasattr(module, 'pos_embed') and
                    hasattr(module, 'interpolate_pos_encoding')):
                continue

            patch_size = module.patch_embed.patch_size
            if isinstance(patch_size, tuple):
                patch_size = patch_size[0]

            w_patches = w // patch_size
            h_patches = h // patch_size
            npatch = w_patches * h_patches

            # Build a dummy x with the right number of patches + 1 cls token
            dim = module.pos_embed.shape[-1]
            dummy_x = torch.zeros(
                1, npatch + 1, dim,
                dtype=module.pos_embed.dtype,
                device=module.pos_embed.device,
            )

            # Compute the interpolated positional embedding once
            with torch.no_grad():
                new_pos_embed = module.interpolate_pos_encoding(dummy_x, w, h)

            # Replace the parameter so tracing never calls F.interpolate
            module.pos_embed = nn.Parameter(new_pos_embed)


    def forward(self, images):
        """
        Wraps the Spa3R forward pass to extract only the encoder output (latents).
        This matches the structure of edge/encoder/spa3r_encoder.py but returns
        a flat tensor suitable for ONNX tracing.

        Args:
            images: Tensor of shape (B, V, C, H, W)
        """
        bs, v, _, h, w = images.shape

        with torch.no_grad():
            aggr_tokens, patch_start_idx = self.model.aggregator(images)
            f_ctx = aggr_tokens[-1][:, :, patch_start_idx:]

            x = self.model.projection(f_ctx)
            bs, v_ctx, n, c = x.shape

            # Position encoding for the transformer
            pos = self.model.position_getter(
                bs, h // self.patch_size, w // self.patch_size, device=x.device
            )
            pos = pos.reshape(bs, 1, n, 2)

            query = self.model.query_embed.weight[None].expand(bs, -1, -1)
            query_pos = torch.zeros_like(query[..., :2]).to(pos) - 1
            
            encoder_input = torch.cat([query, x.flatten(1, 2)], dim=1)
            encoder_pos = torch.cat(
                [query_pos, pos.expand(-1, v_ctx, -1, -1).flatten(1, 2)], dim=1
            )
            
            enc_out = self.model.encoder(encoder_input, pos=encoder_pos)
            latents = enc_out[:, :self.model.num_queries]

        return latents
