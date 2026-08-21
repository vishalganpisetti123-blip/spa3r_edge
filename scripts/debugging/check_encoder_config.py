import sys
sys.path.insert(0, "submodules/vggt")
sys.path.insert(0, ".")
from spa3r.models.spa3r import build_spa3r

model = build_spa3r()

enc = model.encoder
print(f"Encoder depth: {enc.n_blocks}")
print(f"Encoder num_heads: {enc.num_heads}")
print(f"Num register tokens: {enc.num_register_tokens}")
print(f"Has register tokens: {enc.register_tokens is not None}")
print(f"Norm: {enc.norm}")
print(f"Number of blocks: {len(enc.blocks)}")

# Check projection
print(f"\nProjection: {model.projection}")
print(f"Query embed: {model.query_embed}")
print(f"Num queries: {model.num_queries}")
print(f"Patch size: {model.patch_size}")

# Check a single block
blk = enc.blocks[0]
print(f"\nBlock 0:")
print(f"  Attention head_dim: {blk.attn.head_dim}")
print(f"  Attention num_heads: {blk.attn.num_heads}")
print(f"  Attention scale: {blk.attn.scale}")
print(f"  Has RoPE: {blk.attn.rope is not None}")
print(f"  Has qk_norm: {not isinstance(blk.attn.q_norm, type(blk.attn.q_norm).__mro__[-2]())}")
print(f"  q_norm type: {type(blk.attn.q_norm)}")
print(f"  norm type: {type(blk.attn.norm)}")
print(f"  ls1 type: {type(blk.ls1)}")
print(f"  ls2 type: {type(blk.ls2)}")
print(f"  drop_path1 type: {type(blk.drop_path1)}")

