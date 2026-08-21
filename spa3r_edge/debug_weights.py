"""
Debug script to verify if strict=False loading is causing the non-determinism.
This runs the two tests suggested to prove the hypothesis.
"""
import torch
from spa3r.models.spa3r import Spa3R

def main():
    checkpoint_path = "spa3r_weights.ckpt"
    print(f"Loading weights from {checkpoint_path}...")
    
    # Test 1: Check for missing/unexpected weights
    model1 = Spa3R(embed_dim=768, num_queries=256)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state_dict_cleaned = {
        k[len('model.'):]: v for k, v in state_dict.items() if k.startswith('model.')
    }
    if not state_dict_cleaned:
        state_dict_cleaned = state_dict

    missing, unexpected = model1.load_state_dict(state_dict_cleaned, strict=False)
    
    print("\n--- Test 1: Missing / Unexpected Weights ---")
    print(f"Missing (count: {len(missing)}):")
    for m in missing[:10]:
        print(f"  {m}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
        
    print(f"\nUnexpected (count: {len(unexpected)}):")
    for u in unexpected[:10]:
        print(f"  {u}")
    if len(unexpected) > 10:
        print(f"  ... and {len(unexpected) - 10} more")

    # Test 2: Parameter equality between two freshly loaded models
    model2 = Spa3R(embed_dim=768, num_queries=256)
    model2.load_state_dict(state_dict_cleaned, strict=False)
    
    print("\n--- Test 2: Parameter Equality ---")
    mismatch_count = 0
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.equal(p1, p2):
            print(f"Mismatch found: {n1}")
            mismatch_count += 1
            
    if mismatch_count == 0:
        print("All parameters match perfectly. (Issue must be in the forward path)")
    else:
        print(f"Total mismatched parameters: {mismatch_count} (Issue is weight initialization)")

if __name__ == "__main__":
    main()
