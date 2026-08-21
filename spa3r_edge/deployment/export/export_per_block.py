import os
import sys

import numpy as np
import torch
import onnx
import onnxruntime as ort


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(
    0,
    os.path.join(PROJECT_ROOT, "submodules", "vggt")
)


# ============================================================
# Imports
# ============================================================

from spa3r_edge.edge.encoder.spa3r_encoder import Spa3REncoder
from spa3r_edge.deployment.export.per_block_wrappers import (
    StemWrapper,
    BlockWrapper,
    HeadWrapper,
)


# ============================================================
# Main
# ============================================================

def main():

    output_dir = os.path.join(
        PROJECT_ROOT,
        "spa3r_edge",
        "latents",
        "per_block",
    )

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Spa3R Per-Block Encoder Export")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading Spa3R model...")

    pt_backend = Spa3REncoder.build_pytorch_backend(
        checkpoint_path=None
    )

    model = pt_backend.model.eval()

    print("Model loaded successfully.")
    print(f"Number of encoder blocks: {model.encoder.n_blocks}")

    # --------------------------------------------------------
    # Dummy inputs
    # --------------------------------------------------------

    dummy_input = torch.randn(
        1,
        256,
        2048,
        dtype=torch.float32,
    )

    dummy_block_input = torch.randn(
        1,
        516,
        768,
        dtype=torch.float32,
    )

    # ========================================================
    # 1. STEM
    # ========================================================

    print("\n" + "=" * 70)
    print("1. EXPORTING STEM")
    print("=" * 70)

    stem = StemWrapper(
        model,
        height=224,
        width=224,
    ).eval()

    with torch.no_grad():
        stem_out = stem(dummy_input)

    print("Stem output:", tuple(stem_out.shape))

    assert tuple(stem_out.shape) == (
        1,
        516,
        768,
    ), f"Unexpected stem shape: {stem_out.shape}"

    stem_path = os.path.join(
        output_dir,
        "spa3r_encoder_stem.onnx",
    )

    torch.onnx.export(
        stem,
        dummy_input,
        stem_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["features_2048"],
        output_names=["encoder_input"],
        dynamic_axes={
            "features_2048": {0: "batch_size"},
            "encoder_input": {0: "batch_size"},
        },
    )

    print(
        f"Saved: {stem_path} "
        f"({os.path.getsize(stem_path) / (1024 * 1024):.2f} MB)"
    )

    # Position encoding used by every block
    final_pos = stem.final_pos

    print("final_pos shape:", tuple(final_pos.shape))

    # ========================================================
    # 2. TRANSFORMER BLOCKS
    # ========================================================

    print("\n" + "=" * 70)
    print("2. EXPORTING TRANSFORMER BLOCKS")
    print("=" * 70)

    block_paths = []
    block_wrappers = []

    for i in range(model.encoder.n_blocks):

        print(f"\n--- Block {i} ---")

        block_wrapper = BlockWrapper(
            model.encoder.blocks[i],
            final_pos,
        ).eval()

        block_wrappers.append(block_wrapper)

        # Initialize any buffers used by the ONNX patch
        with torch.no_grad():
            block_out = block_wrapper(
                dummy_block_input
            )

        print(
            "Block output:",
            tuple(block_out.shape),
        )

        assert tuple(block_out.shape) == (
            1,
            516,
            768,
        ), (
            f"Unexpected block {i} output shape: "
            f"{block_out.shape}"
        )

        block_path = os.path.join(
            output_dir,
            f"spa3r_encoder_block_{i}.onnx",
        )

        torch.onnx.export(
            block_wrapper,
            dummy_block_input,
            block_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=["hidden_states"],
            output_names=["hidden_states_out"],
            dynamic_axes={
                "hidden_states": {0: "batch_size"},
                "hidden_states_out": {0: "batch_size"},
            },
        )

        print(
            f"Saved: {block_path} "
            f"({os.path.getsize(block_path) / (1024 * 1024):.2f} MB)"
        )

        block_paths.append(block_path)

    # ========================================================
    # 3. HEAD
    # ========================================================

    print("\n" + "=" * 70)
    print("3. EXPORTING HEAD")
    print("=" * 70)

    head_wrapper = HeadWrapper(
        model.encoder,
        model.num_queries,
    ).eval()

    with torch.no_grad():
        head_out = head_wrapper(
            dummy_block_input
        )

    print(
        "Head output:",
        tuple(head_out.shape),
    )

    assert tuple(head_out.shape) == (
        1,
        256,
        768,
    ), f"Unexpected head shape: {head_out.shape}"

    head_path = os.path.join(
        output_dir,
        "spa3r_encoder_head.onnx",
    )

    torch.onnx.export(
        head_wrapper,
        dummy_block_input,
        head_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["hidden_states"],
        output_names=["latents"],
        dynamic_axes={
            "hidden_states": {0: "batch_size"},
            "latents": {0: "batch_size"},
        },
    )

    print(
        f"Saved: {head_path} "
        f"({os.path.getsize(head_path) / (1024 * 1024):.2f} MB)"
    )

    # ========================================================
    # 4. ONNX CHECK
    # ========================================================

    print("\n" + "=" * 70)
    print("4. CHECKING ONNX MODELS")
    print("=" * 70)

    all_paths = [
        stem_path,
        *block_paths,
        head_path,
    ]

    for path in all_paths:

        print(
            f"Checking {os.path.basename(path)}..."
        )

        onnx_model = onnx.load(path)

        onnx.checker.check_model(
            onnx_model
        )

        print("  OK")

    # ========================================================
    # 5. PYTORCH CHAIN
    # ========================================================

    print("\n" + "=" * 70)
    print("5. PYTORCH CHAIN")
    print("=" * 70)

    torch.manual_seed(42)

    test_input = torch.randn(
        1,
        256,
        2048,
        dtype=torch.float32,
    )

    with torch.no_grad():

        pt_x = stem(test_input)

        for i, block_wrapper in enumerate(
            block_wrappers
        ):

            pt_x = block_wrapper(pt_x)

            print(
                f"PyTorch block {i}:",
                tuple(pt_x.shape),
            )

        pt_latents = head_wrapper(
            pt_x
        )

    pt_latents_np = (
        pt_latents
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "PyTorch final output:",
        pt_latents_np.shape,
    )

    # ========================================================
    # 6. ONNX RUNTIME CHAIN
    # ========================================================

    print("\n" + "=" * 70)
    print("6. ONNX RUNTIME CHAIN")
    print("=" * 70)

    providers = [
        "CPUExecutionProvider"
    ]

    sess_stem = ort.InferenceSession(
        stem_path,
        providers=providers,
    )

    sess_blocks = [
        ort.InferenceSession(
            path,
            providers=providers,
        )
        for path in block_paths
    ]

    sess_head = ort.InferenceSession(
        head_path,
        providers=providers,
    )

    # --------------------------------------------------------
    # Stem
    # --------------------------------------------------------

    x = sess_stem.run(
        None,
        {
            sess_stem.get_inputs()[0].name:
            test_input.numpy()
        },
    )[0]

    print(
        "ONNX stem:",
        x.shape,
    )

    # --------------------------------------------------------
    # Blocks
    # --------------------------------------------------------

    for i, sess in enumerate(
        sess_blocks
    ):

        x = sess.run(
            None,
            {
                sess.get_inputs()[0].name:
                x
            },
        )[0]

        print(
            f"ONNX block {i}:",
            x.shape,
        )

    # --------------------------------------------------------
    # Head
    # --------------------------------------------------------

    ort_latents = sess_head.run(
        None,
        {
            sess_head.get_inputs()[0].name:
            x
        },
    )[0]

    print(
        "ONNX final output:",
        ort_latents.shape,
    )

    # ========================================================
    # 7. NUMERICAL PARITY
    # ========================================================

    print("\n" + "=" * 70)
    print("7. NUMERICAL PARITY")
    print("=" * 70)

    max_diff = np.max(
        np.abs(
            pt_latents_np -
            ort_latents
        )
    )

    mean_diff = np.mean(
        np.abs(
            pt_latents_np -
            ort_latents
        )
    )

    print(
        f"Max absolute difference : "
        f"{max_diff:.8f}"
    )

    print(
        f"Mean absolute difference: "
        f"{mean_diff:.8f}"
    )

    # Reasonable tolerance for FP32 ONNX export
    if max_diff < 1e-4:
        print("\n✓ PARITY CHECK PASSED")
    else:
        print(
            "\n✗ PARITY CHECK FAILED"
        )

        raise RuntimeError(
            f"ONNX parity failure: "
            f"max_diff={max_diff}"
        )

    print("\n" + "=" * 70)
    print("ALL EXPORT TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
