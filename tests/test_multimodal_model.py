"""
Unit tests for Multimodal Vision-Language Model Architecture and Patch Adapter.
"""

import torch
import pytest
from train.multimodal_model import (
    MultimodalModelArgs,
    VisionPatchEncoder,
    VisionLanguageAdapter,
    UniversalMultimodalForCausalLM
)


@pytest.fixture
def mini_multimodal_args():
    return MultimodalModelArgs(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        image_size=56,
        patch_size=14,
        vision_hidden_size=32,
        num_image_tokens=4,
        image_token_id=999
    )


def test_vision_patch_encoder_and_adapter(mini_multimodal_args):
    """Verifies patch extraction and vision-to-language projection."""
    encoder = VisionPatchEncoder(
        image_size=mini_multimodal_args.image_size,
        patch_size=mini_multimodal_args.patch_size,
        vision_dim=mini_multimodal_args.vision_hidden_size
    )
    adapter = VisionLanguageAdapter(
        vision_dim=mini_multimodal_args.vision_hidden_size,
        hidden_dim=mini_multimodal_args.hidden_size,
        num_image_tokens=mini_multimodal_args.num_image_tokens
    )

    # 1 image of 3x56x56
    pixel_values = torch.randn(1, 3, 56, 56)
    patches = encoder(pixel_values)
    assert patches.shape == (1, 16, 32)  # (56/14)^2 = 16 patches

    tokens = adapter(patches)
    assert tokens.shape == (1, 4, 64)  # 4 pooled visual tokens in hidden_dim=64


def test_universal_multimodal_causal_lm(mini_multimodal_args):
    """Verifies multimodal causal LM substitutes image token slots and trains with cross-entropy loss."""
    model = UniversalMultimodalForCausalLM(mini_multimodal_args)

    # Sequence with 4 image token placeholders (999)
    input_ids = torch.tensor([
        [10, 20, 999, 999, 999, 999, 30, 40],
        [50, 60, 70, 80, 90, 100, 110, 120]
    ])
    pixel_values = torch.randn(1, 3, 56, 56)
    labels = input_ids.clone()

    res = model(input_ids=input_ids, pixel_values=pixel_values, labels=labels)
    assert "logits" in res
    assert "loss" in res
    assert res["loss"] is not None
    assert res["logits"].shape == (2, 8, 1000)

    # Verify backprop works through both language and vision adapters
    res["loss"].backward()
    assert model.adapter.linear_1.weight.grad is not None
    assert model.vision_encoder.proj.weight.grad is not None
