"""
Unit tests for DeepSeek-style Mixture-of-Experts (MoE) Architecture and Router.
"""

import os
import torch
import pytest
from train.moe_model import (
    MoEArgs,
    FineGrainedSwiGLUExpert,
    MoETopKRouter,
    DeepSeekMoEBlock,
    UniversalMoEForCausalLM
)


@pytest.fixture
def mini_moe_args():
    return MoEArgs(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k_experts=2,
        routed_intermediate_size=32,
        shared_intermediate_size=64,
        aux_loss_alpha=0.01
    )


def test_moe_router_top_k_and_aux_loss():
    """Verifies router top-k selection and auxiliary balancing loss."""
    router = MoETopKRouter(hidden_size=64, num_experts=8, top_k=2, aux_loss_alpha=0.01)
    tokens = torch.randn(10, 64)

    weights, indices, aux_loss = router(tokens)
    assert weights.shape == (10, 2)
    assert indices.shape == (10, 2)
    assert aux_loss is not None
    assert aux_loss.item() >= 0.0

    # Verify weights are normalized to 1.0
    sum_weights = weights.sum(dim=-1)
    assert torch.allclose(sum_weights, torch.ones_like(sum_weights), atol=1e-5)


def test_deepseek_moe_block(mini_moe_args):
    """Verifies MoE block combines shared experts and routed experts."""
    moe = DeepSeekMoEBlock(mini_moe_args)
    x = torch.randn(2, 8, mini_moe_args.hidden_size)

    out, aux_loss = moe(x)
    assert out.shape == x.shape
    assert aux_loss is not None
    assert aux_loss.item() >= 0.0


def test_universal_moe_causal_lm_forward_and_backward(mini_moe_args):
    """Verifies full MoE causal LM forward pass, loss, and gradient flow."""
    model = UniversalMoEForCausalLM(mini_moe_args)
    input_ids = torch.randint(0, 1000, (2, 16))
    labels = input_ids.clone()

    res = model(input_ids=input_ids, labels=labels)
    assert "logits" in res
    assert "loss" in res
    assert "aux_loss" in res
    assert res["loss"] is not None

    # Test backward pass
    res["loss"].backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"Parameter {name} did not receive gradients"
