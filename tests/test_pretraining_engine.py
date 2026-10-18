"""
Unit tests for Foundation Model Architecture, PretrainEngine, and Checkpointing.
"""

import os
import torch
import pytest
from train.model import UniversalLlamaForCausalLM, ModelArgs, calculate_model_flops
from train.pretrain_engine import PretrainEngine, PretrainConfig
from train.launch_distributed_pretraining import create_batch_from_shards


@pytest.fixture
def mini_model_args():
    return ModelArgs(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rope_theta=10000.0
    )


def test_model_forward_and_loss(mini_model_args):
    """Verifies forward pass and autoregressive cross entropy loss."""
    model = UniversalLlamaForCausalLM(mini_model_args)
    input_ids = torch.randint(0, 1000, (2, 32))
    labels = input_ids.clone()

    out = model(input_ids=input_ids, labels=labels)
    assert "logits" in out
    assert "loss" in out
    assert out["logits"].shape == (2, 32, 1000)
    assert out["loss"] is not None
    assert out["loss"].item() > 0.0


def test_calculate_model_flops():
    """Verifies FLOPs computation scaling."""
    flops = calculate_model_flops(
        num_params=8_000_000_000,
        num_layers=32,
        hidden_size=4096,
        seq_len=8192,
        tokens_processed=100_000
    )
    assert flops > 0
    # Expected FLOPs on order of 10^15
    assert flops > 1e14


def test_pretrain_engine_step_and_checkpoint(mini_model_args, tmp_path):
    """Tests training step execution, optimizer updates, and checkpoint serialization."""
    ckpt_dir = str(tmp_path / "checkpoints")
    cfg = PretrainConfig(
        batch_size=2,
        seq_len=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=2,
        max_steps=10,
        device="cpu",
        checkpoint_dir=ckpt_dir,
        save_interval_steps=2
    )

    engine = PretrainEngine(cfg, model_args=mini_model_args)
    input_batch = torch.randint(0, 1000, (2, 16))
    label_batch = input_batch.clone()

    metrics = engine.train_step(input_batch, label_batch)
    assert metrics["step"] == 1
    assert "loss" in metrics
    assert "perplexity" in metrics
    assert "tflops_sec" in metrics
    assert metrics["total_tokens_trained"] == 32

    # Save checkpoint
    saved_path = engine.save_checkpoint("test_ckpt.pt", metrics)
    assert os.path.exists(saved_path)

    # Load into fresh engine
    new_engine = PretrainEngine(cfg, model_args=mini_model_args)
    loaded_data = new_engine.load_checkpoint(saved_path)
    assert loaded_data["step"] == 1
    assert loaded_data["total_tokens_trained"] == 32


def test_create_batch_from_shards(tmp_path):
    """Verifies synthetic fallback token sequence formatting."""
    input_ids, labels = create_batch_from_shards(
        datasets=[],
        batch_size=3,
        seq_len=24,
        vocab_size=500
    )
    assert input_ids.shape == (3, 24)
    assert labels.shape == (3, 24)
    assert torch.equal(input_ids, labels)
