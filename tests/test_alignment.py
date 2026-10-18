"""
Unit tests for Direct Preference Optimization (DPO) and KTO Alignment.
"""

import pytest
import torch
import torch.nn as nn

from train.model import UniversalLlamaForCausalLM, ModelArgs
from alignment.preference_dataset import (
    PreferencePair,
    SWEPreferenceDataExtractor,
    PreferenceBatchCollator
)
from alignment.dpo_engine import DPOEngine
from alignment.dpo_trainer import DPOTrainer


@pytest.fixture
def mini_model_args():
    return ModelArgs(
        vocab_size=500,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128
    )


def test_swe_preference_data_extractor():
    sample_trajectories = [
        {
            "issue_id": "django__django-11001",
            "task_instructions": "Fix query ordering regression",
            "final_patch": "diff --git a/sql/query.py\n+ return order_by.asc()",
            "sandbox_passed": True
        },
        {
            "issue_id": "django__django-11001",
            "task_instructions": "Fix query ordering regression",
            "final_patch": "diff --git a/sql/query.py\n+ raise SyntaxError()",
            "sandbox_passed": False
        },
        {
            "issue_id": "sympy__sympy-13000",
            "task_instructions": "Fix fraction simplification",
            "final_patch": "diff --git a/core/numbers.py\n+ return gcd(p, q)",
            "sandbox_passed": True
        }
    ]

    pairs, kto_examples = SWEPreferenceDataExtractor.extract_from_trajectories(sample_trajectories)
    assert len(pairs) >= 2
    assert len(kto_examples) == 3

    p0 = pairs[0]
    assert "return order_by.asc()" in p0.chosen
    assert "SyntaxError" in p0.rejected


def test_preference_batch_collator():
    class MockTokenizer:
        def encode(self, text: str):
            return [ord(c) % 200 + 10 for c in text[:20]]

    collator = PreferenceBatchCollator(tokenizer=MockTokenizer(), max_seq_len=64)
    pairs = [
        PreferencePair(
            prompt="def solve(a):",
            chosen="return a * 2",
            rejected="return None"
        )
    ]

    batch = collator.collate_dpo_batch(pairs)
    assert "chosen_input_ids" in batch
    assert "chosen_labels" in batch
    assert "rejected_input_ids" in batch
    assert "rejected_labels" in batch

    # Verify prompt tokens are masked with -100 in labels
    chosen_labels = batch["chosen_labels"][0]
    assert (chosen_labels == -100).any()
    # Verify completion tokens are not -100
    assert (chosen_labels != -100).any()


def test_dpo_loss_computation_and_gradients():
    policy_chosen = torch.tensor([-2.0, -1.5], requires_grad=True)
    policy_rejected = torch.tensor([-4.0, -5.0], requires_grad=True)
    ref_chosen = torch.tensor([-3.0, -3.0])
    ref_rejected = torch.tensor([-3.0, -3.0])

    loss, metrics = DPOEngine.dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )

    assert loss.item() > 0
    assert metrics["preference_accuracy"] == 1.0
    assert metrics["reward_margin"] > 0

    loss.backward()
    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None


def test_kto_loss_computation():
    policy_logps = torch.tensor([-1.0, -5.0])
    ref_logps = torch.tensor([-2.0, -2.0])
    is_chosen = torch.tensor([1, 0])

    loss, metrics = DPOEngine.kto_loss(policy_logps, ref_logps, is_chosen, beta=0.1)
    assert loss.item() > 0
    assert "kto_loss" in metrics


def test_dpo_trainer_step(mini_model_args):
    policy_model = UniversalLlamaForCausalLM(mini_model_args)
    trainer = DPOTrainer(policy_model, lr=1e-4, beta=0.1, device="cpu")

    batch = {
        "chosen_input_ids": torch.randint(10, 400, (2, 24)),
        "chosen_labels": torch.cat([
            torch.full((2, 10), -100, dtype=torch.long),
            torch.randint(10, 400, (2, 14))
        ], dim=-1),
        "rejected_input_ids": torch.randint(10, 400, (2, 24)),
        "rejected_labels": torch.cat([
            torch.full((2, 10), -100, dtype=torch.long),
            torch.randint(10, 400, (2, 14))
        ], dim=-1)
    }

    metrics = trainer.train_step(batch)
    assert "dpo_loss" in metrics
    assert "grad_norm" in metrics
    assert "preference_accuracy" in metrics
