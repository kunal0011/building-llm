"""
Unit and Integration Tests for Multi-Stage Context Annealing and Distributed RL Experience Buffer.
"""

import os
import sys
import numpy as np
import pytest

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.long_context.context_annealing_scheduler import (
    ContextAnnealingScheduler,
    AnnealingStage,
    RoPEScalingCalculator,
    NeedleInAHaystackSynthesizer
)
from pipelines.rl.experience_buffer import (
    AgentStepCheckpoint,
    TrajectoryExperience,
    GRPORolloutGroup,
    DistributedExperienceBuffer
)
from pipelines.rl.grpo_reward_engine import GRPORewardEngine


# =====================================================================
# TRACK 1 TESTS: MULTI-STAGE CONTEXT ANNEALING & ROPE SCALING
# =====================================================================

def test_rope_scaling_calculator():
    """Validates inverse frequencies, NTK-aware scaling, and YaRN temperature calculation."""
    calc = RoPEScalingCalculator(head_dim=128, original_max_position=8192, base_theta=500_000.0)

    # Base frequencies
    base_freqs = calc.compute_base_frequencies()
    assert len(base_freqs) == 64
    assert base_freqs[0] == 1.0
    assert base_freqs[-1] < 1e-4

    # NTK-aware scaling for 32k, 128k, 1M
    ntk_32k = calc.compute_ntk_aware_theta(32_768)
    ntk_128k = calc.compute_ntk_aware_theta(131_072)
    ntk_1m = calc.compute_ntk_aware_theta(1_048_576)

    assert ntk_32k > 500_000.0
    assert ntk_128k > ntk_32k
    assert ntk_1m > ntk_128k

    # YaRN frequency interpolation
    yarn_freqs, attn_temp = calc.compute_yarn_frequencies(32_768)
    assert len(yarn_freqs) == 64
    assert attn_temp > 1.0  # Attention scale factor increases with context


def test_needle_in_a_haystack_synthesis():
    """Validates Needle-In-A-Haystack benchmark generation at calibrated depths."""
    synth = NeedleInAHaystackSynthesizer(random_seed=42)
    background = [
        "Paragraph one describes transformer self-attention computations.",
        "Paragraph two discusses Megatron-LM tensor parallel model distribution.",
        "Paragraph three covers Rotary Position Embeddings and YaRN scaling.",
        "Paragraph four elaborates on reinforcement learning with verifiable rewards."
    ]

    haystack = synth.generate_haystack(target_token_count=500, background_corpus=background)
    assert len(haystack) > 100

    # Inject needle at 50% depth
    sample = synth.inject_needle(
        haystack=haystack,
        needle_key="compiler_opt_flag",
        needle_value="OPT_FLAG_773",
        depth_percent=0.50
    )

    assert "OPT_FLAG_773" in sample["prompt"]
    assert sample["ground_truth"] == "OPT_FLAG_773"
    assert sample["depth_percent"] == 0.50
    assert sample["query"] == "What is the secret verification code for compiler_opt_flag?"


def test_context_annealing_curriculum_schedule():
    """Validates the 4-stage curriculum scheduler across 8k, 32k, 128k, and 1M+ tokens."""
    scheduler = ContextAnnealingScheduler()
    report = scheduler.generate_curriculum_report()

    stages = report["curriculum"]
    assert len(stages) == 4
    assert stages[0]["stage"] == "8k_base"
    assert stages[1]["stage"] == "32k_extended"
    assert stages[2]["stage"] == "128k_long_context"
    assert stages[3]["stage"] == "1m_extreme_horizon"

    packer_1m = scheduler.create_batch_packer(AnnealingStage.STAGE_1M)
    assert packer_1m.max_seq_len == 1_048_576


# =====================================================================
# TRACK 2 TESTS: RL EXPERIENCE BUFFER & GRPO ADVANTAGES
# =====================================================================

def test_distributed_experience_buffer_grpo_advantages():
    """Validates GRPO advantage normalization across candidate rollout groups."""
    buffer = DistributedExperienceBuffer(max_capacity_groups=100, gamma=0.95, prm_lambda=0.5)

    # Create candidate rollouts for a prompt: 2 correct, 2 flawed
    steps_good = [
        AgentStepCheckpoint(step_idx=0, thought="Decompose problem.", action_type="plan", action_payload="plan", observation="OK", prm_reward=1.0),
        AgentStepCheckpoint(step_idx=1, thought="Execute test.", action_type="code", action_payload="code", observation="Passed", prm_reward=1.0)
    ]
    steps_bad = [
        AgentStepCheckpoint(step_idx=0, thought="Rush answer.", action_type="plan", action_payload="plan", observation="OK", prm_reward=0.0),
        AgentStepCheckpoint(step_idx=1, thought="Wrong logic.", action_type="code", action_payload="code", observation="Failed", prm_reward=-1.0)
    ]

    r1 = TrajectoryExperience(trajectory_id="t1", prompt="p1", steps=steps_good, final_solution="<thought>ok</thought>", outcome_correctness=1.0, format_reward=1.0, total_reward=1.2)
    r2 = TrajectoryExperience(trajectory_id="t2", prompt="p1", steps=steps_good, final_solution="<thought>ok</thought>", outcome_correctness=1.0, format_reward=0.8, total_reward=1.1)
    r3 = TrajectoryExperience(trajectory_id="t3", prompt="p1", steps=steps_bad, final_solution="broken", outcome_correctness=0.0, format_reward=0.0, total_reward=0.0)
    r4 = TrajectoryExperience(trajectory_id="t4", prompt="p1", steps=steps_bad, final_solution="broken", outcome_correctness=0.0, format_reward=0.0, total_reward=0.0)

    group = buffer.add_rollout_group("prompt_001", "Implement quicksort", [r1, r2, r3, r4])

    assert group.group_size == 4
    # The successful rollouts must have positive advantages
    assert r1.advantage > 0.0
    assert r2.advantage > 0.0
    # The failed rollouts must have negative advantages
    assert r3.advantage < 0.0
    assert r4.advantage < 0.0

    # Step level credit assignment check
    assert len(r1.discounted_step_returns) == 2
    assert r1.discounted_step_returns[1] > r3.discounted_step_returns[1]

    # Mini-batch sampling check
    batches = list(buffer.sample_minibatches(batch_size=2))
    assert len(batches) == 2
    assert len(batches[0]) == 2

    # Summary check
    summary = buffer.export_summary()
    assert summary["total_trajectories"] == 4
    assert summary["pass_rate_accuracy"] == "50.0%"
