"""
Distributed Post-Training / RL Experience Buffer & Trajectory Checkpoint Storage.
Manages multi-turn agent rollout trajectories, step-level PRM annotations,
deterministic outcome verification, and Group Relative Policy Optimization (GRPO) advantage calculation.
"""

import math
import json
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Iterator
import numpy as np

from pipelines.rl.grpo_reward_engine import GRPORewardEngine


@dataclass
class AgentStepCheckpoint:
    """Represents a single step in a multi-turn agent interaction trajectory."""
    step_idx: int
    thought: str
    action_type: str  # e.g., "code_edit", "shell_cmd", "sympy_derivation", "tool_call"
    action_payload: str
    observation: str  # tool output or compiler/interpreter feedback
    prm_reward: float = 0.0  # Dense step reward: +1.0 (productive), 0.0 (neutral), -1.0 (error/loop)
    state_hash: str = ""

    def __post_init__(self):
        if not self.state_hash:
            h = hashlib.sha256(f"{self.step_idx}:{self.action_payload}:{self.observation}".encode()).hexdigest()[:16]
            self.state_hash = h


@dataclass
class TrajectoryExperience:
    """Represents a full multi-turn rollout trajectory for an agent or reasoning task."""
    trajectory_id: str
    prompt: str
    steps: List[AgentStepCheckpoint]
    final_solution: str
    outcome_correctness: float  # 1.0 (tests passed / math verified) or 0.0
    format_reward: float = 0.0
    efficiency_penalty: float = 0.0
    total_reward: float = 0.0
    advantage: float = 0.0
    discounted_step_returns: List[float] = field(default_factory=list)
    worker_node: str = "local"


@dataclass
class GRPORolloutGroup:
    """Represents a group of G candidate rollouts for a single prompt in GRPO."""
    prompt_id: str
    prompt: str
    group_size: int
    rollouts: List[TrajectoryExperience]
    group_mean_reward: float = 0.0
    group_std_reward: float = 0.0


class DistributedExperienceBuffer:
    """
    Capacity-managed Experience Buffer for GRPO and Step-Supervised RL training.
    Supports:
      1. Aggregation of rollout groups across distributed Ray workers
      2. Step-level credit assignment (PRM + discounted outcome)
      3. Mini-batch generation with group-normalized advantages
      4. Serialization to JSONL and binary formats
    """

    def __init__(
        self,
        max_capacity_groups: int = 10_000,
        gamma: float = 0.95,
        prm_lambda: float = 0.5,
        grpo_engine: Optional[GRPORewardEngine] = None
    ):
        self.max_capacity = max_capacity_groups
        self.gamma = gamma
        self.prm_lambda = prm_lambda
        self.grpo_engine = grpo_engine or GRPORewardEngine()
        self.groups: List[GRPORolloutGroup] = []

    def add_rollout_group(
        self,
        prompt_id: str,
        prompt: str,
        rollouts: List[TrajectoryExperience]
    ) -> GRPORolloutGroup:
        """
        Calculates group advantages across the G candidate rollouts and stores in buffer.
        """
        # Prepare dicts for GRPO advantage computation
        rollout_dicts = [
            {
                "idx": i,
                "total_reward": r.total_reward,
                "correctness_reward": r.outcome_correctness,
                "format_reward": r.format_reward,
                "output_text": r.final_solution
            }
            for i, r in enumerate(rollouts)
        ]

        evaluated = self.grpo_engine.compute_group_advantages(rollout_dicts)
        mean_r = evaluated[0]["group_mean_reward"] if evaluated else 0.0
        std_r = evaluated[0]["group_std_reward"] if evaluated else 0.0

        for i, ev in enumerate(evaluated):
            rollouts[i].advantage = ev["advantage"]
            # Compute dense step-level credit assignment
            self._compute_step_level_credits(rollouts[i])

        group = GRPORolloutGroup(
            prompt_id=prompt_id,
            prompt=prompt,
            group_size=len(rollouts),
            rollouts=rollouts,
            group_mean_reward=mean_r,
            group_std_reward=std_r
        )

        if len(self.groups) >= self.max_capacity:
            self.groups.pop(0)  # FIFO eviction

        self.groups.append(group)
        return group

    def _compute_step_level_credits(self, trajectory: TrajectoryExperience) -> None:
        """
        Blends final outcome correctness with dense PRM step supervision:
          R_step(t) = gamma^(T-t) * R_outcome + lambda * R_prm(t)
        """
        T = len(trajectory.steps)
        if T == 0:
            trajectory.discounted_step_returns = [trajectory.outcome_correctness]
            return

        step_returns = []
        for t, step in enumerate(trajectory.steps):
            discount = self.gamma ** (T - 1 - t)
            step_return = (discount * trajectory.outcome_correctness) + (self.prm_lambda * step.prm_reward)
            step_returns.append(round(step_return, 4))

        trajectory.discounted_step_returns = step_returns

    def get_total_trajectories(self) -> int:
        return sum(len(g.rollouts) for g in self.groups)

    def sample_minibatches(
        self,
        batch_size: int = 16,
        shuffle: bool = True
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Yields flat training samples with group-normalized advantages and step credits.
        """
        all_samples: List[Dict[str, Any]] = []

        for grp in self.groups:
            for r in grp.rollouts:
                all_samples.append({
                    "prompt": grp.prompt,
                    "solution": r.final_solution,
                    "advantage": r.advantage,
                    "total_reward": r.total_reward,
                    "outcome_correctness": r.outcome_correctness,
                    "step_count": len(r.steps),
                    "step_credits": r.discounted_step_returns,
                    "worker_node": r.worker_node
                })

        if shuffle:
            np.random.default_rng(42).shuffle(all_samples)

        for i in range(0, len(all_samples), batch_size):
            yield all_samples[i : i + batch_size]

    def export_summary(self) -> Dict[str, Any]:
        """Provides analytical telemetry on buffer contents and advantage distributions."""
        all_advantages = [r.advantage for g in self.groups for r in g.rollouts]
        all_rewards = [r.total_reward for g in self.groups for r in g.rollouts]
        all_correctness = [r.outcome_correctness for g in self.groups for r in g.rollouts]

        return {
            "total_groups": len(self.groups),
            "total_trajectories": len(all_advantages),
            "mean_reward": round(float(np.mean(all_rewards)), 4) if all_rewards else 0.0,
            "pass_rate_accuracy": f"{np.mean(all_correctness) * 100:.1f}%" if all_correctness else "0.0%",
            "advantage_std": round(float(np.std(all_advantages)), 4) if all_advantages else 0.0,
            "max_advantage": round(float(np.max(all_advantages)), 4) if all_advantages else 0.0,
            "min_advantage": round(float(np.min(all_advantages)), 4) if all_advantages else 0.0
        }
