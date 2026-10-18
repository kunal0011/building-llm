"""
Group Relative Policy Optimization (GRPO) Multi-Objective Reward & Advantage Engine.
Calculates verifiable outcome rewards, format compliance rewards, length penalties,
and computes group-normalized advantages without requiring a Critic / Value network.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class GRPORewardEngine:
    """
    GRPO Reward & Advantage Engine (inspired by DeepSeek-R1 / DeepSeekMath).
    For each prompt, evaluates a group of G candidate rollouts:
      1. Verifiable correctness reward (code execution / symbolic math)
      2. Format integrity reward (<thought>...</thought> structure)
      3. Repetition / length efficiency penalty
    Computes group-normalized advantage:
      A_i = (R_i - mean(R)) / (std(R) + epsilon)
    """

    def __init__(
        self,
        weight_correctness: float = 1.0,
        weight_format: float = 0.20,
        weight_efficiency: float = 0.10,
        epsilon: float = 1e-4
    ):
        self.w_corr = weight_correctness
        self.w_fmt = weight_format
        self.w_eff = weight_efficiency
        self.epsilon = epsilon

    def evaluate_format_compliance(self, text: str) -> float:
        """
        Rewards proper structural framing:
        Checks for matching <thought> and </thought> tags.
        """
        score = 0.0
        has_thought_open = "<thought>" in text
        has_thought_close = "</thought>" in text

        if has_thought_open and has_thought_close:
            # Positive score for closed thought block
            score += 1.0
        elif has_thought_open != has_thought_close:
            # Penalty for unclosed syntax
            score -= 1.0

        # Check for tool call tags balance if tool calling is invoked
        if "<tool_call>" in text:
            if "</tool_call>" in text:
                score += 0.5
            else:
                score -= 0.5

        return max(0.0, score / 1.5)

    def evaluate_efficiency_penalty(self, token_count: int, max_expected_tokens: int = 2048) -> float:
        """Penalizes over-extended or looping chain of thought beyond expected threshold."""
        if token_count <= max_expected_tokens:
            return 0.0
        overshoot = token_count - max_expected_tokens
        return min(1.0, overshoot / max_expected_tokens)

    def compute_composite_reward(
        self,
        verifiable_correctness: float,  # 1.0 or 0.0
        output_text: str,
        token_count: int,
        max_expected_tokens: int = 2048
    ) -> Dict[str, float]:
        """Combines verifiable correctness, format compliance, and efficiency into total reward."""
        r_corr = verifiable_correctness
        r_fmt = self.evaluate_format_compliance(output_text)
        p_eff = self.evaluate_efficiency_penalty(token_count, max_expected_tokens)

        total_reward = (self.w_corr * r_corr) + (self.w_fmt * r_fmt) - (self.w_eff * p_eff)

        return {
            "total_reward": float(total_reward),
            "correctness_reward": float(r_corr),
            "format_reward": float(r_fmt),
            "efficiency_penalty": float(p_eff)
        }

    def compute_group_advantages(
        self,
        group_rollouts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Computes group-normalized advantages for G rollouts:
          A_i = (R_i - mean(R)) / (std(R) + epsilon)
        """
        group_size = len(group_rollouts)
        if group_size == 0:
            return []

        rewards = [r["total_reward"] for r in group_rollouts]
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))

        results = []
        for rollout in group_rollouts:
            r = rollout["total_reward"]
            if std_reward < 1e-8:
                # If all rollouts achieved the exact same reward, advantage is zero
                advantage = 0.0
            else:
                advantage = (r - mean_reward) / (std_reward + self.epsilon)

            enriched = dict(rollout)
            enriched["advantage"] = float(round(advantage, 4))
            enriched["group_mean_reward"] = float(round(mean_reward, 4))
            enriched["group_std_reward"] = float(round(std_reward, 4))
            results.append(enriched)

        return results

    def format_grpo_training_batch(
        self,
        prompt: str,
        group_rollouts_with_advantages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Serializes group rollouts into format consumable by GRPO trainer."""
        return {
            "prompt": prompt,
            "group_size": len(group_rollouts_with_advantages),
            "rollouts": [
                {
                    "output_text": item["output_text"],
                    "total_reward": item["total_reward"],
                    "advantage": item["advantage"],
                    "correctness": item["correctness_reward"],
                    "format_score": item["format_reward"]
                }
                for item in group_rollouts_with_advantages
            ]
        }
