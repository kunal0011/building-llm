"""
Multi-Stage Context Annealing Scheduler & RoPE Frequency Scaler.
Orchestrates progressive sequence length expansion (8k -> 32k -> 128k -> 1M+ tokens),
computes NTK-aware and YaRN Rotary Position Embedding (RoPE) frequency scaling,
and synthesizes Needle-In-A-Haystack (NIAH) verification benchmarks.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from pipelines.long_context.document_packer import DocumentPacker


class AnnealingStage(Enum):
    STAGE_8K = "8k_base"
    STAGE_32K = "32k_extended"
    STAGE_128K = "128k_long_context"
    STAGE_1M = "1m_extreme_horizon"


@dataclass
class StageConfiguration:
    stage: AnnealingStage
    context_length: int
    base_rope_theta: float
    target_token_budget: int
    yarn_scale_factor: float
    domain_mix_weights: Dict[str, float]
    flash_attention_block_size: int = 128


class RoPEScalingCalculator:
    """
    Computes Rotary Position Embedding (RoPE) frequency adjustments
    for long-context LLM expansion.
    Supports:
      1. Base frequency recalculation: theta = 500k -> 1M -> 5M -> 10M
      2. NTK-Aware Dynamic RoPE scaling
      3. YaRN (Yet another RoPE extensioN) frequency interpolation
    """

    def __init__(
        self,
        head_dim: int = 128,
        original_max_position: int = 8192,
        base_theta: float = 500_000.0
    ):
        self.head_dim = head_dim
        self.original_max_position = original_max_position
        self.base_theta = base_theta

    def compute_base_frequencies(self, theta: Optional[float] = None) -> np.ndarray:
        """
        Computes standard RoPE inverse frequencies:
          omega_i = 1.0 / (theta ** (2i / d)) for i in [0, d/2 - 1]
        """
        t = theta or self.base_theta
        indices = np.arange(0, self.head_dim, 2, dtype=np.float64)
        inv_freq = 1.0 / (t ** (indices / self.head_dim))
        return inv_freq

    def compute_ntk_aware_theta(self, extended_context_length: int) -> float:
        """
        Computes NTK-Aware scaled base theta for context expansion:
          theta' = theta * s ** (d / (d - 2))
        where s = extended_context / original_max_position
        """
        if extended_context_length <= self.original_max_position:
            return self.base_theta

        scale = extended_context_length / self.original_max_position
        d = float(self.head_dim)
        ntk_theta = self.base_theta * (scale ** (d / (d - 2.0)))
        return float(ntk_theta)

    def compute_yarn_frequencies(
        self,
        extended_context_length: int,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        Computes YaRN (Yet another RoPE extensioN) interpolated frequencies.
        Splits dimensions by wavelength lambda_i = 2 * pi / omega_i:
          - High-frequency dimensions (lambda_i < L_base / beta_fast): no interpolation
          - Low-frequency dimensions (lambda_i > L_base / beta_slow): full linear interpolation by scale s
          - Intermediate dimensions: smooth ramp interpolation gamma in [0, 1]
        Returns (scaled_inv_frequencies, attention_temperature_scale).
        """
        scale = max(1.0, extended_context_length / self.original_max_position)
        base_inv_freq = self.compute_base_frequencies(self.base_theta)

        # Wavelengths for each frequency dimension
        wavelengths = 2.0 * math.pi / base_inv_freq

        # Ramp bounds based on original context
        low_bound = self.original_max_position / beta_fast
        high_bound = self.original_max_position / beta_slow

        scaled_inv_freq = np.zeros_like(base_inv_freq)
        for i, wl in enumerate(wavelengths):
            if wl < low_bound:
                # High frequency: preserve local positional discrimination without stretching
                gamma = 0.0
            elif wl > high_bound:
                # Low frequency: stretch fully by scale factor
                gamma = 1.0
            else:
                # Smooth ramp interpolation between low and high
                gamma = (wl - low_bound) / (high_bound - low_bound)

            # Interpolate inverse frequency: (1 - gamma) * omega + gamma * (omega / scale)
            scaled_inv_freq[i] = (1.0 - gamma) * base_inv_freq[i] + gamma * (base_inv_freq[i] / scale)

        # Temperature scaling factor for attention dot-product: sqrt(t) = 0.1 * ln(s) + 1.0
        attn_temp_scale = 0.1 * math.log(scale) + 1.0 if scale > 1.0 else 1.0

        return scaled_inv_freq, attn_temp_scale


class NeedleInAHaystackSynthesizer:
    """
    Synthesizes Needle-In-A-Haystack (NIAH) long-context evaluation benchmarks.
    Embeds secret target facts ('needles') into large corpus backgrounds ('haystacks')
    at calibrated depths (0% to 100%) across 8k, 32k, 128k, and 1M context windows.
    """

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def generate_haystack(
        self,
        target_token_count: int,
        background_corpus: List[str]
    ) -> str:
        """Builds a continuous long haystack text matching target token count."""
        combined = []
        current_len = 0
        approx_words_needed = int(target_token_count * 0.75)

        idx = 0
        while current_len < approx_words_needed and background_corpus:
            doc = background_corpus[idx % len(background_corpus)]
            combined.append(doc)
            current_len += len(doc.split())
            idx += 1

        return "\n\n".join(combined)

    def inject_needle(
        self,
        haystack: str,
        needle_key: str,
        needle_value: str,
        depth_percent: float = 0.50
    ) -> Dict[str, Any]:
        """
        Embeds a verifiable needle at depth_percent (0.0 = top, 1.0 = bottom) of haystack.
        """
        needle_text = f" <|needle_start|> Special Authorization Guideline: The secret verification code for {needle_key} is {needle_value}. <|needle_end|> "
        paragraphs = haystack.split("\n\n")

        insert_idx = max(0, min(len(paragraphs), int(len(paragraphs) * depth_percent)))
        paragraphs.insert(insert_idx, needle_text)

        synthesized_text = "\n\n".join(paragraphs)
        test_query = f"What is the secret verification code for {needle_key}?"

        return {
            "prompt": synthesized_text,
            "query": test_query,
            "ground_truth": needle_value,
            "depth_percent": depth_percent,
            "needle_key": needle_key,
            "character_length": len(synthesized_text),
            "estimated_tokens": int(len(synthesized_text.split()) * 1.33)
        }


class ContextAnnealingScheduler:
    """
    Curriculum Scheduler for Progressive Multi-Stage Context Annealing.
    Manages transition milestones, RoPE calibrations, and dynamic sequence packing.
    """

    def __init__(self, head_dim: int = 128):
        self.head_dim = head_dim
        self.rope_calculator = RoPEScalingCalculator(head_dim=head_dim, original_max_position=8192)
        self.stages: Dict[AnnealingStage, StageConfiguration] = {
            AnnealingStage.STAGE_8K: StageConfiguration(
                stage=AnnealingStage.STAGE_8K,
                context_length=8_192,
                base_rope_theta=500_000.0,
                target_token_budget=12_000_000_000_000,  # 12T tokens
                yarn_scale_factor=1.0,
                domain_mix_weights={"curated_web": 0.60, "code": 0.20, "stem_math": 0.10, "books": 0.10}
            ),
            AnnealingStage.STAGE_32K: StageConfiguration(
                stage=AnnealingStage.STAGE_32K,
                context_length=32_768,
                base_rope_theta=1_000_000.0,
                target_token_budget=1_500_000_000_000,   # 1.5T tokens
                yarn_scale_factor=2.0,
                domain_mix_weights={"curated_web": 0.35, "code": 0.35, "stem_math": 0.15, "books": 0.15}
            ),
            AnnealingStage.STAGE_128K: StageConfiguration(
                stage=AnnealingStage.STAGE_128K,
                context_length=131_072,
                base_rope_theta=5_000_000.0,
                target_token_budget=500_000_000_000,     # 500B tokens
                yarn_scale_factor=4.0,
                domain_mix_weights={"curated_web": 0.15, "code": 0.40, "stem_math": 0.25, "books_legal": 0.20}
            ),
            AnnealingStage.STAGE_1M: StageConfiguration(
                stage=AnnealingStage.STAGE_1M,
                context_length=1_048_576,
                base_rope_theta=10_000_000.0,
                target_token_budget=100_000_000_000,    # 100B tokens
                yarn_scale_factor=8.0,
                domain_mix_weights={"code_full_repo": 0.45, "stem_monographs": 0.25, "agent_trajectories": 0.20, "niah_synthetic": 0.10}
            )
        }

    def get_stage_config(self, stage: AnnealingStage) -> StageConfiguration:
        return self.stages[stage]

    def compute_rope_for_stage(self, stage: AnnealingStage) -> Dict[str, Any]:
        """Calculates RoPE frequencies and attention temperature for the selected stage."""
        config = self.stages[stage]
        ctx_len = config.context_length
        base_theta = config.base_rope_theta

        ntk_theta = self.rope_calculator.compute_ntk_aware_theta(ctx_len)
        yarn_freqs, attn_temp = self.rope_calculator.compute_yarn_frequencies(ctx_len)

        return {
            "stage": stage.value,
            "context_length": ctx_len,
            "base_theta": base_theta,
            "ntk_scaled_theta": round(ntk_theta, 1),
            "yarn_attn_temp_scale": round(attn_temp, 4),
            "inv_freq_dims": len(yarn_freqs),
            "min_inv_freq": float(np.min(yarn_freqs)),
            "max_inv_freq": float(np.max(yarn_freqs))
        }

    def create_batch_packer(self, stage: AnnealingStage) -> DocumentPacker:
        """Returns DocumentPacker calibrated to the stage's target sequence length."""
        config = self.stages[stage]
        return DocumentPacker(max_seq_len=config.context_length)

    def generate_curriculum_report(self) -> Dict[str, Any]:
        """Compiles a complete curriculum blueprint across all 4 context expansion stages."""
        summary = []
        for stg in [AnnealingStage.STAGE_8K, AnnealingStage.STAGE_32K, AnnealingStage.STAGE_128K, AnnealingStage.STAGE_1M]:
            cfg = self.stages[stg]
            rope_info = self.compute_rope_for_stage(stg)
            summary.append({
                "stage": stg.value,
                "context_length": f"{cfg.context_length:,} tokens",
                "token_budget": f"{cfg.target_token_budget / 1e12:.2f}T tokens" if cfg.target_token_budget >= 1e12 else f"{cfg.target_token_budget / 1e9:.0f}B tokens",
                "base_rope_theta": f"{cfg.base_rope_theta:,.0f}",
                "ntk_scaled_theta": f"{rope_info['ntk_scaled_theta']:,.0f}",
                "yarn_temp_scale": rope_info["yarn_attn_temp_scale"],
                "primary_domain_focus": max(cfg.domain_mix_weights, key=cfg.domain_mix_weights.get),
                "domain_mix": cfg.domain_mix_weights
            })
        return {
            "curriculum": summary,
            "total_annealing_tokens": "14.1T tokens",
            "progressive_factor": "128x expansion (8k -> 1,048,576)"
        }
