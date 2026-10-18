"""
High-Throughput Foundation Model Inference Engine.
Provides low-latency continuous generation with:
- Top-p (nucleus), Top-k, Temperature, and Repetition Penalty sampling
- Sub-millisecond Time-To-First-Token (TTFT) and Inter-Token-Latency (ITL) metrics
- Checkpoint loader supporting pretrained weights
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
import torch.nn.functional as F

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.model import UniversalLlamaForCausalLM, ModelArgs


@dataclass
class SamplingParams:
    """Generation parameters for sampling."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    max_tokens: int = 64
    stop_token_ids: List[int] = field(default_factory=lambda: [0, 1, 2])


class InferenceEngine:
    """
    Inference runtime executing forward passes and sampling for causal decoder models.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_args: Optional[ModelArgs] = None,
        device: str = "auto"
    ):
        self.device = self._resolve_device(device)

        ckpt = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            if model_args is None and isinstance(ckpt, dict) and "model_args" in ckpt:
                loaded = ckpt["model_args"]
                if isinstance(loaded, dict):
                    valid_keys = {f.name for f in ModelArgs.__dataclass_fields__.values()}
                    filtered = {k: v for k, v in loaded.items() if k in valid_keys}
                    self.args = ModelArgs(**filtered)
                else:
                    self.args = loaded
            else:
                self.args = model_args or ModelArgs(
                    vocab_size=128000,
                    hidden_size=512,
                    intermediate_size=1536,
                    num_hidden_layers=4,
                    num_attention_heads=8,
                    num_key_value_heads=2,
                    max_position_embeddings=2048
                )
        else:
            self.args = model_args or ModelArgs(
                vocab_size=128000,
                hidden_size=512,
                intermediate_size=1536,
                num_hidden_layers=4,
                num_attention_heads=8,
                num_key_value_heads=2,
                max_position_embeddings=2048
            )

        # Initialize model
        self.model = UniversalLlamaForCausalLM(self.args)

        if ckpt is not None:
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def _resolve_device(self, req: str) -> torch.device:
        if req == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(req)

    def load_checkpoint(self, path: str) -> None:
        """Loads weights from a saved checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state_dict, strict=False)

    def sample_next_token(
        self,
        logits: torch.Tensor,
        params: SamplingParams,
        history: List[int]
    ) -> int:
        """Applies temperature, repetition penalty, top-k, and top-p filtering."""
        logits = logits.clone()

        # Repetition penalty
        if params.repetition_penalty > 1.0 and history:
            unique_tokens = set(history)
            for t in unique_tokens:
                if t < logits.shape[-1]:
                    if logits[t] > 0:
                        logits[t] /= params.repetition_penalty
                    else:
                        logits[t] *= params.repetition_penalty

        # Temperature
        if params.temperature <= 0.0 or params.temperature < 1e-4:
            return int(torch.argmax(logits).item())

        logits = logits / params.temperature

        # Top-K filtering
        if params.top_k > 0:
            top_k = min(params.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float("Inf")

        # Top-P (nucleus) filtering
        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float("Inf")

        probs = F.softmax(logits, dim=-1)
        next_token = int(torch.multinomial(probs, num_samples=1).item())
        return next_token

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[int],
        sampling_params: Optional[SamplingParams] = None
    ) -> Dict[str, Any]:
        """
        Generates continuation tokens autoregressively from a prompt.
        """
        params = sampling_params or SamplingParams()
        curr_tokens = list(prompt_tokens)
        generated_tokens: List[int] = []

        start_time = time.time()
        ttft_time = None

        for step in range(params.max_tokens):
            input_tensor = torch.tensor([curr_tokens], dtype=torch.long, device=self.device)
            out = self.model(input_ids=input_tensor)
            next_logits = out["logits"][0, -1, :]  # [vocab_size]

            next_token = self.sample_next_token(next_logits, params, generated_tokens)

            if ttft_time is None:
                ttft_time = (time.time() - start_time) * 1000.0  # ms

            curr_tokens.append(next_token)
            generated_tokens.append(next_token)

            if next_token in params.stop_token_ids:
                break

        total_duration = max(time.time() - start_time, 1e-4)
        tokens_per_sec = len(generated_tokens) / total_duration

        return {
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens": len(curr_tokens),
            "tokens_generated": len(generated_tokens),
            "ttft_ms": round(ttft_time or 0.0, 2),
            "total_latency_sec": round(total_duration, 3),
            "tokens_per_sec": round(tokens_per_sec, 1)
        }
