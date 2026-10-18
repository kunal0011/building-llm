"""
DeepSeek-Style Mixture-of-Experts (MoE) Architecture with Fine-Grained & Shared Experts.
Features:
- Fine-grained expert segmentation (e.g. 16/64 routed experts with compact intermediate dimension)
- Dedicated shared experts capturing invariant domain knowledge without routing overhead
- Softmax top-k gating with score re-normalization
- Auxiliary load balancing loss to prevent expert starvation or collapse
- Native integration with 128k BPE vocabulary, GQA, and RMSNorm
"""

import math
import json
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.model import RMSNorm, AttentionWithGQA, ModelArgs


@dataclass
class MoEArgs(ModelArgs):
    """Architectural parameters for DeepSeek-style MoE models."""
    num_routed_experts: int = 16
    num_shared_experts: int = 2
    top_k_experts: int = 4
    routed_intermediate_size: int = 1408
    shared_intermediate_size: int = 2816
    aux_loss_alpha: float = 0.01

    @classmethod
    def from_moe_json(cls, json_path: str) -> "MoEArgs":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            vocab_size=data.get("vocab_size", 128000),
            hidden_size=data.get("hidden_size", 2048),
            intermediate_size=data.get("intermediate_size", 5632),
            num_hidden_layers=data.get("num_hidden_layers", 16),
            num_attention_heads=data.get("num_attention_heads", 16),
            num_key_value_heads=data.get("num_key_value_heads", 4),
            max_position_embeddings=data.get("max_position_embeddings", 8192),
            rope_theta=data.get("rope_theta", 500000.0),
            rms_norm_eps=data.get("rms_norm_eps", 1e-5),
            initializer_range=data.get("initializer_range", 0.02),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            dropout=data.get("attention_dropout", 0.0),
            num_routed_experts=data.get("num_routed_experts", 16),
            num_shared_experts=data.get("num_shared_experts", 2),
            top_k_experts=data.get("top_k_experts", 4),
            routed_intermediate_size=data.get("routed_intermediate_size", 1408),
            shared_intermediate_size=data.get("shared_intermediate_size", 2816),
            aux_loss_alpha=data.get("aux_loss_alpha", 0.01)
        )


class FineGrainedSwiGLUExpert(nn.Module):
    """Single expert implementing SwiGLU activation."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoETopKRouter(nn.Module):
    """
    Softmax Top-K Router with auxiliary load-balancing loss.
    Computes gating probabilities over routed experts and dispatches top-k.
    """
    def __init__(self, hidden_size: int, num_experts: int, top_k: int, aux_loss_alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.aux_loss_alpha = aux_loss_alpha
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Flattened tokens [num_tokens, hidden_size]
        Returns:
            top_k_weights: [num_tokens, top_k] normalized weights
            top_k_indices: [num_tokens, top_k] expert indices
            aux_loss: Scalar auxiliary load-balancing penalty
        """
        logits = self.gate(x)  # [num_tokens, num_experts]
        scores = F.softmax(logits, dim=-1)

        # Select top-k experts per token
        top_k_weights, top_k_indices = torch.topk(scores, self.top_k, dim=-1)

        # Softmax re-normalization of chosen top-k weights
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        # Auxiliary load balancing loss computation:
        # L_aux = alpha * num_experts * sum(f_i * P_i)
        if self.training:
            num_tokens = x.shape[0]
            # Fraction of tokens dispatched to each expert
            expert_mask = torch.zeros_like(scores).scatter_(-1, top_k_indices, 1.0)
            f_i = expert_mask.sum(dim=0) / float(max(1, num_tokens))
            P_i = scores.mean(dim=0)
            aux_loss = self.aux_loss_alpha * float(self.num_experts) * torch.sum(f_i * P_i)
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return top_k_weights, top_k_indices, aux_loss


class DeepSeekMoEBlock(nn.Module):
    """
    Mixture-of-Experts block combining:
    1. Shared Experts (always active, invariant representation)
    2. Fine-grained Routed Experts (top-k sparse activation)
    """
    def __init__(self, args: MoEArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_routed = args.num_routed_experts
        self.num_shared = args.num_shared_experts
        self.top_k = args.top_k_experts

        # Router
        self.router = MoETopKRouter(
            hidden_size=args.hidden_size,
            num_experts=args.num_routed_experts,
            top_k=args.top_k_experts,
            aux_loss_alpha=args.aux_loss_alpha
        )

        # Routed Experts
        self.routed_experts = nn.ModuleList([
            FineGrainedSwiGLUExpert(args.hidden_size, args.routed_intermediate_size)
            for _ in range(args.num_routed_experts)
        ])

        # Shared Experts (always active)
        self.shared_experts = nn.ModuleList([
            FineGrainedSwiGLUExpert(args.hidden_size, args.shared_intermediate_size)
            for _ in range(args.num_shared_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, hidden_size]
        b_sz, seq_len, hidden_dim = x.shape
        flat_x = x.view(-1, hidden_dim)  # [num_tokens, hidden_size]

        # 1. Compute Shared Experts (dense computation)
        shared_output = torch.zeros_like(flat_x)
        for shared_exp in self.shared_experts:
            shared_output = shared_output + shared_exp(flat_x)

        # 2. Route Tokens to Top-K Experts
        top_k_weights, top_k_indices, aux_loss = self.router(flat_x)

        # 3. Compute Routed Experts Output
        routed_output = torch.zeros_like(flat_x)
        for k_idx in range(self.top_k):
            exp_indices = top_k_indices[:, k_idx]  # [num_tokens]
            weights = top_k_weights[:, k_idx].unsqueeze(-1)  # [num_tokens, 1]

            # Unique experts in this top-k position
            for exp_id in range(self.num_routed):
                mask = (exp_indices == exp_id)
                if mask.any():
                    token_subset = flat_x[mask]
                    exp_out = self.routed_experts[exp_id](token_subset)
                    routed_output[mask] += weights[mask] * exp_out

        # Combine shared and routed outputs
        final_output = (shared_output + routed_output).view(b_sz, seq_len, hidden_dim)
        return final_output, aux_loss


class UniversalMoEDecoderLayer(nn.Module):
    """Transformer decoder block with GQA Attention and DeepSeek MoE FFN."""
    def __init__(self, args: MoEArgs):
        super().__init__()
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.self_attn = AttentionWithGQA(args)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.moe = DeepSeekMoEBlock(args)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Residual + Attention
        h = x + self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)
        # Residual + MoE FFN
        moe_out, aux_loss = self.moe(self.post_attention_layernorm(h))
        out = h + moe_out
        return out, aux_loss


class UniversalMoEForCausalLM(nn.Module):
    """
    Foundation Decoder Model with DeepSeek-style Mixture-of-Experts.
    Outputs logits, autoregressive loss, and total auxiliary load-balancing loss.
    """
    def __init__(self, args: MoEArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList([UniversalMoEDecoderLayer(args) for _ in range(args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        if args.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self._init_weights()

    def _init_weights(self):
        std = self.args.initializer_range
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        # Active params per token calculation:
        # Embedding + Norms + GQA + Shared Experts + (Top-K / Total Routed) * Routed Experts
        return {"total_parameters": total}

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        x = self.embed_tokens(input_ids)

        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        for layer in self.layers:
            x, layer_aux_loss = layer(x, attention_mask=attention_mask)
            total_aux_loss = total_aux_loss + layer_aux_loss

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            ce_loss = loss_fn(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            # Total objective = CrossEntropy + Auxiliary Load Balancing Penalty
            loss = ce_loss + total_aux_loss

        return {
            "logits": logits,
            "loss": loss,
            "ce_loss": ce_loss if labels is not None else None,
            "aux_loss": total_aux_loss
        }
