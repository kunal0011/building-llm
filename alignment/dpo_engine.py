"""
Direct Preference Optimization (DPO) and Kahneman-Tversky Optimization (KTO) Loss Engines.
Computes exact implicit reward margins, per-token log probabilities, and policy alignment gradients.
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOEngine:
    """
    Mathematical loss engine for Direct Preference Optimization (Rafailov et al., 2023)
    and Kahneman-Tversky Optimization (Ethayarajh et al., 2024).
    """

    @staticmethod
    def compute_log_probs(
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes per-sequence sum of log probabilities for non-masked tokens (labels != -100).
        
        Args:
            model: Causal decoder language model
            input_ids: [batch_size, seq_len]
            labels: [batch_size, seq_len] with -100 on prompt tokens
        Returns:
            sequence_log_probs: [batch_size]
        """
        outputs = model(input_ids)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs

        # Shift logits and labels for autoregressive next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss_mask = shift_labels != -100
        safe_labels = shift_labels.clone()
        safe_labels[~loss_mask] = 0

        # Compute log softmax over vocabulary
        log_probs = F.log_softmax(shift_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=2, index=safe_labels.unsqueeze(2)).squeeze(2)

        # Zero out prompt tokens
        per_token_logps = per_token_logps * loss_mask.float()
        return per_token_logps.sum(dim=-1)

    @classmethod
    def dpo_loss(
        cls,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
        beta: float = 0.1,
        label_smoothing: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculates DPO loss and implicit reward metrics:
        L_DPO = -E [ log sigma( beta * (log(pi(y_w)/pi_ref(y_w)) - log(pi(y_l)/pi_ref(y_l))) ) ]
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        if label_smoothing > 0:
            losses = (
                -F.logsigmoid(beta * logits) * (1 - label_smoothing)
                - F.logsigmoid(-beta * logits) * label_smoothing
            )
        else:
            losses = -F.logsigmoid(beta * logits)

        loss = losses.mean()

        # Compute implicit rewards for telemetry
        chosen_rewards = (beta * (policy_chosen_logps - ref_chosen_logps)).detach()
        rejected_rewards = (beta * (policy_rejected_logps - ref_rejected_logps)).detach()
        reward_margin = (chosen_rewards - rejected_rewards).mean().item()
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()

        metrics = {
            "dpo_loss": round(loss.item(), 4),
            "chosen_reward_mean": round(chosen_rewards.mean().item(), 4),
            "rejected_reward_mean": round(rejected_rewards.mean().item(), 4),
            "reward_margin": round(reward_margin, 4),
            "preference_accuracy": round(accuracy, 4)
        }

        return loss, metrics

    @classmethod
    def kto_loss(
        cls,
        policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        is_chosen: torch.Tensor,
        beta: float = 0.1,
        kl_penalty: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Kahneman-Tversky Optimization (KTO) loss on individual chosen/rejected examples.
        """
        log_ratios = policy_logps - ref_logps
        scaled_logits = beta * log_ratios

        chosen_loss = 1.0 - torch.sigmoid(scaled_logits - kl_penalty)
        rejected_loss = 1.0 - torch.sigmoid(kl_penalty - scaled_logits)

        is_chosen_bool = is_chosen.bool()
        losses = torch.where(is_chosen_bool, chosen_loss, rejected_loss)
        loss = losses.mean()

        implicit_rewards = (beta * log_ratios).detach()

        metrics = {
            "kto_loss": round(loss.item(), 4),
            "implicit_reward_mean": round(implicit_rewards.mean().item(), 4)
        }
        return loss, metrics
