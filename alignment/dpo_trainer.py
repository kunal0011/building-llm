"""
Production DPO Alignment Trainer.
Orchestrates policy model optimization against a frozen reference model,
monitoring preference margins, KL divergence drift, and training convergence.
"""

import copy
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW

from alignment.dpo_engine import DPOEngine


class DPOTrainer:
    """
    Direct Preference Optimization Trainer.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        lr: float = 1e-5,
        beta: float = 0.1,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        device: str = "auto"
    ):
        self.device = self._resolve_device(device)
        self.beta = beta
        self.max_grad_norm = max_grad_norm

        self.policy_model = policy_model.to(self.device)

        if ref_model is not None:
            self.ref_model = ref_model.to(self.device)
        else:
            # Create frozen reference snapshot
            self.ref_model = copy.deepcopy(policy_model).to(self.device)

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.optimizer = AdamW(
            self.policy_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def _resolve_device(self, req: str) -> torch.device:
        if req == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(req)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Executes a single optimization step on a batch of paired preferences.
        """
        self.policy_model.train()

        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        # 1. Policy forward passes
        policy_chosen_logps = DPOEngine.compute_log_probs(
            self.policy_model, chosen_ids, chosen_labels
        )
        policy_rejected_logps = DPOEngine.compute_log_probs(
            self.policy_model, rejected_ids, rejected_labels
        )

        # 2. Reference forward passes (no autograd)
        with torch.no_grad():
            ref_chosen_logps = DPOEngine.compute_log_probs(
                self.ref_model, chosen_ids, chosen_labels
            )
            ref_rejected_logps = DPOEngine.compute_log_probs(
                self.ref_model, rejected_ids, rejected_labels
            )

        # 3. Calculate DPO loss & metrics
        loss, metrics = DPOEngine.dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=self.beta
        )

        # 4. Backpropagation & optimization step
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        metrics["grad_norm"] = round(float(grad_norm), 4)
        return metrics
