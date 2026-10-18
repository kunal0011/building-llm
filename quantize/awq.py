"""
Activation-Aware Weight Quantization (AWQ) Calibration & Quantization Engine.
Protects salient weight channels by scaling weights according to activation magnitudes,
dramatically reducing perplexity degradation during INT4/INT8 quantization.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from quantize.quant_linear import QuantLinear


class AWQCalibrator:
    """
    Calibrator that captures layer input activations on a representative calibration dataset,
    computes channel-wise saliency metrics, and determines optimal per-channel scales.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def _register_hooks(self, target_modules: List[str]) -> None:
        """Attaches forward hooks to collect activation statistics."""
        self.activations.clear()
        self.hooks.clear()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
                def make_hook(mod_name):
                    def hook(m, inp, outp):
                        # inp[0]: [batch, seq_len, in_features]
                        x = inp[0].detach()
                        # Compute mean absolute magnitude per channel: [in_features]
                        act_magnitude = x.abs().view(-1, x.shape[-1]).mean(dim=0)
                        if mod_name not in self.activations:
                            self.activations[mod_name] = act_magnitude
                        else:
                            self.activations[mod_name] = torch.max(
                                self.activations[mod_name], act_magnitude
                            )
                    return hook

                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def _remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    @torch.no_grad()
    def calibrate(
        self,
        calibration_inputs: List[torch.Tensor],
        target_modules: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Runs calibration forward passes and computes channel activation scales.
        """
        targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self._register_hooks(targets)
        self.model.eval()

        device = next(self.model.parameters()).device

        for batch in calibration_inputs:
            if isinstance(batch, torch.Tensor):
                batch = batch.to(device)
                self.model(batch)
            elif isinstance(batch, dict):
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                self.model(**inputs)

        self._remove_hooks()
        return self.activations

    @staticmethod
    def compute_awq_scales(
        weight: torch.Tensor,
        act_magnitude: torch.Tensor,
        alpha: float = 0.5
    ) -> torch.Tensor:
        """
        Computes optimal channel scales s = act_magnitude^alpha / mean(act_magnitude^alpha)
        weight: [out_features, in_features]
        act_magnitude: [in_features]
        Returns: scales [in_features]
        """
        # Saliency s_X
        s_x = act_magnitude.clamp(min=1e-5)
        # Scale: s = s_X^alpha
        scales = (s_x ** alpha)
        # Normalize so scale mean is 1.0 (preserves global dynamic range)
        scales = scales / (scales.mean() + 1e-8)
        # Clamp scale to prevent extreme numerical variance
        scales = scales.clamp(min=1e-2, max=100.0)
        return scales

    @classmethod
    def quantize_linear_awq(
        cls,
        linear: nn.Linear,
        act_magnitude: Optional[torch.Tensor] = None,
        bits: int = 4,
        group_size: int = 128,
        alpha: float = 0.5
    ) -> QuantLinear:
        """
        Applies AWQ protection to a Linear layer and converts to QuantLinear.
        """
        weight = linear.weight.data.clone()
        out_features, in_features = weight.shape

        if act_magnitude is not None and act_magnitude.shape[0] == in_features:
            scales = cls.compute_awq_scales(weight, act_magnitude, alpha=alpha)
            # W' = W * diag(s)
            # When multiplying W' by (x / s), the result is identical to W * x
            weight_scaled = weight * scales.unsqueeze(0)
        else:
            weight_scaled = weight

        q_layer = QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bits=bits,
            group_size=group_size,
            bias=linear.bias is not None
        )
        if linear.bias is not None:
            q_layer.bias.copy_(linear.bias.data)

        q_layer.quantize_and_pack(weight_scaled)
        return q_layer
