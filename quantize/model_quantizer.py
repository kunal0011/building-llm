"""
Full-Model Post-Training Quantization Engine.
Recursively converts causal LM transformer blocks (attention, FFN, MoE experts)
into packed QuantLinear representations, measures compression ratios, and manages checkpoint serialization.
"""

import os
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn

from quantize.quant_linear import QuantLinear
from quantize.awq import AWQCalibrator


class ModelQuantizer:
    """
    Engine for quantizing transformer language models and MoE architectures.
    """

    @staticmethod
    def calculate_memory_footprint(model: nn.Module) -> Dict[str, Any]:
        """
        Calculates parameter counts and buffer sizes in Megabytes (MB).
        """
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
        total_bytes = param_bytes + buffer_bytes
        total_params = sum(p.numel() for p in model.parameters())

        return {
            "total_params": total_params,
            "param_mb": round(param_bytes / (1024 * 1024), 2),
            "buffer_mb": round(buffer_bytes / (1024 * 1024), 2),
            "total_size_mb": round(total_bytes / (1024 * 1024), 2)
        }

    @classmethod
    def quantize_model(
        cls,
        model: nn.Module,
        bits: int = 4,
        group_size: int = 128,
        calibration_data: Optional[List[torch.Tensor]] = None,
        target_layers: Optional[List[str]] = None,
        exclude_layers: Optional[List[str]] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Replaces nn.Linear layers in model with QuantLinear modules.
        
        Args:
            model: PyTorch model to quantize
            bits: 4 or 8
            group_size: Quantization group size (e.g. 128, 64, 32)
            calibration_data: Optional input batches for AWQ activation profiling
            target_layers: Substring matches for layers to quantize (default: attention and MLP projections)
            exclude_layers: Substrings for layers to keep in full precision (e.g. ['lm_head', 'embed_tokens'])
        """
        initial_metrics = cls.calculate_memory_footprint(model)

        targets = target_layers or ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "w1", "w2", "w3"]
        excludes = exclude_layers or ["lm_head", "embed_tokens"]

        # Run AWQ calibration if data provided
        act_dict: Dict[str, torch.Tensor] = {}
        if calibration_data:
            calibrator = AWQCalibrator(model)
            act_dict = calibrator.calibrate(calibration_data, target_modules=targets)

        quantized_count = 0

        def replace_linear(module: nn.Module, prefix: str = ""):
            nonlocal quantized_count
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name

                # Check if layer should be excluded
                if any(ex in full_name for ex in excludes):
                    continue

                if isinstance(child, nn.Linear) and any(t in name for t in targets):
                    act_mag = act_dict.get(full_name, None)
                    if act_mag is not None:
                        q_linear = AWQCalibrator.quantize_linear_awq(
                            child, act_magnitude=act_mag, bits=bits, group_size=group_size
                        )
                    else:
                        q_linear = QuantLinear.from_linear(child, bits=bits, group_size=group_size)

                    setattr(module, name, q_linear)
                    quantized_count += 1
                else:
                    replace_linear(child, full_name)

        replace_linear(model)

        final_metrics = cls.calculate_memory_footprint(model)
        compression_ratio = round(
            initial_metrics["total_size_mb"] / max(final_metrics["total_size_mb"], 1e-3), 2
        )

        report = {
            "quantized_layers_count": quantized_count,
            "bits": bits,
            "group_size": group_size,
            "initial_size_mb": initial_metrics["total_size_mb"],
            "quantized_size_mb": final_metrics["total_size_mb"],
            "compression_ratio": compression_ratio,
            "memory_saved_percent": round(
                (1.0 - (final_metrics["total_size_mb"] / max(initial_metrics["total_size_mb"], 1e-3))) * 100, 1
            )
        }

        return model, report

    @staticmethod
    def save_quantized(model: nn.Module, report: Dict[str, Any], save_path: str) -> None:
        """Saves quantized model state dictionary and metadata."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "quant_report": report
        }, save_path)
