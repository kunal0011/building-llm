"""
Unit tests for FP8/INT4/INT8 Quantization Engine and AWQ Calibration.
"""

import pytest
import torch
import torch.nn as nn

from train.model import UniversalLlamaForCausalLM, ModelArgs
from train.moe_model import UniversalMoEForCausalLM, MoEArgs
from quantize.quant_linear import QuantLinear
from quantize.awq import AWQCalibrator
from quantize.model_quantizer import ModelQuantizer


@pytest.fixture
def mini_model_args():
    return ModelArgs(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128
    )


@pytest.fixture
def mini_moe_args():
    return MoEArgs(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        num_routed_experts=4,
        num_shared_experts=1,
        top_k_experts=2,
        routed_intermediate_size=64,
        shared_intermediate_size=64
    )


def test_quant_linear_int4_and_int8():
    in_feat, out_feat = 128, 64
    linear = nn.Linear(in_feat, out_feat, bias=True)

    # 1. INT4 with group_size=32
    q_int4 = QuantLinear.from_linear(linear, bits=4, group_size=32)
    assert q_int4.qweight.dtype == torch.uint8
    assert q_int4.qweight.shape == (out_feat, in_feat // 2)

    x = torch.randn(2, 8, in_feat)
    y_orig = linear(x)
    y_int4 = q_int4(x)
    assert y_int4.shape == (2, 8, out_feat)

    mse_int4 = F_mse = ((y_orig - y_int4) ** 2).mean().item()
    assert mse_int4 < 0.05, f"INT4 error too large: {mse_int4}"

    # 2. INT8 with group_size=32
    q_int8 = QuantLinear.from_linear(linear, bits=8, group_size=32)
    assert q_int8.qweight.dtype == torch.int8
    assert q_int8.qweight.shape == (out_feat, in_feat)

    y_int8 = q_int8(x)
    mse_int8 = ((y_orig - y_int8) ** 2).mean().item()
    assert mse_int8 < 0.001, f"INT8 error too large: {mse_int8}"


def test_awq_calibrator_and_salience_scaling(mini_model_args):
    model = UniversalLlamaForCausalLM(mini_model_args)
    calibrator = AWQCalibrator(model)

    calib_inputs = [
        torch.randint(0, 1000, (2, 16)),
        torch.randint(0, 1000, (2, 16))
    ]

    act_dict = calibrator.calibrate(calib_inputs)
    assert len(act_dict) > 0

    # Verify scale calculation
    dummy_weight = torch.randn(64, 64)
    dummy_act = torch.abs(torch.randn(64)) + 0.1
    scales = AWQCalibrator.compute_awq_scales(dummy_weight, dummy_act, alpha=0.5)
    assert scales.shape == (64,)
    assert (scales > 0).all()


def test_model_quantizer_transformer_compression(mini_model_args):
    model = UniversalLlamaForCausalLM(mini_model_args)
    calib_inputs = [torch.randint(0, 1000, (2, 16))]

    quantized_model, report = ModelQuantizer.quantize_model(
        model,
        bits=4,
        group_size=32,
        calibration_data=calib_inputs
    )

    assert report["quantized_layers_count"] > 0
    assert report["bits"] == 4
    assert report["compression_ratio"] >= 1.2
    assert report["memory_saved_percent"] > 15.0

    # Check forward pass
    x = torch.randint(0, 1000, (2, 12))
    out = quantized_model(x)
    assert "logits" in out
    assert out["logits"].shape == (2, 12, mini_model_args.vocab_size)


def test_moe_model_quantization(mini_moe_args):
    moe_model = UniversalMoEForCausalLM(mini_moe_args)
    quantized_moe, report = ModelQuantizer.quantize_model(
        moe_model,
        bits=8,
        group_size=32
    )

    assert report["quantized_layers_count"] > 0

    x = torch.randint(0, 1000, (2, 8))
    out = quantized_moe(x)
    assert "logits" in out
    assert "aux_loss" in out
    assert out["logits"].shape == (2, 8, mini_moe_args.vocab_size)
