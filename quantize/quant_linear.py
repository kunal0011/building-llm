"""
Low-Bit Quantized Linear Layer Architecture (INT4 / INT8 / FP8).
Implements per-channel and per-group asymmetric/symmetric quantization
with packed integer storage and high-throughput on-the-fly dequantization.
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantLinear(nn.Module):
    """
    Quantized Linear Layer storing weights in packed INT4 or INT8 formats.
    
    Attributes:
        in_features: Input dimension
        out_features: Output dimension
        bits: Bit width (4 or 8)
        group_size: Number of consecutive weights sharing a scale factor (-1 for per-channel)
        scales: Scale factors per group
        zeros: Zero-point offsets per group (for asymmetric quantization)
        qweight: Packed quantized weight tensor
        bias: Optional bias tensor
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = False
    ):
        super().__init__()
        assert bits in (4, 8), f"Unsupported bit-width: {bits}. Only 4 and 8 are supported."
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size if group_size > 0 else in_features

        # Number of groups per output channel
        self.num_groups = math.ceil(in_features / self.group_size)

        # Register scale and zero point buffers
        self.register_buffer(
            "scales",
            torch.ones((out_features, self.num_groups), dtype=torch.float32)
        )
        self.register_buffer(
            "zeros",
            torch.zeros((out_features, self.num_groups), dtype=torch.float32)
        )

        # Packed weight tensor storage
        if self.bits == 4:
            # 2 INT4 values packed per uint8 (or int8) byte along in_features dimension
            packed_in_features = math.ceil(in_features / 2)
            self.register_buffer(
                "qweight",
                torch.zeros((out_features, packed_in_features), dtype=torch.uint8)
            )
        else:  # 8-bit
            self.register_buffer(
                "qweight",
                torch.zeros((out_features, in_features), dtype=torch.int8)
            )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer("bias", None)

    @torch.no_grad()
    def quantize_and_pack(self, weight: torch.Tensor) -> None:
        """
        Quantizes full-precision weight [out_features, in_features] and packs into qweight buffer.
        """
        out_features, in_features = weight.shape
        device = weight.device
        dtype = weight.dtype

        # Pad in_features to multiple of group_size if needed
        pad_size = (self.num_groups * self.group_size) - in_features
        if pad_size > 0:
            weight_padded = F.pad(weight, (0, pad_size), "constant", 0)
        else:
            weight_padded = weight

        # Reshape to [out_features, num_groups, group_size]
        w_groups = weight_padded.view(out_features, self.num_groups, self.group_size)

        if self.bits == 4:
            # Asymmetric INT4 quantization into range [0, 15]
            w_min = w_groups.min(dim=-1, keepdim=True).values  # [out_features, num_groups, 1]
            w_max = w_groups.max(dim=-1, keepdim=True).values
            scales = (w_max - w_min).clamp(min=1e-5) / 15.0
            zeros = (-w_min / scales).round().clamp(0, 15)

            # Quantize: q = clamp(round((w - w_min) / scale), 0, 15)
            q = ((w_groups - w_min) / scales).round().clamp(0, 15).to(torch.uint8)
            q_flat = q.view(out_features, -1)[:, :in_features]  # Remove padding

            # Pack 2 x 4-bit integers into 1 x 8-bit byte:
            # low nibble = q[:, 0::2], high nibble = q[:, 1::2]
            pad_pack = in_features % 2
            if pad_pack > 0:
                q_to_pack = F.pad(q_flat, (0, 1), "constant", 0)
            else:
                q_to_pack = q_flat

            low_nibble = q_to_pack[:, 0::2]
            high_nibble = q_to_pack[:, 1::2]
            packed = (low_nibble & 0x0F) | ((high_nibble & 0x0F) << 4)

            self.qweight.copy_(packed)
            self.scales.copy_(scales.squeeze(-1))
            self.zeros.copy_(zeros.squeeze(-1))

        else:  # 8-bit symmetric quantization into [-128, 127]
            w_max = w_groups.abs().max(dim=-1, keepdim=True).values
            scales = w_max.clamp(min=1e-5) / 127.0
            zeros = torch.zeros_like(scales)

            q = (w_groups / scales).round().clamp(-128, 127).to(torch.int8)
            q_flat = q.view(out_features, -1)[:, :in_features]

            self.qweight.copy_(q_flat)
            self.scales.copy_(scales.squeeze(-1))
            self.zeros.copy_(zeros.squeeze(-1))

    def dequantize(self, target_dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Dequantizes stored weights back to target floating-point precision.
        """
        out_features = self.out_features
        in_features = self.in_features

        if self.bits == 4:
            # Unpack 4-bit nibbles
            low_nibble = (self.qweight & 0x0F).to(target_dtype)
            high_nibble = ((self.qweight >> 4) & 0x0F).to(target_dtype)

            # Interleave back to original order: [out_features, packed_dim * 2]
            unpacked = torch.stack([low_nibble, high_nibble], dim=-1).view(out_features, -1)
            unpacked = unpacked[:, :in_features]

            # Broadcast scales and zeros across group dimension
            pad_size = (self.num_groups * self.group_size) - in_features
            if pad_size > 0:
                unpacked_padded = F.pad(unpacked, (0, pad_size), "constant", 0)
            else:
                unpacked_padded = unpacked

            u_groups = unpacked_padded.view(out_features, self.num_groups, self.group_size)
            scales_expanded = self.scales.unsqueeze(-1).to(target_dtype)
            zeros_expanded = self.zeros.unsqueeze(-1).to(target_dtype)

            w_dequant = (u_groups - zeros_expanded) * scales_expanded
            w_out = w_dequant.view(out_features, -1)[:, :in_features]
            return w_out

        else:  # 8-bit
            q = self.qweight.to(target_dtype)
            pad_size = (self.num_groups * self.group_size) - in_features
            if pad_size > 0:
                q_padded = F.pad(q, (0, pad_size), "constant", 0)
            else:
                q_padded = q

            q_groups = q_padded.view(out_features, self.num_groups, self.group_size)
            scales_expanded = self.scales.unsqueeze(-1).to(target_dtype)
            w_dequant = q_groups * scales_expanded
            return w_dequant.view(out_features, -1)[:, :in_features]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic on-the-fly dequantization.
        x: [..., in_features]
        Returns: [..., out_features]
        """
        weight = self.dequantize(target_dtype=x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, weight, bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128
    ) -> "QuantLinear":
        """
        Factory method constructing QuantLinear from a standard PyTorch nn.Linear.
        """
        q_layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            group_size=group_size,
            bias=linear.bias is not None
        )
        if linear.bias is not None:
            q_layer.bias.copy_(linear.bias.data)
        q_layer.quantize_and_pack(linear.weight.data)
        return q_layer

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"bias={self.bias is not None}"
        )
