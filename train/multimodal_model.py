"""
Multimodal Foundation Model Architecture with Interleaved Vision-Language Pretraining.
Features:
- Native integration with Universal BPE special image anchors (<|image_ref_start|>, <|image_ref_end|>)
- Vision patch projection adapter (MLP) mapping spatial embeddings into language space
- Seamless fusion of image tokens into memory-mapped text sequence streams
- Joint autoregressive next-token prediction loss across text, code, and visual tags
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.model import UniversalLlamaForCausalLM, ModelArgs, RMSNorm, DecoderLayer


@dataclass
class MultimodalModelArgs(ModelArgs):
    """Configuration for Multimodal Vision-Language Model."""
    image_size: int = 224
    patch_size: int = 14
    vision_hidden_size: int = 768
    num_image_tokens: int = 16  # Pooled / downsampled visual token budget per image
    image_token_id: int = 127999  # Placeholder token ID representing image embeddings


class VisionPatchEncoder(nn.Module):
    """
    Patch-based Vision Encoder (ViT/CLIP/SigLIP style).
    Converts raw image pixel tensors [B, 3, H, W] into a sequence of patch representations.
    """
    def __init__(self, image_size: int = 224, patch_size: int = 14, in_channels: int = 3, vision_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # 2D patch convolution
        self.proj = nn.Conv2d(in_channels, vision_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.norm = nn.LayerNorm(vision_dim)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [B, C, H, W]
        b_sz = pixel_values.shape[0]
        patches = self.proj(pixel_values)  # [B, vision_dim, H', W']
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, vision_dim]
        return self.norm(patches)


class VisionLanguageAdapter(nn.Module):
    """
    Two-layer GELU MLP projector mapping vision representations to language token space:
    [num_patches, vision_dim] -> [num_image_tokens, hidden_size]
    """
    def __init__(self, vision_dim: int, hidden_dim: int, num_image_tokens: int = 16):
        super().__init__()
        self.num_image_tokens = num_image_tokens
        # Adaptive pooling to fix visual token count per image
        self.pool = nn.AdaptiveAvgPool1d(num_image_tokens)

        self.linear_1 = nn.Linear(vision_dim, hidden_dim)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, vision_embeds: torch.Tensor) -> torch.Tensor:
        # vision_embeds: [B, num_patches, vision_dim]
        # Pool patches down to target sequence length
        transposed = vision_embeds.transpose(1, 2)  # [B, vision_dim, num_patches]
        pooled = self.pool(transposed).transpose(1, 2)  # [B, num_image_tokens, vision_dim]
        projected = self.linear_2(self.act(self.linear_1(pooled)))
        return projected


class UniversalMultimodalForCausalLM(nn.Module):
    """
    Multimodal Foundation Decoder Model.
    Accepts interleaved sequences containing text token IDs and companion image tensors.
    Projects visual patches and substitutes image token embeddings in-place.
    """
    def __init__(self, args: MultimodalModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.image_token_id = args.image_token_id

        # Language backbone
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.num_hidden_layers)])
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # Vision backbone & multimodal adapter
        self.vision_encoder = VisionPatchEncoder(
            image_size=args.image_size,
            patch_size=args.patch_size,
            vision_dim=args.vision_hidden_size
        )
        self.adapter = VisionLanguageAdapter(
            vision_dim=args.vision_hidden_size,
            hidden_dim=args.hidden_size,
            num_image_tokens=args.num_image_tokens
        )

        if args.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Args:
            input_ids: [B, seq_len] text token sequences with optional image_token_id slots
            pixel_values: Optional [B_img, 3, H, W] companion image pixel tensors
            labels: Optional [B, seq_len] target labels
        """
        # 1. Base text token embeddings
        token_embeds = self.embed_tokens(input_ids)  # [B, seq_len, hidden_size]

        # 2. Inject visual tokens if images are present
        if pixel_values is not None and pixel_values.shape[0] > 0:
            vision_features = self.vision_encoder(pixel_values)  # [B_img, num_patches, vision_dim]
            visual_tokens = self.adapter(vision_features)  # [B_img, num_image_tokens, hidden_size]

            # In-place substitution at image token positions
            b_sz, seq_len, _ = token_embeds.shape
            img_idx = 0
            for b in range(b_sz):
                mask = (input_ids[b] == self.image_token_id)
                num_slots = mask.sum().item()
                if num_slots > 0 and img_idx < visual_tokens.shape[0]:
                    # Fill available slots with visual embeddings
                    v_embed = visual_tokens[img_idx]  # [num_image_tokens, hidden_size]
                    indices = mask.nonzero(as_tuple=True)[0]
                    fill_count = min(len(indices), v_embed.shape[0])
                    token_embeds[b, indices[:fill_count], :] = v_embed[:fill_count, :]
                    img_idx += 1

        # 3. Autoregressive Transformer Decoder
        x = token_embeds
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        # 4. Cross-Entropy Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )

        return {"logits": logits, "loss": loss}
