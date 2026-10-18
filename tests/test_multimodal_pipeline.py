"""
Unit and Integration Tests for Multimodal Data Fusion Track.
Validates image filters, CLIP/SigLIP alignment scoring, interleaved formatting, and binary sharding.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.multimodal.image_filter_validator import ImageFilterValidator
from pipelines.multimodal.clip_siglip_aligner import CLIPSigLIPAligner
from pipelines.multimodal.interleaved_formatter import InterleavedMultimodalFormatter
from pipelines.multimodal.multimodal_shard_persister import MultimodalShardPersister
from pipelines.dataloader.mmap_dataset import MMapIndexedDataset


def test_image_filter_validator():
    """Validates URL schemes, dimension bounds, aspect ratios, and boilerplate alt texts."""
    val = ImageFilterValidator()

    # URL Checks
    ok_url, _ = val.validate_url("https://cdn.example.org/images/astronomy/jwst_deep_field.jpg")
    bad_url_ext, _ = val.validate_url("https://cdn.example.org/files/manual.pdf")
    tracking_url, _ = val.validate_url("https://cdn.example.org/tracking/pixel.png")

    assert ok_url is True
    assert bad_url_ext is False
    assert tracking_url is False

    # Dimension Checks
    ok_dim, _ = val.validate_dimensions(1920, 1080)
    tiny_dim, _ = val.validate_dimensions(1, 1)  # tracking pixel
    extreme_aspect, _ = val.validate_dimensions(3000, 100)  # aspect ratio 30.0

    assert ok_dim is True
    assert tiny_dim is False
    assert extreme_aspect is False

    # Alt-text checks
    ok_alt, _ = val.validate_alt_text("High-resolution infrared telescope view of star-forming nebula")
    generic_alt, _ = val.validate_alt_text("image")
    filename_alt, _ = val.validate_alt_text("DSC_00492.JPG")

    assert ok_alt is True
    assert generic_alt is False
    assert filename_alt is False


def test_clip_siglip_semantic_alignment():
    """Validates dual-encoder cosine similarity and surrounding context boosting."""
    aligner = CLIPSigLIPAligner(alignment_threshold=0.24, high_quality_threshold=0.28)

    url = "https://example.org/transformer_architecture.png"
    good_alt = "Detailed architectural schematic illustrating multi-head attention mechanisms and residual connections."
    bad_alt = "Red apple sitting on top of wooden kitchen counter with fruit bowl."
    context = "Self-attention computes pairwise token compatibility scores using query and key projections."

    # Good match
    res_good = aligner.evaluate_image_text_pair(good_alt, url, surrounding_context=context)
    assert res_good["passed"] is True
    assert res_good["effective_score"] >= aligner.alignment_threshold

    # Mismatched pair
    res_bad = aligner.evaluate_image_text_pair(bad_alt, url, surrounding_context="")
    assert res_bad["effective_score"] < res_good["effective_score"]


def test_interleaved_multimodal_formatter():
    """Validates interleaved syntax generation and image manifest compilation."""
    formatter = InterleavedMultimodalFormatter()

    paragraphs = [
        "First section introduces convolutional neural networks and feature extraction.",
        "Second section analyzes spatial kernel convolutions across multi-channel feature maps.",
        "Third section summarizes classification accuracy benchmarks on ImageNet."
    ]
    media_items = [
        {
            "url": "https://example.org/cnn_filters.png",
            "alt_text": "Feature map visualization across convolution layers.",
            "width": 1024,
            "height": 768,
            "alignment_score": 0.35
        }
    ]

    doc = formatter.format_interleaved_document(
        document_id="doc_test_001",
        text_paragraphs=paragraphs,
        media_items=media_items,
        s3_bucket_prefix="s3://ai-data-shards/multimodal/images"
    )

    assert doc["total_images"] == 1
    assert "<|image_ref_start|>" in doc["full_interleaved_text"]
    assert "<|image_ref_end|>" in doc["full_interleaved_text"]
    assert "align=\"0.350\"" in doc["full_interleaved_text"]
    assert len(doc["image_manifest"]) == 1
    assert doc["image_manifest"][0]["width"] == 1024


def test_multimodal_shard_persistence_and_mmap():
    """Validates binary sharding for multimodal token batches and companion visual manifests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        persister = MultimodalShardPersister(output_dir=tmpdir, s3_bucket="ai-data-shards")

        # Fake packed sequences containing multimodal token IDs
        fake_packed = [
            {"input_ids": [1, 58, 100, 101, 102, 59, 2] + [0] * 57},  # 58/59 = <|image_ref_start/end|>
            {"input_ids": [1, 200, 201, 202, 2] + [0] * 59}
        ]
        manifests = [{"doc_id": "doc_01", "images": [{"image_id": "img_01"}]}]

        res = persister.write_shard(
            shard_id=99,
            packed_sequences=fake_packed,
            document_manifests=manifests,
            upload_to_s3=False
        )

        assert res["sequences"] == 2
        assert os.path.exists(f"{res['shard_prefix']}.bin")
        assert os.path.exists(f"{res['shard_prefix']}.idx")
        assert os.path.exists(f"{res['shard_prefix']}_manifest.json")

        # Verify zero-copy DMA access
        ds = MMapIndexedDataset(res["shard_prefix"])
        assert len(ds) == 2
        assert ds[0][1] == 58  # image_ref_start token
        ds.close()
