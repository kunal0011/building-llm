"""
Unit and Integration Tests for Distributed Ray Ingestion Pipeline and S3 Data Lake.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipelines.storage.s3_manager import S3DataManager
from pipelines.deduplication.minhash_lsh import MinHashLSH
from pipelines.tokenization.binary_shard_writer import BinaryShardWriter
from pipelines.dataloader.mmap_dataset import MMapIndexedDataset
from pipelines.ray_distributed_pipeline import run_distributed_pipeline


def test_s3_data_manager_local_emulator():
    """Validates local S3 emulator operations: create bucket, upload bytes, list objects."""
    endpoint = os.getenv("AWS_ENDPOINT_URL") or "http://localhost:4566"
    bucket = "ai-test-bucket"

    s3_mgr = S3DataManager(bucket_name=bucket, endpoint_url=endpoint)
    created = s3_mgr.ensure_bucket_exists()
    assert created is True

    test_bytes = b"Universal LLM pretraining distributed shard content"
    uploaded = s3_mgr.upload_bytes(test_bytes, "test_shard.bin")
    assert uploaded is True

    objects = s3_mgr.list_objects()
    keys = [o["key"] for o in objects]
    assert "test_shard.bin" in keys


def test_minhash_deduplication_cluster_partitioning():
    """Validates MinHash deduplication correctly drops near-duplicate documents across partitions."""
    lsh = MinHashLSH(num_perm=128, num_bands=16, shingle_size=3, similarity_threshold=0.80)

    doc_a = "Advanced multi-agent reasoning models require massive scale pretraining corpora across all world domains and verifiable step rewards for long persistent horizons."
    doc_b = "Advanced multi-agent reasoning models require massive scale pretraining corpora across all world domains and verifiable step rewards for long persistent horizons!" # 99% identical near duplicate
    doc_c = "Differential equations and Riemannian geometry form the foundation of general relativity in theoretical physics." # completely distinct

    sig_a = lsh.compute_minhash_signature(doc_a)
    sig_b = lsh.compute_minhash_signature(doc_b)
    sig_c = lsh.compute_minhash_signature(doc_c)

    lsh.insert("doc_a", sig_a)
    lsh.insert("doc_b", sig_b)
    lsh.insert("doc_c", sig_c)

    clusters = lsh.find_clusters()
    # doc_a and doc_b should cluster together
    all_duplicates = set()
    for dups in clusters.values():
        all_duplicates.update(dups)

    assert len(all_duplicates) >= 1
    assert "doc_c" not in all_duplicates


def test_distributed_shard_serialization_and_dma():
    """Validates binary shard serialization and zero-copy memmap DMA read."""
    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "test_shard")
        writer = BinaryShardWriter(prefix, dtype=np.uint32)

        # Add 10 synthetic token sequences of length 64
        for i in range(10):
            seq = [1000 + i * 10 + j for j in range(64)]
            writer.add_document(seq)
        writer.close()

        assert os.path.exists(f"{prefix}.bin")
        assert os.path.exists(f"{prefix}.idx")

        # Load with MMapIndexedDataset
        ds = MMapIndexedDataset(prefix)
        assert len(ds) == 10
        assert len(ds[0]) == 64
        assert ds[0][0] == 1000
        ds.close()
