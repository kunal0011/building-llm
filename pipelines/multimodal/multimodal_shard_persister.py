"""
Multimodal Memory-Mapped Binary Shard & Image Manifest Persister.
Writes Megatron-LM .bin/.idx binary token shards for interleaved sequences,
generates companion JSON manifests mapping token spans to visual assets, and uploads to S3.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from pipelines.tokenization.binary_shard_writer import BinaryShardWriter
from pipelines.dataloader.mmap_dataset import MMapIndexedDataset
from pipelines.storage.s3_manager import S3DataManager


class MultimodalShardPersister:
    """
    Persists interleaved multimodal token sequences into memory-mapped
    binary arrays (.bin/.idx) and uploads companion image manifests to S3.
    """

    def __init__(
        self,
        output_dir: str = "./data/multimodal_shards",
        s3_bucket: str = "ai-data-shards",
        s3_endpoint: Optional[str] = None
    ):
        self.output_dir = output_dir
        self.s3_bucket = s3_bucket
        self.s3_endpoint = s3_endpoint
        os.makedirs(self.output_dir, exist_ok=True)

    def write_shard(
        self,
        shard_id: int,
        packed_sequences: List[Dict[str, Any]],
        document_manifests: List[Dict[str, Any]],
        upload_to_s3: bool = True
    ) -> Dict[str, Any]:
        """
        Writes .bin and .idx shards locally, validates via MMap, and streams to S3.
        """
        shard_prefix = os.path.join(self.output_dir, f"shard_mm_{shard_id:05d}")

        # 1. Write binary token arrays (.bin/.idx)
        with BinaryShardWriter(shard_prefix, dtype=np.uint32) as writer:
            for seq in packed_sequences:
                writer.add_document(seq["input_ids"])

        bin_path = f"{shard_prefix}.bin"
        idx_path = f"{shard_prefix}.idx"
        manifest_path = f"{shard_prefix}_manifest.json"

        # 2. Write companion multimodal visual manifest JSON
        with open(manifest_path, "w") as f:
            json.dump({
                "shard_id": shard_id,
                "total_sequences": len(packed_sequences),
                "total_tokens": writer.total_tokens,
                "documents": document_manifests
            }, f, indent=2)

        bin_size = os.path.getsize(bin_path)
        idx_size = os.path.getsize(idx_path)
        manifest_size = os.path.getsize(manifest_path)

        # 3. Verify zero-copy DMA access
        ds = MMapIndexedDataset(shard_prefix)
        assert len(ds) == len(packed_sequences), f"Mismatch: {len(ds)} vs {len(packed_sequences)}"
        ds.close()

        # 4. Stream to S3
        uploaded_keys = []
        if upload_to_s3:
            try:
                s3_mgr = S3DataManager(bucket_name=self.s3_bucket, endpoint_url=self.s3_endpoint)
                s3_mgr.ensure_bucket_exists()

                bin_key = f"multimodal/shards/shard_mm_{shard_id:05d}.bin"
                idx_key = f"multimodal/shards/shard_mm_{shard_id:05d}.idx"
                man_key = f"multimodal/shards/shard_mm_{shard_id:05d}_manifest.json"

                u1 = s3_mgr.upload_file_multipart(bin_path, bin_key)
                u2 = s3_mgr.upload_file_multipart(idx_path, idx_key)
                u3 = s3_mgr.upload_file_multipart(manifest_path, man_key)

                if u1 and u2 and u3:
                    uploaded_keys = [bin_key, idx_key, man_key]
            except Exception as e:
                print(f"[Warning] S3 Multimodal Shard Upload: {e}")

        return {
            "shard_id": shard_id,
            "shard_prefix": shard_prefix,
            "sequences": len(packed_sequences),
            "tokens": writer.total_tokens,
            "bin_size_bytes": bin_size,
            "idx_size_bytes": idx_size,
            "manifest_size_bytes": manifest_size,
            "s3_uploaded_keys": uploaded_keys
        }
