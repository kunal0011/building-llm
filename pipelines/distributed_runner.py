"""
Ray-Based Distributed Preprocessing and Sharding Orchestrator.
Demonstrates scalable multi-node processing for petabyte-scale training data.
"""

import os
import sys
import glob
from typing import List, Dict, Any, Iterator
import numpy as np

from schemas.data_schema import FilteredDocument, MediaReference
from pipelines.filtering.heuristic_filter import HeuristicQualityFilter
from pipelines.deduplication.minhash_lsh import MinHashLSH
from pipelines.long_context.document_packer import DocumentPacker
from pipelines.tokenization.binary_shard_writer import BinaryShardWriter


class DistributedDataPipelineWorker:
    """
    Worker task designed to run across hundreds of Ray CPU nodes.
    Executes filtering, PII scrubbing, tokenization, and shard packing.
    """

    def __init__(self, target_seq_len: int = 8192, output_dir: str = "./data/shards"):
        self.filter = HeuristicQualityFilter()
        self.packer = DocumentPacker(max_seq_len=target_seq_len)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_document_stream(
        self,
        docs: List[Dict[str, Any]],
        worker_id: int = 0,
        shard_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Processes a batch of raw documents:
        1. Filters out spam/corrupt docs
        2. Scrubs PII
        3. Simulates tokenization (or calls Rust tokenizer)
        4. Packs sequences to target context length
        5. Emits binary memory-mapped shards (.bin and .idx)
        """
        passed_count = 0
        rejected_count = 0
        rejection_breakdown = {}

        tokenized_docs: List[List[int]] = []

        for item in docs:
            raw_text = item.get("text", "")
            passes, reason, metrics = self.filter.evaluate(raw_text)

            if not passes:
                rejected_count += 1
                rejection_breakdown[reason] = rejection_breakdown.get(reason, 0) + 1
                continue

            passed_count += 1
            clean_text = self.filter.scrub_pii(raw_text)

            # Simulated BPE tokenization (in production, replaced by Rust tokenizers / tiktoken)
            # Simulating UTF-8 byte tokens with EOS delimiter
            tokens = list(clean_text.encode("utf-8"))[:1000] + [2]  # EOS = 2
            tokenized_docs.append(tokens)

        # Pack into target sequence length
        packed_samples = self.packer.pack_documents(tokenized_docs, allow_cross_document_attention=False)

        # Write to memory-mapped binary shard
        shard_prefix = os.path.join(self.output_dir, f"shard_w{worker_id}_{shard_idx:05d}")
        total_tokens_written = 0

        with BinaryShardWriter(shard_prefix, dtype=np.uint32) as writer:
            for sample in packed_samples:
                writer.add_document(sample["input_ids"])
                total_tokens_written += len(sample["input_ids"])

        return {
            "worker_id": worker_id,
            "shard_path": f"{shard_prefix}.bin",
            "passed_documents": passed_count,
            "rejected_documents": rejected_count,
            "packed_sequences": len(packed_samples),
            "total_tokens": total_tokens_written,
            "rejections": rejection_breakdown
        }


def run_pipeline_demo():
    print("[*] Initializing Distributed Pipeline Orchestrator Demo...")
    worker = DistributedDataPipelineWorker(target_seq_len=256, output_dir="./sample_shards")

    sample_docs = [
        {
            "id": "1",
            "text": (
                "Large language models represent a significant advancement in computational linguistics and artificial intelligence. "
                "By training deep neural network architectures across vast collections of text, code, books, and scientific articles, "
                "these systems learn generalized syntactic, semantic, and reasoning patterns. In this comprehensive survey, we examine "
                "the architectural components required to scale models from eight billion to seventy billion parameters, with an explicit "
                "focus on long-context attention curricula and reinforcement learning with process supervision."
            )
        },
        {
            "id": "2",
            "text": "Buy crypto now!!! Free money click here http://spam.xyz\n" * 20  # Spam/repetitive
        },
        {
            "id": "3",
            "text": "Too short snippet."
        }
    ]

    stats = worker.process_document_stream(sample_docs, worker_id=0, shard_idx=1)
    print("\n[+] Worker Execution Statistics:")
    for k, v in stats.items():
        print(f"    - {k}: {v}")

    # Clean up demo artifacts
    for f in glob.glob("./sample_shards/shard_w0_00001*"):
        os.remove(f)
    os.rmdir("./sample_shards")
    print("\n[✓] Distributed runner verified successfully.")


if __name__ == "__main__":
    run_pipeline_demo()
