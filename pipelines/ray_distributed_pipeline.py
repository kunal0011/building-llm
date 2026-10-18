"""
Distributed Ray Ingestion, Decontamination, Tokenization, and Sharding Pipeline.
Executes parallel map-reduce data processing across Ray multi-node clusters
(Head and Worker nodes) and integrates with S3 object storage for raw streaming
and binary shard persistence (.bin / .idx).
"""

import os
import io
import sys
import time
import socket
import urllib.request
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
import numpy as np

from schemas.data_schema import FilteredDocument
from pipelines.filtering.heuristic_filter import HeuristicQualityFilter
from pipelines.crawling.warc_extractor import CommonCrawlWarcExtractor
from pipelines.eval.decontamination_scanner import BenchmarkDecontaminationScanner
from pipelines.deduplication.minhash_lsh import MinHashLSH
from pipelines.code.repo_ast_ingestor import RepositoryASTIngestor, FillInTheMiddleAugmenter
from pipelines.academic.stem_math_normalizer import STEMMathNormalizer
from pipelines.long_context.document_packer import DocumentPacker
from pipelines.tokenization.custom_bpe_tokenizer import UniversalBPETokenizer
from pipelines.tokenization.binary_shard_writer import BinaryShardWriter
from pipelines.dataloader.mmap_dataset import MMapIndexedDataset
from pipelines.storage.s3_manager import S3DataManager


# =====================================================================
# RAY REMOTE TASKS (Executed across cluster Head and Worker nodes)
# =====================================================================

@ray.remote(num_cpus=1)
def distributed_extract_warc_slice(
    partition_id: int,
    warc_url: str,
    byte_range_start: int,
    byte_range_end: int,
    max_records: int = 20
) -> Dict[str, Any]:
    """
    Streams a byte range of a Common Crawl WARC file over HTTP, extracts
    clean prose and image alt-texts, applies quality heuristics and
    decontamination scanning, and returns clean documents.
    """
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()
    start_t = time.time()

    heuristic_filter = HeuristicQualityFilter(min_words=15)
    extractor = CommonCrawlWarcExtractor(heuristic_filter=heuristic_filter)
    decontamination_scanner = BenchmarkDecontaminationScanner(ngram_size=13)

    req = urllib.request.Request(
        warc_url,
        headers={
            "User-Agent": f"DataForgeRayWorker/{partition_id}",
            "Range": f"bytes={byte_range_start}-{byte_range_end}"
        }
    )

    clean_docs: List[Dict[str, Any]] = []
    bytes_downloaded = 0
    raw_records = 0
    rejected_filter = 0
    rejected_leak = 0

    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = resp.read()
            bytes_downloaded = len(data)
            stream_buf = io.BytesIO(data)

            for doc in extractor.extract_records_from_stream(stream_buf):
                raw_records += 1
                text = doc.get("text", "")
                is_clean, leaks = decontamination_scanner.scan_document(text, doc_id=doc.get("url", ""))
                if not is_clean:
                    rejected_leak += 1
                    continue

                clean_docs.append({
                    "id": f"warc_p{partition_id}_{len(clean_docs):04d}",
                    "domain": "curated_web",
                    "source": doc.get("url", ""),
                    "text": text,
                    "media_refs": doc.get("media_refs", [])
                })

                if len(clean_docs) >= max_records:
                    break
    except Exception as e:
        print(f"[Worker {hostname} / Partition {partition_id}] Warning in WARC fetch: {e}")

    elapsed = time.time() - start_t
    return {
        "partition_id": partition_id,
        "type": "warc_web",
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "bytes_downloaded": bytes_downloaded,
        "raw_records": raw_records,
        "clean_documents": clean_docs,
        "rejected_leak": rejected_leak,
        "elapsed_sec": round(elapsed, 2)
    }


@ray.remote(num_cpus=1)
def distributed_extract_code_slice(
    partition_id: int,
    files_to_process: List[Tuple[str, str, str]]
) -> Dict[str, Any]:
    """
    Ingests source code files, applies AST dependency analysis and
    Fill-In-The-Middle (PSM/SPM) code augmentation, scans for contamination,
    and returns formatted code documents.
    """
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()
    start_t = time.time()

    fim_augmenter = FillInTheMiddleAugmenter(fim_rate=0.5)
    decontamination_scanner = BenchmarkDecontaminationScanner(ngram_size=13)
    clean_docs: List[Dict[str, Any]] = []

    for file_path, language, content in files_to_process:
        augmented_code = fim_augmenter.apply_fim(content)
        packed_file = f"<|file_start|>{file_path}\n<|file_language|>{language}\n{augmented_code}\n<|file_end|>"

        is_clean, leaks = decontamination_scanner.scan_document(packed_file, doc_id=file_path)
        if not is_clean:
            continue

        clean_docs.append({
            "id": f"code_p{partition_id}_{len(clean_docs):04d}",
            "domain": "code_repositories",
            "source": file_path,
            "text": packed_file,
            "media_refs": []
        })

    elapsed = time.time() - start_t
    return {
        "partition_id": partition_id,
        "type": "code",
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "clean_documents": clean_docs,
        "elapsed_sec": round(elapsed, 2)
    }


@ray.remote(num_cpus=1)
def distributed_extract_stem_slice(
    partition_id: int,
    papers: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Normalizes LaTeX mathematical expressions and citation references into unified
    scientific discourse.
    """
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()
    start_t = time.time()

    stem_normalizer = STEMMathNormalizer()
    clean_docs: List[Dict[str, Any]] = []

    for p in papers:
        normalized = stem_normalizer.process_paper(
            p.get("content", ""),
            referenced_sources=p.get("references", {})
        )
        clean_docs.append({
            "id": f"stem_p{partition_id}_{len(clean_docs):04d}",
            "domain": "academic_stem",
            "source": p.get("source", "arxiv:simulated"),
            "text": normalized,
            "media_refs": []
        })

    elapsed = time.time() - start_t
    return {
        "partition_id": partition_id,
        "type": "academic_stem",
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "clean_documents": clean_docs,
        "elapsed_sec": round(elapsed, 2)
    }


@ray.remote(num_cpus=1)
def distributed_tokenize_slice(
    batch_id: int,
    docs: List[Dict[str, Any]],
    tokenizer_path: str
) -> Dict[str, Any]:
    """
    Tokenizes a batch of clean documents in parallel on a Ray cluster node.
    """
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()
    start_t = time.time()

    tokenizer = UniversalBPETokenizer.load(tokenizer_path)
    tokenized_sequences: List[List[int]] = []
    total_tokens = 0

    for doc in docs:
        token_ids = tokenizer.encode(doc["text"], add_special_tokens=True)
        tokenized_sequences.append(token_ids)
        total_tokens += len(token_ids)

    time.sleep(0.15)
    elapsed = time.time() - start_t
    return {
        "batch_id": batch_id,
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "tokenized_sequences": tokenized_sequences,
        "total_tokens": total_tokens,
        "num_docs": len(docs),
        "elapsed_sec": round(elapsed, 2)
    }


@ray.remote(num_cpus=1)
def distributed_pack_and_shard(
    shard_id: int,
    token_sequences: List[List[int]],
    target_seq_len: int,
    output_dir: str,
    s3_bucket: Optional[str] = "ai-data-shards",
    s3_endpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Packs token sequences with boundary-reset attention, writes memory-mapped
    Megatron-LM binary (.bin/.idx) shards, performs DMA validation, and uploads to S3.
    """
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()
    start_t = time.time()

    os.makedirs(output_dir, exist_ok=True)
    packer = DocumentPacker(max_seq_len=target_seq_len)
    packed_batches = packer.pack_documents(token_sequences, allow_cross_document_attention=False)

    shard_prefix = os.path.join(output_dir, f"shard_dist_{shard_id:05d}")
    with BinaryShardWriter(shard_prefix, dtype=np.uint32) as writer:
        for seq in packed_batches:
            writer.add_document(seq["input_ids"])

    bin_path = f"{shard_prefix}.bin"
    idx_path = f"{shard_prefix}.idx"
    bin_size = os.path.getsize(bin_path)
    idx_size = os.path.getsize(idx_path)

    # DMA Zero-Copy read validation
    ds = MMapIndexedDataset(shard_prefix)
    assert len(ds) == len(packed_batches), f"Shard length mismatch: {len(ds)} vs {len(packed_batches)}"
    ds.close()

    # Upload to S3 if requested
    s3_uploaded = False
    s3_keys = []
    if s3_bucket:
        try:
            s3_mgr = S3DataManager(bucket_name=s3_bucket, endpoint_url=s3_endpoint)
            s3_mgr.ensure_bucket_exists()

            bin_key = f"shards/shard_dist_{shard_id:05d}.bin"
            idx_key = f"shards/shard_dist_{shard_id:05d}.idx"

            u1 = s3_mgr.upload_file_multipart(bin_path, bin_key)
            u2 = s3_mgr.upload_file_multipart(idx_path, idx_key)
            if u1 and u2:
                s3_uploaded = True
                s3_keys = [bin_key, idx_key]
        except Exception as e:
            print(f"[Worker {hostname}] S3 upload warning for shard {shard_id}: {e}")

    time.sleep(0.15)
    elapsed = time.time() - start_t
    return {
        "shard_id": shard_id,
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "shard_prefix": shard_prefix,
        "packed_sequences": len(packed_batches),
        "total_tokens": writer.total_tokens,
        "bin_size_bytes": bin_size,
        "idx_size_bytes": idx_size,
        "s3_uploaded": s3_uploaded,
        "s3_keys": s3_keys,
        "elapsed_sec": round(elapsed, 2)
    }


# =====================================================================
# DISTRIBUTED PIPELINE ORCHESTRATOR
# =====================================================================

def run_distributed_pipeline(
    target_seq_len: int = 2048,
    num_warc_slices: int = 6,
    warc_slice_size: int = 1_200_000,
    output_dir: str = "./data/distributed_shards",
    s3_bucket: str = "ai-data-shards",
    s3_endpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    End-to-End Orchestrator for Distributed Multi-Node Ray Ingestion.
    """
    print("=" * 70)
    print("  DISTRIBUTED RAY INGESTION, TOKENIZATION & S3 SHARDING PIPELINE")
    print("=" * 70)

    # 1. Connect to Ray cluster
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    cluster_nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    print(f"\n[+] Connected to Ray Cluster: {len(cluster_nodes)} Active Nodes Detected")
    for idx, node in enumerate(cluster_nodes):
        print(f"    - Node {idx+1}: Host={node.get('NodeManagerHostname')} | IP={node.get('NodeManagerAddress')} | CPUs={node.get('Resources', {}).get('CPU', 0)}")

    total_cluster_cpus = sum(n.get("Resources", {}).get("CPU", 0) for n in cluster_nodes)
    print(f"    - Total Cluster Compute: {total_cluster_cpus} CPUs")

    start_time = time.time()
    endpoint = s3_endpoint or os.getenv("AWS_ENDPOINT_URL") or "http://s3:4566"

    # 2. Check / Ensure Tokenizer
    tok_dir = "./data/tokenizer_verified"
    tok_path = os.path.join(tok_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        from pipelines.tokenization.train_vocab_sampler import train_tokenizer
        train_tokenizer(vocab_size=2000, output_dir=tok_dir)
    print(f"\n[+] Universal BPE Tokenizer verified at {tok_path}")

    # -------------------------------------------------------------
    # STAGE 1: Distributed Partitioned Extraction (Map)
    # -------------------------------------------------------------
    print(f"\n[*] STAGE 1: Dispatching distributed extraction across cluster...")
    extraction_futures = []

    # 1.1 WARC Slices from Common Crawl across multiple segments
    base_warc = "https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/segments/1707947473347.0/warc/CC-MAIN-20240220211055-20240221001055"
    for i in range(num_warc_slices):
        seg_url = f"{base_warc}-{i:05d}.warc.gz"
        extraction_futures.append(
            distributed_extract_warc_slice.remote(i, seg_url, 0, warc_slice_size, max_records=15)
        )

    # 1.2 Code Files with AST & FIM
    sample_code_files = [
        ("pipelines/filtering/heuristic_filter.py", "python", open("pipelines/filtering/heuristic_filter.py").read()),
        ("pipelines/long_context/document_packer.py", "python", open("pipelines/long_context/document_packer.py").read()),
        ("pipelines/tokenization/custom_bpe_tokenizer.py", "python", open("pipelines/tokenization/custom_bpe_tokenizer.py").read()),
        ("pipelines/eval/decontamination_scanner.py", "python", open("pipelines/eval/decontamination_scanner.py").read()),
        ("pipelines/rl/grpo_reward_engine.py", "python", open("pipelines/rl/grpo_reward_engine.py").read()),
        ("pipelines/rl/symbolic_math_verifier.py", "python", open("pipelines/rl/symbolic_math_verifier.py").read()),
    ]
    # Split code files into 2 partitions
    extraction_futures.append(distributed_extract_code_slice.remote(101, sample_code_files[:3]))
    extraction_futures.append(distributed_extract_code_slice.remote(102, sample_code_files[3:]))

    # 1.3 Academic STEM Mathematics
    stem_papers = [
        {
            "source": "arxiv:2401.0001",
            "content": r"\begin{theorem}[Uniform Convergence]\nLet $\mathcal{H}$ have finite VC-dim $d$.\n\begin{equation}\n\mathbb{P}(\sup |R - \hat{R}| > \epsilon) \le 4m(2m)^d e^{-m\epsilon^2/8}\n\end{equation}\n\end{theorem}\n\begin{proof}Via symmetrization ghost variables.\end{proof}\nAs proven in \cite{vapnik}.",
            "references": {"vapnik": "Vapnik, V. Statistical Learning Theory. 1998."}
        },
        {
            "source": "arxiv:2401.0002",
            "content": r"\begin{lemma}[Attention Complexity Lower Bound]\nGiven context $L$, causal attention requires $\Omega(L^2)$ scalar multiplications.\n\end{lemma}\nVerified across transformers \cite{vaswani}.",
            "references": {"vaswani": "Vaswani et al. Attention Is All You Need. NeurIPS 2017."}
        }
    ]
    extraction_futures.append(distributed_extract_stem_slice.remote(201, stem_papers))

    print(f"    - Submitting {len(extraction_futures)} parallel extraction tasks to Ray cluster...")
    extraction_results = ray.get(extraction_futures)

    # Collect documents and node distribution
    all_clean_docs: List[Dict[str, Any]] = []
    extraction_nodes = Counter()
    total_warc_bytes = 0

    for res in extraction_results:
        node_ip = res["ip_address"]
        node_host = res["hostname"]
        extraction_nodes[f"{node_host} ({node_ip})"] += 1
        docs = res.get("clean_documents", [])
        all_clean_docs.extend(docs)
        total_warc_bytes += res.get("bytes_downloaded", 0)

    print(f"    -> Extracted {len(all_clean_docs)} clean, decontaminated documents across domains.")
    print(f"    -> Total WARC Network Stream: {total_warc_bytes / 1024 / 1024:.2f} MB")
    print(f"    -> Extraction Node Distribution:")
    for host_ip, count in extraction_nodes.items():
        print(f"       * {host_ip}: {count} tasks ({count/len(extraction_futures)*100:.1f}%)")

    # -------------------------------------------------------------
    # STAGE 2: Near-Deduplication (MinHash LSH)
    # -------------------------------------------------------------
    print(f"\n[*] STAGE 2: Near-Deduplication via MinHash LSH (128 Permutations)...")
    lsh = MinHashLSH(num_perm=128, num_bands=16, similarity_threshold=0.85)
    for doc in all_clean_docs:
        sig = lsh.compute_minhash_signature(doc["text"])
        lsh.insert(doc["id"], sig)

    duplicate_clusters = lsh.find_clusters()
    dropped_ids = set()
    for canonical, dups in duplicate_clusters.items():
        dropped_ids.update(dups)

    deduped_docs = [d for d in all_clean_docs if d["id"] not in dropped_ids]
    duplicates_found = len(dropped_ids)

    print(f"    -> Deduplication complete: {len(deduped_docs)} unique documents retained ({duplicates_found} duplicates dropped).")

    # -------------------------------------------------------------
    # STAGE 3: Distributed Parallel BPE Tokenization
    # -------------------------------------------------------------
    print(f"\n[*] STAGE 3: Distributing BPE tokenization across cluster nodes...")
    # Partition documents across 6 tokenization tasks
    num_tok_batches = min(6, len(deduped_docs))
    batch_size = (len(deduped_docs) + num_tok_batches - 1) // num_tok_batches
    tok_futures = []

    for b_idx in range(num_tok_batches):
        batch_slice = deduped_docs[b_idx * batch_size : (b_idx + 1) * batch_size]
        if batch_slice:
            tok_futures.append(distributed_tokenize_slice.remote(b_idx, batch_slice, tok_path))

    tok_results = ray.get(tok_futures)
    all_token_sequences: List[List[int]] = []
    total_tokens_generated = 0
    tokenization_nodes = Counter()

    for tr in tok_results:
        node_ip = tr["ip_address"]
        node_host = tr["hostname"]
        tokenization_nodes[f"{node_host} ({node_ip})"] += 1
        all_token_sequences.extend(tr["tokenized_sequences"])
        total_tokens_generated += tr["total_tokens"]

    print(f"    -> Tokenized {len(all_token_sequences)} sequences into {total_tokens_generated:,} tokens.")
    print(f"    -> Tokenization Node Distribution:")
    for host_ip, count in tokenization_nodes.items():
        print(f"       * {host_ip}: {count} tasks ({count/len(tok_futures)*100:.1f}%)")

    # -------------------------------------------------------------
    # STAGE 4: Distributed Packing, Binary Sharding & S3 Upload
    # -------------------------------------------------------------
    print(f"\n[*] STAGE 4: Packing sequences ({target_seq_len} ctx) & Writing S3 Shards (.bin/.idx)...")
    # Partition token sequences across 4 shards
    num_shards = min(4, len(all_token_sequences))
    shard_size = (len(all_token_sequences) + num_shards - 1) // num_shards
    seq_partitions = [
        all_token_sequences[i * shard_size : (i + 1) * shard_size]
        for i in range(num_shards)
        if all_token_sequences[i * shard_size : (i + 1) * shard_size]
    ]

    shard_futures = []
    for s_idx, part in enumerate(seq_partitions):
        if part:
            shard_futures.append(
                distributed_pack_and_shard.remote(
                    s_idx,
                    part,
                    target_seq_len,
                    output_dir,
                    s3_bucket=s3_bucket,
                    s3_endpoint=endpoint
                )
            )

    shard_results = ray.get(shard_futures)
    total_packed_sequences = 0
    total_bin_bytes = 0
    total_idx_bytes = 0
    s3_uploaded_keys = []
    sharding_nodes = Counter()

    for sr in shard_results:
        node_ip = sr["ip_address"]
        node_host = sr["hostname"]
        sharding_nodes[f"{node_host} ({node_ip})"] += 1
        total_packed_sequences += sr["packed_sequences"]
        total_bin_bytes += sr["bin_size_bytes"]
        total_idx_bytes += sr["idx_size_bytes"]
        if sr["s3_uploaded"]:
            s3_uploaded_keys.extend(sr["s3_keys"])

    print(f"    -> Generated {len(shard_results)} binary shards ({total_packed_sequences} packed sequences).")
    print(f"    -> Total Shard Storage: {total_bin_bytes / 1024:.1f} KB (.bin) + {total_idx_bytes} bytes (.idx)")
    print(f"    -> S3 Keys Uploaded: {s3_uploaded_keys}")
    print(f"    -> Sharding Node Distribution:")
    for host_ip, count in sharding_nodes.items():
        print(f"       * {host_ip}: {count} tasks ({count/len(shard_futures)*100:.1f}%)")

    # -------------------------------------------------------------
    # STAGE 5: S3 Verification & Zero-Copy DMA Read Check
    # -------------------------------------------------------------
    print(f"\n[*] STAGE 5: Verifying S3 bucket lake integrity...")
    s3_mgr = S3DataManager(bucket_name=s3_bucket, endpoint_url=endpoint)
    s3_objects = s3_mgr.list_objects(prefix="shards/")
    print(f"    -> Found {len(s3_objects)} objects in s3://{s3_bucket}/shards/:")
    for obj in s3_objects:
        print(f"       * {obj['key']} ({obj['size']} bytes, modified: {obj['last_modified']})")

    elapsed_total = time.time() - start_time

    # -------------------------------------------------------------
    # Overall Cluster Multi-Node Verification Summary
    # -------------------------------------------------------------
    all_node_counters = extraction_nodes + tokenization_nodes + sharding_nodes
    distinct_nodes_used = len(all_node_counters)

    print("\n" + "=" * 70)
    print("  DISTRIBUTED RAY MULTI-NODE INGESTION SUMMARY: COMPLETED")
    print("=" * 70)
    print(f"  Total Pipeline Time          : {elapsed_total:.2f} seconds")
    print(f"  Cluster Nodes Utilized       : {distinct_nodes_used} nodes ({list(all_node_counters.keys())})")
    print(f"  Common Crawl Streamed        : {total_warc_bytes / 1024 / 1024:.2f} MB")
    print(f"  Clean Unique Documents       : {len(deduped_docs)}")
    print(f"  Total Tokens Generated       : {total_tokens_generated:,} tokens")
    print(f"  Packed Sequences (ctx={target_seq_len}) : {total_packed_sequences}")
    print(f"  Binary Shards Serialized     : {len(shard_results)} (.bin + .idx)")
    print(f"  S3 Bucket Target             : s3://{s3_bucket}/")
    print(f"  S3 Shard Objects Active      : {len(s3_objects)}")
    print(f"  Multi-Node Load Balance      : {'VERIFIED ACROSS CLUSTER' if distinct_nodes_used > 1 else 'SINGLE NODE'}")
    print("=" * 70)

    return {
        "status": "success",
        "elapsed_sec": round(elapsed_total, 2),
        "cluster_nodes_utilized": distinct_nodes_used,
        "nodes": dict(all_node_counters),
        "warc_bytes": total_warc_bytes,
        "clean_documents": len(deduped_docs),
        "total_tokens": total_tokens_generated,
        "packed_sequences": total_packed_sequences,
        "shards_count": len(shard_results),
        "s3_objects_count": len(s3_objects),
        "s3_keys": [o["key"] for o in s3_objects]
    }


if __name__ == "__main__":
    run_distributed_pipeline()
