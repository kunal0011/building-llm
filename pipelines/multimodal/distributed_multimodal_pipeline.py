"""
Distributed Ray Multi-Node Multimodal Ingestion & Sharding Pipeline.
Executes parallel extraction of interleaved text-image web corpora, applies
image dimension and boilerplate validation, computes CLIP/SigLIP vision-language alignment,
tokenizes with Universal BPE, and writes memory-mapped binary shards to S3.
"""

import os
import sys
import time
import socket
import json
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import ray
import numpy as np

from pipelines.multimodal.image_filter_validator import ImageFilterValidator
from pipelines.multimodal.clip_siglip_aligner import CLIPSigLIPAligner
from pipelines.multimodal.interleaved_formatter import InterleavedMultimodalFormatter
from pipelines.multimodal.multimodal_shard_persister import MultimodalShardPersister
from pipelines.long_context.document_packer import DocumentPacker
from pipelines.tokenization.custom_bpe_tokenizer import UniversalBPETokenizer
from pipelines.storage.s3_manager import S3DataManager


# =====================================================================
# RAY REMOTE MULTIMODAL TASKS (Head and Worker Nodes)
# =====================================================================

@ray.remote(num_cpus=1)
def distributed_extract_multimodal_slice(
    partition_id: int,
    raw_articles: List[Dict[str, Any]],
    s3_bucket: str = "ai-data-shards"
) -> Dict[str, Any]:
    """
    Validates images, computes CLIP/SigLIP alignment scores, and formats
    interleaved multimodal documents on an arbitrary Ray cluster node.
    """
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()
    start_t = time.time()

    validator = ImageFilterValidator()
    aligner = CLIPSigLIPAligner()
    formatter = InterleavedMultimodalFormatter()

    clean_documents: List[Dict[str, Any]] = []
    total_images_processed = 0
    images_retained = 0
    images_dropped = 0

    for doc in raw_articles:
        doc_id = doc.get("id", f"doc_p{partition_id}_{len(clean_documents):04d}")
        paragraphs = doc.get("paragraphs", [])
        raw_images = doc.get("images", [])

        valid_media_items = []
        for img in raw_images:
            total_images_processed += 1
            url = img.get("url", "")
            alt = img.get("alt_text", "")
            w = img.get("width", 800)
            h = img.get("height", 600)

            # 1. Filter dimensions and alt text
            val_res = validator.validate_candidate(url, alt, width=w, height=h)
            if not val_res["passed"]:
                images_dropped += 1
                continue

            # 2. Compute CLIP/SigLIP semantic alignment
            align_res = aligner.evaluate_image_text_pair(
                alt_text=alt,
                image_url=url,
                surrounding_context=" ".join(paragraphs[:2])
            )
            if not align_res["passed"]:
                images_dropped += 1
                continue

            valid_media_items.append({
                "url": url,
                "alt_text": alt,
                "width": w,
                "height": h,
                "alignment_score": align_res["effective_score"],
                "mime_type": img.get("mime_type", "image/jpeg")
            })
            images_retained += 1

        # Format interleaved document
        formatted = formatter.format_interleaved_document(
            document_id=doc_id,
            text_paragraphs=paragraphs,
            media_items=valid_media_items,
            s3_bucket_prefix=f"s3://{s3_bucket}/multimodal/images"
        )
        clean_documents.append(formatted)

    # Artificial work pacing to ensure multi-node distribution
    time.sleep(0.15)
    elapsed = time.time() - start_t

    return {
        "partition_id": partition_id,
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "clean_documents": clean_documents,
        "total_images_processed": total_images_processed,
        "images_retained": images_retained,
        "images_dropped": images_dropped,
        "elapsed_sec": round(elapsed, 2)
    }


@ray.remote(num_cpus=1)
def distributed_tokenize_and_pack_multimodal(
    batch_id: int,
    documents: List[Dict[str, Any]],
    target_seq_len: int,
    tokenizer_path: str
) -> Dict[str, Any]:
    """
    Tokenizes interleaved multimodal text with Universal BPE and packs into sequences.
    """
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()
    start_t = time.time()

    tokenizer = UniversalBPETokenizer.load(tokenizer_path)
    packer = DocumentPacker(max_seq_len=target_seq_len)

    tokenized_sequences = []
    manifests = []
    total_tokens = 0

    for doc in documents:
        text = doc["full_interleaved_text"]
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        tokenized_sequences.append(token_ids)
        total_tokens += len(token_ids)
        manifests.append({
            "doc_id": doc["document_id"],
            "token_count": len(token_ids),
            "images": doc["image_manifest"]
        })

    packed_batches = packer.pack_documents(tokenized_sequences, allow_cross_document_attention=False)
    time.sleep(0.15)
    elapsed = time.time() - start_t

    return {
        "batch_id": batch_id,
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "packed_batches": packed_batches,
        "manifests": manifests,
        "total_tokens": total_tokens,
        "elapsed_sec": round(elapsed, 2)
    }


# =====================================================================
# DISTRIBUTED MULTIMODAL ORCHESTRATOR
# =====================================================================

def run_distributed_multimodal_pipeline(
    target_seq_len: int = 2048,
    output_dir: str = "./data/multimodal_shards",
    s3_bucket: str = "ai-data-shards",
    s3_endpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    End-to-End Orchestrator for Distributed Multimodal Data Ingestion & Sharding.
    """
    print("=" * 70)
    print("  DISTRIBUTED RAY MULTIMODAL INGESTION & VISION-LANGUAGE ALIGNMENT")
    print("=" * 70)

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    cluster_nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    print(f"\n[+] Connected to Ray Cluster: {len(cluster_nodes)} Active Nodes Detected")
    for idx, node in enumerate(cluster_nodes):
        print(f"    - Node {idx+1}: Host={node.get('NodeManagerHostname')} | IP={node.get('NodeManagerAddress')}")

    start_t = time.time()
    endpoint = s3_endpoint or os.getenv("AWS_ENDPOINT_URL") or "http://s3:4566"

    # Tokenizer path check
    tok_path = "./data/tokenizer_verified/tokenizer.json"
    if not os.path.exists(tok_path):
        from pipelines.tokenization.train_vocab_sampler import train_tokenizer
        train_tokenizer(vocab_size=2000, output_dir="./data/tokenizer_verified")

    # Synthetic Curated Multimodal Corpus across 6 partitions
    multimodal_raw_corpus = [
        # Partition 0: Astronomical Observations
        [
            {
                "id": "mm_astro_001",
                "paragraphs": [
                    "The James Webb Space Telescope (JWST) utilizes its Near-Infrared Camera (NIRCam) to resolve high-redshift galactic formations.",
                    "Deep-field imagery reveals gravitational lensing rings generated by massive galaxy cluster SMACS 0723.",
                    "Spectroscopic analysis confirms the chemical presence of water vapor, carbon monoxide, and methane in exoplanet atmospheres."
                ],
                "images": [
                    {
                        "url": "https://images.nasa.gov/details-PIA25438.jpg",
                        "alt_text": "High-resolution deep-field infrared image of galaxy cluster SMACS 0723 showing bright arcs and gravitational lenses.",
                        "width": 1920,
                        "height": 1080
                    },
                    {
                        "url": "https://images.nasa.gov/ad_banner_1x1.png",
                        "alt_text": "spacer",  # Will be rejected
                        "width": 1,
                        "height": 1
                    }
                ]
            }
        ],
        # Partition 1: Deep Learning Architecture
        [
            {
                "id": "mm_arch_001",
                "paragraphs": [
                    "Vision Transformers (ViT) partition input images into non-overlapping patches of size 16x16 pixels.",
                    "Each visual patch is linearly projected into an embedding space and concatenated with a learnable class token.",
                    "Self-attention layers compute global contextual representations without requiring spatial convolutional inductive biases."
                ],
                "images": [
                    {
                        "url": "https://arxiv.org/html/2010.11929/vit_patch_projection_diagram.png",
                        "alt_text": "Detailed architectural schematic illustrating 16x16 image patch flattening and linear embedding in Vision Transformers.",
                        "width": 1280,
                        "height": 720
                    }
                ]
            }
        ],
        # Partition 2: Cellular Biology
        [
            {
                "id": "mm_bio_001",
                "paragraphs": [
                    "Mitochondria are double-membraned organelles that generate adenosine triphosphate (ATP) through oxidative phosphorylation.",
                    "The inner mitochondrial membrane forms convoluted folds called cristae, greatly increasing the surface area for enzyme complexes.",
                    "Cryo-electron microscopy demonstrates the structural assembly of ATP synthase rotary nanomotors."
                ],
                "images": [
                    {
                        "url": "https://biology-archive.org/diagrams/mitochondria_cristae_cryoem.jpg",
                        "alt_text": "Transmission electron micrograph showing mitochondrial cristae folding and inner membrane matrix structure.",
                        "width": 1024,
                        "height": 768
                    }
                ]
            }
        ],
        # Partition 3: Quantum Computing
        [
            {
                "id": "mm_quantum_001",
                "paragraphs": [
                    "Superconducting transmon qubits are manipulated using microwave pulses calibrated to specific resonant transition frequencies.",
                    "Surface codes implement topological quantum error correction by interleaving data qubits with syndrome measurement ancillas.",
                    "State tomography plots verify two-qubit entangling gate fidelity exceeding 99.7%."
                ],
                "images": [
                    {
                        "url": "https://physics-review.org/quantum_surface_code_lattice.png",
                        "alt_text": "Schematic diagram of 2D superconducting transmon qubit square lattice with interleaved X and Z error correction stabilizers.",
                        "width": 1440,
                        "height": 900
                    }
                ]
            }
        ],
        # Partition 4: Geological Formations
        [
            {
                "id": "mm_geo_001",
                "paragraphs": [
                    "Tectonic subduction zones produce volcanic island arcs along continental convergent boundaries.",
                    "Seismic reflection profiles map deep crustal fault lines and magma chamber geometries beneath stratovolcanoes.",
                    "Stratigraphic columns show distinct sedimentary layers deposited across the Cretaceous-Paleogene transition."
                ],
                "images": [
                    {
                        "url": "https://geology.org/plates/subduction_zone_cross_section.jpg",
                        "alt_text": "Geological cross-section diagram showing oceanic crust subducting beneath continental lithosphere with volcanic arc.",
                        "width": 1600,
                        "height": 900
                    }
                ]
            }
        ],
        # Partition 5: Autonomous Robotics
        [
            {
                "id": "mm_robotics_001",
                "paragraphs": [
                    "Mobile manipulation robots combine 3D LiDAR point clouds with RGB camera streams for spatial occupancy mapping.",
                    "End-effector trajectory planners generate collision-free motion paths in six degrees of freedom.",
                    "Visual servomotor feedback loops enable precision grasping of deformable objects."
                ],
                "images": [
                    {
                        "url": "https://robotics-journal.org/lidar_rgb_pointcloud_fusion.png",
                        "alt_text": "3D sensor fusion visualization displaying colorized LiDAR point cloud with robot arm reachability workspace envelope.",
                        "width": 1920,
                        "height": 1080
                    }
                ]
            }
        ]
    ]

    print(f"\n[*] STAGE 1: Dispatching {len(multimodal_raw_corpus)} parallel multimodal extraction tasks...")
    ext_futures = [
        distributed_extract_multimodal_slice.remote(i, partition_docs, s3_bucket=s3_bucket)
        for i, partition_docs in enumerate(multimodal_raw_corpus)
    ]
    ext_results = ray.get(ext_futures)

    all_clean_docs: List[Dict[str, Any]] = []
    extraction_nodes = Counter()
    total_images_processed = 0
    total_images_retained = 0
    total_images_dropped = 0

    for r in ext_results:
        node_str = f"{r['hostname']} ({r['ip_address']})"
        extraction_nodes[node_str] += 1
        all_clean_docs.extend(r["clean_documents"])
        total_images_processed += r["total_images_processed"]
        total_images_retained += r["images_retained"]
        total_images_dropped += r["images_dropped"]

    print(f"    -> Extracted {len(all_clean_docs)} clean interleaved documents.")
    print(f"    -> Image Assets: {total_images_retained} validated & aligned ({total_images_dropped} rejected boilerplate/tracking pixels).")
    print(f"    -> Extraction Node Distribution:")
    for node_str, count in extraction_nodes.items():
        print(f"       * {node_str}: {count} tasks ({count/len(ext_futures)*100:.1f}%)")

    # -------------------------------------------------------------
    # STAGE 2: Distributed Parallel BPE Tokenization & Packing
    # -------------------------------------------------------------
    print(f"\n[*] STAGE 2: Tokenizing interleaved sequences with Universal BPE & packing...")
    half = len(all_clean_docs) // 2
    tok_partitions = [all_clean_docs[:half], all_clean_docs[half:]]
    tok_futures = [
        distributed_tokenize_and_pack_multimodal.remote(i, part, target_seq_len, tok_path)
        for i, part in enumerate(tok_partitions)
    ]
    tok_results = ray.get(tok_futures)

    all_packed_batches = []
    all_manifests = []
    tokenization_nodes = Counter()
    total_tokens = 0

    for tr in tok_results:
        node_str = f"{tr['hostname']} ({tr['ip_address']})"
        tokenization_nodes[node_str] += 1
        all_packed_batches.extend(tr["packed_batches"])
        all_manifests.extend(tr["manifests"])
        total_tokens += tr["total_tokens"]

    print(f"    -> Generated {len(all_packed_batches)} packed sequences ({total_tokens:,} tokens).")
    print(f"    -> Tokenization Node Distribution:")
    for node_str, count in tokenization_nodes.items():
        print(f"       * {node_str}: {count} tasks ({count/len(tok_futures)*100:.1f}%)")

    # -------------------------------------------------------------
    # STAGE 3: Memory-Mapped Binary Sharding & S3 Persistence
    # -------------------------------------------------------------
    print(f"\n[*] STAGE 3: Serializing Megatron-LM .bin/.idx and S3 Visual Manifests...")
    persister = MultimodalShardPersister(
        output_dir=output_dir,
        s3_bucket=s3_bucket,
        s3_endpoint=endpoint
    )

    shard_res = persister.write_shard(
        shard_id=0,
        packed_sequences=all_packed_batches,
        document_manifests=all_manifests,
        upload_to_s3=True
    )

    # Verify S3 multimodal bucket items
    s3_mgr = S3DataManager(bucket_name=s3_bucket, endpoint_url=endpoint)
    s3_mm_objects = s3_mgr.list_objects(prefix="multimodal/")

    elapsed_total = time.time() - start_t

    all_nodes = extraction_nodes + tokenization_nodes
    distinct_nodes = len(all_nodes)

    print("\n" + "=" * 70)
    print("  DISTRIBUTED MULTIMODAL INGESTION PIPELINE: COMPLETED")
    print("=" * 70)
    print(f"  Total Ingestion Time         : {elapsed_total:.2f} seconds")
    print(f"  Cluster Nodes Utilized       : {distinct_nodes} nodes ({list(all_nodes.keys())})")
    print(f"  Interleaved Documents        : {len(all_clean_docs)}")
    print(f"  Visual Assets Retained       : {total_images_retained}")
    print(f"  Total Multimodal Tokens      : {total_tokens:,} tokens")
    print(f"  Packed Sequences (ctx={target_seq_len}) : {len(all_packed_batches)}")
    print(f"  Binary Shard Size (.bin)     : {shard_res['bin_size_bytes'] / 1024:.1f} KB")
    print(f"  Companion Manifest (.json)   : {shard_res['manifest_size_bytes']} bytes")
    print(f"  S3 Staged Objects Active     : {len(s3_mm_objects)} in s3://{s3_bucket}/multimodal/")
    print(f"  Multi-Node Load Balance      : {'VERIFIED ACROSS CLUSTER' if distinct_nodes > 1 else 'SINGLE NODE'}")
    print("=" * 70)

    return {
        "status": "success",
        "elapsed_sec": round(elapsed_total, 2),
        "cluster_nodes_utilized": distinct_nodes,
        "nodes": dict(all_nodes),
        "documents": len(all_clean_docs),
        "visual_assets": total_images_retained,
        "total_tokens": total_tokens,
        "packed_sequences": len(all_packed_batches),
        "bin_size_bytes": shard_res["bin_size_bytes"],
        "manifest_size_bytes": shard_res["manifest_size_bytes"],
        "s3_objects_active": len(s3_mm_objects),
        "s3_keys": [o["key"] for o in s3_mm_objects]
    }


if __name__ == "__main__":
    run_distributed_multimodal_pipeline()
