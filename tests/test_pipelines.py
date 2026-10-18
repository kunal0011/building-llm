"""
End-to-End Test Suite for Training Data Processing Pipelines.
"""

import os
import shutil
import tempfile
import struct
import numpy as np

from schemas.data_schema import (
    RawDocument, FilteredDocument, RepositoryCodebaseDocument,
    RepositoryFile, LongHorizonAgentTrajectory
)
from pipelines.filtering.heuristic_filter import HeuristicQualityFilter
from pipelines.deduplication.minhash_lsh import MinHashLSH
from pipelines.long_context.document_packer import DocumentPacker
from pipelines.agent_synthesis.trajectory_builder import AgentTrajectoryBuilder
from pipelines.tokenization.binary_shard_writer import BinaryShardWriter
from pipelines.tokenization.custom_bpe_tokenizer import UniversalBPETokenizer
from pipelines.crawling.warc_extractor import CommonCrawlWarcExtractor
from pipelines.storage.s3_manager import S3DataManager
from warcio.warcwriter import WARCWriter
from warcio.statusandheaders import StatusAndHeaders


def test_heuristic_filter():
    print("[+] Testing HeuristicQualityFilter...")
    filter_engine = HeuristicQualityFilter(min_words=10)

    # High quality document
    good_text = (
        "The quick brown fox jumps over the lazy dog. Artificial intelligence requires "
        "comprehensive, high-quality pre-training corpora encompassing scientific knowledge, "
        "literature, algorithmic reasoning, and diverse human languages to generalize effectively."
    )
    passed, reason, metrics = filter_engine.evaluate(good_text)
    assert passed, f"Good text failed: {reason}"
    assert metrics["word_count"] > 10

    # Low quality document (excessive repetition)
    repetitive_text = "This is a spam banner.\n" * 20
    passed, reason, _ = filter_engine.evaluate(repetitive_text)
    assert not passed, "Repetitive text should have been rejected"
    assert "repetitive" in reason

    # PII Scrubbing
    pii_text = "Contact researcher at alice.smith@lab.org or call +1-555-123-4567 for dataset access."
    scrubbed = filter_engine.scrub_pii(pii_text)
    assert "<|PII_EMAIL|>" in scrubbed
    assert "<|PII_PHONE|>" in scrubbed
    print("  -> Heuristic filter passed!")


def test_minhash_lsh():
    print("[+] Testing MinHashLSH Near-Deduplication...")
    lsh = MinHashLSH(num_perm=64, num_bands=8, shingle_size=3)

    doc1 = "The solar system consists of the Sun and the astronomical objects gravitationally bound in orbit around it."
    doc2 = "The solar system consists of the Sun and the astronomical objects gravitationally bound in orbit around it today."
    doc3 = "Quantum chromodynamics is the theoretical physics framework describing strong interactions between quarks and gluons."

    sig1 = lsh.compute_minhash_signature(doc1)
    sig2 = lsh.compute_minhash_signature(doc2)
    sig3 = lsh.compute_minhash_signature(doc3)

    sim_1_2 = lsh.jaccard_similarity(sig1, sig2)
    sim_1_3 = lsh.jaccard_similarity(sig1, sig3)

    assert sim_1_2 >= 0.45, f"Near duplicates should have high similarity: {sim_1_2}"
    assert sim_1_3 < 0.2, f"Unrelated docs should have low similarity: {sim_1_3}"

    lsh.insert("doc_1", sig1)
    lsh.insert("doc_2", sig2)
    lsh.insert("doc_3", sig3)

    clusters = lsh.find_clusters()
    assert "doc_1" in clusters or "doc_2" in clusters, "Doc 1 and Doc 2 should be in a duplicate cluster"
    print("  -> MinHash LSH deduplication passed!")


def test_document_packer():
    print("[+] Testing Long-Context DocumentPacker...")
    packer = DocumentPacker(max_seq_len=64, eos_token_id=2, pad_token_id=0)

    # 3 small tokenized documents
    doc_a = [10, 11, 12, 13, 2] # 5 tokens
    doc_b = [20, 21, 22, 23, 24, 25, 2] # 7 tokens
    doc_c = [30, 31, 32, 2] # 4 tokens

    packed = packer.pack_documents([doc_a, doc_b, doc_c], allow_cross_document_attention=False)
    assert len(packed) == 1
    sample = packed[0]
    assert len(sample["input_ids"]) == 64
    assert sample["num_documents"] == 3
    assert len(sample["document_boundaries"]) == 3
    # Check that position IDs reset to 0 at doc boundaries
    assert sample["position_ids"][0] == 0
    boundary_1_start = sample["document_boundaries"][1][0]
    assert sample["position_ids"][boundary_1_start] == 0
    print("  -> Document packer passed!")


def test_trajectory_builder():
    print("[+] Testing AgentTrajectoryBuilder...")
    builder = AgentTrajectoryBuilder()

    trajectory = builder.build_refactoring_trajectory(
        repo_name="org/data-loader",
        issue_description="DataLoader crashes on empty byte streams",
        target_files=["dataloader/reader.py"],
        reproduced_error_log="ValueError: Empty stream encountered",
        diff_patch="--- a/reader.py\n+++ b/reader.py\n@@ -10,1 +10,2 @@\n+ if len(stream) == 0: return b''",
        test_command="pytest tests/test_reader.py"
    )

    assert trajectory["verifiable_reward"] == 1.0
    assert trajectory["total_steps"] >= 5
    assert trajectory["domain"] == "software_engineering_agent"

    checkpoint = builder.build_persistent_horizon_checkpoint(
        session_id="session-9982",
        current_day=3,
        active_goals=["Run regression benchmark across 10k repositories"],
        completed_milestones=["Setup distributed Ray cluster", "Tokenize 1T tokens"],
        scratchpad_state="Active nodes: 64. Memory utilization: 42%."
    )
    assert "Day 3" in checkpoint["input_prompt"]
    assert "session-9982" in checkpoint["target_response"]
    print("  -> Agent trajectory builder passed!")


def test_binary_shard_writer():
    print("[+] Testing BinaryShardWriter (.bin and .idx)...")
    temp_dir = tempfile.mkdtemp()
    prefix = os.path.join(temp_dir, "test_shard_000")

    try:
        with BinaryShardWriter(prefix, dtype=np.uint32) as writer:
            writer.add_document([101, 102, 103, 104, 2])
            writer.add_document([201, 202, 2])

        assert os.path.exists(f"{prefix}.bin")
        assert os.path.exists(f"{prefix}.idx")

        # Verify .bin size (8 uint32 tokens * 4 bytes = 32 bytes)
        bin_size = os.path.getsize(f"{prefix}.bin")
        assert bin_size == 8 * 4, f"Expected 32 bytes, got {bin_size}"

        # Verify .idx header
        with open(f"{prefix}.idx", "rb") as f:
            header = f.read(8)
            assert header == b"MMIDXB\x00\x00"
            version = struct.unpack("<Q", f.read(8))[0]
            assert version == 1
            dtype_code = struct.unpack("<B", f.read(1))[0]
            assert dtype_code == 6 # uint32
            num_docs = struct.unpack("<Q", f.read(8))[0]
            assert num_docs == 2
            total_tokens = struct.unpack("<Q", f.read(8))[0]
            assert total_tokens == 8

        print("  -> Binary shard writer passed!")
    finally:
        shutil.rmtree(temp_dir)


def test_bpe_tokenizer():
    print("[+] Testing UniversalBPETokenizer...")
    corpus = [
        "The quick brown fox jumps over the lazy dog.",
        "Repository refactoring requires deep code dependency analysis.",
        "def compute_gradients(loss, weights): return loss.backward()",
        "<|repo_start|> project_apollo <|file_start|> main.py <|file_end|> <|repo_end|>",
        "<|thought|> Evaluating next step for agent <|thought_end|>",
        "<|image_ref_start|> System diagram <|image_ref_end|>"
    ] * 10

    tokenizer = UniversalBPETokenizer(vocab_size=1000)
    tokenizer.train_from_iterator(iter(corpus), min_frequency=1)

    assert tokenizer.get_vocab_size() >= 100

    test_str = "<|repo_start|> test_repo <|thought|> Analyzing files <|thought_end|>"
    token_ids = tokenizer.encode(test_str, add_special_tokens=True)
    assert token_ids[0] == tokenizer.token_to_id("<|begin_of_text|>")
    assert token_ids[-1] == tokenizer.token_to_id("<|end_of_text|>")

    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    assert "<|repo_start|>" in decoded
    assert "<|thought|>" in decoded
    print("  -> Custom BPE Tokenizer passed!")


def test_warc_extractor():
    print("[+] Testing CommonCrawlWarcExtractor...")
    import io
    custom_filter = HeuristicQualityFilter(min_words=20)
    extractor = CommonCrawlWarcExtractor(heuristic_filter=custom_filter, min_doc_length=50)

    out_buf = io.BytesIO()
    writer = WARCWriter(out_buf, gzip=True)

    html_content = (
        b"<html><head><title>Research Article</title></head><body>"
        b"<h1>Distributed AI Systems</h1>"
        b"<p>Artificial intelligence pre-training demands reliable, high-throughput data processing pipelines. "
        b"This paper surveys data extraction architectures, deduplication methods, and distributed storage across "
        b"scientific literature, books, code, and worldwide knowledge sources to generalize effectively.</p>"
        b'<img src="https://example.org/diagram.jpg" alt="System Architecture Diagram">'
        b"</body></html>"
    )

    headers = StatusAndHeaders("200", [
        ("Content-Type", "text/html; charset=utf-8"),
        ("Content-Length", str(len(html_content)))
    ], protocol="HTTP/1.1")

    record = writer.create_warc_record(
        uri="https://example.org/article.html",
        record_type="response",
        payload=io.BytesIO(html_content),
        http_headers=headers
    )
    writer.write_record(record)
    out_buf.seek(0)

    extracted = list(extractor.extract_records_from_stream(out_buf))
    assert len(extracted) == 1, f"Expected 1 extracted record, got {len(extracted)}"
    doc = extracted[0]
    assert "Artificial intelligence" in doc["text"]
    assert len(doc["media_refs"]) == 1
    assert doc["media_refs"][0]["alt_text"] == "System Architecture Diagram"
    print("  -> WARC extractor passed!")


def test_s3_manager():
    print("[+] Testing S3DataManager...")
    mgr = S3DataManager(bucket_name="commoncrawl", unsigned_access=True)
    assert mgr.s3_client is not None
    assert mgr.bucket_name == "commoncrawl"
    print("  -> S3 Data Manager passed!")


if __name__ == "__main__":
    test_heuristic_filter()
    test_minhash_lsh()
    test_document_packer()
    test_trajectory_builder()
    test_binary_shard_writer()
    test_bpe_tokenizer()
    test_warc_extractor()
    test_s3_manager()
    print("\n[SUCCESS] All pipeline modules verified successfully!")
