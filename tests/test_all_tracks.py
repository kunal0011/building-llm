"""
Comprehensive Integration Test Suite for Tracks 1-4:
Track 1: Repository AST Ingestion & FIM Augmentation
Track 2: Benchmark De-Contamination Scanner
Track 3: STEM & Mathematical Paper Normalizer
Track 4: Zero-Copy MMap Dataset & Curriculum Batcher
"""

import os
import shutil
import tempfile
import numpy as np

from pipelines.code.repo_ast_ingestor import RepositoryASTIngestor, FillInTheMiddleAugmenter
from pipelines.eval.decontamination_scanner import BenchmarkDecontaminationScanner
from pipelines.academic.stem_math_normalizer import STEMMathNormalizer
from pipelines.dataloader.mmap_dataset import MMapIndexedDataset, CurriculumMultiDomainBatcher
from pipelines.tokenization.binary_shard_writer import BinaryShardWriter


def test_track1_repo_ast_and_fim():
    print("[+] Testing Track 1: Repository AST Ingestor & FIM Augmenter...")
    temp_repo = tempfile.mkdtemp()
    try:
        # Create a sample multi-file Python repository with dependencies
        os.makedirs(os.path.join(temp_repo, "src"), exist_ok=True)

        # File A: utils.py (Depth 0 - no in-repo deps)
        with open(os.path.join(temp_repo, "src", "utils.py"), "w") as f:
            f.write("def helper(): return 42\n")

        # File B: core.py (Imports utils -> Depth 1)
        with open(os.path.join(temp_repo, "src", "core.py"), "w") as f:
            f.write("from src import utils\ndef run(): return utils.helper()\n")

        # File C: README.md
        with open(os.path.join(temp_repo, "README.md"), "w") as f:
            f.write("# Demo Repo\nRepository documentation.")

        ingestor = RepositoryASTIngestor()
        repo_doc = ingestor.ingest_local_directory(temp_repo, repo_name="demo_repo")

        assert repo_doc.repo_name == "demo_repo"
        assert len(repo_doc.file_hierarchy) == 3
        assert "Demo Repo" in repo_doc.readme_content

        serialized = repo_doc.serialize_packed_repo()
        assert "<|repo_start|>demo_repo" in serialized
        assert "<|file_start|>src/utils.py" in serialized
        assert "<|repo_end|>" in serialized

        # Test Fill-In-The-Middle
        fim = FillInTheMiddleAugmenter(fim_rate=1.0)
        code_sample = "line 1\nline 2\nline 3\nline 4\nline 5\n"
        fim_transformed = fim.apply_fim(code_sample)
        assert ("<|fim_prefix|>" in fim_transformed and
                "<|fim_middle|>" in fim_transformed and
                "<|fim_suffix|>" in fim_transformed)
        print("  -> Track 1 (AST Ingestion & FIM) verified successfully!")
    finally:
        shutil.rmtree(temp_repo)


def test_track2_decontamination_scanner():
    print("[+] Testing Track 2: Benchmark Decontamination Scanner...")
    scanner = BenchmarkDecontaminationScanner(ngram_size=13)

    # Clean document
    clean_text = (
        "Distributed computing frameworks leverage consensus algorithms like Raft and Paxos "
        "to maintain replicated state machines across clusters with Byzantine fault tolerance."
    )
    is_clean, detections = scanner.scan_document(clean_text, doc_id="doc_001")
    assert is_clean, f"Clean text was falsely flagged: {detections}"
    assert len(detections) == 0

    # Contaminated document (leaking GSM8K prompt)
    contaminated_text = (
        "Here is a common math problem: Janet's ducks lay 16 eggs per day. "
        "She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
        "How many eggs are left?"
    )
    is_clean, detections = scanner.scan_document(contaminated_text, doc_id="doc_leak")
    assert not is_clean, "Contaminated text was not detected"
    assert len(detections) >= 1
    assert detections[0]["benchmark"] == "GSM8K"

    # Contaminated document (leaking HumanEval prompt)
    he_text = "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    pass"
    is_clean, detections = scanner.scan_document(he_text, doc_id="doc_he")
    assert not is_clean, "HumanEval leakage not detected"
    assert any(d["benchmark"] == "HumanEval" for d in detections)

    print("  -> Track 2 (Decontamination Scanner) verified successfully!")


def test_track3_stem_math_normalizer():
    print("[+] Testing Track 3: STEM & Mathematical Paper Normalizer...")
    normalizer = STEMMathNormalizer()

    latex_input = r"""
\begin{theorem}[Convergence Rate]
For convex optimization, the gradient descent sequence converges with rate:
\begin{equation}
\label{eq:rate}
\displaystyle f(x_k) - f^* \le \frac{L \|x_0 - x^*\|^2}{2k} \quad
\end{equation}
\end{theorem}
\begin{proof}
By expanding the Lipschitz condition, we obtain the inequality directly.
\end{proof}
As established in \cite{nesterov2004, boyd2004}.
"""
    references = {
        "nesterov2004": "Nesterov, Y. Introductory Lectures on Convex Optimization. Springer, 2004.",
        "boyd2004": "Boyd, S., & Vandenberghe, L. Convex Optimization. Cambridge Univ Press, 2004."
    }

    normalized = normalizer.process_paper(latex_input, referenced_sources=references)

    # Verify theorem & proof tagging
    assert "<|theorem_start|> [Convergence Rate]" in normalized
    assert "<|theorem_end|>" in normalized
    assert "<|proof_start|>" in normalized
    assert "<|proof_end|>" in normalized

    # Verify math macro cleanup
    assert r"\label{eq:rate}" not in normalized
    assert r"\displaystyle" not in normalized
    assert "$$" in normalized

    # Verify citation weaving
    assert "<|citation_context_graph|>" in normalized
    assert "Nesterov, Y." in normalized
    assert "Boyd, S." in normalized

    print("  -> Track 3 (STEM Math Normalizer) verified successfully!")


def test_track4_mmap_dataset_and_batcher():
    print("[+] Testing Track 4: MMap Indexed Dataset & Curriculum Batcher...")
    temp_dir = tempfile.mkdtemp()
    try:
        # Create two domain binary shards: Web and Code
        web_prefix = os.path.join(temp_dir, "web_shard")
        code_prefix = os.path.join(temp_dir, "code_shard")

        # 1. Write Web Shard (3 documents)
        with BinaryShardWriter(web_prefix, dtype=np.uint32) as writer:
            writer.add_document([1001, 1002, 1003, 2])
            writer.add_document([1004, 1005, 2])
            writer.add_document([1006, 1007, 1008, 1009, 2])

        # 2. Write Code Shard (2 documents)
        with BinaryShardWriter(code_prefix, dtype=np.uint32) as writer:
            writer.add_document([2001, 2002, 2003, 2004, 2])
            writer.add_document([2005, 2006, 2])

        # 3. Test MMap Loading
        ds_web = MMapIndexedDataset(web_prefix)
        ds_code = MMapIndexedDataset(code_prefix)

        assert len(ds_web) == 3
        assert len(ds_code) == 2

        # Check random access zero-copy slice
        doc_0 = ds_web[0]
        assert isinstance(doc_0, np.ndarray)
        assert list(doc_0) == [1001, 1002, 1003, 2]

        doc_code_1 = ds_code[1]
        assert list(doc_code_1) == [2005, 2006, 2]

        # 4. Test Multi-Domain Curriculum Batcher
        domain_datasets = {"curated_web": ds_web, "code_repositories": ds_code}
        domain_weights = {"curated_web": 0.70, "code_repositories": 0.30}

        batcher = CurriculumMultiDomainBatcher(
            domain_datasets=domain_datasets,
            domain_weights=domain_weights,
            batch_size=10,
            seed=42
        )

        batch = batcher.sample_batch()
        assert len(batch) == 10
        web_samples = sum(1 for domain, _ in batch if domain == "curated_web")
        code_samples = sum(1 for domain, _ in batch if domain == "code_repositories")

        assert web_samples > 0
        assert code_samples > 0
        print(f"  -> Curriculum batch sampled: {web_samples} Web, {code_samples} Code")

        ds_web.close()
        ds_code.close()
        print("  -> Track 4 (MMap Dataset & Curriculum Batcher) verified successfully!")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_track1_repo_ast_and_fim()
    test_track2_decontamination_scanner()
    test_track3_stem_math_normalizer()
    test_track4_mmap_dataset_and_batcher()
    print("\n[ALL TRACKS VERIFIED] Tracks 1, 2, 3, and 4 passed successfully!")
