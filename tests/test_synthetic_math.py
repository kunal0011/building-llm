"""
Unit tests for Synthetic Mathematical Olympiad & Formal Verification Generator.
"""

import os
import tempfile
import pytest

from pipelines.academic.synthetic_math_olympiad import (
    SyntheticOlympiadGenerator,
    FormalMathVerifier,
    FormalMathCorpusPacker,
    OlympiadProblem
)


@pytest.fixture
def generator():
    return SyntheticOlympiadGenerator(seed=123)


@pytest.fixture
def verifier():
    return FormalMathVerifier()


def test_olympiad_problem_generation_domains(generator):
    p_nt = generator.generate_number_theory_problem()
    assert p_nt.domain == "number_theory"
    assert len(p_nt.thought_chain) >= 4
    assert int(p_nt.canonical_answer) > 0
    assert "Mathlib" in p_nt.formal_lean4_proof

    p_alg = generator.generate_algebra_problem()
    assert p_alg.domain == "algebra"
    assert "Vieta" in p_alg.formal_lean4_proof
    assert int(p_alg.canonical_answer) is not None

    p_comb = generator.generate_combinatorics_problem()
    assert p_comb.domain == "combinatorics"
    assert "recurrence" in p_comb.thought_chain[0].lower() or "characteristic" in p_comb.thought_chain[0].lower()

    p_calc = generator.generate_calculus_problem()
    assert p_calc.domain == "calculus"
    assert "/" in p_calc.canonical_answer or p_calc.canonical_answer.isdigit()


def test_formal_math_verifier(generator, verifier):
    suite = generator.generate_suite(count=4)
    assert len(suite) == 4

    for prob in suite:
        report = verifier.verify_problem(prob)
        assert report["overall_verified"] is True
        assert report["symbolic_answer_valid"] is True
        assert report["lean4_syntax_valid"] is True
        assert report["isabelle_syntax_valid"] is True
        assert report["thought_chain_depth"] >= 3


def test_corpus_packer_formatting(generator):
    prob = generator.generate_algebra_problem()
    doc = FormalMathCorpusPacker.format_problem_document(prob)

    assert "<|begin_of_text|>" in doc
    assert "<|end_of_text|>" in doc
    assert "<|thought|>" in doc
    assert "<|thought_end|>" in doc
    assert "Lean 4" in doc
    assert "Isabelle" in doc


def test_binary_shard_export(generator):
    suite = generator.generate_suite(count=3)

    class MockTokenizer:
        def encode(self, text: str):
            return [ord(c) % 500 for c in text[:50]]

    with tempfile.TemporaryDirectory() as tmpdir:
        prefix = os.path.join(tmpdir, "olympiad_math_shard")
        result = FormalMathCorpusPacker.export_to_binary_shard(
            suite, output_prefix=prefix, tokenizer=MockTokenizer()
        )

        assert os.path.exists(result["bin_path"])
        assert os.path.exists(result["idx_path"])
        assert result["num_documents"] == 3
        assert result["total_tokens"] > 0
