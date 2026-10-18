"""
Benchmark De-Contamination and Test Leakage Prevention Scanner.
Scans pre-training documents against evaluation test splits (HumanEval, MBPP,
SWE-bench, GSM8K, MATH, MMLU) using 13-gram exact matches and normalized shingling.
"""

import re
import hashlib
import struct
from typing import List, Dict, Any, Set, Tuple, Optional


class BenchmarkDecontaminationScanner:
    """
    13-gram exact match and n-gram overlap decontamination scanner.
    Ensures pre-training data contains zero test-set contamination from
    coding, math, reasoning, and SWE benchmarks.
    """

    DEFAULT_BENCHMARK_PROMPTS = {
        # Coding Benchmarks
        "HumanEval": [
            "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
            "def separate_paren_groups(paren_string: str) -> List[str]:",
            "def truncate_number(number: float) -> float:",
            "def below_zero(operations: List[int]) -> bool:",
            "def mean_absolute_deviation(numbers: List[float]) -> float:"
        ],
        "MBPP": [
            "Write a function to find the minimum cost path to reach (m, n) from (0, 0) in a given cost matrix.",
            "Write a python function to find the sum of all odd length subarrays of a given array.",
            "Write a function to find the longest common subsequence of two given strings."
        ],
        # Math & Logic Benchmarks
        "GSM8K": [
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.",
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "Josh decides to try flipping a house. He buys a house for $80,000 and puts $50,000 into repairs."
        ],
        "MATH": [
            "Let P(x) = x^4 + 2x^3 - 7x^2 - 8x + 12. Find all real roots of the polynomial.",
            "In right triangle ABC, angle C is 90 degrees. If sin A = 3/5, find tan B.",
            "Compute the sum of all positive integers n such that n divides 2024."
        ],
        # SWE Agent Benchmarks
        "SWE-bench": [
            "pytest tests/test_ast.py -k test_parse_imports_cycle",
            "django.core.exceptions.FieldError: Unknown field(s) specified for Model",
            "sympy.core.sympify.SympifyError: could not parse string expression",
            "requests.exceptions.ConnectionError: Max retries exceeded with url"
        ],
        # General Multitask Knowledge
        "MMLU": [
            "Which of the following describes the primary function of the Golgi apparatus?",
            "In contract law, promissory estoppel prevents a party from withdrawing a promise if another party has reasonably relied upon it.",
            "What is the time complexity of searching for an element in a balanced red-black tree with n nodes?"
        ]
    }

    def __init__(self, ngram_size: int = 13, custom_benchmark_prompts: Optional[Dict[str, List[str]]] = None):
        self.ngram_size = ngram_size
        self.benchmarks = custom_benchmark_prompts or self.DEFAULT_BENCHMARK_PROMPTS

        # 64-bit integer hash index: hash -> (benchmark_name, ngram_text)
        self.ngram_index: Dict[int, Tuple[str, str]] = {}
        self._build_index()

    def _normalize_text(self, text: str) -> List[str]:
        """Lowercases, strips punctuation, and extracts word token sequence."""
        return re.findall(r"\b\w+\b", text.lower())

    def _hash_ngram(self, ngram_words: Tuple[str, ...]) -> int:
        """Computes 64-bit integer hash for word tuple."""
        key = " ".join(ngram_words).encode("utf-8")
        return struct.unpack("<Q", hashlib.md5(key).digest()[:8])[0]

    def _build_index(self) -> None:
        """Indexes all benchmark test questions into the 13-gram hash table."""
        for bench_name, prompts in self.benchmarks.items():
            for prompt in prompts:
                words = self._normalize_text(prompt)
                if len(words) < self.ngram_size:
                    # If prompt is shorter than 13 words, index all available words
                    if len(words) >= 4:
                        h = self._hash_ngram(tuple(words))
                        self.ngram_index[h] = (bench_name, " ".join(words))
                    continue

                for i in range(len(words) - self.ngram_size + 1):
                    ngram = tuple(words[i:i + self.ngram_size])
                    h = self._hash_ngram(ngram)
                    self.ngram_index[h] = (bench_name, " ".join(ngram))

    def scan_document(self, text: str, doc_id: str = "") -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Scans document text against the decontamination index.
        Returns:
            (is_clean, list_of_detected_contaminations)
        """
        words = self._normalize_text(text)
        if len(words) < 4:
            return True, []

        detections = []
        seen_benchmarks = set()

        # Check full n-grams
        for i in range(len(words) - min(len(words), self.ngram_size) + 1):
            ngram = tuple(words[i:i + self.ngram_size])
            h = self._hash_ngram(ngram)
            if h in self.ngram_index:
                bench_name, matched_snippet = self.ngram_index[h]
                if bench_name not in seen_benchmarks:
                    seen_benchmarks.add(bench_name)
                    detections.append({
                        "doc_id": doc_id,
                        "benchmark": bench_name,
                        "matched_ngram": matched_snippet,
                        "word_position": i
                    })

        # Also check shorter exact matches (for 4-12 word short prompts)
        for i in range(len(words) - 3):
            for k in range(4, min(self.ngram_size, len(words) - i + 1)):
                sub_ngram = tuple(words[i:i + k])
                h = self._hash_ngram(sub_ngram)
                if h in self.ngram_index:
                    bench_name, matched_snippet = self.ngram_index[h]
                    if bench_name not in seen_benchmarks:
                        seen_benchmarks.add(bench_name)
                        detections.append({
                            "doc_id": doc_id,
                            "benchmark": bench_name,
                            "matched_ngram": matched_snippet,
                            "word_position": i
                        })

        is_clean = len(detections) == 0
        return is_clean, detections
