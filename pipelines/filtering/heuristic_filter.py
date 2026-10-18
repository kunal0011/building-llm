"""
High-Performance Heuristic, Quality, and Safety Filter for Pre-training Data.
Implements frontier-grade filters (inspired by FineWeb, RefinedWeb, and Llama 3 pipelines).
"""

import re
from typing import Tuple, Dict, Any, List, Set
from collections import Counter


class HeuristicQualityFilter:
    """
    Evaluates text against strict structural, linguistic, and statistical heuristics.
    Rejects low-quality web scrapes, SEO spam, automated dumps, and corrupted data.
    """

    def __init__(
        self,
        min_words: int = 50,
        max_words: int = 150_000,
        min_mean_word_length: float = 3.0,
        max_mean_word_length: float = 12.0,
        max_symbol_to_word_ratio: float = 0.35,
        max_bullet_line_ratio: float = 0.85,
        max_repeated_line_ratio: float = 0.30,
        max_repeated_ngram_ratio: float = 0.20,
        min_stopword_ratio: float = 0.05,
    ):
        self.min_words = min_words
        self.max_words = max_words
        self.min_mean_word_length = min_mean_word_length
        self.max_mean_word_length = max_mean_word_length
        self.max_symbol_to_word_ratio = max_symbol_to_word_ratio
        self.max_bullet_line_ratio = max_bullet_line_ratio
        self.max_repeated_line_ratio = max_repeated_line_ratio
        self.max_repeated_ngram_ratio = max_repeated_ngram_ratio
        self.min_stopword_ratio = min_stopword_ratio

        # Common English stop words for linguistic integrity verification
        self.common_stopwords: Set[str] = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me"
        }

        # PII Regex patterns for redaction
        self.pii_patterns = {
            "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
            "ipv4": re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            "phone_intl": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            "credit_card": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        }

    def scrub_pii(self, text: str) -> str:
        """Replaces identified PII with structured generic replacement tokens."""
        text = self.pii_patterns["email"].sub("<|PII_EMAIL|>", text)
        text = self.pii_patterns["ipv4"].sub("<|PII_IP|>", text)
        text = self.pii_patterns["phone_intl"].sub("<|PII_PHONE|>", text)
        text = self.pii_patterns["credit_card"].sub("<|PII_CARD|>", text)
        return text

    def evaluate(self, text: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Assesses document quality.
        Returns:
            (passes_filter, rejection_reason, metrics_dict)
        """
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if not lines:
            return False, "empty_document", {}

        words = re.findall(r"\b\w+\b", text)
        word_count = len(words)

        metrics = {
            "word_count": word_count,
            "line_count": len(lines),
        }

        # 1. Volume constraints
        if word_count < self.min_words:
            return False, f"too_short_{word_count}_words", metrics
        if word_count > self.max_words:
            return False, f"too_long_{word_count}_words", metrics

        # 2. Mean word length (catches minified code or random strings)
        total_chars = sum(len(w) for w in words)
        mean_word_len = total_chars / max(word_count, 1)
        metrics["mean_word_length"] = round(mean_word_len, 2)
        if not (self.min_mean_word_length <= mean_word_len <= self.max_mean_word_length):
            return False, f"abnormal_mean_word_length_{mean_word_len:.2f}", metrics

        # 3. Symbol to word ratio (catches ascii art, dumps, broken encoding)
        symbols = re.findall(r"[^A-Za-z0-9\s]", text)
        symbol_ratio = len(symbols) / max(word_count, 1)
        metrics["symbol_to_word_ratio"] = round(symbol_ratio, 3)
        if symbol_ratio > self.max_symbol_to_word_ratio:
            return False, f"high_symbol_ratio_{symbol_ratio:.3f}", metrics

        # 4. Bullet / list line ratio (catches navigation menus, sitemaps, footers)
        bullet_lines = sum(1 for line in lines if line.startswith(("-", "*", "•", "1.", "2.", ">")))
        bullet_ratio = bullet_lines / len(lines)
        metrics["bullet_ratio"] = round(bullet_ratio, 3)
        if bullet_ratio > self.max_bullet_line_ratio:
            return False, f"menu_or_list_dump_{bullet_ratio:.2f}", metrics

        # 5. Repeated lines ratio (catches copyright notices, banners, repeated watermarks)
        line_counts = Counter(lines)
        duplicated_lines = sum(c for line, c in line_counts.items() if c > 1)
        repeated_line_ratio = duplicated_lines / len(lines)
        metrics["repeated_line_ratio"] = round(repeated_line_ratio, 3)
        if repeated_line_ratio > self.max_repeated_line_ratio:
            return False, f"repetitive_lines_{repeated_line_ratio:.2f}", metrics

        # 6. Repetition check via 4-grams (catches infinite loop hallucinations or scraping loops)
        if word_count >= 50:
            ngrams = [tuple(words[i:i+4]) for i in range(len(words) - 3)]
            ngram_counts = Counter(ngrams)
            repeated_ngrams = sum(c for ng, c in ngram_counts.items() if c > 1)
            ngram_ratio = repeated_ngrams / len(ngrams)
            metrics["repeated_4gram_ratio"] = round(ngram_ratio, 3)
            if ngram_ratio > self.max_repeated_ngram_ratio:
                return False, f"repetitive_ngrams_{ngram_ratio:.2f}", metrics

        # 7. Linguistic integrity check (stopwords check for natural language text)
        lower_words = [w.lower() for w in words]
        stopwords_found = sum(1 for w in lower_words if w in self.common_stopwords)
        stopword_ratio = stopwords_found / max(word_count, 1)
        metrics["stopword_ratio"] = round(stopword_ratio, 3)
        # Note: We enforce stopword ratio primarily on prose/literature/web text (not code files)
        if stopword_ratio < self.min_stopword_ratio:
            # We mark but do not immediately reject if it could be code; passed to domain router
            metrics["low_stopword_warning"] = True

        return True, "passed", metrics
