"""
CLIP / SigLIP Vision-Language Alignment & Semantic Scoring Engine.
Evaluates cosine similarity between visual feature representations and candidate text descriptions.
Filters out unaligned images and enriches marginal samples with surrounding document context.
"""

import math
import hashlib
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class CLIPSigLIPAligner:
    """
    Dual-Encoder Vision-Language Alignment Scorer (inspired by CLIP and SigLIP).
    Computes normalized embedding projections and cosine similarity:
      S_align = (e_txt . e_img) / (|e_txt| * |e_img|)
    Enforces alignment threshold tau = 0.24 for frontier multimodal pretraining.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        alignment_threshold: float = 0.24,
        high_quality_threshold: float = 0.28,
        random_seed: int = 42
    ):
        self.embedding_dim = embedding_dim
        self.alignment_threshold = alignment_threshold
        self.high_quality_threshold = high_quality_threshold
        self.rng = np.random.default_rng(random_seed)

        # Pre-generate deterministic projection matrices for character & word n-grams
        self._projection_weights = self.rng.normal(0, 1.0 / math.sqrt(embedding_dim), (2048, embedding_dim))

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Embeds text into a normalized d-dimensional semantic feature vector.
        Uses subword hash-kernel projection with L2 normalization.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            vec = np.zeros(self.embedding_dim, dtype=np.float32)
            vec[0] = 1.0
            return vec

        # Accumulate projection rows based on word hash indices
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        for w in words:
            h = int(hashlib.md5(w.encode()).hexdigest()[:8], 16) % 2048
            vec += self._projection_weights[h]

        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec = vec / norm
        return vec

    def _embed_image_metadata(
        self,
        image_url: str,
        alt_text: str,
        image_features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Derives or accepts vision embedding vector for the image asset.
        """
        if image_features is not None:
            vec = image_features.astype(np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 1e-8 else vec

        # Deterministic simulation of ViT-L/14 patch projection for data pipeline testing
        url_hash = int(hashlib.sha256(image_url.encode()).hexdigest()[:8], 16) % 2048
        alt_vec = self._embed_text(alt_text)

        # Visual feature is correlated with valid visual descriptors in alt_text plus image hash
        base_visual = self._projection_weights[url_hash]
        fused = (0.75 * alt_vec) + (0.25 * base_visual)
        norm = np.linalg.norm(fused)
        return fused / norm if norm > 1e-8 else fused

    def compute_alignment_score(
        self,
        text: str,
        image_url: str,
        alt_text: str,
        image_features: Optional[np.ndarray] = None
    ) -> float:
        """
        Computes cosine similarity between text embedding and visual embedding:
          score in [-1.0, 1.0]
        """
        txt_vec = self._embed_text(text)
        img_vec = self._embed_image_metadata(image_url, alt_text, image_features)

        score = float(np.dot(txt_vec, img_vec))
        return max(-1.0, min(1.0, score))

    def evaluate_image_text_pair(
        self,
        alt_text: str,
        image_url: str,
        surrounding_context: str = "",
        image_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluates vision-language alignment.
        If alt-text alone yields marginal alignment (< threshold), incorporates
        surrounding paragraph context to verify multimodal relevance.
        """
        direct_score = self.compute_alignment_score(alt_text, image_url, alt_text, image_features)

        # Contextual alignment incorporating surrounding paragraph
        contextual_score = direct_score
        used_context = False
        if surrounding_context and direct_score < self.high_quality_threshold:
            combined_text = f"{alt_text}. {surrounding_context}"
            contextual_score = self.compute_alignment_score(combined_text, image_url, alt_text, image_features)
            used_context = True

        effective_score = max(direct_score, contextual_score)
        passed = effective_score >= self.alignment_threshold
        is_high_quality = effective_score >= self.high_quality_threshold

        return {
            "passed": passed,
            "is_high_quality": is_high_quality,
            "effective_score": round(effective_score, 4),
            "direct_alt_score": round(direct_score, 4),
            "contextual_score": round(contextual_score, 4) if used_context else None,
            "used_surrounding_context": used_context,
            "threshold": self.alignment_threshold
        }
