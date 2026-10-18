"""
MinHash and Locality-Sensitive Hashing (LSH) for Scalable Near-Deduplication.
Identifies near-duplicate documents across petabyte-scale pre-training datasets.
"""

import hashlib
import struct
import re
from typing import List, Set, Dict, Tuple, Iterator
from collections import defaultdict


class MinHashLSH:
    """
    MinHash LSH implementation for document similarity detection.
    Computes k MinHash signatures using independent linear hash permutations:
      h_i(x) = (a_i * x + b_i) mod p
    Partitions signatures into b bands of r rows.
    """

    def __init__(
        self,
        num_perm: int = 128,
        num_bands: int = 16,
        shingle_size: int = 5,
        similarity_threshold: float = 0.80
    ):
        assert num_perm % num_bands == 0, "num_perm must be divisible by num_bands"
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        self.shingle_size = shingle_size
        self.similarity_threshold = similarity_threshold

        # Large Mersenne prime for hashing
        self.prime = (1 << 61) - 1

        # Deterministic generation of linear permutation parameters (a, b)
        self.permutations: List[Tuple[int, int]] = []
        for i in range(num_perm):
            # Seeded hash derivations
            a = int(hashlib.sha256(f"a_seed_{i}".encode()).hexdigest()[:15], 16) % (self.prime - 1) + 1
            b = int(hashlib.sha256(f"b_seed_{i}".encode()).hexdigest()[:15], 16) % self.prime
            self.permutations.append((a, b))

        # Inverted index: (band_idx, bucket_hash) -> list of doc_ids
        self.buckets: Dict[Tuple[int, int], List[str]] = defaultdict(list)

    def _get_shingles(self, text: str) -> Set[int]:
        """Extracts normalized n-gram word shingles converted to 64-bit integer hashes."""
        tokens = re.findall(r"\b\w+\b", text.lower())
        if len(tokens) < self.shingle_size:
            tokens.extend(["<pad>"] * (self.shingle_size - len(tokens)))

        shingles = set()
        for i in range(len(tokens) - self.shingle_size + 1):
            shingle_str = " ".join(tokens[i:i + self.shingle_size])
            # Hash to 64-bit uint
            shingle_hash = struct.unpack("<Q", hashlib.md5(shingle_str.encode()).digest()[:8])[0]
            shingles.add(shingle_hash)
        return shingles

    def compute_minhash_signature(self, text: str) -> List[int]:
        """Computes the MinHash signature array for a document."""
        shingles = self._get_shingles(text)
        if not shingles:
            return [0] * self.num_perm

        signature = []
        for a, b in self.permutations:
            min_val = self.prime
            for sh in shingles:
                val = (a * sh + b) % self.prime
                if val < min_val:
                    min_val = val
            signature.append(min_val)
        return signature

    def insert(self, doc_id: str, signature: List[int]) -> None:
        """Indexes document signature into LSH bands."""
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_chunk = tuple(signature[start:end])
            bucket_hash = hash(band_chunk)
            self.buckets[(band_idx, bucket_hash)].append(doc_id)

    def find_clusters(self) -> Dict[str, Set[str]]:
        """
        Builds connected components of duplicate clusters.
        Returns a mapping from canonical doc_id to set of duplicate doc_ids.
        """
        # Graph adjacency
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        for doc_list in self.buckets.values():
            if len(doc_list) > 1:
                first = doc_list[0]
                for other in doc_list[1:]:
                    adjacency[first].add(other)
                    adjacency[other].add(first)

        visited: Set[str] = set()
        clusters: Dict[str, Set[str]] = {}

        for node in list(adjacency.keys()):
            if node not in visited:
                component = set()
                queue = [node]
                visited.add(node)
                while queue:
                    curr = queue.pop(0)
                    component.add(curr)
                    for neighbor in adjacency[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                # Pick lexicographically lowest doc_id as the canonical keeper
                canonical = sorted(component)[0]
                component.remove(canonical)
                if component:
                    clusters[canonical] = component

        return clusters

    @staticmethod
    def jaccard_similarity(sig_a: List[int], sig_b: List[int]) -> float:
        """Estimates Jaccard similarity between two MinHash signatures."""
        assert len(sig_a) == len(sig_b)
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / len(sig_a)
