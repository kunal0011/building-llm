"""
Long-Context Document Packer with Attention Boundary Generation.
Supports progressive context lengths (8k, 32k, 128k, 1M+ tokens).
"""

from typing import List, Dict, Any, Tuple
import numpy as np


class DocumentPacker:
    """
    Packs variable-length documents into fixed-length context windows
    using First-Fit Decreasing (FFD) bin-packing.
    Generates segment IDs and cumulative position IDs for block-diagonal attention.
    """

    def __init__(
        self,
        max_seq_len: int = 32_768,
        eos_token_id: int = 2,
        bos_token_id: int = 1,
        pad_token_id: int = 0
    ):
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id

    def pack_documents(
        self,
        tokenized_docs: List[List[int]],
        allow_cross_document_attention: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Packs tokenized documents into fixed max_seq_len sequences.

        Returns a list of packed sample dictionaries containing:
          - input_ids: Array of length max_seq_len
          - position_ids: Positional indices (reset to 0 at doc boundaries if cross-doc attention is disabled)
          - document_boundaries: List of (start_idx, end_idx) tuples for FlashAttention / cuDNN varlen flash
        """
        # Sort documents by length descending for efficient First-Fit Decreasing packing
        indexed_docs = sorted(
            [(len(doc), doc) for doc in tokenized_docs],
            key=lambda x: x[0],
            reverse=True
        )

        packed_sequences: List[Dict[str, Any]] = []

        # Current open bins: list of (remaining_space, list_of_docs)
        bins: List[Tuple[int, List[List[int]]]] = []

        for length, doc in indexed_docs:
            # If a single document exceeds max_seq_len, it is chunked
            if length > self.max_seq_len:
                for chunk_start in range(0, length, self.max_seq_len):
                    chunk = doc[chunk_start:chunk_start + self.max_seq_len]
                    bins.append((self.max_seq_len - len(chunk), [chunk]))
                continue

            placed = False
            for i in range(len(bins)):
                rem_space, doc_list = bins[i]
                if rem_space >= length:
                    doc_list.append(doc)
                    bins[i] = (rem_space - length, doc_list)
                    placed = True
                    break

            if not placed:
                bins.append((self.max_seq_len - length, [doc]))

        # Convert bins into tensor-ready arrays
        for _, doc_list in bins:
            packed_ids: List[int] = []
            position_ids: List[int] = []
            boundaries: List[Tuple[int, int]] = []

            for doc in doc_list:
                start_idx = len(packed_ids)
                packed_ids.extend(doc)
                end_idx = len(packed_ids)
                boundaries.append((start_idx, end_idx))

                if allow_cross_document_attention:
                    # Continuous monotonic position IDs
                    position_ids.extend(range(start_idx, end_idx))
                else:
                    # Reset position IDs to 0 at the beginning of each document
                    position_ids.extend(range(0, end_idx - start_idx))

            # Pad remaining space
            pad_len = self.max_seq_len - len(packed_ids)
            if pad_len > 0:
                packed_ids.extend([self.pad_token_id] * pad_len)
                position_ids.extend([0] * pad_len)

            packed_sequences.append({
                "input_ids": packed_ids[:self.max_seq_len],
                "position_ids": position_ids[:self.max_seq_len],
                "document_boundaries": boundaries,
                "num_documents": len(boundaries),
                "pad_tokens": pad_len
            })

        return packed_sequences
