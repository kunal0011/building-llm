"""
Binary Memory-Mapped Shard Builder for Distributed LLM Training.
Produces Megatron-LM / Nanotron compatible binary (.bin) and index (.idx) formats.
"""

import os
import struct
from typing import List, BinaryIO
import numpy as np


class BinaryShardWriter:
    """
    Writes token sequences into flat binary files (.bin) and companion index files (.idx).
    Enables zero-copy memory-mapped ingestion across hundreds of distributed GPUs.
    Supports uint16 (vocab < 65,536) and uint32/int32 (vocab >= 65,536, e.g. Llama-3 128k).
    """

    MAGIC_HEADER = b"MMIDXB\x00\x00"  # Memory-Mapped Index Binary format
    VERSION = 1

    def __init__(self, output_prefix: str, dtype: np.dtype = np.uint32):
        self.output_prefix = output_prefix
        self.bin_path = f"{output_prefix}.bin"
        self.idx_path = f"{output_prefix}.idx"
        self.dtype = dtype

        self.bin_file: BinaryIO = open(self.bin_path, "wb")
        self.doc_offsets: List[int] = [0]
        self.doc_sizes: List[int] = []
        self.total_tokens: int = 0

    def add_document(self, token_ids: List[int]) -> None:
        """Appends a single tokenized document to the binary shard."""
        if not token_ids:
            return

        arr = np.array(token_ids, dtype=self.dtype)
        self.bin_file.write(arr.tobytes())

        doc_len = len(token_ids)
        self.doc_sizes.append(doc_len)
        self.total_tokens += doc_len
        self.doc_offsets.append(self.total_tokens)

    def close(self) -> None:
        """Closes the binary file and serializes the index metadata (.idx)."""
        self.bin_file.close()

        # Dtype encoding: 1 = uint8, 2 = int8, 3 = int16, 4 = uint16, 5 = int32, 6 = uint32, 7 = int64
        dtype_code = 6 if self.dtype == np.uint32 else 5 if self.dtype == np.int32 else 4

        with open(self.idx_path, "wb") as idx_file:
            # Header
            idx_file.write(self.MAGIC_HEADER)
            idx_file.write(struct.pack("<Q", self.VERSION))
            idx_file.write(struct.pack("<B", dtype_code))

            # Number of documents and total tokens
            num_docs = len(self.doc_sizes)
            idx_file.write(struct.pack("<Q", num_docs))
            idx_file.write(struct.pack("<Q", self.total_tokens))

            # Document sequence sizes (int32 per doc)
            sizes_arr = np.array(self.doc_sizes, dtype=np.int32)
            idx_file.write(sizes_arr.tobytes())

            # Document offsets (int64 per doc pointer)
            offsets_arr = np.array(self.doc_offsets, dtype=np.int64)
            idx_file.write(offsets_arr.tobytes())

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
