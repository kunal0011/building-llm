"""
Production 128k Byte-Pair Encoding (BPE) Tokenizer Architecture.
Engineered using Hugging Face's Rust-backed tokenizers library with custom regex splitting,
byte-level fallback, and reserved tokens for SWE, Agentic Planning, and Multimodal Anchors.
"""

import os
from typing import List, Dict, Any, Optional, Iterator
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex


class UniversalBPETokenizer:
    """
    128k vocabulary Byte-Level BPE Tokenizer.
    Features:
      - ByteLevel pre-tokenization with LLaMA-3 / GPT-4 style regex splitting
      - Full byte fallback (no unknown tokens for arbitrary binary/unicode sequences)
      - Specialized control tokens for Repository-level code, Agent thoughts/tools,
        multi-day state persistence, and multimodal anchors.
    """

    SPECIAL_TOKENS = [
        # Foundational boundary tokens
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|pad|>",
        "<|unk|>",

        # Repository-level SWE structure tokens
        "<|repo_start|>",
        "<|repo_end|>",
        "<|repo_readme|>",
        "<|repo_readme_end|>",
        "<|repo_overview|>",
        "<|repo_overview_end|>",
        "<|file_start|>",
        "<|file_end|>",
        "<|file_language|>",

        # Code Fill-In-The-Middle (FIM) tokens
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",

        # Long-Horizon Agent thought & tool execution tokens
        "<|thought|>",
        "<|thought_end|>",
        "<|tool_call|>",
        "<|tool_call_end|>",
        "<|tool_output|>",
        "<|tool_output_end|>",

        # Multi-day persistent horizon & state checkpoints
        "<|state_checkpoint_restore|>",
        "<|end_checkpoint|>",

        # Multimodal spatial & temporal anchor placeholders
        "<|image_ref_start|>",
        "<|image_ref_end|>",
        "<|audio_ref_start|>",
        "<|audio_ref_end|>",
        "<|video_frame_start|>",
        "<|video_frame_end|>"
    ]

    # Modern frontier regex split pattern (handles contractions, numbers, punctuation, and whitespace)
    SPLIT_REGEX = (
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    )

    def __init__(self, vocab_size: int = 128_000):
        self.vocab_size = vocab_size
        self.tokenizer = self._build_empty_tokenizer()

    def _build_empty_tokenizer(self) -> Tokenizer:
        """Initializes empty BPE model with ByteLevel pre-tokenization and split regex."""
        tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(self.SPLIT_REGEX), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        tokenizer.decoder = decoders.ByteLevel()
        return tokenizer

    def train_from_iterator(
        self,
        iterator: Iterator[str],
        save_directory: Optional[str] = None,
        min_frequency: int = 2
    ) -> None:
        """
        Trains the 128k BPE vocabulary on an iterator of text documents.
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        self.tokenizer.train_from_iterator(iterator, trainer=trainer)

        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
            self.save(save_directory)

    def save(self, directory: str) -> str:
        """Saves tokenizer.json and vocabulary files."""
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "tokenizer.json")
        self.tokenizer.save(path)
        return path

    @classmethod
    def load(cls, file_path: str) -> "UniversalBPETokenizer":
        """Loads a pre-trained tokenizer from file."""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(file_path)
        instance.vocab_size = instance.tokenizer.get_vocab_size()
        return instance

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encodes text string into list of token IDs."""
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        if add_special_tokens:
            bos_id = self.token_to_id("<|begin_of_text|>")
            eos_id = self.token_to_id("<|end_of_text|>")
            token_ids = [bos_id] + token_ids + [eos_id]
        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decodes token IDs back into string."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        """Looks up integer ID of token."""
        return self.tokenizer.token_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        """Looks up string representation of token ID."""
        return self.tokenizer.id_to_token(token_id)

    def get_vocab_size(self) -> int:
        """Returns active vocabulary size."""
        return self.tokenizer.get_vocab_size()
