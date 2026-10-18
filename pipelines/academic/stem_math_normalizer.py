"""
STEM & Mathematical Paper Normalizer for ArXiv and PubMed Central.
Normalizes LaTeX equations, strips presentation macros, tags theorem/proof blocks,
and resolves academic citation graphs for extended-context reasoning.
"""

import re
from typing import List, Dict, Any, Tuple, Optional


class STEMMathNormalizer:
    """
    Standardizes LaTeX math and scientific discourse for LLM pre-training.
    Eliminates compilation artifacts while preserving symbolic reasoning signals.
    """

    # Environments for mathematical derivations
    MATH_ENVIRONMENTS = [
        "equation", "equation*", "align", "align*", "gather", "gather*",
        "multline", "multline*", "split", "cases"
    ]

    # Theorem and proof environments to wrap with structured control tokens
    THEOREM_ENVIRONMENTS = {
        "theorem": ("<|theorem_start|>", "<|theorem_end|>"),
        "lemma": ("<|lemma_start|>", "<|lemma_end|>"),
        "corollary": ("<|corollary_start|>", "<|corollary_end|>"),
        "definition": ("<|definition_start|>", "<|definition_end|>"),
        "proof": ("<|proof_start|>", "<|proof_end|>"),
    }

    def __init__(self):
        # Regex to strip non-semantic rendering macros
        self.strip_macros = [
            (re.compile(r"\\label\{[^}]*\}"), ""),
            (re.compile(r"\\nonumber"), ""),
            (re.compile(r"\\tag\{[^}]*\}"), ""),
            (re.compile(r"\\displaystyle"), ""),
            (re.compile(r"\\quad"), " "),
            (re.compile(r"\\qquad"), "  "),
            (re.compile(r"\\,"), " "),
            (re.compile(r"\\!"), ""),
        ]

    def normalize_math_expressions(self, text: str) -> str:
        """Cleans and standardizes inline and display LaTeX math equations."""
        # 1. Clean presentation macros
        for pattern, replacement in self.strip_macros:
            text = pattern.sub(replacement, text)

        # 2. Normalize display math delimiters: \[ ... \] -> $$ ... $$
        text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)

        # 3. Standardize math environment blocks into clean $$ blocks
        for env in self.MATH_ENVIRONMENTS:
            pattern = re.compile(rf"\\begin\{{{re.escape(env)}\}}(.*?)\\end\{{{re.escape(env)}\}}", re.DOTALL)
            text = pattern.sub(r"$$\1$$", text)

        # 4. Collapse redundant whitespace inside $$...$$
        def clean_display_math(match: re.Match) -> str:
            inner = match.group(1).strip()
            # Collapse multiple spaces and blank lines within equation
            inner = re.sub(r"[ \t]+", " ", inner)
            inner = re.sub(r"\n{2,}", "\n", inner)
            return f"\n$$\n{inner}\n$$\n"

        text = re.sub(r"\$\$(.*?)\$\$", clean_display_math, text, flags=re.DOTALL)

        # 5. Clean inline math ($...$)
        def clean_inline_math(match: re.Match) -> str:
            inner = match.group(1).strip()
            return f"${inner}$"

        text = re.sub(r"(?<!\$)\$([^\$\n]+)\$(?!\$)", clean_inline_math, text)

        return text

    def tag_theorems_and_proofs(self, text: str) -> str:
        """Injects control tokens around formal mathematical reasoning structures."""
        for env, (start_token, end_token) in self.THEOREM_ENVIRONMENTS.items():
            pattern = re.compile(rf"\\begin\{{{env}\}}(?:\[(.*?)\])?(.*?)\\end\{{{env}\}}", re.DOTALL | re.IGNORECASE)

            def replacer(match: re.Match) -> str:
                title = match.group(1) or ""
                body = match.group(2).strip()
                title_str = f" [{title}]" if title else ""
                return f"\n{start_token}{title_str}\n{body}\n{end_token}\n"

            text = pattern.sub(replacer, text)
        return text

    def weave_citation_context(
        self,
        main_paper_text: str,
        referenced_papers: Dict[str, str]
    ) -> str:
        """
        Synthesizes a master long-context document linking cited references directly
        into the main text, creating rich cross-document reasoning data.
        """
        # Find citations: \cite{ref_key} or (Author et al., 2024)
        cited_keys = set(re.findall(r"\\cite\{([^}]+)\}", main_paper_text))
        flat_keys = set()
        for k in cited_keys:
            for item in k.split(","):
                flat_keys.add(item.strip())

        appendix_chunks = []
        for key in flat_keys:
            if key in referenced_papers:
                appendix_chunks.append(
                    f"<|cited_reference key=\"{key}\">\n"
                    f"{referenced_papers[key].strip()}\n"
                    f"<|end_cited_reference|>"
                )

        if not appendix_chunks:
            return main_paper_text

        combined = (
            f"{main_paper_text}\n\n"
            f"<|citation_context_graph|>\n"
            + "\n\n".join(appendix_chunks) + "\n"
            f"<|end_citation_context_graph|>"
        )
        return combined

    def process_paper(
        self,
        raw_latex: str,
        referenced_sources: Optional[Dict[str, str]] = None
    ) -> str:
        """Executes full normalization pipeline for an academic paper."""
        text = self.normalize_math_expressions(raw_latex)
        text = self.tag_theorems_and_proofs(text)
        if referenced_sources:
            text = self.weave_citation_context(text, referenced_sources)
        return text.strip()
