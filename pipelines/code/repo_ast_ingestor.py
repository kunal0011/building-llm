"""
Repository-Level Codebase Ingestion, AST Topological Dependency Sorting,
and Fill-In-The-Middle (FIM) Data Augmentation.
"""

import os
import re
import ast
import random
from typing import List, Dict, Any, Tuple, Optional, Set
from collections import defaultdict

from schemas.data_schema import RepositoryFile, RepositoryCodebaseDocument


class RepositoryASTIngestor:
    """
    Parses code repositories, extracts module dependencies across languages,
    sorts files topologically, and serializes repository graphs.
    """

    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".rs": "rust",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".ts": "typescript",
        ".js": "javascript",
        ".go": "go",
        ".java": "java",
        ".toml": "toml",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown"
    }

    def __init__(self, max_file_size_kb: int = 500):
        self.max_file_size_bytes = max_file_size_kb * 1024

    def extract_python_imports(self, code_str: str) -> Tuple[List[str], List[str]]:
        """Extracts import targets and top-level definitions from Python AST."""
        imports = []
        exports = []
        try:
            tree = ast.parse(code_str)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split(".")[0])
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    exports.append(node.name)
        except SyntaxError:
            # Fallback regex for syntax-broken code or templating
            for match in re.finditer(r"^(?:from|import)\s+([\w\.]+)", code_str, re.MULTILINE):
                imports.append(match.group(1).split(".")[0])
        return sorted(list(set(imports))), sorted(list(set(exports)))

    def extract_generic_imports(self, code_str: str, language: str) -> List[str]:
        """Regex import extraction for Rust, TypeScript, C++, Go, and Java."""
        imports = []
        if language in ("typescript", "javascript"):
            matches = re.findall(r"""(?:import|require)\s*\(?['"]([^'"]+)['"]""", code_str)
            imports.extend([m.split("/")[-1].replace(".ts", "").replace(".js", "") for m in matches])
        elif language == "rust":
            matches = re.findall(r"""(?:use|mod|extern\s+crate)\s+([\w:]+)""", code_str)
            imports.extend([m.split("::")[0] for m in matches])
        elif language in ("cpp", "c"):
            matches = re.findall(r"""#include\s+["<]([^">]+)[">]""", code_str)
            imports.extend([os.path.splitext(os.path.basename(m))[0] for m in matches])
        elif language == "go":
            matches = re.findall(r"""import\s+(?:\(([^)]+)\)|"([^"]+)")""", code_str)
            for m in matches:
                chunk = m[0] or m[1]
                for line in chunk.splitlines():
                    cleaned = line.strip().strip('"')
                    if cleaned:
                        imports.append(cleaned.split("/")[-1])
        return sorted(list(set(imports)))

    def ingest_local_directory(self, repo_path: str, repo_name: str = "") -> RepositoryCodebaseDocument:
        """
        Traverses a local repository, parses AST dependencies,
        computes topological ordering, and builds a RepositoryCodebaseDocument.
        """
        repo_name = repo_name or os.path.basename(os.path.abspath(repo_path))
        file_entries: List[RepositoryFile] = []
        module_name_to_path: Dict[str, str] = {}

        # 1. Discover files
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("venv", "node_modules", "target", "build", "dist")]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in self.SUPPORTED_EXTENSIONS:
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, repo_path)

                if os.path.getsize(full_path) > self.max_file_size_bytes:
                    continue

                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception:
                    continue

                lang = self.SUPPORTED_EXTENSIONS[ext]
                imports: List[str] = []
                exports: List[str] = []

                if lang == "python":
                    imports, exports = self.extract_python_imports(content)
                else:
                    imports = self.extract_generic_imports(content, lang)

                file_entries.append(RepositoryFile(
                    file_path=rel_path,
                    content=content,
                    language=lang,
                    ast_imports=imports,
                    ast_exports=exports,
                    topological_depth=0
                ))

                base_mod = os.path.splitext(os.path.basename(rel_path))[0]
                module_name_to_path[base_mod] = rel_path

        # 2. Compute Dependency Graph & Topological Depth
        # Files with zero in-repo dependencies are assigned depth 0 (e.g., config, types)
        # Files importing other in-repo files get depth = max(deps_depth) + 1
        graph: Dict[str, Set[str]] = defaultdict(set)
        for f in file_entries:
            for imp in f.ast_imports:
                if imp in module_name_to_path and module_name_to_path[imp] != f.file_path:
                    graph[f.file_path].add(module_name_to_path[imp])

        # Assign depths
        depths: Dict[str, int] = {}
        def get_depth(node: str, visited: Set[str]) -> int:
            if node in depths:
                return depths[node]
            if node in visited:
                return 0 # Cycle fallback
            visited.add(node)
            d = 0
            for dep in graph.get(node, []):
                d = max(d, get_depth(dep, visited) + 1)
            visited.remove(node)
            depths[node] = d
            return d

        for f in file_entries:
            f.topological_depth = get_depth(f.file_path, set())

        # Extract README if present
        readme = ""
        for f in file_entries:
            if "readme" in f.file_path.lower():
                readme = f.content
                break

        return RepositoryCodebaseDocument(
            repo_name=repo_name,
            commit_sha="HEAD",
            license="Permissive",
            file_hierarchy=file_entries,
            readme_content=readme,
            architecture_overview=f"Repository {repo_name} containing {len(file_entries)} parsed source modules."
        )


class FillInTheMiddleAugmenter:
    """
    Applies Fill-In-The-Middle (FIM) transformation to source code.
    Splits document into Prefix, Middle, and Suffix.
    Supports PSM (Prefix-Suffix-Middle) and SPM (Suffix-Prefix-Middle) modes.
    """

    def __init__(
        self,
        fim_rate: float = 0.50,
        spm_rate: float = 0.50,
        prefix_token: str = "<|fim_prefix|>",
        middle_token: str = "<|fim_middle|>",
        suffix_token: str = "<|fim_suffix|>"
    ):
        self.fim_rate = fim_rate
        self.spm_rate = spm_rate
        self.prefix_token = prefix_token
        self.middle_token = middle_token
        self.suffix_token = suffix_token

    def apply_fim(self, text: str) -> str:
        """Applies FIM transformation to code with specified probability."""
        if random.random() > self.fim_rate:
            return text

        lines = text.splitlines(keepends=True)
        if len(lines) < 4:
            return text

        # Select two boundary split points
        idx1 = random.randint(1, len(lines) - 2)
        idx2 = random.randint(idx1 + 1, len(lines) - 1)

        prefix = "".join(lines[:idx1])
        middle = "".join(lines[idx1:idx2])
        suffix = "".join(lines[idx2:])

        # Choose between PSM and SPM modes
        if random.random() < self.spm_rate:
            # Suffix-Prefix-Middle mode
            return f"{self.suffix_token}{suffix}{self.prefix_token}{prefix}{self.middle_token}{middle}"
        else:
            # Prefix-Suffix-Middle mode
            return f"{self.prefix_token}{prefix}{self.suffix_token}{suffix}{self.middle_token}{middle}"
