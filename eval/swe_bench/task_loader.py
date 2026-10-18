"""
SWE-bench Evaluation Benchmark Task Loader.
Manages repository specifications, reproduction scripts, and test suite patches
across diverse software engineering domains.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class SWEBenchInstance:
    """Standardized representation of a SWE-bench evaluation instance."""
    instance_id: str
    repo: str
    problem_statement: str
    affected_file: str
    initial_file_content: str
    buggy_snippet: str
    patched_snippet: str
    reproduction_script: str
    test_code: str
    gold_patch: str = ""
    hints: str = ""
    environment_setup_cmd: str = "python3 -m pip install -e ."


class SWEBenchTaskLoader:
    """Loads standardized SWE-bench verified benchmark tasks."""

    @staticmethod
    def get_verified_benchmark_suite() -> List[SWEBenchInstance]:
        """Provides a verified benchmark suite covering core open-source repositories."""
        return [
            SWEBenchInstance(
                instance_id="django__django-16379",
                repo="django/django",
                problem_statement="KeyError thrown when accessing evicted items in high-concurrency LRU cache.",
                affected_file="django/core/cache/locmem.py",
                initial_file_content=(
                    "class LocMemCache:\n"
                    "    def __init__(self):\n"
                    "        self._cache = {}\n"
                    "    def get(self, key):\n"
                    "        return self._cache[key]\n"
                ),
                buggy_snippet="    def get(self, key):\n        return self._cache[key]",
                patched_snippet="    def get(self, key, default=None):\n        return self._cache.get(key, default)",
                gold_patch="    def get(self, key, default=None):\n        return self._cache.get(key, default)",
                reproduction_script=(
                    "from django.core.cache.locmem import LocMemCache\n"
                    "c = LocMemCache()\n"
                    "try:\n"
                    "    v = c.get('missing_key')\n"
                    "    assert False, 'Expected KeyError on missing key'\n"
                    "except KeyError:\n"
                    "    pass\n"
                ),
                test_code=(
                    "from django.core.cache.locmem import LocMemCache\n"
                    "c = LocMemCache()\n"
                    "assert c.get('missing_key', 'fallback') == 'fallback'\n"
                    "c._cache['present'] = 100\n"
                    "assert c.get('present') == 100\n"
                ),
                hints="Ensure cache.get accepts an optional default value parameter."
            ),
            SWEBenchInstance(
                instance_id="psf__requests-7102",
                repo="psf/requests",
                problem_statement="Single float passed to timeout argument raises TypeError when unpacking connect/read tuple.",
                affected_file="requests/adapters.py",
                initial_file_content=(
                    "class HTTPAdapter:\n"
                    "    def send(self, request, timeout=None):\n"
                    "        connect, read = timeout\n"
                    "        return f'Connected in {connect}s, Read in {read}s'\n"
                ),
                buggy_snippet="        connect, read = timeout",
                patched_snippet=(
                    "        if isinstance(timeout, (int, float)):\n"
                    "            connect, read = float(timeout), float(timeout)\n"
                    "        else:\n"
                    "            connect, read = float(timeout[0]), float(timeout[1])"
                ),
                gold_patch=(
                    "        if isinstance(timeout, (int, float)):\n"
                    "            connect, read = float(timeout), float(timeout)\n"
                    "        else:\n"
                    "            connect, read = float(timeout[0]), float(timeout[1])"
                ),
                reproduction_script=(
                    "from requests.adapters import HTTPAdapter\n"
                    "adapter = HTTPAdapter()\n"
                    "try:\n"
                    "    adapter.send(None, timeout=5.0)\n"
                    "    assert False, 'Expected TypeError'\n"
                    "except TypeError:\n"
                    "    pass\n"
                ),
                test_code=(
                    "from requests.adapters import HTTPAdapter\n"
                    "adapter = HTTPAdapter()\n"
                    "assert adapter.send(None, timeout=5.0) == 'Connected in 5.0s, Read in 5.0s'\n"
                    "assert adapter.send(None, timeout=(2.5, 7.5)) == 'Connected in 2.5s, Read in 7.5s'\n"
                ),
                hints="Allow timeout to be either a numeric scalar or a 2-tuple."
            ),
            SWEBenchInstance(
                instance_id="pallets__flask-4981",
                repo="pallets/flask",
                problem_statement="Trailing slash URL redirect strips query parameters from request path.",
                affected_file="flask/helpers.py",
                initial_file_content=(
                    "def redirect_canonical(path):\n"
                    "    return path.rstrip('/')\n"
                ),
                buggy_snippet="def redirect_canonical(path):\n    return path.rstrip('/')",
                patched_snippet=(
                    "def redirect_canonical(path):\n"
                    "    parts = path.split('?', 1)\n"
                    "    base = parts[0].rstrip('/')\n"
                    "    return f'{base}?{parts[1]}' if len(parts) > 1 else base"
                ),
                gold_patch=(
                    "def redirect_canonical(path):\n"
                    "    parts = path.split('?', 1)\n"
                    "    base = parts[0].rstrip('/')\n"
                    "    return f'{base}?{parts[1]}' if len(parts) > 1 else base"
                ),
                reproduction_script=(
                    "from flask.helpers import redirect_canonical\n"
                    "res = redirect_canonical('/api/v1/users/?page=2')\n"
                    "assert res == '/api/v1/users', 'Query params were not stripped as expected in buggy version'\n"
                ),
                test_code=(
                    "from flask.helpers import redirect_canonical\n"
                    "assert redirect_canonical('/api/v1/users/?page=2') == '/api/v1/users?page=2'\n"
                    "assert redirect_canonical('/docs/') == '/docs'\n"
                ),
                hints="Split path on query string separator prior to stripping trailing slash."
            ),
            SWEBenchInstance(
                instance_id="psf__black-3042",
                repo="psf/black",
                problem_statement="AttributeError on NoneType node value when formatting empty class docstring.",
                affected_file="black/nodes.py",
                initial_file_content=(
                    "def extract_docstring(node):\n"
                    "    return node.value.value.strip()\n"
                ),
                buggy_snippet="def extract_docstring(node):\n    return node.value.value.strip()",
                patched_snippet=(
                    "def extract_docstring(node):\n"
                    "    if getattr(node, 'value', None) and getattr(node.value, 'value', None):\n"
                    "        return node.value.value.strip()\n"
                    "    return ''"
                ),
                gold_patch=(
                    "def extract_docstring(node):\n"
                    "    if getattr(node, 'value', None) and getattr(node.value, 'value', None):\n"
                    "        return node.value.value.strip()\n"
                    "    return ''"
                ),
                reproduction_script=(
                    "from black.nodes import extract_docstring\n"
                    "class EmptyNode: value = None\n"
                    "try:\n"
                    "    extract_docstring(EmptyNode())\n"
                    "    assert False, 'Expected AttributeError'\n"
                    "except AttributeError:\n"
                    "    pass\n"
                ),
                test_code=(
                    "from black.nodes import extract_docstring\n"
                    "class EmptyNode: value = None\n"
                    "assert extract_docstring(EmptyNode()) == ''\n"
                    "class Val: value = '  doc  '\n"
                    "class FullNode: value = Val()\n"
                    "assert extract_docstring(FullNode()) == 'doc'\n"
                ),
                hints="Add null checks with getattr before accessing nested node.value attributes."
            )
        ]

    @staticmethod
    def load_from_jsonl(file_path: str) -> List[SWEBenchInstance]:
        """Loads instances from a JSONL file."""
        instances = []
        if not os.path.exists(file_path):
            return instances
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    instances.append(SWEBenchInstance(
                        instance_id=data["instance_id"],
                        repo=data["repo"],
                        problem_statement=data["problem_statement"],
                        affected_file=data["affected_file"],
                        initial_file_content=data["initial_file_content"],
                        buggy_snippet=data.get("buggy_snippet", ""),
                        patched_snippet=data.get("patched_snippet", ""),
                        reproduction_script=data["reproduction_script"],
                        test_code=data["test_code"],
                        gold_patch=data.get("gold_patch", ""),
                        hints=data.get("hints", "")
                    ))
        return instances
