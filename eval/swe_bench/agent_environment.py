"""
Isolated Sandbox Environment for SWE-bench Multi-Turn Agent Evaluation.
Sets up isolated task workspaces, tracks file system modifications, executes bash
commands and file editor actions, and provides deterministic verification.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import time
from typing import Dict, Any, List, Optional, Tuple

from eval.swe_bench.task_loader import SWEBenchInstance


class AgentEnvironment:
    """
    Subprocess-isolated workspace executing agent tools on a specific SWE-bench task.
    """

    def __init__(self, task: SWEBenchInstance, timeout_sec: float = 10.0, max_output_chars: int = 10_000):
        self.task = task
        self.timeout_sec = timeout_sec
        self.max_output_chars = max_output_chars
        self.workspace_dir = tempfile.mkdtemp(prefix=f"swe_eval_{task.instance_id.replace('__', '_')}_")
        self._init_workspace()

    def _init_workspace(self) -> None:
        """Initializes the repository directory and writes initial source code."""
        # Create directory tree for affected file with package __init__.py files
        target_path = os.path.join(self.workspace_dir, self.task.affected_file)
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(self.task.initial_file_content)

        # Create __init__.py along package hierarchy
        rel_dir = os.path.dirname(self.task.affected_file)
        curr = self.workspace_dir
        for part in rel_dir.split(os.sep):
            if part:
                curr = os.path.join(curr, part)
                init_file = os.path.join(curr, "__init__.py")
                if not os.path.exists(init_file):
                    with open(init_file, "w", encoding="utf-8") as f:
                        f.write("# Package init\n")

        # Write reproduction test script
        repro_path = os.path.join(self.workspace_dir, "reproduce_issue.py")
        with open(repro_path, "w", encoding="utf-8") as f:
            f.write(self.task.reproduction_script)

        # Write golden regression test suite
        test_path = os.path.join(self.workspace_dir, "test_suite.py")
        with open(test_path, "w", encoding="utf-8") as f:
            f.write(self.task.test_code)

        # Initialize git repo in workspace for clean diff tracking
        try:
            subprocess.run(["git", "init"], cwd=self.workspace_dir, capture_output=True, check=False)
            subprocess.run(["git", "config", "user.name", "SWE-Evaluator"], cwd=self.workspace_dir, capture_output=True, check=False)
            subprocess.run(["git", "config", "user.email", "eval@agent.org"], cwd=self.workspace_dir, capture_output=True, check=False)
            subprocess.run(["git", "add", "."], cwd=self.workspace_dir, capture_output=True, check=False)
            subprocess.run(["git", "commit", "-m", "Initial baseline state"], cwd=self.workspace_dir, capture_output=True, check=False)
        except Exception:
            pass

    def execute_bash(self, command: str) -> Dict[str, Any]:
        """Executes a shell command in the isolated workspace directory."""
        start_t = time.time()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
                env={
                    "PYTHONPATH": self.workspace_dir,
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1"
                }
            )
            elapsed = time.time() - start_t
            output = (proc.stdout + proc.stderr)[:self.max_output_chars]
            return {
                "exit_code": proc.returncode,
                "output": output,
                "duration_sec": round(elapsed, 3),
                "timed_out": False
            }
        except subprocess.TimeoutExpired:
            return {
                "exit_code": 124,
                "output": f"Command timed out after {self.timeout_sec} seconds.",
                "duration_sec": self.timeout_sec,
                "timed_out": True
            }
        except Exception as e:
            return {
                "exit_code": 1,
                "output": str(e),
                "duration_sec": 0.0,
                "timed_out": False
            }

    def execute_file_editor(
        self,
        action: str,  # "view", "str_replace", "create", "write"
        path: str,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        content: Optional[str] = None,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """Interacts with source files: viewing lines or replacing substrings."""
        full_path = os.path.join(self.workspace_dir, path)

        if action == "view":
            if not os.path.exists(full_path):
                return {"success": False, "error": f"File '{path}' does not exist."}
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            start = (start_line - 1) if start_line and start_line > 0 else 0
            end = end_line if end_line and end_line <= len(lines) else len(lines)
            selected = lines[start:end]
            numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(selected, start=start)]
            return {
                "success": True,
                "content": "".join(numbered),
                "total_lines": len(lines)
            }

        elif action == "str_replace":
            if not os.path.exists(full_path):
                return {"success": False, "error": f"File '{path}' does not exist."}
            if old_str is None or new_str is None:
                return {"success": False, "error": "Both 'old_str' and 'new_str' must be specified for str_replace."}
            with open(full_path, "r", encoding="utf-8") as f:
                file_text = f.read()

            if old_str not in file_text:
                return {"success": False, "error": f"Target substring not found in '{path}'."}

            occurrences = file_text.count(old_str)
            if occurrences > 1:
                return {"success": False, "error": f"Target substring found {occurrences} times. Must be unique."}

            # Check for Python indentation/syntax safety
            modified_text = file_text.replace(old_str, new_str, 1)
            if path.endswith(".py"):
                try:
                    import ast
                    ast.parse(modified_text)
                except (IndentationError, SyntaxError) as e:
                    return {
                        "success": False,
                        "error": f"{type(e).__name__}: {str(e)} on line {getattr(e, 'lineno', '?')}. File not modified."
                    }

            with open(full_path, "w", encoding="utf-8") as f:
                f.write(modified_text)

            return {
                "success": True,
                "message": f"Successfully replaced 1 occurrence in '{path}'."
            }

        elif action in ("create", "write"):
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content or "")
            return {"success": True, "message": f"Wrote {len(content or '')} chars to '{path}'."}

        return {"success": False, "error": f"Unknown action '{action}'."}

    def get_git_diff(self) -> str:
        """Retrieves the unified diff of changes made by the agent."""
        res = self.execute_bash("git diff")
        return res["output"]

    def run_reproduction_test(self) -> Dict[str, Any]:
        """Runs the standalone reproduction script (expected to fail on buggy code, pass on patched code)."""
        return self.execute_bash(f"python3 reproduce_issue.py")

    def run_regression_test_suite(self) -> Dict[str, Any]:
        """Runs the official evaluation regression test suite to verify 0/1 correctness."""
        return self.execute_bash(f"python3 test_suite.py")

    def cleanup(self) -> None:
        """Removes the temporary workspace directory."""
        if os.path.exists(self.workspace_dir):
            try:
                shutil.rmtree(self.workspace_dir)
            except Exception:
                pass
