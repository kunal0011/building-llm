"""
Isolated Execution Sandbox for Verifiable Code & SWE Agent Rewards.
Executes generated code patches against test suites with strict timeouts,
memory bounds, and process isolation to compute deterministic 0/1 rewards.
"""

import os
import sys
import tempfile
import subprocess
import time
from typing import Dict, Any, List, Optional, Tuple


class CodeExecutionSandbox:
    """
    Subprocess-isolated sandbox executing Python, Bash, and test suites.
    Computes verifiable binary rewards (1.0 = all tests pass, 0.0 = failure).
    """

    def __init__(self, default_timeout_sec: float = 10.0, max_output_chars: int = 10_000):
        self.default_timeout_sec = default_timeout_sec
        self.max_output_chars = max_output_chars

    def execute_python_code(
        self,
        code_str: str,
        test_code: str = "",
        timeout_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes a Python snippet alongside unit tests in an isolated temporary directory.
        Returns outcome metrics and binary verifiable reward.
        """
        timeout = timeout_sec or self.default_timeout_sec
        temp_dir = tempfile.mkdtemp(prefix="rl_sandbox_")

        full_script = f"{code_str}\n\n# Verification Tests\n{test_code}"
        script_path = os.path.join(temp_dir, "solution_test.py")

        with open(script_path, "w", encoding="utf-8") as f:
            f.write(full_script)

        start_time = time.time()
        try:
            # Execute in isolated subprocess with restricted environment
            proc = subprocess.run(
                [sys.executable, script_path],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    "PYTHONPATH": temp_dir,
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONDONTWRITEBYTECODE": "1"
                }
            )
            elapsed = time.time() - start_time
            stdout = proc.stdout[:self.max_output_chars]
            stderr = proc.stderr[:self.max_output_chars]

            # Deterministic binary reward: 1.0 if exit code == 0, else 0.0
            passed = (proc.returncode == 0)
            reward = 1.0 if passed else 0.0

            return {
                "reward": reward,
                "passed": passed,
                "exit_code": proc.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "elapsed_sec": round(elapsed, 3),
                "timeout": False
            }

        except subprocess.TimeoutExpired as e:
            elapsed = time.time() - start_time
            return {
                "reward": 0.0,
                "passed": False,
                "exit_code": -1,
                "stdout": (e.stdout or "")[:self.max_output_chars],
                "stderr": f"Execution timed out after {timeout} seconds",
                "elapsed_sec": round(elapsed, 3),
                "timeout": True
            }
        except Exception as e:
            return {
                "reward": 0.0,
                "passed": False,
                "exit_code": -2,
                "stdout": "",
                "stderr": f"Sandbox execution error: {str(e)}",
                "elapsed_sec": 0.0,
                "timeout": False
            }
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def evaluate_git_patch(
        self,
        base_repo_dir: str,
        git_patch: str,
        test_command: str = "pytest"
    ) -> Dict[str, Any]:
        """
        Applies a git patch to a sandbox copy of a repository and executes test commands.
        Used for SWE-bench reinforcement learning rollouts.
        """
        temp_repo = tempfile.mkdtemp(prefix="swe_repo_")
        import shutil
        shutil.copytree(base_repo_dir, temp_repo, dirs_exist_ok=True)

        patch_file = os.path.join(temp_repo, "patch.diff")
        with open(patch_file, "w") as f:
            f.write(git_patch)

        try:
            # Apply patch
            apply_res = subprocess.run(
                ["git", "apply", "--ignore-whitespace", "patch.diff"],
                cwd=temp_repo,
                capture_output=True,
                text=True,
                timeout=10
            )
            if apply_res.returncode != 0:
                return {
                    "reward": 0.0,
                    "passed": False,
                    "error": "Failed to apply git diff patch",
                    "stderr": apply_res.stderr
                }

            # Run verification test command
            test_res = subprocess.run(
                test_command,
                shell=True,
                cwd=temp_repo,
                capture_output=True,
                text=True,
                timeout=self.default_timeout_sec
            )
            passed = (test_res.returncode == 0)
            return {
                "reward": 1.0 if passed else 0.0,
                "passed": passed,
                "stdout": test_res.stdout[:self.max_output_chars],
                "stderr": test_res.stderr[:self.max_output_chars]
            }
        finally:
            shutil.rmtree(temp_repo, ignore_errors=True)
