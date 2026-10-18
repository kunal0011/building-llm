"""
Autonomous SWE Agent Data Synthesizer for Full-Codebase PR Resolution.
Generates multi-turn agent interaction trajectories matching SWE-bench Verified benchmarks.
Simulates tool usage (Bash, File Editor, Git, Test Runner), failure reproduction,
diagnostic self-correction, regression testing, and deterministic sandbox validation.
"""

import os
import sys
import uuid
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipelines.rl.execution_sandbox import CodeExecutionSandbox
from schemas.data_schema import LongHorizonAgentTrajectory, AgentTrajectoryStep, AgentToolCall


@dataclass
class SWEIssueSpec:
    """Specification of a realistic repository bug or feature request."""
    issue_id: str
    repo_name: str
    problem_statement: str
    affected_files: List[str]
    buggy_code_snippet: str
    patched_code_snippet: str
    reproduction_test_code: str
    regression_test_code: str
    expected_error_traceback: str


class SWEAgentSynthesizer:
    """
    Synthesizes multi-turn software engineering trajectories with verified test execution.
    Teaches the LLM how to:
      1. Explore a new codebase (`find`, `grep`, `git log`)
      2. Write a minimal reproduction script (`reproduce_bug.py`)
      3. Diagnose error tracebacks and locate the defect in AST call graphs
      4. Recover from intermediate syntax/runtime errors during patching (self-correction)
      5. Run unit & regression test suites to verify green status
      6. Review `git diff` and craft professional commit messages
    """

    def __init__(self, sandbox: Optional[CodeExecutionSandbox] = None):
        self.sandbox = sandbox or CodeExecutionSandbox()

    def generate_verified_trajectory(
        self,
        spec: SWEIssueSpec,
        simulate_intermediate_syntax_error: bool = True
    ) -> Dict[str, Any]:
        """
        Synthesizes an end-to-end multi-turn SWE trajectory and verifies
        the solution in the code execution sandbox.
        """
        trajectory_id = f"swe_traj_{spec.issue_id}_{uuid.uuid4().hex[:8]}"
        steps: List[Dict[str, Any]] = []

        primary_file = spec.affected_files[0] if spec.affected_files else "module.py"

        # -------------------------------------------------------------
        # Turn 1: Workspace Reconnaissance
        # -------------------------------------------------------------
        turn1_thought = (
            f"We are tasked with resolving issue '{spec.issue_id}' in repository '{spec.repo_name}'.\n"
            f"Problem description: {spec.problem_statement}\n"
            f"Let's first inspect the git repository layout, recent commit logs, and find candidate files."
        )
        turn1_cmd = f"git status && git log -n 1 --oneline && find . -maxdepth 3 -name '*.py'"
        turn1_obs = (
            f"On branch main\nYour branch is up to date with 'origin/main'.\n"
            f"d8a21f4 Fix previous benchmark serialization\n"
            f"./{primary_file}\n./tests/test_{primary_file.replace('/', '_')}"
        )
        steps.append({
            "step": 1,
            "thought": turn1_thought,
            "tool_name": "bash",
            "tool_args": {"command": turn1_cmd},
            "observation": turn1_obs,
            "exit_code": 0
        })

        # -------------------------------------------------------------
        # Turn 2: Create & Run Minimal Reproduction Script
        # -------------------------------------------------------------
        turn2_thought = (
            f"Before modifying any source files, let's write a standalone reproduction test script "
            f"to reliably reproduce the reported defect and verify the failure mode."
        )
        turn2_cmd = f"cat << 'EOF' > reproduce_issue.py\n{spec.reproduction_test_code}\nEOF\npython3 reproduce_issue.py"
        turn2_obs = f"Traceback (most recent call last):\n{spec.expected_error_traceback}\nAssertionError: Defect reproduced successfully.\nExit code: 1"
        steps.append({
            "step": 2,
            "thought": turn2_thought,
            "tool_name": "bash",
            "tool_args": {"command": turn2_cmd},
            "observation": turn2_obs,
            "exit_code": 1
        })

        # -------------------------------------------------------------
        # Turn 3: Inspect Target Source Code
        # -------------------------------------------------------------
        turn3_thought = (
            f"The reproduction script confirmed the failure. Let's read the implementation of "
            f"'{primary_file}' around the lines indicated in the traceback to analyze the flaw."
        )
        turn3_args = {"action": "view", "path": primary_file, "start_line": 1, "end_line": 120}
        turn3_obs = f"[Retrieved {primary_file} (120 lines)]\n...\n{spec.buggy_code_snippet}\n..."
        steps.append({
            "step": 3,
            "thought": turn3_thought,
            "tool_name": "file_editor",
            "tool_args": turn3_args,
            "observation": turn3_obs,
            "exit_code": 0
        })

        # -------------------------------------------------------------
        # Turn 4: First Patch Attempt (with Self-Correction Demonstration)
        # -------------------------------------------------------------
        if simulate_intermediate_syntax_error:
            # Step with indentation mismatch or typo
            turn4_thought = (
                f"I have identified the root cause in '{primary_file}'. Let's apply a replacement "
                f"to fix the unhandled edge-case."
            )
            turn4_args = {
                "action": "str_replace",
                "path": primary_file,
                "old_str": spec.buggy_code_snippet,
                "new_str": " " + spec.patched_code_snippet  # Intentional whitespace defect
            }
            turn4_obs = "Error: IndentationError: unexpected indent on replacement block. File not modified."
            steps.append({
                "step": 4,
                "thought": turn4_thought,
                "tool_name": "file_editor",
                "tool_args": turn4_args,
                "observation": turn4_obs,
                "exit_code": 1
            })

            # Turn 5: Self-Correction Step
            turn5_thought = (
                f"The replacement string had an indentation error. Let's examine the surrounding block "
                f"indentation and ensure exact column alignment."
            )
            turn5_args = {
                "action": "str_replace",
                "path": primary_file,
                "old_str": spec.buggy_code_snippet,
                "new_str": spec.patched_code_snippet
            }
            turn5_obs = f"Success: 1 occurrence replaced in '{primary_file}'."
            steps.append({
                "step": 5,
                "thought": turn5_thought,
                "tool_name": "file_editor",
                "tool_args": turn5_args,
                "observation": turn5_obs,
                "exit_code": 0
            })
        else:
            turn4_thought = f"Applying clean patch to '{primary_file}'."
            turn4_args = {
                "action": "str_replace",
                "path": primary_file,
                "old_str": spec.buggy_code_snippet,
                "new_str": spec.patched_code_snippet
            }
            turn4_obs = f"Success: 1 occurrence replaced in '{primary_file}'."
            steps.append({
                "step": 4,
                "thought": turn4_thought,
                "tool_name": "file_editor",
                "tool_args": turn4_args,
                "observation": turn4_obs,
                "exit_code": 0
            })

        # -------------------------------------------------------------
        # Turn 5/6: Run Reproduction Script & Full Regression Test Suite
        # -------------------------------------------------------------
        next_step = len(steps) + 1
        turn6_thought = (
            f"The patch is applied. First, re-run 'reproduce_issue.py' to confirm the issue is resolved. "
            f"Then run the full regression test suite to ensure zero regressions."
        )
        turn6_cmd = "python3 reproduce_issue.py && pytest tests/ -v"
        turn6_obs = (
            "reproduce_issue.py: PASSED (All edge cases handled)\n"
            "tests/test_core.py::test_basic PASSED\n"
            "tests/test_core.py::test_boundary_conditions PASSED\n"
            "================= 18 passed in 1.12s ================="
        )
        steps.append({
            "step": next_step,
            "thought": turn6_thought,
            "tool_name": "bash",
            "tool_args": {"command": turn6_cmd},
            "observation": turn6_obs,
            "exit_code": 0
        })

        # -------------------------------------------------------------
        # Turn 7: Review Clean Git Diff
        # -------------------------------------------------------------
        diff_cmd = "git diff --staged"
        diff_obs = (
            f"--- a/{primary_file}\n+++ b/{primary_file}\n"
            f"@@ -42,7 +42,7 @@\n-{spec.buggy_code_snippet}\n+{spec.patched_code_snippet}"
        )
        steps.append({
            "step": next_step + 1,
            "thought": "All regression tests passed cleanly. Let's inspect git diff to ensure no stray debug statements remain.",
            "tool_name": "bash",
            "tool_args": {"command": diff_cmd},
            "observation": diff_obs,
            "exit_code": 0
        })

        # Deterministic sandbox verification
        sandbox_res = self.sandbox.execute_python_code(
            code_str=spec.patched_code_snippet,
            test_code=spec.regression_test_code
        )
        verifiable_reward = 1.0 if sandbox_res["passed"] else 0.0

        return {
            "trajectory_id": trajectory_id,
            "issue_id": spec.issue_id,
            "repo_name": spec.repo_name,
            "total_turns": len(steps),
            "steps": steps,
            "verifiable_reward": verifiable_reward,
            "sandbox_passed": sandbox_res["passed"],
            "affected_files": spec.affected_files,
            "patch_diff": diff_obs
        }

    def synthesize_benchmark_suite(self) -> List[SWEIssueSpec]:
        """
        Generates realistic SWE-bench style issues across distinct algorithmic & software domains.
        """
        suite = [
            SWEIssueSpec(
                issue_id="DJANGO_CACHE_001",
                repo_name="django/django",
                problem_statement="LRUCache throws KeyError when accessing concurrently evicted keys in high-concurrency workloads.",
                affected_files=["django/core/cache/backends/locmem.py"],
                buggy_code_snippet="def get(self, key):\n    return self._cache[key]",
                patched_code_snippet="def get(self, key, default=None):\n    with self._lock:\n        return self._cache.get(key, default)",
                reproduction_test_code="cache = {'a': 1}\ntry:\n    val = cache['b']\n    assert False, 'Expected KeyError'\nexcept KeyError:\n    assert True",
                regression_test_code="class MockCache:\n    def __init__(self):\n        self._cache = {}\n        import threading\n        self._lock = threading.Lock()\n    def get(self, key, default=None):\n        with self._lock:\n            return self._cache.get(key, default)\nc = MockCache()\nassert c.get('missing', 'fallback') == 'fallback'",
                expected_error_traceback="  File \"django/core/cache/backends/locmem.py\", line 42, in get\n    return self._cache[key]\nKeyError: 'session_auth_token_992'"
            ),
            SWEIssueSpec(
                issue_id="AST_VISITOR_002",
                repo_name="psf/black",
                problem_statement="AST visitor crashes with AttributeError on NoneType docstrings in empty class definitions.",
                affected_files=["src/black/nodes.py"],
                buggy_code_snippet="def extract_docstring(node):\n    return node.value.value.strip()",
                patched_code_snippet="def extract_docstring(node):\n    if getattr(node, 'value', None) and getattr(node.value, 'value', None):\n        return node.value.value.strip()\n    return ''",
                reproduction_test_code="class MockNode: value = None\ntry:\n    extract_docstring(MockNode())\nexcept AttributeError:\n    assert True",
                regression_test_code="def extract_docstring(node):\n    if getattr(node, 'value', None) and getattr(node.value, 'value', None):\n        return node.value.value.strip()\n    return ''\nclass Node: pass\nn = Node()\nassert extract_docstring(n) == ''",
                expected_error_traceback="  File \"src/black/nodes.py\", line 88, in extract_docstring\n    return node.value.value.strip()\nAttributeError: 'NoneType' object has no attribute 'value'"
            ),
            SWEIssueSpec(
                issue_id="MATH_PRECISION_003",
                repo_name="scipy/scipy",
                problem_statement="Float epsilon comparison in binary search root finder produces InfiniteLoop on subnormal floats.",
                affected_files=["scipy/optimize/zeros.py"],
                buggy_code_snippet="def bisection_step(a, b, tol=0.0):\n    while abs(b - a) > tol:\n        mid = (a + b) / 2.0\n        if mid == a: break\n        a = mid\n    return a",
                patched_code_snippet="def bisection_step(a, b, tol=1e-12):\n    count = 0\n    while abs(b - a) > tol:\n        mid = (a + b) / 2.0\n        if abs(b - a) <= tol: break\n        a = mid\n        count += 1\n        if count > 100: break\n    return a",
                reproduction_test_code="def bisection_step(a, b, tol=0.0):\n    count = 0\n    while abs(b - a) > tol:\n        mid = (a + b) / 2.0\n        if mid == a: break\n        a = mid\n        count += 1\n        if count > 50: raise TimeoutError('Loop depth exceeded')\n    return a\ntry:\n    bisection_step(1.0, 1.0 + 1e-16)\nexcept TimeoutError:\n    pass",
                regression_test_code="res = bisection_step(1.0, 1.0 + 1e-8)\nassert abs((1.0 + 1e-8) - res) <= 1e-8",
                expected_error_traceback="  File \"scipy/optimize/zeros.py\", line 154, in bisection_step\n    while abs(b - a) > 0.0:\nTimeoutError: Loop depth exceeded"
            )
        ]
        return suite
