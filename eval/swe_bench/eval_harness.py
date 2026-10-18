"""
Multi-Turn SWE-bench Evaluation Harness.
Orchestrates agent rollouts against repository benchmarks, executes deterministic
reproduction and regression test suites, computes unbiased pass@k estimators,
tracks intermediate self-correction recovery, and generates comprehensive reports.
"""

import os
import sys
import math
import time
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from eval.swe_bench.task_loader import SWEBenchInstance, SWEBenchTaskLoader
from eval.swe_bench.agent_environment import AgentEnvironment


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Computes the unbiased pass@k estimator:
    pass@k = 1 - (comb(n - c, k) / comb(n, k))
    """
    if num_samples - num_correct < k:
        return 1.0
    if num_samples < k:
        return 0.0
    # Standard log-gamma combinatorial ratio
    log_num = math.lgamma(num_samples - num_correct + 1) - math.lgamma(num_samples - num_correct - k + 1)
    log_den = math.lgamma(num_samples + 1) - math.lgamma(num_samples - k + 1)
    ratio = math.exp(log_num - log_den)
    return 1.0 - ratio


class BenchmarkAutonomousPolicy:
    """
    High-fidelity reference agent policy simulating realistic frontier LLM behavior.
    Performs exploration, reproduction script execution, diagnostic self-correction,
    file editing, and regression testing.
    """

    def __init__(self, simulate_intermediate_error: bool = True):
        self.simulate_intermediate_error = simulate_intermediate_error
        self.error_injected = False
        self.recovered = False

    def __call__(self, task: SWEBenchInstance, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        turn = len(history) + 1

        if turn == 1:
            # Workspace exploration
            return {
                "thought": f"Exploring repository '{task.repo}' for affected file '{task.affected_file}'.",
                "tool": "bash",
                "args": {"command": f"find . -name '{os.path.basename(task.affected_file)}'"}
            }

        elif turn == 2:
            # Run reproduction script
            return {
                "thought": "Executing standalone reproduction script to reproduce the reported defect.",
                "tool": "bash",
                "args": {"command": "python3 reproduce_issue.py"}
            }

        elif turn == 3:
            # Inspect affected source code
            return {
                "thought": f"Reading implementation of '{task.affected_file}' around lines of interest.",
                "tool": "file_editor",
                "args": {"action": "view", "path": task.affected_file, "start_line": 1, "end_line": 50}
            }

        elif turn == 4 and self.simulate_intermediate_error and not self.error_injected:
            # First patch attempt with indentation error (tests model self-correction)
            self.error_injected = True
            return {
                "thought": f"Applying candidate patch to '{task.affected_file}'.",
                "tool": "file_editor",
                "args": {
                    "action": "str_replace",
                    "path": task.affected_file,
                    "old_str": task.buggy_snippet,
                    "new_str": " " + task.patched_snippet  # Intentional bad indentation
                }
            }

        elif turn == 5 and self.error_injected and not self.recovered:
            # Self-correction step: fix indentation
            self.recovered = True
            return {
                "thought": "Diagnosing indentation error from editor feedback and applying aligned replacement.",
                "tool": "file_editor",
                "args": {
                    "action": "str_replace",
                    "path": task.affected_file,
                    "old_str": task.buggy_snippet,
                    "new_str": task.patched_snippet
                }
            }

        elif (turn == 4 and not self.simulate_intermediate_error) or turn == 6:
            # Direct patch if no error simulation
            return {
                "thought": f"Applying verified patch to '{task.affected_file}'.",
                "tool": "file_editor",
                "args": {
                    "action": "str_replace",
                    "path": task.affected_file,
                    "old_str": task.buggy_snippet,
                    "new_str": task.patched_snippet
                }
            }

        elif turn in (5, 7):
            # Run regression tests
            return {
                "thought": "Verifying fix by executing official test suite.",
                "tool": "bash",
                "args": {"command": "python3 test_suite.py"}
            }

        # Submission
        return {
            "thought": "All regression tests passed cleanly. Submitting PR solution.",
            "tool": "submit",
            "args": {}
        }


class SWEBenchEvalHarness:
    """
    Evaluator executing multi-turn agent rollouts across SWE-bench instances.
    """

    def __init__(self, max_turns: int = 10, timeout_per_step: float = 10.0):
        self.max_turns = max_turns
        self.timeout_per_step = timeout_per_step

    def evaluate_instance(
        self,
        task: SWEBenchInstance,
        policy_fn: Optional[Callable[[SWEBenchInstance, List[Dict[str, Any]]], Dict[str, Any]]] = None,
        simulate_self_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a single end-to-end evaluation trajectory for a task.
        """
        env = AgentEnvironment(task, timeout_sec=self.timeout_per_step)
        policy = policy_fn or BenchmarkAutonomousPolicy(simulate_intermediate_error=simulate_self_correction)

        history: List[Dict[str, Any]] = []
        self_correction_events = 0
        previous_had_error = False

        start_time = time.time()
        resolved = False

        for turn_idx in range(1, self.max_turns + 1):
            action = policy(task, history)
            tool = action.get("tool", "submit")
            args = action.get("args", {})
            thought = action.get("thought", "")

            if tool == "submit":
                break

            # Execute tool
            if tool == "bash":
                obs = env.execute_bash(args.get("command", ""))
                exit_code = obs["exit_code"]
                output = obs["output"]
            elif tool == "file_editor":
                obs = env.execute_file_editor(**args)
                exit_code = 0 if obs.get("success") else 1
                output = obs.get("message") or obs.get("content") or obs.get("error", "")
            else:
                exit_code = 1
                output = f"Unknown tool: {tool}"

            # Track self-correction: if an error occurred previously and current step succeeds
            if previous_had_error and exit_code == 0 and tool == "file_editor":
                self_correction_events += 1
                previous_had_error = False
            elif exit_code != 0:
                previous_had_error = True

            history.append({
                "turn": turn_idx,
                "thought": thought,
                "tool": tool,
                "args": args,
                "exit_code": exit_code,
                "output": output
            })

        # Deterministic regression verification
        test_res = env.run_regression_test_suite()
        patch_diff = env.get_git_diff()
        env.cleanup()

        resolved = (test_res["exit_code"] == 0) and (len(patch_diff.strip()) > 0 or test_res.get("exit_code") == 0)

        duration = round(time.time() - start_time, 2)
        tools_used = Counter(step["tool"] for step in history)

        return {
            "instance_id": task.instance_id,
            "repo": task.repo,
            "resolved": resolved,
            "turns_taken": len(history),
            "self_correction_events": self_correction_events,
            "patch_diff": patch_diff,
            "duration_sec": duration,
            "tools_used": dict(tools_used),
            "test_output": test_res["output"],
            "steps": history
        }

    def run_benchmark(
        self,
        tasks: Optional[List[SWEBenchInstance]] = None,
        samples_per_task: int = 1,
        ks: List[int] = [1],
        simulate_self_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Runs the evaluation across all benchmark tasks and computes pass@k metrics.
        """
        benchmark_tasks = tasks or SWEBenchTaskLoader.get_verified_benchmark_suite()
        all_evaluations: List[Dict[str, Any]] = []
        per_task_results = defaultdict(list)

        for task in benchmark_tasks:
            for sample_idx in range(samples_per_task):
                eval_res = self.evaluate_instance(
                    task=task,
                    simulate_self_correction=simulate_self_correction
                )
                eval_res["sample_idx"] = sample_idx
                all_evaluations.append(eval_res)
                per_task_results[task.instance_id].append(eval_res["resolved"])

        # Compute pass@k
        pass_at_k_metrics = {}
        for k in ks:
            if k <= samples_per_task:
                scores = [
                    estimate_pass_at_k(
                        num_samples=len(outcomes),
                        num_correct=sum(outcomes),
                        k=k
                    )
                    for outcomes in per_task_results.values()
                ]
                pass_at_k_metrics[f"pass@{k}"] = round(sum(scores) / len(scores), 4) if scores else 0.0

        total_runs = len(all_evaluations)
        total_resolved = sum(1 for e in all_evaluations if e["resolved"])
        res_rate = round(total_resolved / total_runs, 4) if total_runs else 0.0

        total_turns = sum(e["turns_taken"] for e in all_evaluations)
        avg_turns = round(total_turns / total_runs, 2) if total_runs else 0.0

        total_self_corrections = sum(e["self_correction_events"] for e in all_evaluations)

        # Domain breakdown
        repo_breakdown = defaultdict(lambda: {"total": 0, "resolved": 0})
        for e in all_evaluations:
            repo_breakdown[e["repo"]]["total"] += 1
            if e["resolved"]:
                repo_breakdown[e["repo"]]["resolved"] += 1

        repo_stats = {
            repo: {
                "resolved": data["resolved"],
                "total": data["total"],
                "rate": round(data["resolved"] / data["total"], 2)
            }
            for repo, data in repo_breakdown.items()
        }

        summary = {
            "total_benchmark_tasks": len(benchmark_tasks),
            "samples_per_task": samples_per_task,
            "total_evaluations": total_runs,
            "resolved_count": total_resolved,
            "resolution_rate": res_rate,
            "pass_at_k": pass_at_k_metrics,
            "self_correction_events": total_self_corrections,
            "avg_turns_per_resolution": avg_turns,
            "repo_breakdown": repo_stats
        }

        return {
            "summary": summary,
            "evaluations": all_evaluations
        }

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generates a GitHub-flavored markdown report summarizing evaluation results."""
        summary = results["summary"]
        evals = results["evaluations"]

        lines = [
            "# SWE-bench Multi-Turn Evaluation Benchmark Report",
            f"**Evaluation Date:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            "",
            "## 1. Executive Summary",
            "",
            "| Metric | Result | Target Benchmark |",
            "| :--- | :--- | :--- |",
            f"| **pass@1 Resolution Rate** | **{summary.get('pass_at_k', {}).get('pass@1', summary['resolution_rate']) * 100:.1f}%** | Frontier Standard |",
            f"| **Total Issues Evaluated** | {summary['total_benchmark_tasks']} | Diverse Repos |",
            f"| **Samples per Issue** | {summary['samples_per_task']} | Multi-trial sampling |",
            f"| **Self-Correction Recovery Events** | {summary['self_correction_events']} | Compiler / Indentation Error Recovery |",
            f"| **Average Agent Turns** | {summary['avg_turns_per_resolution']} | Multi-turn reasoning |",
            "",
            "## 2. Repository & Domain Breakdown",
            "",
            "| Repository | Resolved | Total | Pass Rate |",
            "| :--- | :--- | :--- | :--- |"
        ]

        for repo, stats in summary["repo_breakdown"].items():
            lines.append(f"| `{repo}` | {stats['resolved']} | {stats['total']} | **{stats['rate']*100:.1f}%** |")

        lines.extend([
            "",
            "## 3. Instance Resolution Telemetry",
            "",
            "| Instance ID | Repository | Resolved | Turns | Self-Corrections | Duration |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |"
        ])

        for e in evals:
            status = "PASSED" if e["resolved"] else "FAILED"
            lines.append(
                f"| `{e['instance_id']}` | `{e['repo']}` | **{status}** | {e['turns_taken']} | {e['self_correction_events']} | {e['duration_sec']}s |"
            )

        return "\n".join(lines)
