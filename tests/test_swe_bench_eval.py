"""
Unit tests for SWE-bench Multi-Turn Evaluation Harness, Task Loader, and Agent Environment.
"""

import os
import pytest
from eval.swe_bench.task_loader import SWEBenchTaskLoader, SWEBenchInstance
from eval.swe_bench.agent_environment import AgentEnvironment
from eval.swe_bench.eval_harness import SWEBenchEvalHarness, estimate_pass_at_k, BenchmarkAutonomousPolicy


def test_task_loader_suite():
    tasks = SWEBenchTaskLoader.get_verified_benchmark_suite()
    assert len(tasks) >= 4
    for t in tasks:
        assert t.instance_id
        assert t.repo
        assert t.affected_file
        assert t.buggy_snippet
        assert t.patched_snippet
        assert t.reproduction_script
        assert t.test_code


def test_estimate_pass_at_k():
    # When all samples correct, pass@k is 1.0
    assert estimate_pass_at_k(num_samples=5, num_correct=5, k=1) == 1.0
    assert estimate_pass_at_k(num_samples=5, num_correct=5, k=3) == 1.0
    # When 0 correct, pass@k is 0.0
    assert estimate_pass_at_k(num_samples=5, num_correct=0, k=1) == 0.0
    # When 5 out of 10 correct, pass@1 is 0.5
    assert abs(estimate_pass_at_k(num_samples=10, num_correct=5, k=1) - 0.5) < 1e-4


def test_agent_environment_isolated_lifecycle():
    task = SWEBenchTaskLoader.get_verified_benchmark_suite()[0]
    env = AgentEnvironment(task)
    workspace = env.workspace_dir
    assert os.path.exists(workspace)

    # 1. Verify reproduction script fails on initial buggy state
    repro_res = env.run_reproduction_test()
    # In initial state, reproduction script should fail (exit_code != 0) or catch expected defect
    # 2. View file lines
    view_res = env.execute_file_editor(action="view", path=task.affected_file)
    assert view_res["success"] is True
    assert "LocMemCache" in view_res["content"]

    # 3. Patch file with str_replace
    replace_res = env.execute_file_editor(
        action="str_replace",
        path=task.affected_file,
        old_str=task.buggy_snippet,
        new_str=task.patched_snippet
    )
    assert replace_res["success"] is True

    # 4. Regression tests should pass cleanly
    regress_res = env.run_regression_test_suite()
    assert regress_res["exit_code"] == 0

    # 5. Diff should be non-empty
    diff = env.get_git_diff()
    assert len(diff.strip()) > 0
    assert "def get(self, key, default=None):" in diff

    # 6. Cleanup
    env.cleanup()
    assert not os.path.exists(workspace)


def test_eval_harness_single_instance_with_self_correction():
    task = SWEBenchTaskLoader.get_verified_benchmark_suite()[0]
    harness = SWEBenchEvalHarness(max_turns=10)
    result = harness.evaluate_instance(task, simulate_self_correction=True)

    assert result["instance_id"] == task.instance_id
    assert result["resolved"] is True
    assert result["turns_taken"] == 7
    assert result["self_correction_events"] >= 1
    assert "bash" in result["tools_used"]
    assert "file_editor" in result["tools_used"]


def test_eval_harness_benchmark_run_and_markdown_report():
    tasks = SWEBenchTaskLoader.get_verified_benchmark_suite()[:2]
    harness = SWEBenchEvalHarness(max_turns=10)
    results = harness.run_benchmark(tasks=tasks, samples_per_task=1, ks=[1], simulate_self_correction=True)

    summary = results["summary"]
    assert summary["total_benchmark_tasks"] == 2
    assert summary["resolved_count"] == 2
    assert summary["resolution_rate"] == 1.0
    assert summary["pass_at_k"]["pass@1"] == 1.0
    assert summary["self_correction_events"] >= 2

    # Check Markdown report formatting
    report = harness.generate_markdown_report(results)
    assert "# SWE-bench Multi-Turn Evaluation Benchmark Report" in report
    assert "pass@1 Resolution Rate" in report
    assert "100.0%" in report
