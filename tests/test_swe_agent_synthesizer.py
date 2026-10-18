"""
Tests for Autonomous SWE Agent Synthesizer and Distributed Ray Collector.
Verifies multi-turn tool interaction, self-correction recovery, regression testing,
unified git diff formatting, and cluster telemetry.
"""

import os
import json
import pytest
from pipelines.agent_synthesis.swe_agent_synthesizer import SWEAgentSynthesizer, SWEIssueSpec
from pipelines.agent_synthesis.distributed_swe_collector import (
    DistributedSWECollector,
    get_extended_swe_benchmark_suite
)


@pytest.fixture
def sample_issue():
    return SWEIssueSpec(
        issue_id="TEST_CACHE_001",
        repo_name="org/fastcache",
        problem_statement="KeyError raised on concurrent access to expired keys.",
        affected_files=["fastcache/cache.py"],
        buggy_code_snippet="def get(k): return cache[k]",
        patched_code_snippet="def get(k, default=None):\n    return cache.get(k, default)",
        reproduction_test_code="cache = {'a': 1}\ntry:\n    val = cache['missing']\nexcept KeyError:\n    pass",
        regression_test_code="cache = {'a': 1}\ndef get(k, default=None):\n    return cache.get(k, default)\nassert get('missing', 42) == 42",
        expected_error_traceback="KeyError: 'missing'"
    )


def test_swe_agent_trajectory_with_self_correction(sample_issue):
    """Verifies that the agent executes 7 turns when recovering from a syntax error."""
    synthesizer = SWEAgentSynthesizer()
    traj = synthesizer.generate_verified_trajectory(
        spec=sample_issue,
        simulate_intermediate_syntax_error=True
    )

    assert traj["issue_id"] == "TEST_CACHE_001"
    assert traj["total_turns"] == 7
    assert traj["sandbox_passed"] is True
    assert traj["verifiable_reward"] == 1.0
    assert len(traj["steps"]) == 7

    # Turn 1: Reconnaissance
    assert traj["steps"][0]["tool_name"] == "bash"
    assert traj["steps"][0]["exit_code"] == 0

    # Turn 2: Reproduction script fails with defect
    assert traj["steps"][1]["tool_name"] == "bash"
    assert traj["steps"][1]["exit_code"] == 1

    # Turn 3: Inspect target source
    assert traj["steps"][2]["tool_name"] == "file_editor"
    assert traj["steps"][2]["exit_code"] == 0

    # Turn 4: Flawed patch attempt (indentation/syntax)
    assert traj["steps"][3]["tool_name"] == "file_editor"
    assert traj["steps"][3]["exit_code"] == 1
    assert "IndentationError" in traj["steps"][3]["observation"]

    # Turn 5: Self-correction patch success
    assert traj["steps"][4]["tool_name"] == "file_editor"
    assert traj["steps"][4]["exit_code"] == 0
    assert "Success" in traj["steps"][4]["observation"]

    # Turn 6: Re-run reproduction script & full test suite
    assert traj["steps"][5]["tool_name"] == "bash"
    assert traj["steps"][5]["exit_code"] == 0

    # Turn 7: Review clean git diff
    assert traj["steps"][6]["tool_name"] == "bash"
    assert "git diff" in traj["steps"][6]["tool_args"]["command"]
    assert "--- a/fastcache/cache.py" in traj["steps"][6]["observation"]


def test_swe_agent_trajectory_single_shot(sample_issue):
    """Verifies trajectory without intermediate failure takes 6 turns."""
    synthesizer = SWEAgentSynthesizer()
    traj = synthesizer.generate_verified_trajectory(
        spec=sample_issue,
        simulate_intermediate_syntax_error=False
    )

    assert traj["total_turns"] == 6
    assert traj["sandbox_passed"] is True
    assert traj["verifiable_reward"] == 1.0

    # Turn 4 should succeed immediately
    assert traj["steps"][3]["tool_name"] == "file_editor"
    assert traj["steps"][3]["exit_code"] == 0


def test_benchmark_suite_verification():
    """Verifies all synthetic issues in benchmark suite pass deterministic sandbox execution."""
    synthesizer = SWEAgentSynthesizer()
    suite = synthesizer.synthesize_benchmark_suite()
    assert len(suite) >= 3

    for spec in suite:
        res = synthesizer.generate_verified_trajectory(spec, simulate_intermediate_syntax_error=True)
        assert res["sandbox_passed"] is True, f"Failed for {spec.issue_id}"
        assert res["verifiable_reward"] == 1.0


def test_extended_benchmark_suite():
    """Verifies extended benchmark suite covers various algorithmic and SWE domains."""
    suite = get_extended_swe_benchmark_suite()
    assert len(suite) >= 6
    repo_names = {s.repo_name for s in suite}
    assert "django/django" in repo_names
    assert "psf/black" in repo_names
    assert "scipy/scipy" in repo_names
    assert "pallets/flask" in repo_names
    assert "psf/requests" in repo_names
    assert "sqlalchemy/sqlalchemy" in repo_names


def test_distributed_swe_collector_local_execution(sample_issue):
    """Tests distributed SWE collector orchestrating parallel execution across tasks."""
    collector = DistributedSWECollector()
    result = collector.collect(specs=[sample_issue], simulate_self_correction=True)

    summary = result["summary"]
    assert summary["total_trajectories"] == 1
    assert summary["verified_passed"] == 1
    assert summary["pass_rate"] == 1.0
    assert summary["total_agent_turns"] == 7
    assert summary["self_correction_events"] == 1
    assert "bash" in summary["tool_distribution"]
    assert "file_editor" in summary["tool_distribution"]
    assert summary["unique_nodes_utilized"] >= 1

    # Test persistence to local JSONL and S3 data lake
    persistence_info = collector.persist_to_s3(result, filename_prefix="test_swe_traj")
    assert os.path.exists(persistence_info["local_path"])
    assert persistence_info["trajectory_count"] == 1
    assert persistence_info["file_size_bytes"] > 0
    assert "s3://ai-data-shards/swe_bench/" in persistence_info["s3_uri"]

    # Verify JSONL lines can be re-loaded
    with open(persistence_info["local_path"], "r", encoding="utf-8") as f:
        loaded = json.loads(f.readline())
        assert loaded["issue_id"] == "TEST_CACHE_001"
        assert loaded["sandbox_passed"] is True
