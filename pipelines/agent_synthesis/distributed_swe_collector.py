"""
Distributed Ray SWE Agent Trajectory Collector.
Orchestrates parallel multi-turn software engineering trajectory synthesis across
Ray Head and Worker cluster nodes. Dispatches SWE-bench verified tasks, validates
code patches with deterministic sandbox execution, tracks cluster node telemetry,
and persists training corpora to S3.
"""

import os
import sys
import time
import socket
import json
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import ray

from pipelines.agent_synthesis.swe_agent_synthesizer import SWEAgentSynthesizer, SWEIssueSpec
from pipelines.rl.execution_sandbox import CodeExecutionSandbox
from pipelines.storage.s3_manager import S3DataManager


# =====================================================================
# RAY REMOTE SWE SYNTHESIS TASK (Executes across cluster Head and Worker nodes)
# =====================================================================

@ray.remote(num_cpus=1)
def distributed_synthesize_swe_task(
    spec_dict: Dict[str, Any],
    simulate_intermediate_syntax_error: bool = True
) -> Dict[str, Any]:
    """
    Executes a SWE-bench trajectory synthesis and deterministic sandbox verification
    on an assigned Ray cluster node. Captures telemetry including node ID, IP, and hostname.
    """
    start_t = time.time()
    node_ip = "127.0.0.1"
    hostname = socket.gethostname()
    node_id = "local"

    try:
        ctx = ray.get_runtime_context()
        node_id = str(ctx.get_node_id())
        node_ip = str(ray.util.get_node_ip_address())
    except Exception:
        pass

    # Reconstruct SWEIssueSpec
    spec = SWEIssueSpec(
        issue_id=spec_dict["issue_id"],
        repo_name=spec_dict["repo_name"],
        problem_statement=spec_dict["problem_statement"],
        affected_files=spec_dict["affected_files"],
        buggy_code_snippet=spec_dict["buggy_code_snippet"],
        patched_code_snippet=spec_dict["patched_code_snippet"],
        reproduction_test_code=spec_dict["reproduction_test_code"],
        regression_test_code=spec_dict["regression_test_code"],
        expected_error_traceback=spec_dict["expected_error_traceback"]
    )

    synthesizer = SWEAgentSynthesizer()
    trajectory = synthesizer.generate_verified_trajectory(
        spec=spec,
        simulate_intermediate_syntax_error=simulate_intermediate_syntax_error
    )

    duration_ms = round((time.time() - start_t) * 1000, 2)

    trajectory["telemetry"] = {
        "node_id": node_id,
        "node_ip": node_ip,
        "hostname": hostname,
        "duration_ms": duration_ms
    }

    return trajectory


# =====================================================================
# EXTENDED BENCHMARK SUITE GENERATOR
# =====================================================================

def get_extended_swe_benchmark_suite() -> List[SWEIssueSpec]:
    """
    Provides a comprehensive suite of SWE issues spanning core domains:
    concurrency, AST parsing, numerical precision, web routing, network timeouts,
    testing fixtures, indexing, and transactional retry loops.
    """
    return [
        SWEIssueSpec(
            issue_id="DJANGO_CACHE_001",
            repo_name="django/django",
            problem_statement="LRUCache throws KeyError when accessing concurrently evicted keys.",
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
            problem_statement="AST visitor crashes with AttributeError on NoneType docstrings.",
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
            problem_statement="Binary search root finder loops on subnormal float epsilon comparison.",
            affected_files=["scipy/optimize/zeros.py"],
            buggy_code_snippet="def bisection_step(a, b, tol=0.0):\n    while abs(b - a) > tol:\n        mid = (a + b) / 2.0\n        if mid == a: break\n        a = mid\n    return a",
            patched_code_snippet="def bisection_step(a, b, tol=1e-12):\n    count = 0\n    while abs(b - a) > tol:\n        mid = (a + b) / 2.0\n        if abs(b - a) <= tol: break\n        a = mid\n        count += 1\n        if count > 100: break\n    return a",
            reproduction_test_code="def bisection_step(a, b, tol=0.0):\n    count = 0\n    while abs(b - a) > tol:\n        mid = (a + b) / 2.0\n        if mid == a: break\n        a = mid\n        count += 1\n        if count > 50: raise TimeoutError('Loop depth exceeded')\n    return a\ntry:\n    bisection_step(1.0, 1.0 + 1e-16)\nexcept TimeoutError:\n    pass",
            regression_test_code="res = bisection_step(1.0, 1.0 + 1e-8)\nassert abs((1.0 + 1e-8) - res) <= 1e-8",
            expected_error_traceback="  File \"scipy/optimize/zeros.py\", line 154, in bisection_step\n    while abs(b - a) > 0.0:\nTimeoutError: Loop depth exceeded"
        ),
        SWEIssueSpec(
            issue_id="FLASK_ROUTING_004",
            repo_name="pallets/flask",
            problem_statement="URL path normalization drops query string during trailing slash redirect.",
            affected_files=["flask/helpers.py"],
            buggy_code_snippet="def normalize_path(path):\n    return path.rstrip('/')",
            patched_code_snippet="def normalize_path(path, keep_query=True):\n    parts = path.split('?', 1)\n    base = parts[0].rstrip('/')\n    return f\"{base}?{parts[1]}\" if len(parts) > 1 and keep_query else base",
            reproduction_test_code="p = '/users/?page=2'\nassert normalize_path(p) != '/users?page=2'",
            regression_test_code="def normalize_path(path, keep_query=True):\n    parts = path.split('?', 1)\n    base = parts[0].rstrip('/')\n    return f\"{base}?{parts[1]}\" if len(parts) > 1 and keep_query else base\nassert normalize_path('/api/v1/?limit=10') == '/api/v1?limit=10'",
            expected_error_traceback="  File \"flask/helpers.py\", line 21, in normalize_path\nAssertionError: Query parameters stripped"
        ),
        SWEIssueSpec(
            issue_id="REQUESTS_TIMEOUT_005",
            repo_name="psf/requests",
            problem_statement="Connect timeout tuple unpacking raises TypeError on single float.",
            affected_files=["requests/adapters.py"],
            buggy_code_snippet="def parse_timeout(timeout):\n    connect, read = timeout\n    return connect, read",
            patched_code_snippet="def parse_timeout(timeout):\n    if isinstance(timeout, (int, float)):\n        return float(timeout), float(timeout)\n    return float(timeout[0]), float(timeout[1])",
            reproduction_test_code="try:\n    parse_timeout(5.0)\n    assert False, 'Expected TypeError'\nexcept TypeError:\n    assert True",
            regression_test_code="def parse_timeout(timeout):\n    if isinstance(timeout, (int, float)):\n        return float(timeout), float(timeout)\n    return float(timeout[0]), float(timeout[1])\nassert parse_timeout(5.0) == (5.0, 5.0)\nassert parse_timeout((3.0, 10.0)) == (3.0, 10.0)",
            expected_error_traceback="  File \"requests/adapters.py\", line 112, in parse_timeout\nTypeError: cannot unpack non-iterable float object"
        ),
        SWEIssueSpec(
            issue_id="SQL_RETRY_006",
            repo_name="sqlalchemy/sqlalchemy",
            problem_statement="Transaction retry loop crashes immediately on OperationalError without checking deadlock code.",
            affected_files=["sqlalchemy/engine/base.py"],
            buggy_code_snippet="def execute_with_retry(fn):\n    return fn()",
            patched_code_snippet="def execute_with_retry(fn, max_retries=3):\n    for attempt in range(max_retries):\n        try:\n            return fn()\n        except Exception as e:\n            if attempt == max_retries - 1: raise\n            time.sleep(0.05 * (2 ** attempt))",
            reproduction_test_code="call_count = 0\ndef flaky():\n    global call_count; call_count += 1\n    if call_count < 2: raise RuntimeError('Deadlock')\n    return 'OK'\ntry:\n    execute_with_retry(flaky)\nexcept RuntimeError:\n    assert True",
            regression_test_code="import time\ndef execute_with_retry(fn, max_retries=3):\n    for attempt in range(max_retries):\n        try:\n            return fn()\n        except Exception as e:\n            if attempt == max_retries - 1: raise\n            time.sleep(0.01 * (2 ** attempt))\nstate = {'attempts': 0}\ndef flaky():\n    state['attempts'] += 1\n    if state['attempts'] < 2: raise RuntimeError('Deadlock')\n    return 'SUCCESS'\nres = execute_with_retry(flaky)\nassert res == 'SUCCESS' and state['attempts'] == 2",
            expected_error_traceback="  File \"sqlalchemy/engine/base.py\", line 77, in execute_with_retry\nRuntimeError: Deadlock detected in concurrent transaction"
        )
    ]


# =====================================================================
# DISTRIBUTED SWE COLLECTOR
# =====================================================================

class DistributedSWECollector:
    """
    Cluster coordinator for synthesizing SWE-bench agent trajectories.
    Supports Ray cluster execution across multiple EC2/Docker nodes or local execution.
    """

    def __init__(
        self,
        ray_address: Optional[str] = None,
        s3_bucket: str = "ai-data-shards",
        s3_prefix: str = "swe_bench",
        s3_manager: Optional[S3DataManager] = None
    ):
        self.ray_address = ray_address or os.getenv("RAY_ADDRESS", None)
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_manager = s3_manager or S3DataManager(bucket_name=s3_bucket)
        self._init_ray()

    def _init_ray(self):
        """Initializes or connects to Ray cluster."""
        if not ray.is_initialized():
            if self.ray_address:
                try:
                    ray.init(address=self.ray_address, ignore_reinit_error=True)
                except Exception:
                    ray.init(ignore_reinit_error=True, num_cpus=2)
            else:
                ray.init(ignore_reinit_error=True, num_cpus=2)

    def collect(
        self,
        specs: Optional[List[SWEIssueSpec]] = None,
        count: Optional[int] = None,
        simulate_self_correction: bool = True
    ) -> Dict[str, Any]:
        """
        Dispatches SWE trajectory synthesis across the Ray cluster nodes.
        Returns aggregated dataset and cluster distribution telemetry.
        """
        target_specs = specs or get_extended_swe_benchmark_suite()
        if count and count < len(target_specs):
            target_specs = target_specs[:count]

        # Convert to serializable dicts
        spec_dicts = [
            {
                "issue_id": s.issue_id,
                "repo_name": s.repo_name,
                "problem_statement": s.problem_statement,
                "affected_files": s.affected_files,
                "buggy_code_snippet": s.buggy_code_snippet,
                "patched_code_snippet": s.patched_code_snippet,
                "reproduction_test_code": s.reproduction_test_code,
                "regression_test_code": s.regression_test_code,
                "expected_error_traceback": s.expected_error_traceback
            }
            for s in target_specs
        ]

        # Dispatch Ray remote tasks
        futures = [
            distributed_synthesize_swe_task.remote(
                spec_dict=sd,
                simulate_intermediate_syntax_error=simulate_self_correction
            )
            for sd in spec_dicts
        ]

        # Gather results
        trajectories = ray.get(futures)

        # Aggregate cluster metrics
        node_ips = [t["telemetry"]["node_ip"] for t in trajectories]
        hostnames = [t["telemetry"]["hostname"] for t in trajectories]
        node_counts = dict(Counter(node_ips))

        total_turns = sum(t["total_turns"] for t in trajectories)
        verified_count = sum(1 for t in trajectories if t["sandbox_passed"])
        pass_rate = round(verified_count / len(trajectories), 4) if trajectories else 0.0

        tool_counter = Counter()
        self_correction_events = 0
        for traj in trajectories:
            for s in traj["steps"]:
                tool_counter[s["tool_name"]] += 1
                if s["exit_code"] != 0 and s["tool_name"] == "file_editor":
                    self_correction_events += 1

        summary = {
            "total_trajectories": len(trajectories),
            "verified_passed": verified_count,
            "pass_rate": pass_rate,
            "total_agent_turns": total_turns,
            "avg_turns_per_trajectory": round(total_turns / len(trajectories), 2) if trajectories else 0.0,
            "self_correction_events": self_correction_events,
            "tool_distribution": dict(tool_counter),
            "cluster_node_distribution": node_counts,
            "unique_nodes_utilized": len(node_counts),
            "hostnames_utilized": list(set(hostnames))
        }

        return {
            "summary": summary,
            "trajectories": trajectories
        }

    def persist_to_s3(
        self,
        collection_result: Dict[str, Any],
        filename_prefix: str = "swe_trajectories"
    ) -> Dict[str, Any]:
        """
        Saves trajectories to a local JSONL file and uploads to the S3 data lake.
        """
        timestamp = int(time.time())
        local_dir = "/tmp/swe_agent_shards"
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, f"{filename_prefix}_{timestamp}.jsonl")

        with open(local_path, "w", encoding="utf-8") as f:
            for traj in collection_result["trajectories"]:
                f.write(json.dumps(traj) + "\n")

        s3_key = f"{self.s3_prefix}/{filename_prefix}_{timestamp}.jsonl"
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"

        upload_success = False
        try:
            self.s3_manager.upload_file(local_path, s3_key)
            upload_success = True
        except Exception as e:
            # Fallback or local emulator not ready
            pass

        return {
            "local_path": local_path,
            "s3_key": s3_key,
            "s3_uri": s3_uri,
            "s3_uploaded": upload_success,
            "file_size_bytes": os.path.getsize(local_path) if os.path.exists(local_path) else 0,
            "trajectory_count": len(collection_result["trajectories"])
        }
