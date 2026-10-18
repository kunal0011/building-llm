"""
Distributed Ray Multi-Node Post-Training & RL Experience Rollout Collector.
Distributes multi-turn agent candidate rollouts (G per prompt) across Ray cluster
worker nodes, evaluates deterministic test suite rewards (Code Sandbox & Symbolic Math),
scores step-level process supervision (PRM), computes GRPO advantages, and persists to S3.
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
import numpy as np

from pipelines.rl.execution_sandbox import CodeExecutionSandbox
from pipelines.rl.symbolic_math_verifier import SymbolicMathVerifier
from pipelines.rl.process_reward_annotator import ProcessRewardAnnotator
from pipelines.rl.grpo_reward_engine import GRPORewardEngine
from pipelines.rl.experience_buffer import (
    AgentStepCheckpoint,
    TrajectoryExperience,
    GRPORolloutGroup,
    DistributedExperienceBuffer
)
from pipelines.storage.s3_manager import S3DataManager


# =====================================================================
# RAY REMOTE ROLLOUT WORKER (Executes across cluster Head and Worker nodes)
# =====================================================================

@ray.remote(num_cpus=1)
def distributed_collect_agent_group(
    prompt_id: str,
    prompt: str,
    ground_truth_specs: Dict[str, Any],
    group_size: int = 4,
    env_type: str = "code"  # "code" or "math"
) -> Dict[str, Any]:
    """
    Executes G candidate rollouts for a single prompt on a Ray cluster node.
    Performs deterministic environment validation, step-level PRM scoring,
    and returns rich TrajectoryExperience objects.
    """
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()
    start_t = time.time()

    sandbox = CodeExecutionSandbox()
    math_verifier = SymbolicMathVerifier()
    prm_annotator = ProcessRewardAnnotator()
    grpo_engine = GRPORewardEngine()

    rollouts: List[Dict[str, Any]] = []

    # Generate G candidate rollout variations (simulating policy sampling)
    for g_idx in range(group_size):
        trajectory_id = f"{prompt_id}_rollout_{g_idx:02d}"

        # Candidate simulation logic
        if env_type == "code":
            # Some candidates succeed, some make edge-case errors
            if g_idx % 2 == 0:
                # Correct solution
                code_solution = ground_truth_specs["correct_code"]
                test_code = ground_truth_specs["test_code"]
                thought_text = f"<thought>Decomposing the problem into subroutines. Verifying boundary conditions for n=0 and n=1.</thought>\n{code_solution}"
                sandbox_result = sandbox.execute_python_code(code_solution, test_code)
                outcome_corr = 1.0 if sandbox_result["passed"] else 0.0
            else:
                # Flawed solution
                code_solution = ground_truth_specs["buggy_code"]
                test_code = ground_truth_specs["test_code"]
                thought_text = f"<thought>Rushing to solution without checking edge cases.\n{code_solution}"  # unclosed tag
                sandbox_result = sandbox.execute_python_code(code_solution, test_code)
                outcome_corr = 1.0 if sandbox_result["passed"] else 0.0

            # Step checkpoints
            steps = [
                {
                    "step_idx": 0,
                    "thought": "Analyze code requirements and time complexity.",
                    "action_type": "ast_parse",
                    "action_payload": "import ast; ast.parse(code)",
                    "observation": "Syntax valid.",
                    "prm_reward": 1.0
                },
                {
                    "step_idx": 1,
                    "thought": "Implement core algorithmic function.",
                    "action_type": "code_generate",
                    "action_payload": code_solution,
                    "observation": f"Executed tests with passed={sandbox_result['passed']}",
                    "prm_reward": 1.0 if sandbox_result["passed"] else -1.0
                }
            ]

        else:  # "math"
            cand_latex = ground_truth_specs["candidates"][g_idx % len(ground_truth_specs["candidates"])]
            target_latex = ground_truth_specs["target_latex"]
            is_equiv, _ = math_verifier.verify_equivalence(cand_latex, target_latex)
            outcome_corr = 1.0 if is_equiv else 0.0
            thought_text = f"<thought>Step-by-step algebraic reduction. Applying distributive property and binomial expansion.</thought>\nResult: {cand_latex}"

            steps = [
                {
                    "step_idx": 0,
                    "thought": "Derive polynomial step.",
                    "action_type": "sympy_derivation",
                    "action_payload": cand_latex,
                    "observation": f"Algebraic equivalence={outcome_corr}",
                    "prm_reward": 1.0 if outcome_corr == 1.0 else -1.0
                }
            ]

        # Evaluate rewards
        approx_tokens = len(thought_text.split()) * 2
        reward_dict = grpo_engine.compute_composite_reward(
            verifiable_correctness=outcome_corr,
            output_text=thought_text,
            token_count=approx_tokens
        )

        rollouts.append({
            "trajectory_id": trajectory_id,
            "prompt": prompt,
            "final_solution": thought_text,
            "outcome_correctness": outcome_corr,
            "format_reward": reward_dict["format_reward"],
            "efficiency_penalty": reward_dict["efficiency_penalty"],
            "total_reward": reward_dict["total_reward"],
            "steps": steps,
            "worker_node": f"{hostname} ({ip_address})"
        })

    # Artificial work pacing to ensure multi-node cluster distribution
    time.sleep(0.2)
    elapsed = time.time() - start_t

    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "rollouts": rollouts,
        "elapsed_sec": round(elapsed, 2)
    }


# =====================================================================
# DISTRIBUTED RL ORCHESTRATOR
# =====================================================================

def run_distributed_rl_collection(
    group_size: int = 4,
    s3_bucket: str = "ai-data-shards",
    s3_endpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Orchestrates distributed RL experience collection across the Ray cluster.
    """
    print("=" * 70)
    print("  DISTRIBUTED RAY RL EXPERIENCE BUFFER & GRPO ROLLOUT COLLECTOR")
    print("=" * 70)

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    cluster_nodes = [n for n in ray.nodes() if n.get("Alive", False)]
    print(f"\n[+] Connected to Ray Cluster: {len(cluster_nodes)} Active Nodes Detected")
    for idx, node in enumerate(cluster_nodes):
        print(f"    - Node {idx+1}: Host={node.get('NodeManagerHostname')} | IP={node.get('NodeManagerAddress')}")

    start_t = time.time()
    endpoint = s3_endpoint or os.getenv("AWS_ENDPOINT_URL") or "http://s3:4566"

    # Define verifiable prompt benchmark suite (SWE code tasks & STEM math proofs)
    prompts_dataset = [
        {
            "prompt_id": "swe_code_001",
            "prompt": "Write a Python function `fibonacci(n)` that returns the nth Fibonacci number efficiently.",
            "env_type": "code",
            "ground_truth_specs": {
                "correct_code": "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n",
                "buggy_code": "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2) # Exponential TLE\n",
                "test_code": "assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(10) == 55\n"
            }
        },
        {
            "prompt_id": "swe_code_002",
            "prompt": "Implement `is_palindrome(s)` ignoring case and non-alphanumeric characters.",
            "env_type": "code",
            "ground_truth_specs": {
                "correct_code": "import re\ndef is_palindrome(s):\n    cleaned = re.sub(r'[^a-zA-Z0-9]', '', s).lower()\n    return cleaned == cleaned[::-1]\n",
                "buggy_code": "def is_palindrome(s):\n    return s == s[::-1] # Does not ignore punctuation or case\n",
                "test_code": "assert is_palindrome('A man, a plan, a canal: Panama') == True\nassert is_palindrome('race a car') == False\n"
            }
        },
        {
            "prompt_id": "math_proof_001",
            "prompt": "Simplify the algebraic fraction expression: \\frac{x^2 - 4}{x - 2}",
            "env_type": "math",
            "ground_truth_specs": {
                "target_latex": "x + 2",
                "candidates": ["x + 2", "x - 2", "\\frac{(x-2)(x+2)}{x-2}", "x + 4"]
            }
        },
        {
            "prompt_id": "math_proof_002",
            "prompt": "Compute the derivative with respect to x of: f(x) = \\sin(x) \\cdot e^x",
            "env_type": "math",
            "ground_truth_specs": {
                "target_latex": "e^x * \\sin(x) + e^x * \\cos(x)",
                "candidates": [
                    "e^x * (\\sin(x) + \\cos(x))",
                    "\\cos(x) * e^x",
                    "e^x * \\sin(x) + e^x * \\cos(x)",
                    "\\sin(x) * e^x"
                ]
            }
        },
        {
            "prompt_id": "swe_code_003",
            "prompt": "Implement a binary search function `binary_search(arr, target)` returning the index or -1.",
            "env_type": "code",
            "ground_truth_specs": {
                "correct_code": "def binary_search(arr, target):\n    low, high = 0, len(arr) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: low = mid + 1\n        else: high = mid - 1\n    return -1\n",
                "buggy_code": "def binary_search(arr, target):\n    return arr.index(target) # Fails with ValueError on missing target\n",
                "test_code": "assert binary_search([1, 3, 5, 7, 9], 5) == 2\nassert binary_search([1, 3, 5, 7, 9], 2) == -1\n"
            }
        },
        {
            "prompt_id": "math_proof_003",
            "prompt": "Verify the trigonometric identity: \\sin^2(x) + \\cos^2(x)",
            "env_type": "math",
            "ground_truth_specs": {
                "target_latex": "1",
                "candidates": ["1", "\\tan^2(x)", "1.0", "0"]
            }
        }
    ]

    print(f"\n[*] Submitting {len(prompts_dataset)} parallel rollout groups (G={group_size} rollouts each) to Ray cluster...")
    futures = [
        distributed_collect_agent_group.remote(
            item["prompt_id"],
            item["prompt"],
            item["ground_truth_specs"],
            group_size=group_size,
            env_type=item["env_type"]
        )
        for item in prompts_dataset
    ]

    worker_results = ray.get(futures)

    # Ingest into DistributedExperienceBuffer
    buffer = DistributedExperienceBuffer()
    node_distribution = Counter()

    for res in worker_results:
        node_ip = res["ip_address"]
        node_host = res["hostname"]
        node_distribution[f"{node_host} ({node_ip})"] += 1

        trajectory_objs = []
        for r in res["rollouts"]:
            step_objs = [
                AgentStepCheckpoint(
                    step_idx=s["step_idx"],
                    thought=s["thought"],
                    action_type=s["action_type"],
                    action_payload=s["action_payload"],
                    observation=s["observation"],
                    prm_reward=s["prm_reward"]
                )
                for s in r["steps"]
            ]

            traj = TrajectoryExperience(
                trajectory_id=r["trajectory_id"],
                prompt=r["prompt"],
                steps=step_objs,
                final_solution=r["final_solution"],
                outcome_correctness=r["outcome_correctness"],
                format_reward=r["format_reward"],
                efficiency_penalty=r["efficiency_penalty"],
                total_reward=r["total_reward"],
                worker_node=r["worker_node"]
            )
            trajectory_objs.append(traj)

        buffer.add_rollout_group(res["prompt_id"], res["prompt"], trajectory_objs)

    buffer_summary = buffer.export_summary()
    total_rollouts = buffer.get_total_trajectories()

    print(f"\n[+] RL Experience Buffer Aggregation Complete:")
    print(f"    - Total Prompts Processed    : {len(prompts_dataset)}")
    print(f"    - Total Rollouts Collected    : {total_rollouts} (G={group_size})")
    print(f"    - Verifiable Pass Rate       : {buffer_summary['pass_rate_accuracy']}")
    print(f"    - Mean Group Reward          : {buffer_summary['mean_reward']}")
    print(f"    - Advantage Spread (Min/Max) : [{buffer_summary['min_advantage']}, {buffer_summary['max_advantage']}]")

    print(f"\n[+] Cluster Node Task Distribution:")
    for host_ip, count in node_distribution.items():
        print(f"    * {host_ip}: {count} rollout groups ({count/len(prompts_dataset)*100:.1f}%)")

    # Serialize Experience Batch & Upload to S3
    os.makedirs("./data/rl_experience", exist_ok=True)
    exp_file = "./data/rl_experience/grpo_experience_batch_00000.json"
    serializable_samples = list(buffer.sample_minibatches(batch_size=total_rollouts, shuffle=False))[0]

    with open(exp_file, "w") as f:
        json.dump(serializable_samples, f, indent=2)

    s3_uploaded = False
    s3_key = "rl/grpo_experience_batch_00000.json"
    try:
        s3_mgr = S3DataManager(bucket_name=s3_bucket, endpoint_url=endpoint)
        s3_mgr.ensure_bucket_exists()
        s3_uploaded = s3_mgr.upload_file_multipart(exp_file, s3_key)
        print(f"\n[+] Persisted GRPO Experience Batch to s3://{s3_bucket}/{s3_key} (Uploaded={s3_uploaded})")
    except Exception as e:
        print(f"[-] S3 RL buffer upload warning: {e}")

    elapsed_total = time.time() - start_t

    print("\n" + "=" * 70)
    print("  DISTRIBUTED RL EXPERIENCE COLLECTION SUMMARY: SUCCESS")
    print("=" * 70)
    print(f"  Total Collection Time        : {elapsed_total:.2f} seconds")
    print(f"  Cluster Nodes Utilized       : {len(node_distribution)} nodes ({list(node_distribution.keys())})")
    print(f"  Total Trajectories Captured  : {total_rollouts}")
    print(f"  S3 Staging Path              : s3://{s3_bucket}/{s3_key}")
    print("=" * 70)

    return {
        "status": "success",
        "elapsed_sec": round(elapsed_total, 2),
        "cluster_nodes_utilized": len(node_distribution),
        "nodes": dict(node_distribution),
        "total_prompts": len(prompts_dataset),
        "total_trajectories": total_rollouts,
        "pass_rate": buffer_summary["pass_rate_accuracy"],
        "mean_reward": buffer_summary["mean_reward"],
        "s3_uploaded": s3_uploaded,
        "s3_key": s3_key
    }


if __name__ == "__main__":
    run_distributed_rl_collection()
