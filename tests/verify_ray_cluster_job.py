"""
Verification Script for Distributed Ray Multi-Node Cluster Execution.
Submits parallel tasks and proves that computation executes across both
the Ray Head node and the Ray Worker node.
"""

import os
import sys
import socket
import time
from collections import Counter
from typing import Dict, Any

import ray


@ray.remote
def distributed_tokenization_task(task_id: int, sample_text: str) -> Dict[str, Any]:
    """Task executed on an arbitrary node in the Ray cluster."""
    ctx = ray.get_runtime_context()
    node_id = ctx.get_node_id()
    hostname = socket.gethostname()
    ip_address = ray.util.get_node_ip_address()

    # Perform tokenization and filtering simulation
    words = sample_text.split()
    tokens = [hash(w) % 128000 for w in words]

    # Artificial work to ensure tasks span multiple workers
    time.sleep(0.5)

    return {
        "task_id": task_id,
        "node_id": node_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "token_count": len(tokens),
        "status": "success"
    }


def main():
    print("=" * 65)
    print("  VERIFYING RAY MULTI-NODE CLUSTER JOB EXECUTION")
    print("=" * 65)

    # Connect to Ray cluster
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    cluster_nodes = ray.nodes()
    alive_nodes = [n for n in cluster_nodes if n.get("Alive", False)]

    print(f"\n[+] Cluster Topology Detected: {len(alive_nodes)} Active Nodes")
    for idx, node in enumerate(alive_nodes):
        node_id = node.get("NodeID")
        ip = node.get("NodeManagerAddress")
        hostname = node.get("NodeManagerHostname")
        resources = node.get("Resources", {})
        cpus = resources.get("CPU", 0)
        print(f"    - Node {idx+1}: ID={node_id[:16]}... | IP={ip} | Host={hostname} | CPUs={cpus}")

    assert len(alive_nodes) >= 2, f"Expected at least 2 nodes (head + worker), found {len(alive_nodes)}"

    # Dispatch 16 parallel tasks to force distribution across nodes
    num_tasks = 16
    print(f"\n[*] Dispatching {num_tasks} parallel tokenization tasks across cluster...")

    sample_sentence = "Universal LLM pre-training requires distributed compute across multi-node clusters."
    futures = [distributed_tokenization_task.remote(i, sample_sentence * 5) for i in range(num_tasks)]

    results = ray.get(futures)

    # Analyze node execution distribution
    nodes_used = Counter(r["node_id"] for r in results)
    hosts_used = Counter(r["hostname"] for r in results)

    print("\n[+] Task Distribution Across Cluster Nodes:")
    for node_id, count in nodes_used.items():
        matching_host = next(r["hostname"] for r in results if r["node_id"] == node_id)
        matching_ip = next(r["ip_address"] for r in results if r["node_id"] == node_id)
        print(f"    - Node {node_id[:16]}... ({matching_host} / {matching_ip}): Executed {count} tasks ({count/num_tasks*100:.1f}%)")

    distinct_nodes = len(nodes_used)
    print(f"\n[✓] Number of distinct compute nodes used: {distinct_nodes}")
    if distinct_nodes > 1:
        print("[SUCCESS] Verified: Ray job distributed successfully across both head and worker nodes!")
    else:
        print("[WARNING] Tasks executed on a single node. Check resource scheduling.")

    print("=" * 65)


if __name__ == "__main__":
    main()
