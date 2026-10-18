"""
Entrypoint for distributed Ray SWE Agent Trajectory synthesis job.
Can be executed directly or submitted via Ray JobSubmissionClient.
"""

import os
import sys
import json
import time

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import ray
from pipelines.agent_synthesis.distributed_swe_collector import (
    DistributedSWECollector,
    get_extended_swe_benchmark_suite
)


def main():
    print("=" * 80)
    print("STARTING DISTRIBUTED RAY SWE-BENCH TRAJECTORY SYNTHESIS JOB")
    print("=" * 80)

    # Initialize Ray
    if not ray.is_initialized():
        ray_address = os.getenv("RAY_ADDRESS", None)
        if ray_address:
            print(f"[Ray Cluster] Connecting to Ray cluster at {ray_address}...")
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            print("[Ray Cluster] Initializing local Ray context...")
            ray.init(ignore_reinit_error=True)

    cluster_resources = ray.cluster_resources()
    nodes = ray.nodes()
    print(f"[Ray Cluster] Connected nodes: {len(nodes)}")
    for n in nodes:
        print(f"  - Node {n['NodeID'][:8]} | IP: {n['NodeManagerAddress']} | Alive: {n['Alive']}")

    specs = get_extended_swe_benchmark_suite()
    print(f"\n[SWE Synthesizer] Loaded {len(specs)} realistic benchmark issues across domains.")

    s3_endpoint = os.getenv("AWS_ENDPOINT_URL") or "http://s3:4566"
    print(f"[Storage] Connecting to S3 Data Lake (endpoint: {s3_endpoint})...")

    collector = DistributedSWECollector()
    start_t = time.time()
    results = collector.collect(specs=specs, simulate_self_correction=True)
    elapsed = round(time.time() - start_t, 2)

    summary = results["summary"]
    print("\n" + "=" * 80)
    print(f"DISTRIBUTED SWE TRAJECTORY SYNTHESIS COMPLETED IN {elapsed}s")
    print("=" * 80)
    print(f"Total Trajectories Generated : {summary['total_trajectories']}")
    print(f"Deterministic Verifications  : {summary['verified_passed']} / {summary['total_trajectories']} ({summary['pass_rate'] * 100:.1f}%)")
    print(f"Total Agent Multi-Turn Steps : {summary['total_agent_turns']}")
    print(f"Avg Turns per Trajectory     : {summary['avg_turns_per_trajectory']}")
    print(f"Self-Correction Occurrences  : {summary['self_correction_events']}")
    print(f"Tool Distribution            : {summary['tool_distribution']}")
    print(f"Cluster Node Distribution    : {summary['cluster_node_distribution']}")
    print(f"Unique Nodes Utilized        : {summary['unique_nodes_utilized']}")

    persist_info = collector.persist_to_s3(results, filename_prefix="swe_trajectories_cluster")
    print(f"\n[Persistence] Shard saved locally to : {persist_info['local_path']}")
    print(f"[Persistence] Uploaded to S3         : {persist_info['s3_uri']} (Size: {persist_info['file_size_bytes']} bytes)")

    # Print JSON line for client parsing
    print("\n--- JSON_SUMMARY_START ---")
    print(json.dumps({
        "summary": summary,
        "persistence": persist_info,
        "elapsed_sec": elapsed
    }))
    print("--- JSON_SUMMARY_END ---")


if __name__ == "__main__":
    main()
