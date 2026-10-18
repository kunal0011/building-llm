"""
SWE-bench Evaluation CLI Runner.
Executes multi-turn evaluation of agents across SWE-bench Verified tasks
and generates structured JSON & Markdown benchmark reports.
"""

import os
import sys
import json
import argparse
import time

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from eval.swe_bench.eval_harness import SWEBenchEvalHarness
from eval.swe_bench.task_loader import SWEBenchTaskLoader


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Multi-Turn Agent Evaluation Runner")
    parser.add_argument("--samples-per-task", type=int, default=1, help="Samples per issue for pass@k estimation")
    parser.add_argument("--output-dir", type=str, default="data/eval_reports", help="Directory to store evaluation reports")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum agent turns per task")
    parser.add_argument("--no-error-simulation", action="store_true", help="Disable synthetic error injection for self-correction")
    args = parser.parse_args()

    print("=" * 80)
    print("STARTING SWE-BENCH VERIFIED MULTI-TURN EVALUATION HARNESS")
    print("=" * 80)
    print(f"Samples per Task : {args.samples_per_task}")
    print(f"Max Turns        : {args.max_turns}")
    print(f"Output Directory : {args.output_dir}")

    tasks = SWEBenchTaskLoader.get_verified_benchmark_suite()
    print(f"\n[Task Loader] Loaded {len(tasks)} benchmark instances across open-source repositories:")
    for t in tasks:
        print(f"  - [{t.repo}] {t.instance_id}: {t.problem_statement[:60]}...")

    harness = SWEBenchEvalHarness(max_turns=args.max_turns)
    start_t = time.time()
    results = harness.run_benchmark(
        tasks=tasks,
        samples_per_task=args.samples_per_task,
        ks=[1, 2, 5],
        simulate_self_correction=not args.no_error_simulation
    )
    elapsed = round(time.time() - start_t, 2)

    summary = results["summary"]
    print("\n" + "=" * 80)
    print(f"SWE-BENCH EVALUATION COMPLETED IN {elapsed}s")
    print("=" * 80)
    print(f"Total Benchmark Issues       : {summary['total_benchmark_tasks']}")
    print(f"Total Evaluations Executed   : {summary['total_evaluations']}")
    print(f"Resolved Successfully        : {summary['resolved_count']} / {summary['total_evaluations']}")
    print(f"pass@1 Resolution Rate       : {summary.get('pass_at_k', {}).get('pass@1', summary['resolution_rate']) * 100:.1f}%")
    if "pass@2" in summary.get("pass_at_k", {}):
        print(f"pass@2 (Multi-trial)         : {summary['pass_at_k']['pass@2'] * 100:.1f}%")
    print(f"Self-Correction Events Fixed : {summary['self_correction_events']}")
    print(f"Average Agent Turns / Task   : {summary['avg_turns_per_resolution']}")
    print(f"Repository Breakdown         : {summary['repo_breakdown']}")

    # Save reports
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "swe_bench_report.json")
    md_path = os.path.join(args.output_dir, "swe_bench_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md_report = harness.generate_markdown_report(results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    print(f"\n[Reports] JSON Report saved to : {json_path}")
    print(f"[Reports] Markdown Report saved: {md_path}")


if __name__ == "__main__":
    main()
