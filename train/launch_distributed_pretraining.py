"""
Distributed Pretraining CLI Launcher.
Executes multi-GPU / multi-node foundation model pretraining directly from
Megatron-LM memory-mapped binary token shards.
"""

import os
import sys
import glob
import time
import argparse
import torch
import numpy as np

# Ensure workspace root in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from train.model import ModelArgs
from train.pretrain_engine import PretrainEngine, PretrainConfig
from pipelines.dataloader.mmap_dataset import MMapIndexedDataset


def get_available_shard_prefixes(shards_dir: str) -> list[str]:
    """Finds all valid .bin/.idx pairs in the directory."""
    prefixes = []
    if not os.path.exists(shards_dir):
        return prefixes
    bin_files = glob.glob(os.path.join(shards_dir, "*.bin"))
    for b in bin_files:
        prefix = b[:-4]
        if os.path.exists(f"{prefix}.idx"):
            prefixes.append(prefix)
    return prefixes


def create_batch_from_shards(
    datasets: list[MMapIndexedDataset],
    batch_size: int,
    seq_len: int,
    vocab_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples zero-copy token slices from memory-mapped shards and formats input/label batches.
    """
    batch_tokens = []
    if datasets:
        for _ in range(batch_size):
            ds = datasets[np.random.randint(0, len(datasets))]
            doc_idx = np.random.randint(0, len(ds))
            doc = ds[doc_idx]
            if len(doc) >= seq_len:
                start = np.random.randint(0, len(doc) - seq_len + 1)
                seq = doc[start : start + seq_len]
            else:
                # Pad or repeat document to fill seq_len
                repeats = (seq_len // len(doc)) + 1
                seq = np.tile(doc, repeats)[:seq_len]
            batch_tokens.append(seq)
    else:
        # Fallback synthetic tokens if shards are empty
        for _ in range(batch_size):
            seq = np.random.randint(10, vocab_size - 10, size=(seq_len,), dtype=np.int64)
            batch_tokens.append(seq)

    arr = np.array(batch_tokens, dtype=np.int64)
    input_ids = torch.from_numpy(arr)
    # Target labels equal input_ids for autoregressive next-token prediction
    labels = input_ids.clone()
    return input_ids, labels


def main():
    parser = argparse.ArgumentParser(description="Distributed Foundation Model Pretraining Launcher")
    parser.add_argument("--config", type=str, default="configs/model_8b_config.json", help="Model JSON config")
    parser.add_argument("--shards-dir", type=str, default="data/distributed_shards", help="Binary shards directory")
    parser.add_argument("--batch-size", type=int, default=2, help="Micro batch size per device")
    parser.add_argument("--seq-len", type=int, default=1024, help="Context length sequence length")
    parser.add_argument("--accum-steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--steps", type=int, default=10, help="Total training steps to execute")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Linear warmup steps")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints", help="Checkpoint output folder")
    parser.add_argument("--small-scale", action="store_true", help="Scale down layers for rapid verification/testing")
    args = parser.parse_args()

    print("=" * 80)
    print("UNIVERSAL FOUNDATION MODEL PRETRAINING LAUNCHER")
    print("=" * 80)
    print(f"Architecture Config : {args.config}")
    print(f"Shards Source Dir   : {args.shards_dir}")
    print(f"Batch Size (Micro)  : {args.batch_size}")
    print(f"Sequence Length     : {args.seq_len} tokens")
    print(f"Gradient Accum      : {args.accum_steps} steps (Effective batch size = {args.batch_size * args.accum_steps})")
    print(f"Peak Learning Rate  : {args.lr}")

    # Discover shards
    prefixes = get_available_shard_prefixes(args.shards_dir)
    if not prefixes:
        # Check live_shards directory fallback
        prefixes = get_available_shard_prefixes("data/live_shards")

    datasets = []
    total_tokens_available = 0
    for p in prefixes:
        try:
            ds = MMapIndexedDataset(p)
            datasets.append(ds)
            total_tokens_available += ds.total_tokens
        except Exception as e:
            print(f"[Warning] Failed loading shard {p}: {e}")

    print(f"[Data Lake] Active MMap Shards Loaded: {len(datasets)} ({total_tokens_available:,} tokens memory-mapped)")

    # Model configuration
    model_args = ModelArgs.from_json(args.config) if os.path.exists(args.config) else ModelArgs()
    if args.small_scale:
        model_args.hidden_size = 512
        model_args.intermediate_size = 1536
        model_args.num_hidden_layers = 4
        model_args.num_attention_heads = 8
        model_args.num_key_value_heads = 2
        model_args.max_position_embeddings = args.seq_len

    pretrain_cfg = PretrainConfig(
        model_config_path=args.config,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.steps,
        checkpoint_dir=args.checkpoint_dir,
        save_interval_steps=max(5, args.steps // 2)
    )

    engine = PretrainEngine(pretrain_cfg, model_args=model_args)
    print(f"[Model Engine] Device: {engine.device} | Parameter Count: {engine.param_count:,}")

    print("\nStarting Training Iterations:")
    print("-" * 80)
    start_all = time.time()
    for step in range(args.steps):
        input_ids, labels = create_batch_from_shards(datasets, args.batch_size, args.seq_len, model_args.vocab_size)
        metrics = engine.train_step(input_ids, labels)

        if step % 2 == 0 or step == args.steps - 1:
            print(
                f"Step {metrics['step']:03d}/{args.steps} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"PPL: {metrics['perplexity']:.2f} | "
                f"LR: {metrics['lr']:.2e} | "
                f"Speed: {metrics['tokens_per_sec']:.1f} tok/s | "
                f"TFLOPs: {metrics['tflops_sec']:.2f} | "
                f"Tokens: {metrics['total_tokens_trained']:,}"
            )

    elapsed = time.time() - start_all
    print("-" * 80)
    print(f"[Done] Completed {args.steps} steps in {elapsed:.2f}s ({engine.total_tokens_trained:,} tokens trained).")

    # Persist final checkpoint
    final_ckpt = engine.save_checkpoint("checkpoint_final.pt")
    print(f"[Persistence] Final checkpoint saved: {final_ckpt}")

    # Clean up mmap handles
    for ds in datasets:
        ds.close()


if __name__ == "__main__":
    main()
