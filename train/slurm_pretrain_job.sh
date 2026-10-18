#!/bin/bash
# ==============================================================================
# SLURM Multi-Node Supercluster Pretraining Job Script
# Target: 64 Nodes x 8 H100 SXM5 GPUs = 512 GPUs (15T Token Pre-training)
# ==============================================================================
#SBATCH --job-name=frontier-llm-pretrain
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --exclusive
#SBATCH --time=14-00:00:00
#SBATCH --partition=gpu-h100
#SBATCH --output=logs/pretrain_%j.out
#SBATCH --error=logs/pretrain_%j.err

set -e
mkdir -p logs data/checkpoints

# 1. Environment & High-Bandwidth Interconnect Tuning
export OMP_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_P2P_DISABLE=0
export NCCL_BUFFSIZE=16777216
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# 2. Master Node Rendezvous Setup
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "[SLURM] Cluster Head Node: $head_node ($head_node_ip)"

# 3. Distributed PyTorch Launcher via torchrun
srun torchrun \
    --nnodes=64 \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:29500 \
    train/launch_distributed_pretraining.py \
    --config configs/model_8b_config.json \
    --shards-dir data/distributed_shards \
    --batch-size 4 \
    --seq-len 8192 \
    --accum-steps 8 \
    --lr 3e-4 \
    --steps 500000 \
    --checkpoint-dir data/checkpoints
