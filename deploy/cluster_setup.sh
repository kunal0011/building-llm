#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo "Initializing High-Throughput Ray Worker Node for LLM Data"
echo "=========================================================="

# 1. Linux Kernel & Networking Optimizations for 100Gbps ENA/InfiniBand
echo "[*] Applying kernel network and memory optimizations..."
sudo sysctl -w net.core.somaxconn=65535
sudo sysctl -w net.ipv4.tcp_max_syn_backlog=65535
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 67108864"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 67108864"
sudo sysctl -w vm.max_map_count=1048576
sudo sysctl -w fs.file-max=2097152

# 2. Check for Local NVMe Instance Storage
echo "[*] Checking local NVMe drives for high-speed scratch space..."
NVME_DRIVE=$(lsblk -d -o NAME,TYPE,SIZE | grep -E "nvme[1-9]n1" | awk '{print $1}' | head -n 1 || true)

if [ -n "${NVME_DRIVE}" ]; then
    echo "[+] Found local NVMe instance store: /dev/${NVME_DRIVE}"
    sudo mkfs.ext4 -F "/dev/${NVME_DRIVE}"
    sudo mkdir -p /mnt/ray_scratch
    sudo mount -o noatime,nodiratime,discard "/dev/${NVME_DRIVE}" /mnt/ray_scratch
    sudo chmod 777 /mnt/ray_scratch
    export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
    echo "[+] Mounted /mnt/ray_scratch for zero-latency memory-mapped binary sharding"
else
    echo "[-] No ephemeral NVMe drive found. Defaulting to /tmp/scratch"
    sudo mkdir -p /tmp/scratch
    sudo chmod 777 /tmp/scratch
fi

# 3. Environment Variables
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export AWS_DEFAULT_REGION="us-east-1"

echo "[✓] Worker node initialization complete."
