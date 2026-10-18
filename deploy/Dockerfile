# Production Multi-Node Ray Worker Container for LLM Training Data Pipelines
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    RAY_DISABLE_DOCKER_CPU_WARNING=1

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install high-throughput Python data stack
RUN pip install --no-cache-dir \
    "ray[default,data]>=2.35.0" \
    "tokenizers>=0.19.0" \
    "trafilatura>=1.12.0" \
    "warcio>=1.8.0" \
    "boto3>=1.34.0" \
    "botocore>=1.34.0" \
    "polars>=1.0.0" \
    "pyarrow>=17.0.0" \
    "numpy>=1.26.0" \
    "pyyaml>=6.0.1" \
    "fastapi>=0.110.0" \
    "uvicorn>=0.28.0" \
    "sympy>=1.12.0" \
    "moto[server]>=5.0.0"

# Copy training data pipelines, web dashboard, and configurations
COPY pipelines/ /workspace/pipelines/
COPY schemas/ /workspace/schemas/
COPY configs/ /workspace/configs/
COPY tests/ /workspace/tests/
COPY deploy/ /workspace/deploy/
COPY web/ /workspace/web/

CMD ["bash"]
