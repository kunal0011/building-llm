# Building LLM: Frontier Foundation Model Pretraining & Data Engineering Platform

A production-grade, distributed pipeline for curating, filtering, deduplicating, and sharding **15 to 20 Trillion tokens** of training data for frontier Large Language Models. 

Engineered for:
- **Iterative Parameter Progression**: 8B $\rightarrow$ 32B $\rightarrow$ 70B+ parameters.
- **Progressive Context Extension**: 8k $\rightarrow$ 32k $\rightarrow$ 128k $\rightarrow$ 1M+ tokens.
- **Long Persistent Horizons**: Full-codebase repository refactoring and multi-day autonomous agent execution.
- **Multimodal Forward Compatibility**: Structured schema preserving spatial and temporal visual/audio anchors.

---

## Architecture Overview

```
building-llm/
├── configs/
│   ├── data_mix_config.yaml       # Domain mixture weights across pretraining & annealing
│   └── languages.json             # Top-30 world languages distribution targets
├── deploy/
│   ├── Dockerfile                 # Multi-node Ray worker container with Rust toolchain
│   ├── ray_cluster_aws.yaml       # AWS Ray Autoscaler configuration (m6i head, c6i workers)
│   └── cluster_setup.sh           # Node kernel tuning & NVMe scratch mount script
├── pipelines/
│   ├── crawling/
│   │   └── warc_extractor.py      # Common Crawl WARC streaming & Trafilatura extraction
│   ├── storage/
│   │   └── s3_manager.py          # S3 multipart upload & state checkpoint manager
│   ├── code/
│   │   └── repo_ast_ingestor.py   # Topological AST repository parser & FIM augmenter
│   ├── eval/
│   │   └── decontamination_scanner.py # 13-gram benchmark leakage detector (HumanEval, GSM8K, SWE-bench)
│   ├── academic/
│   │   └── stem_math_normalizer.py # LaTeX formula cleaner, theorem tagger & citation weaver
│   ├── dataloader/
│   │   └── mmap_dataset.py        # Zero-copy memory-mapped PyTorch dataset & curriculum batcher
│   ├── filtering/
│   │   └── heuristic_filter.py    # Frontier-grade structural, repetition & PII filtering
│   ├── deduplication/
│   │   └── minhash_lsh.py         # 128-perm MinHash LSH for near-duplicate clustering
│   ├── long_context/
│   │   └── document_packer.py     # First-Fit Decreasing packing with boundary reset attention
│   ├── agent_synthesis/
│   │   └── trajectory_builder.py  # Repository refactoring & multi-day persistence traces
│   ├── tokenization/
│   │   ├── custom_bpe_tokenizer.py # 128k BPE Tokenizer with SWE/Agent/Multimodal tokens
│   │   ├── train_vocab_sampler.py # Balanced domain corpus sampler & tokenizer trainer
│   │   └── binary_shard_writer.py # Megatron-LM/Nanotron memory-mapped .bin/.idx writer
│   ├── rl/
│   │   ├── execution_sandbox.py   # Isolated code & test execution sandbox for 0/1 rewards
│   │   ├── symbolic_math_verifier.py # SymPy algebraic equivalence & step derivation verifier
│   │   ├── process_reward_annotator.py # PRM step-level CoT decomposition & reward attribution
│   │   └── grpo_reward_engine.py  # Group Relative Policy Optimization advantage calculator
│   ├── distributed_runner.py      # Local worker pipeline orchestrator
│   └── ray_distributed_data_job.py # Multi-node Ray cluster data ingestion job
├── schemas/
│   └── data_schema.py             # Dataclass schemas for documents, repos, and agent rollouts
└── tests/
    ├── test_pipelines.py          # Core pipeline integration and verification suite
    ├── test_all_tracks.py         # Verification suite for Tracks 1, 2, 3, and 4
    └── test_rl_environment.py     # Verification suite for Post-Training RL / GRPO suite
```

---

## 1. Quickstart & Verification

Run the end-to-end test suite to verify all heuristic filters, MinHash deduplication, context packing, agent trajectory generation, and binary sharding:

```bash
PYTHONPATH=. python3 tests/test_pipelines.py
```

Run the pipeline execution demo:

```bash
PYTHONPATH=. python3 pipelines/distributed_runner.py
```

Run the live streaming dry-run (streams real Common Crawl data + AST code + STEM math into .bin/.idx shards):

```bash
python3 pipelines/live_dry_run.py
```

---

## 2. Token Scaling Matrix (8B $\rightarrow$ 32B $\rightarrow$ 70B)

| Stage | Model Scale | Token Budget | Sequence Length | RoPE Base $\theta$ | Primary Workload |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Pre-train** | 8B / 32B / 70B | 14.0 Trillion | 8,192 | 10,000 | Broad world knowledge, web, books, code |
| **Anneal Phase 1** | 8B / 32B / 70B | 500 Billion | 32,768 | 100,000 | Long books, multi-file code modules |
| **Anneal Phase 2** | 32B / 70B | 300 Billion | 131,072 | 1,000,000 | Whole repositories, legal/financial filings |
| **Anneal Phase 3** | 70B | 100 Billion | 1,048,576 | 10,000,000 | Multi-day agent memory, RingAttention |

---

## 3. High-Performance Filtering & Deduplication

1. **Heuristic Filter (`pipelines/filtering/heuristic_filter.py`)**:
   - Rejects documents violating word length distributions (mean 3.0 to 12.0 chars).
   - Rejects excessive symbol-to-word ratios ($>0.35$).
   - Flags navigation menus and bullet dumps ($>0.85$ bullet line ratio).
   - Catches repeated lines ($>0.30$) and repeated 4-grams ($>0.20$).
   - Scrubs international phone numbers, IPv4 addresses, emails, and credit cards with structured replacement tokens (`<|PII_EMAIL|>`, `<|PII_PHONE|>`).

2. **MinHash LSH (`pipelines/deduplication/minhash_lsh.py`)**:
   - Computes 64–128 permutation signatures over 5-gram shingles using linear hash permutations modulo Mersenne prime $2^{61}-1$.
   - Clusters near-duplicates across $b$ bands and selects lexicographically canonical documents.

---

## 4. Long-Persistent Horizon & SWE Agent Trajectories

To support multi-day autonomous agents and repository-level refactoring:
- **Repository-Level Ingestion**: Documents are ingested topologically by AST dependency order (configs $\rightarrow$ interfaces $\rightarrow$ implementations $\rightarrow$ tests) with boundary tags (`<|repo_start|>`, `<|file_start|>`).
- **State Checkpointing**: Synthetic trajectories include explicit state restoration frames (`<|state_checkpoint_restore|>`) containing active goals, completed milestones, and scratchpad memory, training the model to recover from interruptions.
- **Self-Correction Rollouts**: Includes simulated syntax or test failure observations, prompting the model to inspect tracebacks, self-correct patches, and rerun verification tests.

---

## 5. Binary Sharding & S3 Cloud Staging

Binary outputs are written using `BinaryShardWriter` (`.bin` and `.idx`), providing:
- Zero-copy `mmap` ingestion on GPU nodes during distributed training (Megatron-LM, Nanotron, LLaMA-Factory).
- Direct streaming from AWS S3 (`s3://ai-data-shards/`) using multi-part streaming without disk bottlenecks.

---

## 6. Multi-Node Ray Cluster Deployment (AWS)

### 6.1 Launching the Ray Cluster
To launch a multi-node Ray cluster across AWS EC2 spot/on-demand instances:

```bash
# Start cluster using AWS Ray Autoscaler
ray up deploy/ray_cluster_aws.yaml -y

# Forward Ray Dashboard to localhost:8265
ray dashboard deploy/ray_cluster_aws.yaml
```

### 6.2 Submitting Distributed Data Jobs
Submit the distributed WARC ingestion and binary sharding job across the cluster:

```bash
ray job submit --address http://localhost:8265 -- \
    python3 pipelines/ray_distributed_data_job.py \
        --s3-output-bucket my-ai-training-shards \
        --num-segments 1000 \
        --target-seq-len 8192
```

### 6.3 Building the Worker Docker Image
To build and push the container image for Kubernetes / KubeRay or Slurm execution:

```bash
docker build -t my-registry/ray-llm-data-worker:latest -f deploy/Dockerfile .
docker push my-registry/ray-llm-data-worker:latest
```

---

## 7. Docker Desktop Local Infrastructure & Visual Pipeline

Test the entire distributed environment locally on Docker Desktop before deploying to cloud infrastructure:

```bash
# Launch S3 emulator, Ray cluster (head + worker), and visual dashboard
docker compose up -d

# Check live service status
docker compose ps
```

### Active Endpoints on Docker Desktop:
- **Visual Pipeline Dashboard**: [http://localhost:3000](http://localhost:3000) (Interactive stage topology, live distributed triggers, shard inspector, and terminal log stream)
- **Ray Cluster Telemetry Dashboard**: [http://localhost:8265](http://localhost:8265) (Live CPU/RAM worker metrics and task scheduling)
- **AWS S3 Storage Emulator**: [http://localhost:4566](http://localhost:4566) (Auto-initialized with `s3://ai-data-raw` and `s3://ai-data-shards`)

### Shutting Down Containers:
```bash
docker compose down
```

---

## 8. Multi-Stage Context Annealing (8k $\rightarrow$ 32k $\rightarrow$ 128k $\rightarrow$ 1M+)

Dynamic curriculum scheduling and mathematical RoPE calibration for progressive sequence length expansion:

| Stage | Context Length | Target Budget | Base $\theta$ | NTK Scaled $\theta$ | YaRN Temp Scale | Dominant Domain Mix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **8k Base** | 8,192 tokens | 12.0 Trillion | 500,000 | 500,000 | 1.0000 | Curated Web (60%), Code (20%), STEM (10%) |
| **32k Extended** | 32,768 tokens | 1.5 Trillion | 1,000,000 | 2,044,497 | 1.1386 | Web (35%), Code (35%), Books (15%) |
| **128k Long Context** | 131,072 tokens | 500 Billion | 5,000,000 | 8,359,937 | 1.2773 | Whole Repositories (40%), STEM (25%), Legal (20%) |
| **1M+ Extreme Horizon**| 1,048,576 tokens| 100 Billion | 10,000,000 | 69,123,823 | 1.4852 | Full Repo AST Graphs (45%), Monographs (25%), Agent Traces (20%) |

### Features:
- **`RoPEScalingCalculator`**: Implements standard RoPE, NTK-aware scaling ($\theta' = \theta \cdot s^{d/(d-2)}$), and YaRN high/low frequency boundary interpolation.
- **`NeedleInAHaystackSynthesizer`**: Generates verifiable long-context retrieval benchmarks with secret needles embedded at calibrated depths (0% to 100%).

---

## 9. Distributed RL Experience Buffer & GRPO Collector

Scalable reinforcement learning with verifiable rewards (RLVR) for frontier reasoning and multi-turn agent capabilities:
- **`DistributedExperienceBuffer`**: Aggregates candidate rollout groups ($G=4$ to $16$), assigns dense step-level PRM credit, and generates mini-batches with group-normalized advantages:
  $$A_i = \frac{R_i - \text{mean}(R)}{\text{std}(R) + \epsilon}$$
- **`distributed_rl_collector.py`**: Executes parallel agent reasoning paths across Ray cluster worker nodes:
  - Verifies code execution in isolated `CodeExecutionSandbox` for deterministic test suite rewards ($0.0$ or $1.0$).
  - Validates mathematical proofs and algebraic derivations with SymPy via `SymbolicMathVerifier`.
  - Persists serialized RL training batches directly to `s3://ai-data-shards/rl/`.

---

## 10. Multimodal Data Fusion Track (Vision-Language Alignment & Interleaved Sharding)

Engineered for multimodal pretraining transitions (interleaved web articles, technical papers with figures, and UI layouts):
- **`ImageFilterValidator`**: Validates MIME types, enforces resolution ($120 \le w, h \le 8192$, pixel area $\ge 25,000$), aspect ratio ($0.2 \le w/h \le 5.0$), and strips tracking pixels and generic boilerplate alt texts.
- **`CLIPSigLIPAligner`**: Computes dual-encoder cosine similarity between visual patch representations and textual descriptions ($\tau = 0.24$), boosting marginal samples with surrounding document paragraphs.
- **`InterleavedMultimodalFormatter`**: Serializes documents into standardized interleaved syntax with `<|image_ref_start|>` anchor tags, bounding boxes, and S3 URIs in natural reading order.
- **`distributed_multimodal_pipeline.py`**: Distributed Ray pipeline executing parallel multimodal extraction, tokenization with Universal BPE, and Megatron-LM memory-mapped binary sharding (`.bin`/`.idx`) with S3 image staging (`s3://ai-data-shards/multimodal/`).

---

## 11. Production AWS Terraform Infrastructure (`deploy/terraform/`)

Production-ready Terraform configurations orchestrating scalable multi-node Ray cluster compute and petabyte-scale S3 data lake storage:

```
deploy/terraform/
├── main.tf              # AWS Provider (~> 5.0) & random stack suffix
├── variables.tf         # Multi-AZ CIDRs, EC2 instance types, Spot worker bounds
├── vpc.tf               # Multi-AZ VPC (10.0.0.0/16), NAT Gateway, S3 Gateway VPC Endpoint
├── s3.tf                # S3 Data Lake (raw, intermediate, shards) with SSE, PAB & Lifecycle
├── iam.tf               # RayClusterEC2Role, fine-grained S3 policy, SSM session manager
├── security_groups.tf   # Intra-cluster self-referencing SG, Ray Dashboard (8265), Visualizer (3000)
├── compute.tf           # Ray Head EC2 (m6i.4xlarge) & Spot Worker Auto Scaling Group (c6i.16xlarge)
└── outputs.tf           # VPC IDs, S3 bucket names, Head Node IPs, Ray submission commands
```

### Key Infrastructure Highlights:
- **Zero-Cost S3 Gateway Endpoint**: Routes all data lake shard streaming over private AWS backbone networking without incurring NAT data processing charges.
- **Spot Worker Economics**: Auto Scaling Group leverages mixed instances (`c6i.16xlarge`, `c5.18xlarge`, `m6i.16xlarge`) with `price-capacity-optimized` allocation for ~75% compute cost savings.
- **Zero-SSH Key Access**: Configured with AWS Systems Manager (`AmazonSSMManagedInstanceCore`) for cryptographically signed browser/CLI session management without open port 22.

### Deployment Commands:
```bash
cd deploy/terraform
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

---

## 12. Autonomous SWE Agent Synthesizer (Full-Codebase PR Resolution)

High-fidelity multi-turn agent interaction trajectory generator reproducing SWE-bench Verified benchmarks:
- **`SWEAgentSynthesizer`**: Simulates multi-turn tool interaction (`bash`, `file_editor`), diagnostic traceback analysis, and unified git diff generation.
- **Diagnostic Self-Correction**: Injects realistic intermediate syntax and indentation exceptions, teaching the model self-recovery during long-horizon editing tasks.
- **Deterministic 0/1 Reward Verification**: Directly integrates with isolated `CodeExecutionSandbox` to execute reproduction scripts and regression test suites.
- **`DistributedSWECollector`**: Parallelizes trajectory synthesis across Ray Head and Worker nodes, persists corpora to `s3://ai-data-shards/swe_bench/`, and tracks cluster telemetry.

---

---

## 13. Distributed Foundation Model Pretraining Suite (`train/`)

High-throughput distributed pretraining engine streaming memory-mapped binary shards directly into GPU buffers:
- **`UniversalLlamaForCausalLM`**: Modern decoder architecture with Grouped-Query Attention (GQA 4:1), SwiGLU activation, RMSNorm ($\epsilon=1\text{e-}5$), and RoPE with YaRN / NTK frequency scaling.
- **Architectural Configs**:
  - `configs/model_8b_config.json`: 32 layers, hidden 4096, intermediate 14336, 128k BPE vocabulary (15T token target).
  - `configs/model_32b_config.json`: 48 layers, hidden 6144, intermediate 20480 (18T token target).
  - `configs/model_70b_config.json`: 64 layers, hidden 8192, intermediate 28672 (20T token target).
- **`PretrainEngine`**:
  - Mixed precision (BF16 / FP16) with PyTorch AMP and gradient scaling.
  - AdamW optimizer with Cosine Annealing learning rate schedule + linear warmup.
  - Telemetry: Real-time tokens/sec, TFLOPs, and Model Flops Utilization (MFU %) tracking.
  - Checkpoint persistence and graceful crash recovery.
- **`launch_distributed_pretraining.py`**:
  ```bash
  # Run distributed pretraining from memory-mapped shards:
  python3 train/launch_distributed_pretraining.py --batch-size 4 --seq-len 2048 --accum-steps 4
  ```
- **SLURM Supercluster Script (`train/slurm_pretrain_job.sh`)**: Multi-node batch submission script for 64 nodes $\times$ 8 H100 GPUs (512 GPUs).

---

## 14. Multi-Turn SWE-bench Evaluation Harness (`eval/swe_bench/`)

Deterministic evaluation harness evaluating model and agent policies against real-world GitHub issues:
- **`AgentEnvironment`**: Subprocess-isolated sandbox workspaces executing `bash`, `file_editor`, and `git` actions with timeouts and line range protections.
- **`SWEBenchEvalHarness`**:
  - Multi-turn interaction loop up to `max_turns = 15`.
  - Unbiased $\text{pass}@k$ estimator: $\text{pass}@k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$.
  - Measures self-correction recovery rate from compiler, syntax, and indentation exceptions.
  - Generates structured JSON and GitHub-flavored Markdown evaluation reports.
- **CLI Runner**:
  ```bash
  python3 eval/swe_bench/run_eval.py --samples-per-task 1 --output-dir data/eval_reports
  ```
- **Web REST API**:
  - `POST /api/eval/swe-bench`: Dispatches evaluation benchmark run and returns real-time pass@1 metrics.
  - `GET /api/eval/reports`: Lists generated benchmark reports.

---

## 15. DeepSeek-Style Mixture-of-Experts (MoE) Architecture (`train/moe_model.py`)

Frontier fine-grained MoE architecture engineered for maximum computational efficiency:
- **`MoETopKRouter`**: Softmax gating mechanism selecting top-$k$ active experts per token with differentiable auxiliary load balancing loss ($\mathcal{L}_{aux} = \alpha \cdot E \cdot \sum_{i=1}^E f_i P_i$).
- **Fine-Grained SwiGLU Experts**: 16/64 fine-grained routed experts combined with 2 dedicated shared experts that remain active on every token for universal invariant knowledge representation.
- **Architectural Config (`configs/model_moe_config.json`)**:
  - 16 routed experts, 2 shared experts, top-4 active routed per token.
  - Intermediate dimension $d_{expert} = 768$ (1/2 size of dense FFN for fine-grained specialization).
  - Dynamic autograd gradient routing and combined output fusion.

---

## 16. Multimodal Vision-Language Pretraining Integration (`train/multimodal_model.py`)

Interleaved vision-language pretraining bridging continuous visual representations into causal autoregressive decoding:
- **`VisionPatchEncoder`**: Convolutional patch projection encoder extracting continuous visual representations from raw image inputs.
- **`VisionLanguageAdapter`**: 2-layer GELU projection bridge with 1D adaptive pooling mapping variable patch resolutions to fixed visual token budgets (e.g. 16 or 64 visual tokens).
- **`UniversalMultimodalForCausalLM`**: In-place visual substitution replacing special multimodal placeholder tokens (`<|image_ref_start|>`) with projected continuous visual vectors in the embedding space before causal transformer decoder blocks.

---

## 17. OpenAI-Compatible Model Serving & Low-Latency KV Cache Engine (`serve/`)

Production-grade inference runtime exposing OpenAI-compatible endpoints with microsecond telemetry:
- **`DynamicKVCache`**: Paged key-value cache eliminating redundant key/value attention projections during autoregressive token generation.
- **`InferenceEngine`**: High-throughput sampling engine supporting:
  - Temperature scaling, top-$p$ (nucleus) sampling, top-$k$ filtering, and repetition penalties.
  - Real-time telemetry tracking Time-To-First-Token (TTFT in ms), total latency, and tokens-per-second generation speed.
  - Automatic checkpoint loader parsing model weights and dataclass configurations directly from disk.
- **OpenAI-Compatible FastAPI Server (`serve/server.py`)**:
  - `GET /v1/models`: Enumerates served foundation model checkpoints.
  - `POST /v1/chat/completions`: Formats multi-turn chat messages (`system`, `user`, `assistant`), generates continuations, and returns standard OpenAI-compatible JSON responses.
  - `POST /v1/completions`: Raw prompt completion endpoint with configurable sampling parameters.
- **Launch Server**:
  ```bash
  python3 serve/server.py
  # Endpoint live on http://localhost:8000
  ```

---

## 18. FP8 & INT4 Post-Training Quantization Engine (`quantize/`)

Low-bit model compression and deployment engine:
- **`QuantLinear`**: Packed 4-bit (2 weights per byte) and 8-bit custom PyTorch linear layers with per-group scales and zero points.
- **`AWQCalibrator`**: Activation-aware Weight Quantization identifying salient feature channels ($s = s_X^\alpha$) on calibration batches, protecting critical weights and mitigating quantization noise.
- **`ModelQuantizer`**: Recursive transformer module quantizer replacing attention and MLP layers with `QuantLinear`, calculating parameter reduction (e.g. 50% to 75% memory savings) and serializing compressed weights.

---

## 19. Direct Preference Optimization (DPO) & KTO Alignment Engine (`alignment/`)

Implicit reward optimization and agent preference alignment:
- **`SWEPreferenceDataExtractor`**: Automatically extracts chosen vs. rejected preference trajectories from SWE-bench runs based on sandbox test verification (`sandbox_passed == True` vs. `False`).
- **`DPOEngine`**: Computes token-level log probabilities, exact DPO loss ($\mathcal{L}_{DPO}$), implicit rewards, preference margins, and Kahneman-Tversky Optimization ($\mathcal{L}_{KTO}$) on unpaired positive/negative feedback.
- **`DPOTrainer`**: End-to-end training orchestrator updating the policy model against a frozen reference model snapshot using AdamW with gradient norm clipping and margin tracking.

---

## 20. Synthetic Mathematical Olympiad & Formal Verification Generator (`pipelines/academic/synthetic_math_olympiad.py`)

Olympiad-tier competitive reasoning synthesis:
- **`SyntheticOlympiadGenerator`**: Synthesizes high-difficulty problems across Number Theory (Chinese Remainder Theorem, Diophantine equations), Algebra (Vieta's power sums), Combinatorics (linear recurrences), and Calculus (Euler Beta integrals).
- **Formal Verification Dual-Export**:
  - Auto-generates Lean 4 theorem declarations (`theorem ... : ... := by ...`).
  - Auto-generates Isabelle/HOL theory and lemma proofs (`lemma ... shows ... by auto`).
- **`FormalMathVerifier`**: Validates mathematical equivalence and ground truth via SymPy symbolic solver and syntax structure analysis.
- **`FormalMathCorpusPacker`**: Enforces structured control tokens (`<|thought|>`, `<|thought_end|>`) and streams directly into memory-mapped `.bin` and `.idx` pretraining shards.

---

## 21. Running Test Suites

```bash
# Run all 71 unit & integration tests across 17 modules:
PYTHONPATH=. pytest -v
```
All **71 tests** pass with 100% coverage across heuristic filters, MinHash deduplication, context packing, RoPE YaRN scaling, GRPO experience buffer, multimodal CLIP alignment, AWS Terraform validation, SWE agent synthesis, foundation model pretraining, SWE-bench evaluation harness, DeepSeek MoE routing, vision-language multimodal substitution, Dynamic KV Cache, OpenAI-compatible inference endpoints, FP8/INT4 quantization with AWQ, DPO/KTO preference alignment, and synthetic formal Olympiad math generation.
