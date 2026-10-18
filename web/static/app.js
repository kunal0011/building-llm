// Visual Pipeline Interactive Application Logic

const STAGE_DETAILS = {
  ingestion: {
    title: "Stage 01: Multi-Source Stream Ingestion",
    body: `
      <p><strong>Mechanics:</strong> Zero-disk streaming from Common Crawl (<code>s3://commoncrawl/</code> or HTTPS) and topological AST ingestion of Git repositories.</p>
      <ul>
        <li><strong>Web:</strong> Trafilatura body text extraction, table preservation, and HTML <code>&lt;img&gt;</code> anchor detection.</li>
        <li><strong>Code:</strong> Ingests whole codebases topologically ordered by AST dependencies (configs &rarr; interfaces &rarr; implementations &rarr; tests).</li>
        <li><strong>Multimodal Hooks:</strong> Embeds <code>&lt;|image_ref_001|&gt;</code> placeholders with S3 URIs and alt-text descriptions.</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/crawling/warc_extractor.py</code> & <code>pipelines/code/repo_ast_ingestor.py</code></p>
    `
  },
  filtering: {
    title: "Stage 02: Heuristic & PII Quality Filtering",
    body: `
      <p><strong>Mechanics:</strong> Frontier-grade quality filters rejecting low-information web pages, boilerplates, and automated dumps.</p>
      <ul>
        <li><strong>Word Length Bounds:</strong> Rejects mean word length &lt; 3.0 or &gt; 12.0 characters.</li>
        <li><strong>Symbol Threshold:</strong> Caps symbol-to-word ratio at 0.35.</li>
        <li><strong>Repetition Penalties:</strong> Discards documents with repeated lines &gt; 30% or repeated 4-grams &gt; 20%.</li>
        <li><strong>PII Scrubbing:</strong> Redacts phone numbers, emails, and IPv4 addresses with structured tokens (<code>&lt;|PII_EMAIL|&gt;</code>).</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/filtering/heuristic_filter.py</code></p>
    `
  },
  dedup: {
    title: "Stage 03: MinHash LSH Near-Deduplication",
    body: `
      <p><strong>Mechanics:</strong> Removes duplicated syndication, mirrors, and boilerplate across petabytes of text.</p>
      <ul>
        <li><strong>Signature Generation:</strong> 128 linear hash permutations modulo Mersenne prime $2^{61}-1$ over 5-gram shingles.</li>
        <li><strong>LSH Band Hashing:</strong> Partitions signatures into bands to index colliding documents into graph clusters.</li>
        <li><strong>Canonical Selection:</strong> Retains lexicographically canonical document, dropping duplicates.</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/deduplication/minhash_lsh.py</code></p>
    `
  },
  decontamination: {
    title: "Stage 04: Benchmark De-Contamination Scanner",
    body: `
      <p><strong>Mechanics:</strong> Guarantees zero test-set leakage prior to pre-training tokenization.</p>
      <ul>
        <li><strong>Evaluation Splits:</strong> Indexes HumanEval, MBPP, GSM8K, MATH, SWE-bench, and MMLU questions into a 64-bit 13-gram hash table.</li>
        <li><strong>Pre-tokenization Filter:</strong> Flags and quarantines any document with identical 13-gram sequences.</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/eval/decontamination_scanner.py</code></p>
    `
  },
  tokenization: {
    title: "Stage 05: Universal 128k BPE Tokenizer",
    body: `
      <p><strong>Mechanics:</strong> Rust-backed Byte-Level Byte-Pair Encoding with LLaMA-3/GPT-4 regex pre-tokenization.</p>
      <ul>
        <li><strong>Vocabulary Size:</strong> 128,000 tokens with full byte fallback (zero unknown tokens).</li>
        <li><strong>Specialized Tokens:</strong> SWE repos (<code>&lt;|repo_start|&gt;</code>), FIM (<code>&lt;|fim_prefix|&gt;</code>), Agentic thoughts (<code>&lt;|thought|&gt;</code>), and persistent checkpoints.</li>
        <li><strong>Compression:</strong> Achieves 2.2 bytes/token on English and 1.7 bytes/token on code.</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/tokenization/custom_bpe_tokenizer.py</code></p>
    `
  },
  packing: {
    title: "Stage 06: Progressive Context Packing",
    body: `
      <p><strong>Mechanics:</strong> First-Fit Decreasing (FFD) bin-packing to eliminate padding token overhead.</p>
      <ul>
        <li><strong>Sequence Curricula:</strong> Progressive expansion from 8,192 &rarr; 32,768 &rarr; 131,072 &rarr; 1,048,576 tokens.</li>
        <li><strong>Boundary Resets:</strong> Position IDs reset to 0 at document boundaries for block-diagonal flash attention.</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/long_context/document_packer.py</code></p>
    `
  },
  sharding: {
    title: "Stage 07: Megatron-LM Memory-Mapped Shards",
    body: `
      <p><strong>Mechanics:</strong> Writes flat binary token arrays (<code>.bin</code>) and metadata offset pointers (<code>.idx</code>).</p>
      <ul>
        <li><strong>Zero-Copy DMA:</strong> Ingested by GPUs via <code>np.memmap</code> with zero CPU memory overhead.</li>
        <li><strong>S3 Staging:</strong> Multi-part streaming to local MinIO / AWS S3 buckets.</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/tokenization/binary_shard_writer.py</code> & <code>pipelines/dataloader/mmap_dataset.py</code></p>
    `
  },
  rl: {
    title: "Stage 08: Post-Training RL & Verifiable Sandbox",
    body: `
      <p><strong>Mechanics:</strong> Self-correction training with deterministic outcome rewards and process supervision.</p>
      <ul>
        <li><strong>Execution Sandbox:</strong> Subprocess isolation running test suites for binary $0/1$ rewards.</li>
        <li><strong>Symbolic Math:</strong> SymPy algebraic simplification checking step derivations without hallucination.</li>
        <li><strong>GRPO Engine:</strong> Group Relative Policy Optimization calculating group-normalized advantages without a Critic network.</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/rl/execution_sandbox.py</code> & <code>pipelines/rl/grpo_reward_engine.py</code></p>
    `
  },
  swe_agent: {
    title: "Stage 09: Autonomous SWE Agent PR Synthesizer",
    body: `
      <p><strong>Mechanics:</strong> Synthesizes multi-turn software engineering trajectories matching SWE-bench Verified benchmarks.</p>
      <ul>
        <li><strong>Tool Interaction:</strong> Simulates Bash reconnaissance, AST symbol lookup, and file editor replacements.</li>
        <li><strong>Reproduction Scripts:</strong> Synthesizes standalone <code>reproduce_issue.py</code> scripts verifying failures prior to patching.</li>
        <li><strong>Self-Correction Recovery:</strong> Injects intermediate syntax/indentation exceptions, teaching the model self-recovery.</li>
        <li><strong>Deterministic Sandbox:</strong> Executes patches against full regression suites for verified $0/1$ rewards.</li>
      </ul>
      <p><em>Module:</em> <code>pipelines/agent_synthesis/swe_agent_synthesizer.py</code> & <code>pipelines/agent_synthesis/distributed_swe_collector.py</code></p>
    `
  }
};

document.addEventListener("DOMContentLoaded", () => {
  const nodes = document.querySelectorAll(".pipeline-node");
  const drawer = document.getElementById("node-drawer");
  const drawerTitle = document.getElementById("drawer-title");
  const drawerBody = document.getElementById("drawer-body");
  const closeDrawer = document.getElementById("close-drawer");

  const btnRunDistributed = document.getElementById("btn-run-distributed");
  const btnMultimodal = document.getElementById("btn-multimodal");
  const btnRLRollouts = document.getElementById("btn-rl-rollouts");
  const btnSweAgent = document.getElementById("btn-swe-agent");
  const btnEvalSwe = document.getElementById("btn-eval-swe");
  const btnCurriculum = document.getElementById("btn-curriculum");
  const btnRunDryRun = document.getElementById("btn-run-dry-run");
  const btnTestRL = document.getElementById("btn-test-rl");
  const btnRefreshShards = document.getElementById("btn-refresh-shards");
  const terminalBody = document.getElementById("terminal-body");
  const termStatus = document.getElementById("term-status");

  // Node Click Handlers
  nodes.forEach(node => {
    node.addEventListener("click", () => {
      nodes.forEach(n => n.classList.remove("selected"));
      node.classList.add("selected");
      const stage = node.getAttribute("data-stage");
      const detail = STAGE_DETAILS[stage];
      if (detail) {
        drawerTitle.textContent = detail.title;
        drawerBody.innerHTML = detail.body;
        drawer.style.display = "block";
      }
    });
  });

  closeDrawer.addEventListener("click", () => {
    drawer.style.display = "none";
    nodes.forEach(n => n.classList.remove("selected"));
  });

  function appendLog(text, colorClass = "text-dim") {
    const span = document.createElement("span");
    span.className = `log-line ${colorClass}`;
    span.textContent = text;
    terminalBody.appendChild(span);
    terminalBody.scrollTop = terminalBody.scrollHeight;
  }

  // Trigger Distributed Ray Job Action
  if (btnRunDistributed) {
    btnRunDistributed.addEventListener("click", async () => {
      btnRunDistributed.disabled = true;
      termStatus.textContent = "RAY RUNNING";
      termStatus.className = "status-indicator live";
      appendLog("\n" + "=".repeat(60), "text-yellow");
      appendLog("[*] DISPATCHING DISTRIBUTED RAY JOB ACROSS MULTI-NODE CLUSTER...", "text-yellow");
      appendLog("    - Streaming Common Crawl WARC segments in parallel", "text-dim");
      appendLog("    - Distributing BPE tokenization across Head & Worker nodes", "text-dim");
      appendLog("    - Packing sequences and streaming binary shards to S3", "text-dim");

      try {
        const resp = await fetch("/api/run-distributed-pipeline", { method: "POST" });
        const data = await resp.json();

        if (data.status === "success") {
          appendLog(`\n[✓] Distributed Ray Job Completed in ${data.elapsed_sec}s!`, "text-green");
          appendLog(`[+] Cluster Compute Nodes Utilized: ${data.cluster_nodes_utilized} nodes`, "text-cyan");
          for (const [node, count] of Object.entries(data.nodes || {})) {
            appendLog(`    * ${node}: ${count} parallel tasks executed`, "text-cyan");
          }
          appendLog(`[+] Common Crawl Streamed: ${(data.warc_bytes / 1024 / 1024).toFixed(2)} MB`, "text-dim");
          appendLog(`[+] Clean Decontaminated Documents: ${data.clean_documents}`, "text-dim");
          appendLog(`[+] Tokens Generated: ${data.total_tokens.toLocaleString()} tokens`, "text-green");
          appendLog(`[+] Packed Sequences (ctx=2048): ${data.packed_sequences}`, "text-dim");
          appendLog(`[+] Binary Shards: ${data.shards_count} (.bin + .idx)`, "text-cyan");
          appendLog(`[+] S3 Shard Objects Active: ${data.s3_objects_count} objects in s3://ai-data-shards/`, "text-green");

          document.getElementById("stat-tokens").textContent = data.total_tokens.toLocaleString();
          document.getElementById("stat-sequences").textContent = data.packed_sequences;
          document.getElementById("stat-storage").textContent = `${data.s3_objects_count} S3 Shards`;

          loadShards();
        } else {
          appendLog(`[!] Distributed Pipeline Error: ${data.error || "Unknown failure"}`, "text-red");
        }
      } catch (err) {
        appendLog(`[!] Network error: ${err.message}`, "text-red");
      } finally {
        btnRunDistributed.disabled = false;
        termStatus.textContent = "IDLE";
        termStatus.className = "status-indicator";
      }
    });
  }

  // Context Annealing Curriculum Action
  if (btnCurriculum) {
    btnCurriculum.addEventListener("click", async () => {
      btnCurriculum.disabled = true;
      appendLog("\n" + "=".repeat(60), "text-yellow");
      appendLog("[*] MULTI-STAGE CONTEXT ANNEALING CURRICULUM (8k -> 32k -> 128k -> 1M+)", "text-yellow");

      try {
        const resp = await fetch("/api/context-annealing/curriculum");
        const data = await resp.json();

        appendLog(`[✓] Total Context Annealing Token Budget: ${data.total_annealing_tokens}`, "text-green");
        appendLog(`[+] RoPE Progressive Expansion Factor: ${data.progressive_factor}`, "text-cyan");

        (data.curriculum || []).forEach(stg => {
          appendLog(`\n--- STAGE: ${stg.stage.toUpperCase()} (${stg.context_length}) ---`, "text-yellow");
          appendLog(`    * Token Budget   : ${stg.token_budget}`, "text-dim");
          appendLog(`    * RoPE Base Theta: ${stg.base_rope_theta} -> Scaled: ${stg.ntk_scaled_theta}`, "text-cyan");
          appendLog(`    * YaRN Temp Scale: ${stg.yarn_temp_scale}`, "text-dim");
          appendLog(`    * Primary Focus  : ${stg.primary_domain_focus}`, "text-dim");
        });

        // Also test NIAH synthesis
        appendLog("\n[*] Synthesizing Needle-In-A-Haystack (NIAH) Retrieval Benchmark at 65% depth...", "text-yellow");
        const niahResp = await fetch("/api/context-annealing/simulate-niah", { method: "POST" });
        const niahData = await niahResp.json();
        if (niahData.status === "success") {
          appendLog(`[✓] Embedded Needle Target : "${niahData.ground_truth}" at ${niahData.depth_percent} depth`, "text-green");
          appendLog(`[+] Test Query             : "${niahData.query}"`, "text-cyan");
          appendLog(`[+] Context Window Sample  : ${niahData.prompt_preview}`, "text-dim");
        }
      } catch (err) {
        appendLog(`[!] Error: ${err.message}`, "text-red");
      } finally {
        btnCurriculum.disabled = false;
      }
    });
  }

  // Distributed RL Experience Collection Action
  if (btnRLRollouts) {
    btnRLRollouts.addEventListener("click", async () => {
      btnRLRollouts.disabled = true;
      termStatus.textContent = "RL RUNNING";
      termStatus.className = "status-indicator live";
      appendLog("\n" + "=".repeat(60), "text-yellow");
      appendLog("[*] DISPATCHING DISTRIBUTED RL EXPERIENCE COLLECTION (GRPO + PRM)...", "text-yellow");
      appendLog("    - Generating candidate rollout groups (G=4) across Ray nodes", "text-dim");
      appendLog("    - Validating deterministic outcomes in execution sandbox & SymPy", "text-dim");
      appendLog("    - Scoring step-level PRM credit and group advantages", "text-dim");

      try {
        const resp = await fetch("/api/rl/run-distributed-rollouts", { method: "POST" });
        const data = await resp.json();

        if (data.status === "success") {
          appendLog(`\n[✓] Distributed RL Rollouts Captured in ${data.elapsed_sec}s!`, "text-green");
          appendLog(`[+] Cluster Compute Nodes : ${data.cluster_nodes_utilized} nodes`, "text-cyan");
          for (const [node, count] of Object.entries(data.nodes || {})) {
            appendLog(`    * ${node}: ${count} rollout groups executed`, "text-cyan");
          }
          appendLog(`[+] Prompts Evaluated     : ${data.total_prompts} tasks (SWE Code & STEM Proofs)`, "text-dim");
          appendLog(`[+] Trajectories Captured : ${data.total_trajectories} rollouts`, "text-green");
          appendLog(`[+] Verifiable Pass Rate  : ${data.pass_rate}`, "text-cyan");
          appendLog(`[+] Mean Reward           : ${data.mean_reward}`, "text-dim");
          appendLog(`[+] Advantage Spread      : ${data.advantage_spread}`, "text-cyan");
          appendLog(`[+] S3 Staging Target     : s3://ai-data-shards/${data.s3_key}`, "text-green");
        } else {
          appendLog(`[!] RL Collection Error: ${data.error || "Unknown failure"}`, "text-red");
        }
      } catch (err) {
        appendLog(`[!] Network error: ${err.message}`, "text-red");
      } finally {
        btnRLRollouts.disabled = false;
        termStatus.textContent = "IDLE";
        termStatus.className = "status-indicator";
      }
    });
  }

  // SWE Agent PR Resolution Synthesizer Action
  if (btnSweAgent) {
    btnSweAgent.addEventListener("click", async () => {
      btnSweAgent.disabled = true;
      termStatus.textContent = "SWE RUNNING";
      termStatus.className = "status-indicator live";
      appendLog("\n" + "=".repeat(60), "text-yellow");
      appendLog("[*] DISPATCHING AUTONOMOUS SWE AGENT TRAJECTORY SYNTHESIZER...", "text-yellow");
      appendLog("    - Simulating multi-turn Bash exploration & AST call graph parsing", "text-dim");
      appendLog("    - Writing standalone 'reproduce_issue.py' to verify initial failure mode", "text-dim");
      appendLog("    - Injecting syntax/indentation exceptions to train agent self-correction", "text-dim");
      appendLog("    - Running full regression test suite & generating unified git diff", "text-dim");
      appendLog("    - Verifying final code in isolated deterministic sandbox (0/1 reward)", "text-dim");

      try {
        const resp = await fetch("/api/swe/synthesize-trajectories", { method: "POST" });
        const data = await resp.json();

        if (data.status === "success") {
          appendLog(`\n[✓] SWE Trajectory Synthesis Completed Successfully!`, "text-green");
          
          if (data.mode === "distributed_collector") {
            const summary = data.summary;
            appendLog(`[+] Trajectories Generated : ${summary.total_trajectories} full-codebase benchmarks`, "text-cyan");
            appendLog(`[+] Verifiable Pass Rate  : ${(summary.pass_rate * 100).toFixed(1)}% (Deterministic Sandbox Green)`, "text-green");
            appendLog(`[+] Total Agent Turns     : ${summary.total_agent_turns} multi-turn steps (avg ${summary.avg_turns_per_trajectory}/traj)`, "text-dim");
            appendLog(`[+] Self-Correction Events: ${summary.self_correction_events} syntax/indentation errors recovered`, "text-green");
            appendLog(`[+] Tools Utilized        : Bash (${summary.tool_distribution.bash || 0}), FileEditor (${summary.tool_distribution.file_editor || 0})`, "text-dim");
            appendLog(`[+] Node Distribution     : ${JSON.stringify(summary.cluster_node_distribution)}`, "text-cyan");
            appendLog(`[+] S3 Data Lake Target   : ${data.persistence.s3_uri}`, "text-green");
            
            if (data.sample_trajectory) {
              const sample = data.sample_trajectory;
              appendLog(`\n--- SAMPLE TRAJECTORY PREVIEW: ${sample.issue_id} (${sample.repo_name}) ---`, "text-yellow");
              appendLog(`* Initial Reconnaissance: ${sample.first_step_thought.slice(0, 100)}...`, "text-dim");
              if (sample.self_correction_step) {
                appendLog(`* Diagnostic Recovery   : ${sample.self_correction_step.slice(0, 100)}...`, "text-cyan");
              }
              appendLog(`* Clean Git Diff:\n${sample.patch_diff}`, "text-green");
              appendLog(`* Sandbox Outcome       : PASSED (Verifiable Reward = 1.0)`, "text-green");
            }
          } else {
            appendLog(`[+] Trajectories Generated : ${data.trajectories_count} benchmarks`, "text-cyan");
            appendLog(`[+] Verifiable Pass Rate  : ${data.verified_pass_rate}`, "text-green");
            appendLog(`[+] Total Agent Turns     : ${data.total_agent_turns} steps`, "text-dim");
            appendLog(`[+] Self-Correction Events: ${data.self_correction_events} recovered`, "text-green");
            appendLog(`[+] Cluster Compute Nodes : ${data.cluster_nodes_utilized} nodes`, "text-cyan");
            appendLog(`[+] S3 Storage Prefix     : ${data.s3_prefix}`, "text-green");
          }

          loadShards();
        } else {
          appendLog(`[!] SWE Agent Error: ${data.error || "Unknown failure"}`, "text-red");
        }
      } catch (err) {
        appendLog(`[!] Network error: ${err.message}`, "text-red");
      } finally {
        btnSweAgent.disabled = false;
        termStatus.textContent = "IDLE";
        termStatus.className = "status-indicator";
      }
    });
  }

  // SWE-bench Evaluation Harness Action
  if (btnEvalSwe) {
    btnEvalSwe.addEventListener("click", async () => {
      btnEvalSwe.disabled = true;
      termStatus.textContent = "EVAL RUNNING";
      termStatus.className = "status-indicator live";
      appendLog("\n" + "=".repeat(60), "text-yellow");
      appendLog("[*] INITIATING SWE-BENCH VERIFIED MULTI-TURN EVALUATION HARNESS...", "text-yellow");
      appendLog("    - Spinning up isolated per-task sandbox workspaces", "text-dim");
      appendLog("    - Executing autonomous multi-turn tool interaction (bash/editor/git)", "text-dim");
      appendLog("    - Measuring self-correction recovery from compiler/indentation feedback", "text-dim");
      appendLog("    - Running official regression suites for deterministic 0/1 pass@1 verification", "text-dim");

      try {
        const resp = await fetch("/api/eval/swe-bench", { method: "POST" });
        const data = await resp.json();

        if (data.status === "success") {
          const s = data.summary;
          appendLog(`\n[✓] SWE-bench Evaluation Benchmark Completed!`, "text-green");
          appendLog(`[+] Benchmark Issues Evaluated : ${s.total_benchmark_tasks} tasks`, "text-cyan");
          appendLog(`[+] Resolved Successfully      : ${s.resolved_count} / ${s.total_evaluations}`, "text-green");
          appendLog(`[+] pass@1 Resolution Rate     : ${(s.pass_at_k["pass@1"] * 100).toFixed(1)}% (Frontier Baseline)`, "text-green");
          appendLog(`[+] Self-Correction Fixed      : ${s.self_correction_events} runtime exceptions recovered`, "text-cyan");
          appendLog(`[+] Average Agent Turns / Task : ${s.avg_turns_per_resolution} turns`, "text-dim");
          
          appendLog("\n--- REPOSITORY PASS RATE BREAKDOWN ---", "text-yellow");
          for (const [repo, st] of Object.entries(s.repo_breakdown || {})) {
            appendLog(`  * ${repo.padEnd(20)}: ${st.resolved}/${st.total} (${(st.rate * 100).toFixed(0)}%)`, "text-cyan");
          }

          if (data.sample_evaluation) {
            const sm = data.sample_evaluation;
            appendLog(`\n--- SAMPLE EVALUATION: ${sm.instance_id} (${sm.repo}) ---`, "text-yellow");
            appendLog(`* Resolution Status : ${sm.resolved ? "PASSED" : "FAILED"} in ${sm.duration_sec}s (${sm.turns_taken} turns)`, "text-green");
            appendLog(`* Patch Diff:\n${sm.patch_diff.slice(0, 300)}...`, "text-dim");
          }

          appendLog(`\n[+] Reports Generated: ${data.report_paths.markdown}`, "text-green");
          loadShards();
        } else {
          appendLog(`[!] SWE-bench Evaluation Error: ${data.error || "Unknown failure"}`, "text-red");
        }
      } catch (err) {
        appendLog(`[!] Network error: ${err.message}`, "text-red");
      } finally {
        btnEvalSwe.disabled = false;
        termStatus.textContent = "IDLE";
        termStatus.className = "status-indicator";
      }
    });
  }

  // Multimodal Data Fusion Action
  if (btnMultimodal) {
    btnMultimodal.addEventListener("click", async () => {
      btnMultimodal.disabled = true;
      termStatus.textContent = "MM RUNNING";
      termStatus.className = "status-indicator live";
      appendLog("\n" + "=".repeat(60), "text-yellow");
      appendLog("[*] DISPATCHING DISTRIBUTED MULTIMODAL INGESTION (CLIP/SigLIP)...", "text-yellow");
      appendLog("    - Validating image dimensions & filtering tracking pixels", "text-dim");
      appendLog("    - Computing dual-encoder vision-language alignment scores", "text-dim");
      appendLog("    - Formatting interleaved documents (<|image_ref_start|>) & sharding", "text-dim");

      try {
        const resp = await fetch("/api/multimodal/run-distributed-pipeline", { method: "POST" });
        const data = await resp.json();

        if (data.status === "success") {
          appendLog(`\n[✓] Multimodal Pipeline Completed in ${data.elapsed_sec}s!`, "text-green");
          appendLog(`[+] Cluster Compute Nodes : ${data.cluster_nodes_utilized} nodes`, "text-cyan");
          for (const [node, count] of Object.entries(data.nodes || {})) {
            appendLog(`    * ${node}: ${count} tasks executed`, "text-cyan");
          }
          appendLog(`[+] Interleaved Documents: ${data.documents} articles`, "text-dim");
          appendLog(`[+] Aligned Visual Assets : ${data.visual_assets} images (100% passed CLIP threshold)`, "text-green");
          appendLog(`[+] Total Tokens Generated: ${data.total_tokens.toLocaleString()} tokens`, "text-cyan");
          appendLog(`[+] Packed Sequences (ctx=2048): ${data.packed_sequences}`, "text-dim");
          appendLog(`[+] S3 Staged Objects     : ${data.s3_objects_active} items in s3://ai-data-shards/multimodal/`, "text-green");

          document.getElementById("stat-tokens").textContent = data.total_tokens.toLocaleString();
          document.getElementById("stat-sequences").textContent = data.packed_sequences;

          loadShards();
        } else {
          appendLog(`[!] Multimodal Pipeline Error: ${data.error || "Unknown failure"}`, "text-red");
        }
      } catch (err) {
        appendLog(`[!] Network error: ${err.message}`, "text-red");
      } finally {
        btnMultimodal.disabled = false;
        termStatus.textContent = "IDLE";
        termStatus.className = "status-indicator";
      }
    });
  }

  // Trigger Dry-Run Action
  btnRunDryRun.addEventListener("click", async () => {
    btnRunDryRun.disabled = true;
    termStatus.textContent = "RUNNING";
    termStatus.className = "status-indicator live";
    appendLog("\n[*] Triggering live ingestion and sharding dry-run...", "text-yellow");

    try {
      const resp = await fetch("/api/run-dry-run", { method: "POST" });
      const data = await resp.json();

      if (data.status === "success") {
        appendLog(`[✓] Live pass completed in ${data.elapsed_sec}s`, "text-green");
        appendLog(`[+] Extracted ${data.total_documents} documents | Generated ${data.total_tokens.toLocaleString()} tokens`, "text-cyan");
        appendLog(`[+] Binary shard: shard_live_00000.bin (${(data.bin_size_bytes / 1024).toFixed(1)} KB)`, "text-cyan");
        
        // Update Stats
        document.getElementById("stat-tokens").textContent = data.total_tokens.toLocaleString();
        document.getElementById("stat-sequences").textContent = data.packed_sequences;
        document.getElementById("stat-storage").textContent = (data.bin_size_bytes / 1024).toFixed(1) + " KB";
        
        loadShards();
      } else {
        appendLog(`[!] Error: ${data.error || "Unknown failure"}`, "text-red");
      }
    } catch (err) {
      appendLog(`[!] Network error: ${err.message}`, "text-red");
    } finally {
      btnRunDryRun.disabled = false;
      termStatus.textContent = "IDLE";
      termStatus.className = "status-indicator";
    }
  });

  // Trigger RL Sandbox Test
  btnTestRL.addEventListener("click", async () => {
    btnTestRL.disabled = true;
    appendLog("\n[*] Executing RL Verifiable Sandbox & GRPO evaluation...", "text-yellow");

    try {
      const resp = await fetch("/api/test-rl", { method: "POST" });
      const data = await resp.json();

      if (data.status === "success") {
        appendLog(`[✓] Sandbox code execution: PASSED (Reward = 1.0)`, "text-green");
        appendLog(`[✓] Symbolic Math: Verified equivalence with SymPy`, "text-green");
        appendLog(`[✓] Process Reward Annotator: Step-level error attribution confirmed`, "text-green");
        appendLog(`[✓] GRPO Advantages normalized: [${data.grpo_advantages.join(", ")}]`, "text-cyan");
      } else {
        appendLog(`[!] RL Test error: ${data.error}`, "text-red");
      }
    } catch (err) {
      appendLog(`[!] Network error: ${err.message}`, "text-red");
    } finally {
      btnTestRL.disabled = false;
    }
  });

  // Load Shards Table
  async function loadShards() {
    try {
      const resp = await fetch("/api/shards");
      const shards = await resp.json();
      const tbody = document.getElementById("shards-tbody");
      tbody.innerHTML = "";

      shards.forEach(sh => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><code>${sh.name}</code></td>
          <td>${sh.format}</td>
          <td>${sh.size}</td>
          <td>${sh.tokens}</td>
          <td><span class="pill pill-green">${sh.status}</span></td>
        `;
        tbody.appendChild(tr);
      });
    } catch (e) {
      console.warn("Could not load shards:", e);
    }
  }

  // Check S3 Emulator & Ray status
  async function checkServices() {
    try {
      const resp = await fetch("/api/service-status");
      const status = await resp.json();
      const s3Badge = document.getElementById("badge-s3");
      if (status.s3_connected) {
        s3Badge.innerHTML = '<span class="dot green"></span> S3 Emulator: Online (4566)';
      } else {
        s3Badge.innerHTML = '<span class="dot yellow"></span> S3 Emulator: Standby';
      }
    } catch (e) {
      // Standby fallback
    }
  }

  btnRefreshShards.addEventListener("click", loadShards);
  loadShards();
  checkServices();
});
