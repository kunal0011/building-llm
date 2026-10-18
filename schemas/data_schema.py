"""
Production Data Schemas for Pre-training, Long-Context Packing,
Agent Trajectories, and Multimodal-Ready Ingestion.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import json
import uuid
import time


@dataclass
class MediaReference:
    """Represents an embedded multimodal asset (image, table, audio clip, video frame)."""
    placeholder_token: str
    s3_uri: str
    alt_text: str
    mime_type: str
    bounding_box: Optional[List[int]] = None  # [ymin, xmin, ymax, xmax]
    timestamp_ms: Optional[Tuple[int, int]] = None  # (start_ms, end_ms) for audio/video


@dataclass
class RawDocument:
    """Raw crawled or ingested document before quality filtering."""
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_url: str = ""
    domain: str = ""
    raw_text: str = ""
    content_hash: str = ""
    crawled_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    media_refs: List[MediaReference] = field(default_factory=list)


@dataclass
class FilteredDocument:
    """Document after heuristic, language, quality, and PII filtering."""
    doc_id: str
    clean_text: str
    language: str
    quality_score: float
    domain_category: str  # e.g., 'code', 'stem_paper', 'book', 'web_encyclopedia'
    token_count_estimate: int
    media_refs: List[MediaReference] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl(self) -> str:
        return json.dumps({
            "doc_id": self.doc_id,
            "text": self.clean_text,
            "language": self.language,
            "quality_score": self.quality_score,
            "domain_category": self.domain_category,
            "token_count": self.token_count_estimate,
            "media_refs": [ref.__dict__ for ref in self.media_refs],
            "metadata": self.metadata
        }, ensure_ascii=False)


@dataclass
class RepositoryFile:
    """Single file in a repository with AST and dependency metadata."""
    file_path: str
    content: str
    language: str
    ast_imports: List[str] = field(default_factory=list)
    ast_exports: List[str] = field(default_factory=list)
    topological_depth: int = 0


@dataclass
class RepositoryCodebaseDocument:
    """Entire repository structured topologically for full-codebase refactoring training."""
    repo_name: str
    commit_sha: str
    license: str
    file_hierarchy: List[RepositoryFile] = field(default_factory=list)
    readme_content: str = ""
    architecture_overview: str = ""
    total_tokens: int = 0

    def serialize_packed_repo(self) -> str:
        """Serializes repository files in dependency order with clear delimiter markers."""
        chunks = [
            f"<|repo_start|>{self.repo_name}",
            f"<|repo_readme|>\n{self.readme_content}\n<|repo_readme_end|>"
        ]
        if self.architecture_overview:
            chunks.append(f"<|repo_overview|>\n{self.architecture_overview}\n<|repo_overview_end|>")
        
        # Sort files by topological dependency depth
        sorted_files = sorted(self.file_hierarchy, key=lambda f: f.topological_depth)
        for f in sorted_files:
            chunks.append(
                f"<|file_start|>{f.file_path}\n"
                f"<|file_language|>{f.language}\n"
                f"{f.content}\n"
                f"<|file_end|>"
            )
        chunks.append("<|repo_end|>")
        return "\n".join(chunks)


@dataclass
class AgentToolCall:
    """Tool invocation within an agent execution step."""
    call_id: str
    tool_name: str
    arguments: Dict[str, Any]
    result: str
    execution_time_ms: int
    exit_code: int = 0  # 0 = success, non-zero = failure/error


@dataclass
class AgentStateSnapshot:
    """Memory and execution state snapshot across multi-day / multi-turn operations."""
    active_goals: List[str]
    completed_milestones: List[str]
    environment_variables: Dict[str, str]
    open_file_buffers: Dict[str, str]
    scratchpad_notes: str


@dataclass
class AgentTrajectoryStep:
    """A single step in a long-horizon agent execution trajectory."""
    step_index: int
    thought_reasoning: str  # Chain of thought / planning
    tool_call: Optional[AgentToolCall] = None
    observation: str = ""  # Result from tool or environment
    state_snapshot: Optional[AgentStateSnapshot] = None
    is_recovery_step: bool = False  # True if this step was initiated to recover from an error


@dataclass
class LongHorizonAgentTrajectory:
    """Full end-to-end multi-step persistent agent trajectory for SFT and RL."""
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_prompt: str = ""
    initial_environment_state: Dict[str, Any] = field(default_factory=dict)
    steps: List[AgentTrajectoryStep] = field(default_factory=list)
    task_success: bool = False
    reward_score: float = 0.0  # 1.0 for passing full test suite, 0.0 otherwise
    total_turns: int = 0
    total_tokens: int = 0

    def to_training_sample(self) -> Dict[str, Any]:
        """Converts trajectory into format consumable by SFT / RL trainer."""
        messages = [
            {"role": "system", "content": "You are an autonomous long-horizon agent capable of planning, executing tools, inspecting environments, handling failures, and verifying results."},
            {"role": "user", "content": self.task_prompt}
        ]
        for step in self.steps:
            assistant_content = f"<thought>\n{step.thought_reasoning}\n</thought>"
            if step.tool_call:
                assistant_content += f"\n<tool_call>\n{json.dumps({'name': step.tool_call.tool_name, 'args': step.tool_call.arguments})}\n</tool_call>"
            messages.append({"role": "assistant", "content": assistant_content})
            
            if step.tool_call:
                messages.append({
                    "role": "tool",
                    "content": f"<tool_output exit_code=\"{step.tool_call.exit_code}\">\n{step.observation}\n</tool_output>"
                })
        return {
            "trajectory_id": self.trajectory_id,
            "messages": messages,
            "success": self.task_success,
            "reward": self.reward_score,
            "total_turns": len(self.steps)
        }
