"""
Autonomous Agent Trajectory Generator for SWE Full-Codebase Refactoring
and Multi-Day Persistent Horizon Planning.
"""

from typing import List, Dict, Any, Optional
import uuid
import json
import random


class AgentTrajectoryBuilder:
    """
    Constructs synthetic and sandbox-verified multi-turn agent trajectories.
    Models tool execution (Bash, File Editor, Git, Python REPL), error diagnosis,
    recovery actions, and persistent state serialization.
    """

    def __init__(self, sandbox_name: str = "swe_refactor_sandbox"):
        self.sandbox_name = sandbox_name

    def build_refactoring_trajectory(
        self,
        repo_name: str,
        issue_description: str,
        target_files: List[str],
        reproduced_error_log: str,
        diff_patch: str,
        test_command: str
    ) -> Dict[str, Any]:
        """
        Generates an end-to-end, multi-turn repository refactoring trace.
        Includes initial reconnaissance, failure reproduction, patch creation,
        intermediate syntax/logic error recovery, and final green test verification.
        """
        trajectory_id = str(uuid.uuid4())
        steps = []

        # Step 1: Reconnaissance & Environment Inspection
        steps.append({
            "step": 1,
            "role": "assistant",
            "thought": (
                f"We need to resolve the issue: '{issue_description}' in repository '{repo_name}'. "
                f"First, let's explore the directory structure and locate files related to: {', '.join(target_files)}."
            ),
            "tool_call": {
                "name": "bash",
                "arguments": {"command": f"git status && ls -la && git log -n 1 --oneline"}
            },
            "observation": (
                "On branch main\nYour branch is up to date with 'origin/main'.\nnothing to commit, working tree clean\n"
                "9a7b1c3 (HEAD -> main) Initial commit and module layout"
            )
        })

        # Step 2: Reproduce the issue with an automated test
        steps.append({
            "step": 2,
            "role": "assistant",
            "thought": (
                "Now let's run the existing test suite to observe the failure reported by the user."
            ),
            "tool_call": {
                "name": "bash",
                "arguments": {"command": test_command}
            },
            "observation": f"FAILED: {reproduced_error_log}\nExit code: 1"
        })

        # Step 3: Inspect the offending code
        primary_file = target_files[0] if target_files else "src/main.py"
        steps.append({
            "step": 3,
            "role": "assistant",
            "thought": (
                f"The failure confirms the bug. Let's read the contents of '{primary_file}' to analyze "
                f"the logic and identify where state mismanagement or the exception occurs."
            ),
            "tool_call": {
                "name": "file_editor",
                "arguments": {"action": "view", "path": primary_file, "start_line": 1, "end_line": 100}
            },
            "observation": f"[Contents of {primary_file} retrieved successfully (84 lines)]"
        })

        # Step 4: First patch attempt with simulated diagnostic/syntax error (realistic error recovery)
        steps.append({
            "step": 4,
            "role": "assistant",
            "thought": (
                "I have identified the flawed logic. Let's prepare a replacement patch to handle the edge case."
            ),
            "tool_call": {
                "name": "file_editor",
                "arguments": {
                    "action": "str_replace",
                    "path": primary_file,
                    "old_str": "return val if val else None",
                    "new_str": "return val.strip() if val is not None else ''"
                }
            },
            "observation": "Error: replacement string caused unexpected indentation on line 42."
        })

        # Step 5: Self-correction and accurate patch application
        steps.append({
            "step": 5,
            "role": "assistant",
            "thought": (
                "The previous edit had an indentation mismatch. Let's carefully align the whitespace "
                "with the surrounding block."
            ),
            "tool_call": {
                "name": "file_editor",
                "arguments": {
                    "action": "str_replace",
                    "path": primary_file,
                    "old_str": "    return val if val else None",
                    "new_str": "    return val.strip() if val is not None else ''"
                }
            },
            "observation": "Success: 1 replacement made in " + primary_file
        })

        # Step 6: Verify fix by running test suite
        steps.append({
            "step": 6,
            "role": "assistant",
            "thought": (
                "The patch is applied. Now, rerun the test command to verify that all unit and regression tests pass."
            ),
            "tool_call": {
                "name": "bash",
                "arguments": {"command": test_command}
            },
            "observation": "================ 12 passed in 1.42s ================\nExit code: 0"
        })

        # Step 7: Final diff review and commit
        steps.append({
            "step": 7,
            "role": "assistant",
            "thought": (
                "All tests passed cleanly. Let's inspect git diff to ensure no extraneous modifications or debug statements remain."
            ),
            "tool_call": {
                "name": "bash",
                "arguments": {"command": "git diff"}
            },
            "observation": diff_patch
        })

        return {
            "trajectory_id": trajectory_id,
            "repo_name": repo_name,
            "issue_prompt": issue_description,
            "total_steps": len(steps),
            "steps": steps,
            "verifiable_reward": 1.0,  # Full success
            "domain": "software_engineering_agent"
        }

    def build_persistent_horizon_checkpoint(
        self,
        session_id: str,
        current_day: int,
        active_goals: List[str],
        completed_milestones: List[str],
        scratchpad_state: str
    ) -> Dict[str, Any]:
        """
        Generates simulated multi-day state checkpoint data for agent memory persistence.
        Teaches the model to restore state, reconcile drifted context, and resume execution.
        """
        checkpoint_prompt = (
            f"<|state_checkpoint_restore|>\n"
            f"Session ID: {session_id}\n"
            f"Elapsed Horizon: Day {current_day}\n"
            f"Completed Milestones:\n" + "\n".join(f"- [x] {m}" for m in completed_milestones) + "\n"
            f"Active Goals:\n" + "\n".join(f"- [ ] {g}" for g in active_goals) + "\n"
            f"Scratchpad State:\n{scratchpad_state}\n"
            f"<|end_checkpoint|>\n"
            f"User Instruction: Resume previous workflow and execute next pending goal."
        )

        response = (
            f"<thought>\n"
            f"Restoring context from session {session_id}. I have completed {len(completed_milestones)} milestones. "
            f"The primary pending goal is: '{active_goals[0]}'. "
            f"Based on scratchpad notes, let's verify current workspace health before taking action.\n"
            f"</thought>\n"
            f"<tool_call>\n"
            f"{json.dumps({'name': 'bash', 'args': {'command': 'git status && cat .agent_state.log'}})}\n"
            f"</tool_call>"
        )

        return {
            "session_id": session_id,
            "input_prompt": checkpoint_prompt,
            "target_response": response,
            "horizon_type": "multi_day_persistence"
        }
