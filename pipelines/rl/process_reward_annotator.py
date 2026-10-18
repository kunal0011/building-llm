"""
Process Reward Model (PRM) Step-Level Annotator.
Decomposes long chain-of-thought (CoT) trajectories into discrete reasoning steps,
evaluates intermediate validity, and assigns dense step-level rewards.
"""

import re
from typing import List, Dict, Any, Tuple, Optional


class ProcessRewardAnnotator:
    """
    Step-level Process Reward Annotator.
    Segments model chain-of-thought into reasoning steps and annotates each step
    with discrete reward labels (+1 = Valid, 0 = Neutral/Elaboration, -1 = Error/Hallucination).
    """

    STEP_DELIMITERS = [
        re.compile(r"<step>(.*?)</step>", re.DOTALL),
        re.compile(r"(?:^|\n)(?:Step\s+\d+:?|\d+\.\s+)", re.IGNORECASE),
        re.compile(r"\n\n+")
    ]

    def __init__(self, step_reward_valid: float = 1.0, step_reward_error: float = -1.0):
        self.step_reward_valid = step_reward_valid
        self.step_reward_error = step_reward_error

    def segment_reasoning_steps(self, chain_of_thought_text: str) -> List[str]:
        """Splits chain of thought into discrete logical reasoning steps."""
        # 1. Check for explicit <step> tags
        explicit_steps = re.findall(r"<step>(.*?)</step>", chain_of_thought_text, re.DOTALL)
        if explicit_steps:
            return [s.strip() for s in explicit_steps if s.strip()]

        # 2. Check for numbered steps (1. ..., 2. ...)
        lines = chain_of_thought_text.splitlines()
        steps = []
        current_step: List[str] = []

        for line in lines:
            if re.match(r"^(?:Step\s+\d+:?|\d+\.)\s+", line.strip(), re.IGNORECASE):
                if current_step:
                    steps.append("\n".join(current_step).strip())
                    current_step = []
            current_step.append(line)

        if current_step:
            steps.append("\n".join(current_step).strip())

        if len(steps) > 1:
            return steps

        # 3. Fallback to paragraph boundaries
        paragraphs = [p.strip() for p in re.split(r"\n\n+", chain_of_thought_text) if p.strip()]
        return paragraphs if paragraphs else [chain_of_thought_text.strip()]

    def annotate_trajectory(
        self,
        chain_of_thought_text: str,
        first_error_index: Optional[int] = None,
        terminal_outcome_success: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Annotates each reasoning step with a process reward.
        If first_error_index is provided, all steps up to that index receive +1.0,
        the error step receives -1.0, and subsequent steps are marked invalidated.
        """
        steps = self.segment_reasoning_steps(chain_of_thought_text)
        annotated = []

        for idx, step_text in enumerate(steps):
            if first_error_index is not None:
                if idx < first_error_index:
                    reward = self.step_reward_valid
                    label = "correct"
                elif idx == first_error_index:
                    reward = self.step_reward_error
                    label = "first_error"
                else:
                    reward = self.step_reward_error
                    label = "cascaded_error"
            else:
                # Terminal outcome based heuristic fallback
                reward = self.step_reward_valid if terminal_outcome_success else self.step_reward_error
                label = "correct" if terminal_outcome_success else "flawed"

            annotated.append({
                "step_index": idx,
                "step_text": step_text,
                "process_reward": reward,
                "label": label
            })

        return annotated

    def format_prm_training_sample(
        self,
        prompt: str,
        annotated_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Formats annotated step data into training samples for Process Reward Models."""
        return {
            "prompt": prompt,
            "steps": [
                {
                    "content": s["step_text"],
                    "reward": s["process_reward"],
                    "label": s["label"]
                }
                for s in annotated_steps
            ],
            "total_steps": len(annotated_steps),
            "contains_error": any(s["process_reward"] < 0 for s in annotated_steps)
        }
