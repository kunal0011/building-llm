"""
Preference Dataset & Batch Collator for DPO and KTO Alignment.
Extracts paired preference trajectories (chosen vs rejected) and unpaired binary feedback
from SWE-bench agent runs and reasoning rollouts.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import torch


@dataclass
class PreferencePair:
    """Paired comparison data structure for DPO."""
    prompt: str
    chosen: str
    rejected: str
    margin: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KTOExample:
    """Unpaired binary feedback data structure for KTO."""
    prompt: str
    response: str
    is_chosen: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class SWEPreferenceDataExtractor:
    """
    Extracts high-quality preference pairs from SWE-bench agent trajectories.
    Trajectores that successfully resolve reproduction tests (sandbox_passed=True)
    are treated as chosen, while failed or exception-halted rollouts are rejected.
    """

    @classmethod
    def extract_from_trajectories(
        cls,
        trajectories: List[Dict[str, Any]]
    ) -> Tuple[List[PreferencePair], List[KTOExample]]:
        """
        Groups trajectories by issue_id and pairs passing vs. failing completions.
        """
        by_issue: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        for traj in trajectories:
            issue_id = traj.get("issue_id", "default_task")
            passed = bool(traj.get("sandbox_passed", False))

            if issue_id not in by_issue:
                by_issue[issue_id] = {"passed": [], "failed": []}

            if passed:
                by_issue[issue_id]["passed"].append(traj)
            else:
                by_issue[issue_id]["failed"].append(traj)

        preference_pairs: List[PreferencePair] = []
        kto_examples: List[KTOExample] = []

        for issue_id, groups in by_issue.items():
            passed_list = groups["passed"]
            failed_list = groups["failed"]

            # Unpaired KTO examples
            for p in passed_list:
                prompt_text = p.get("task_instructions", p.get("problem_statement", f"Issue: {issue_id}"))
                resp_text = p.get("final_patch", p.get("solution_content", "Resolved patch"))
                kto_examples.append(KTOExample(prompt=prompt_text, response=resp_text, is_chosen=True, metadata={"issue_id": issue_id}))

            for f_item in failed_list:
                prompt_text = f_item.get("task_instructions", f_item.get("problem_statement", f"Issue: {issue_id}"))
                resp_text = f_item.get("final_patch", f_item.get("solution_content", "Failed patch"))
                kto_examples.append(KTOExample(prompt=prompt_text, response=resp_text, is_chosen=False, metadata={"issue_id": issue_id}))

            # Paired DPO comparisons
            for p in passed_list:
                prompt_text = p.get("task_instructions", p.get("problem_statement", f"Issue: {issue_id}"))
                chosen_text = p.get("final_patch", p.get("solution_content", "Resolved patch"))

                if failed_list:
                    for f_item in failed_list:
                        rejected_text = f_item.get("final_patch", f_item.get("solution_content", "Syntax error in patch"))
                        preference_pairs.append(
                            PreferencePair(
                                prompt=prompt_text,
                                chosen=chosen_text,
                                rejected=rejected_text,
                                margin=1.0,
                                metadata={"issue_id": issue_id}
                            )
                        )
                else:
                    # Synthetic negative if only positive exists (e.g. unpatched or empty)
                    preference_pairs.append(
                        PreferencePair(
                            prompt=prompt_text,
                            chosen=chosen_text,
                            rejected="def fix():\n    pass # Unimplemented fallback",
                            margin=0.8,
                            metadata={"issue_id": issue_id}
                        )
                    )

        return preference_pairs, kto_examples


class PreferenceBatchCollator:
    """
    Tokenizes and collates preference pairs into model tensors.
    Constructs prompt + completion tokens and sets labels to -100 for prompt tokens
    so loss is strictly evaluated on completion tokens.
    """

    def __init__(self, tokenizer: Any, max_seq_len: int = 512, pad_token_id: int = 0):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def _encode_sequence(self, prompt: str, completion: str) -> Tuple[List[int], List[int]]:
        prompt_tokens = self.tokenizer.encode(prompt) if hasattr(self.tokenizer, "encode") else [ord(c) % 500 for c in prompt]
        completion_tokens = self.tokenizer.encode(completion) if hasattr(self.tokenizer, "encode") else [ord(c) % 500 for c in completion]

        # Truncate if exceeds max_seq_len
        avail_len = max(self.max_seq_len - len(prompt_tokens), 8)
        completion_tokens = completion_tokens[:avail_len]

        input_ids = prompt_tokens + completion_tokens
        # Labels: -100 for prompt (ignored in loss), actual token IDs for completion
        labels = ([-100] * len(prompt_tokens)) + list(completion_tokens)
        return input_ids, labels

    def collate_dpo_batch(self, pairs: List[PreferencePair]) -> Dict[str, torch.Tensor]:
        """
        Collates chosen and rejected pairs into padded PyTorch tensors.
        """
        chosen_input_ids = []
        chosen_labels = []
        rejected_input_ids = []
        rejected_labels = []

        for pair in pairs:
            c_ids, c_lbls = self._encode_sequence(pair.prompt, pair.chosen)
            r_ids, r_lbls = self._encode_sequence(pair.prompt, pair.rejected)

            chosen_input_ids.append(c_ids)
            chosen_labels.append(c_lbls)
            rejected_input_ids.append(r_ids)
            rejected_labels.append(r_lbls)

        # Pad to max length in batch
        max_c_len = min(max(len(ids) for ids in chosen_input_ids), self.max_seq_len)
        max_r_len = min(max(len(ids) for ids in rejected_input_ids), self.max_seq_len)
        max_len = max(max_c_len, max_r_len)

        def pad_tensor(seqs: List[List[int]], pad_val: int) -> torch.Tensor:
            batch_size = len(seqs)
            out = torch.full((batch_size, max_len), pad_val, dtype=torch.long)
            for i, seq in enumerate(seqs):
                trunc = seq[:max_len]
                out[i, :len(trunc)] = torch.tensor(trunc, dtype=torch.long)
            return out

        return {
            "chosen_input_ids": pad_tensor(chosen_input_ids, self.pad_token_id),
            "chosen_labels": pad_tensor(chosen_labels, -100),
            "rejected_input_ids": pad_tensor(rejected_input_ids, self.pad_token_id),
            "rejected_labels": pad_tensor(rejected_labels, -100)
        }
