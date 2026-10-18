"""
Alignment Package: Direct Preference Optimization (DPO) and KTO Alignment.
"""

from alignment.preference_dataset import (
    PreferencePair,
    KTOExample,
    SWEPreferenceDataExtractor,
    PreferenceBatchCollator
)
from alignment.dpo_engine import DPOEngine
from alignment.dpo_trainer import DPOTrainer

__all__ = [
    "PreferencePair",
    "KTOExample",
    "SWEPreferenceDataExtractor",
    "PreferenceBatchCollator",
    "DPOEngine",
    "DPOTrainer"
]
