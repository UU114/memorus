"""Decay engine — temporal weight management for memory lifecycle."""

from memx.core.engines.decay.engine import (
    BulletDecayInfo,
    DecayEngine,
    DecayResult,
    DecaySweepResult,
)
from memx.core.engines.decay.formulas import boosted_weight, exponential_decay

__all__ = [
    "BulletDecayInfo",
    "DecayEngine",
    "DecayResult",
    "DecaySweepResult",
    "boosted_weight",
    "exponential_decay",
]
