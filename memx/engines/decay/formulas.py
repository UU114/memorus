"""Pure-function decay formulas for temporal weight computation.

All functions are stateless and side-effect free, making them easy to
test and compose independently of the DecayEngine orchestrator.
"""

from __future__ import annotations


def exponential_decay(age_days: float, half_life: float) -> float:
    """Compute base decay factor using exponential half-life model.

    Formula: 2^(-age_days / half_life)

    Returns 1.0 when age_days <= 0 (freshly created or clock skew).
    The result is always in the range (0.0, 1.0].
    """
    if age_days <= 0.0:
        return 1.0
    if half_life <= 0.0:
        # Guard against invalid config; treat as instant decay
        return 0.0
    return float(2.0 ** (-age_days / half_life))


def boosted_weight(base: float, boost_factor: float, recall_count: int) -> float:
    """Apply recall-frequency boost to a base decay value.

    Formula: base * (1 + boost_factor * recall_count)

    The result is clamped to [0.0, 1.0].
    """
    if recall_count < 0:
        recall_count = 0
    raw = base * (1.0 + boost_factor * recall_count)
    return max(0.0, min(1.0, raw))
