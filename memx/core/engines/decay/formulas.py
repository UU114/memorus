"""Pure-function decay formulas for temporal weight computation.

All functions are stateless and side-effect free, making them easy to
test and compose independently of the DecayEngine orchestrator.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def exponential_decay(age_days: float, half_life: float) -> float:
    """Compute base decay factor using exponential half-life model.

    Formula: 2^(-age_days / half_life)

    Returns 1.0 when age_days <= 0 (freshly created or clock skew).
    The result is always in the range (0.0, 1.0].
    """
    if age_days <= 0.0:
        logger.debug("exponential_decay: age<=0 -> 1.0")
        return 1.0
    if half_life <= 0.0:
        # Guard against invalid config; treat as instant decay
        logger.debug("exponential_decay: half_life<=0 -> 0.0 (instant decay)")
        return 0.0
    result = float(2.0 ** (-age_days / half_life))
    logger.debug("exponential_decay: age=%.2f half_life=%.1f -> %.6f", age_days, half_life, result)
    return result


def boosted_weight(base: float, boost_factor: float, recall_count: int) -> float:
    """Apply recall-frequency boost to a base decay value.

    Formula: base * (1 + boost_factor * recall_count)

    The result is clamped to [0.0, 1.0].
    """
    if recall_count < 0:
        recall_count = 0
    raw = base * (1.0 + boost_factor * recall_count)
    clamped = max(0.0, min(1.0, raw))
    logger.debug(
        "boosted_weight: base=%.6f boost=%.2f recalls=%d raw=%.6f clamped=%.6f",
        base, boost_factor, recall_count, raw, clamped,
    )
    return clamped
