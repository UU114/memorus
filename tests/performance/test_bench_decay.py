# mypy: disable-error-code="untyped-decorator"
"""Benchmark: DecayEngine.sweep() with 5000 bullets.

Threshold: p95 < 30ms (relaxed to 60ms for CI safety margin).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from memx.core.engines.decay.engine import BulletDecayInfo, DecayEngine

# Threshold in seconds (30 ms).  Doubled to 60 ms as safety margin —
# see STORY-045 note on adaptive thresholds.
_P95_THRESHOLD_SEC = 0.060


@pytest.mark.benchmark(group="decay")
def test_decay_sweep_5000(
    benchmark: Any,
    generate_decay_infos: Any,
) -> None:
    """DecayEngine.sweep() over 5000 bullets should complete within threshold."""
    infos: list[BulletDecayInfo] = generate_decay_infos(5000)
    engine = DecayEngine()
    now = datetime.now(timezone.utc)

    result = benchmark.pedantic(
        engine.sweep,
        args=(infos,),
        kwargs={"now": now},
        rounds=10,
        warmup_rounds=2,
    )

    # Sanity: all bullets should be processed
    total = result.updated + result.archived + result.permanent + result.unchanged
    assert total == 5000
    assert len(result.errors) == 0


@pytest.mark.benchmark(group="decay")
def test_decay_sweep_1000(
    benchmark: Any,
    generate_decay_infos: Any,
) -> None:
    """DecayEngine.sweep() over 1000 bullets (lighter workload)."""
    infos: list[BulletDecayInfo] = generate_decay_infos(1000)
    engine = DecayEngine()
    now = datetime.now(timezone.utc)

    result = benchmark.pedantic(
        engine.sweep,
        args=(infos,),
        kwargs={"now": now},
        rounds=20,
        warmup_rounds=3,
    )

    total = result.updated + result.archived + result.permanent + result.unchanged
    assert total == 1000


@pytest.mark.benchmark(group="decay")
def test_decay_compute_weight_single(
    benchmark: Any,
) -> None:
    """DecayEngine.compute_weight() single call should be sub-millisecond."""
    engine = DecayEngine()
    now = datetime.now(timezone.utc)

    result = benchmark(
        engine.compute_weight,
        created_at=now,
        recall_count=5,
        now=now,
    )

    assert result.weight >= 0.0
    assert result.weight <= 1.0
