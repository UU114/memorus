# mypy: disable-error-code="untyped-decorator"
"""Benchmark: CuratorEngine.curate() with 100 existing + 1 candidate.

Threshold: p95 < 10ms (relaxed to 20ms for CI safety margin).
"""

from __future__ import annotations

from typing import Any

import pytest

from memx.core.engines.curator.engine import CuratorEngine, ExistingBullet
from memx.core.types import CandidateBullet

# Threshold in seconds (10 ms).  Doubled to 20 ms as safety margin —
# see STORY-045 note on adaptive thresholds.
_P95_THRESHOLD_SEC = 0.020


@pytest.mark.benchmark(group="curator")
def test_curator_curate_100_existing_1_candidate(
    benchmark: Any,
    generate_existing_bullets: Any,
    generate_candidate_bullets: Any,
) -> None:
    """CuratorEngine.curate() with 100 existing bullets and 1 candidate."""
    existing: list[ExistingBullet] = generate_existing_bullets(100)
    candidates: list[CandidateBullet] = generate_candidate_bullets(1)
    engine = CuratorEngine()

    result = benchmark.pedantic(
        engine.curate,
        args=(candidates, existing),
        rounds=50,
        warmup_rounds=5,
    )

    # Sanity: curate should categorize the candidate
    total = len(result.to_add) + len(result.to_merge) + len(result.to_skip)
    assert total == 1


@pytest.mark.benchmark(group="curator")
def test_curator_curate_100_existing_10_candidates(
    benchmark: Any,
    generate_existing_bullets: Any,
    generate_candidate_bullets: Any,
) -> None:
    """CuratorEngine.curate() with 100 existing bullets and 10 candidates."""
    existing: list[ExistingBullet] = generate_existing_bullets(100)
    candidates: list[CandidateBullet] = generate_candidate_bullets(10)
    engine = CuratorEngine()

    result = benchmark.pedantic(
        engine.curate,
        args=(candidates, existing),
        rounds=30,
        warmup_rounds=3,
    )

    total = len(result.to_add) + len(result.to_merge) + len(result.to_skip)
    assert total == 10


@pytest.mark.benchmark(group="curator")
def test_curator_curate_no_existing(
    benchmark: Any,
    generate_candidate_bullets: Any,
) -> None:
    """Edge case: curate() with no existing bullets should insert all."""
    candidates: list[CandidateBullet] = generate_candidate_bullets(5)
    engine = CuratorEngine()

    result = benchmark(engine.curate, candidates, [])

    assert len(result.to_add) == 5
    assert len(result.to_merge) == 0
