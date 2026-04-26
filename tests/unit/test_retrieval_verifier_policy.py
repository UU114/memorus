"""STORY-R103 — Stale policy dispatch tests (flag / demote / drop).

Shared 6-bullet fixture: 2 verified / 2 stale / 2 not_applicable.
Each test sets ``VerificationConfig.policy`` and asserts:

* result count per policy
* final_score ordering under demote
* dropped_stale_count under drop
* unverifiable edge cases (demote scales, drop keeps)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from memorus.core.config import VerificationConfig
from memorus.core.engines.generator.engine import GeneratorEngine
from memorus.core.engines.generator.score_merger import ScoredBullet
from memorus.core.engines.verifier.engine import VerificationOutcome
from memorus.core.pipeline.retrieval import RetrievalPipeline
from memorus.core.types import BulletMetadata, VerifiedStatus
from memorus.core.utils.token_counter import TokenBudgetTrimmer


# ---------------------------------------------------------------------------
# Stub VerificationEngine that replays a pre-seeded outcome in submission
# order. The pipeline always submits bullets in the same order as the
# trimmed list, and the loader preserves that order, so positional replay
# is stable for these tests.
# ---------------------------------------------------------------------------


class _PositionalEngine:
    """Drop-in for ``VerificationEngine`` that returns outcomes in order.

    The pipeline calls ``verify_many`` with a list of ``BulletMetadata`` in
    the same positional order as ``trimmed``. Caller seeds ``outcomes`` in
    the same order so test authoring stays declarative.
    """

    def __init__(self, outcomes: list[tuple[VerifiedStatus, float | None]]):
        self._outcomes = outcomes
        self._cursor = 0

    def verify_many(
        self, bullets: list[BulletMetadata]
    ) -> list[VerificationOutcome]:
        from datetime import datetime, timezone

        out: list[VerificationOutcome] = []
        for _ in bullets:
            status, score = self._outcomes[self._cursor]
            self._cursor += 1
            out.append(
                VerificationOutcome(
                    verified_status=status,
                    trust_score=score,
                    verified_at=datetime.now(timezone.utc),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Shared 6-bullet fixture
# ---------------------------------------------------------------------------


def _scored(bid: str, score: float) -> ScoredBullet:
    return ScoredBullet(
        bullet_id=bid,
        content=f"bullet {bid}",
        final_score=score,
        keyword_score=0.0,
        semantic_score=score,
        decay_weight=1.0,
        recency_boost=1.0,
    )


# IDs are ordered V1 > V2 > S1 > S2 > N1 > N2 initial scores so we can tell
# whether demote re-sorts. All scores are distinct.
_IDS = [
    ("V1", VerifiedStatus.VERIFIED, 1.0, 0.90),
    ("V2", VerifiedStatus.VERIFIED, 1.0, 0.85),
    ("S1", VerifiedStatus.STALE, 0.3, 0.80),
    ("S2", VerifiedStatus.STALE, 0.3, 0.75),
    ("N1", VerifiedStatus.NOT_APPLICABLE, None, 0.70),
    ("N2", VerifiedStatus.NOT_APPLICABLE, None, 0.65),
]


@pytest.fixture
def six_bullet_fixture() -> tuple[list[ScoredBullet], dict[str, BulletMetadata], _PositionalEngine]:
    scored = [_scored(bid, score) for bid, _, _, score in _IDS]
    # BulletMetadata carries no bullet_id field — the loader maps id->model.
    # Non-empty metadata is enough for the pipeline to treat these as
    # "valid rows" and dispatch them to verify_many in positional order.
    bullets = {
        bid: BulletMetadata(anchors=[]) for bid, _, _, _ in _IDS
    }
    engine = _PositionalEngine(
        outcomes=[(status, score) for _, status, score, _ in _IDS]
    )
    return scored, bullets, engine


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------


def _build_pipeline(
    scored: list[ScoredBullet],
    loaded_bullets: dict[str, BulletMetadata],
    verifier: Any,
    policy: str,
    stale_trust_score: float = 0.3,
    unverifiable_trust_score: float = 0.7,
) -> RetrievalPipeline:
    generator = MagicMock(spec=GeneratorEngine)
    # Return a fresh copy each call so the pipeline can mutate in place.
    generator.search.return_value = list(scored)
    generator.mode = "full"

    trimmer = MagicMock(spec=TokenBudgetTrimmer)
    trimmer.trim.side_effect = lambda x: x
    trimmer.token_budget = 1000

    def loader(ids: list[str]) -> list[BulletMetadata | None]:
        return [loaded_bullets.get(i) for i in ids]

    cfg = VerificationConfig(
        enabled=True,
        ttl_seconds=0,
        stale_trust_score=stale_trust_score,
        unverifiable_trust_score=unverifiable_trust_score,
        policy=policy,  # type: ignore[arg-type]
    )

    return RetrievalPipeline(
        generator=generator,
        trimmer=trimmer,
        decay_engine=None,
        verification_engine=verifier,
        bullet_loader=loader,
        verification_config=cfg,
    )


# ---------------------------------------------------------------------------
# Core three-branch tests
# ---------------------------------------------------------------------------


def test_policy_flag_keeps_all_and_preserves_scores(
    six_bullet_fixture: tuple[list[ScoredBullet], dict[str, BulletMetadata], _PositionalEngine],
) -> None:
    scored, bullets, engine = six_bullet_fixture
    # Snapshot expected final scores BEFORE the pipeline mutates in place.
    expected_scores = {sb.bullet_id: sb.final_score for sb in scored}

    pipeline = _build_pipeline(scored, bullets, engine, policy="flag")

    result = pipeline.search(query="q", bullets=[], limit=10)

    # AC: 6 rows, scores unchanged vs R102
    assert len(result.results) == 6
    assert result.dropped_stale_count == 0
    for r in result.results:
        assert r.final_score == pytest.approx(expected_scores[r.bullet_id])
    # Statuses are still attached (flag keeps R102 wiring).
    by_id = {r.bullet_id: r for r in result.results}
    assert by_id["S1"].verified_status == "stale"
    assert by_id["V1"].verified_status == "verified"
    assert by_id["N1"].verified_status == "not_applicable"


def test_policy_demote_scales_stale_and_resorts(
    six_bullet_fixture: tuple[
        list[ScoredBullet], dict[str, BulletMetadata], _SeededEngine
    ],
) -> None:
    scored, bullets, engine = six_bullet_fixture
    pipeline = _build_pipeline(
        scored, bullets, engine, policy="demote", stale_trust_score=0.3
    )

    result = pipeline.search(query="q", bullets=[], limit=10)

    # AC: 6 rows, stale × 0.3, not_applicable untouched
    assert len(result.results) == 6
    assert result.dropped_stale_count == 0

    by_id = {r.bullet_id: r for r in result.results}
    # Stale multiplied by 0.3
    assert by_id["S1"].final_score == pytest.approx(0.80 * 0.3)
    assert by_id["S2"].final_score == pytest.approx(0.75 * 0.3)
    # Verified untouched
    assert by_id["V1"].final_score == pytest.approx(0.90)
    assert by_id["V2"].final_score == pytest.approx(0.85)
    # NOT_APPLICABLE untouched — demote never scales N_A rows
    assert by_id["N1"].final_score == pytest.approx(0.70)
    assert by_id["N2"].final_score == pytest.approx(0.65)

    # Demote re-sorts so S1/S2 (now 0.24/0.225) sink below N1/N2 (0.70/0.65).
    ordered_ids = [r.bullet_id for r in result.results]
    # Top-4 should be V1, V2, N1, N2 (stale sank)
    assert ordered_ids[:4] == ["V1", "V2", "N1", "N2"]
    # Stale rows end up at the bottom
    assert set(ordered_ids[4:]) == {"S1", "S2"}


def test_policy_drop_filters_only_stale(
    six_bullet_fixture: tuple[
        list[ScoredBullet], dict[str, BulletMetadata], _SeededEngine
    ],
) -> None:
    scored, bullets, engine = six_bullet_fixture
    pipeline = _build_pipeline(scored, bullets, engine, policy="drop")

    result = pipeline.search(query="q", bullets=[], limit=10)

    # AC: only stale removed, not_applicable kept.
    assert len(result.results) == 4
    assert result.dropped_stale_count == 2

    kept_ids = {r.bullet_id for r in result.results}
    assert kept_ids == {"V1", "V2", "N1", "N2"}

    # Kept rows are untouched (no demote in drop mode).
    by_id = {r.bullet_id: r for r in result.results}
    assert by_id["V1"].final_score == pytest.approx(0.90)
    assert by_id["N2"].final_score == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# Edge: unverifiable rows under demote + drop
# ---------------------------------------------------------------------------


def test_demote_scales_unverifiable_by_distinct_multiplier() -> None:
    """Unverifiable rows multiply by ``unverifiable_trust_score`` (default 0.7),
    NOT by ``stale_trust_score``."""
    scored = [
        _scored("V", 1.00),
        _scored("U", 0.90),
        _scored("S", 0.80),
    ]
    bullets = {bid: BulletMetadata(anchors=[]) for bid in "VUS"}
    engine = _PositionalEngine(
        outcomes=[
            (VerifiedStatus.VERIFIED, 1.0),
            (VerifiedStatus.UNVERIFIABLE, 0.7),
            (VerifiedStatus.STALE, 0.3),
        ]
    )
    pipeline = _build_pipeline(
        scored,
        bullets,
        engine,
        policy="demote",
        stale_trust_score=0.3,
        unverifiable_trust_score=0.7,
    )

    result = pipeline.search(query="q", bullets=[], limit=10)

    by_id = {r.bullet_id: r for r in result.results}
    assert by_id["V"].final_score == pytest.approx(1.00)
    assert by_id["U"].final_score == pytest.approx(0.90 * 0.7)
    assert by_id["S"].final_score == pytest.approx(0.80 * 0.3)
    assert result.dropped_stale_count == 0


def test_drop_keeps_unverifiable_and_not_applicable() -> None:
    """Drop policy only removes ``stale``; unverifiable + not_applicable stay."""
    scored = [
        _scored("V", 0.90),
        _scored("U", 0.80),
        _scored("S", 0.70),
        _scored("N", 0.60),
    ]
    bullets = {bid: BulletMetadata(anchors=[]) for bid in "VUSN"}
    engine = _PositionalEngine(
        outcomes=[
            (VerifiedStatus.VERIFIED, 1.0),
            (VerifiedStatus.UNVERIFIABLE, 0.7),
            (VerifiedStatus.STALE, 0.3),
            (VerifiedStatus.NOT_APPLICABLE, None),
        ]
    )
    pipeline = _build_pipeline(scored, bullets, engine, policy="drop")

    result = pipeline.search(query="q", bullets=[], limit=10)

    kept_ids = [r.bullet_id for r in result.results]
    assert kept_ids == ["V", "U", "N"]
    assert result.dropped_stale_count == 1


# ---------------------------------------------------------------------------
# Regression: verifier disabled path is unaffected by policy value.
# ---------------------------------------------------------------------------


def test_verifier_off_policy_drop_is_no_op() -> None:
    """When no VerificationEngine is wired, policy branching never runs."""
    scored = [_scored("a", 0.9), _scored("b", 0.8)]
    # loader is irrelevant here — verifier is None so _run_verifier bails out.
    pipeline = _build_pipeline(
        scored,
        loaded_bullets={},
        verifier=None,
        policy="drop",
    )

    result = pipeline.search(query="q", bullets=[], limit=10)

    # All rows preserved, no drops counted — policy is a no-op without signal.
    assert len(result.results) == 2
    assert result.dropped_stale_count == 0
    for r in result.results:
        assert r.verified_status is None
        assert r.trust_score is None
