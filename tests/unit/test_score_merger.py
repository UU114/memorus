"""Unit tests for memx.engines.generator.score_merger — ScoreMerger."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from memx.config import RetrievalConfig
from memx.engines.generator.score_merger import (
    MAX_KEYWORD_SCORE,
    BulletInfo,
    ScoreMerger,
    ScoredBullet,
)

# ── Helper fixtures ──────────────────────────────────────────────────────

_NOW = datetime(2026, 2, 27, 12, 0, 0, tzinfo=timezone.utc)


def _make_info(
    bid: str,
    content: str = "",
    days_ago: float = 0.0,
    decay_weight: float = 1.0,
) -> BulletInfo:
    """Create a BulletInfo with created_at set to *days_ago* before _NOW."""
    created = _NOW - timedelta(days=days_ago)
    return BulletInfo(
        bullet_id=bid,
        content=content,
        created_at=created,
        decay_weight=decay_weight,
    )


# ── Weight normalization tests ───────────────────────────────────────────


class TestWeightNormalization:
    """Weights must auto-normalize to sum 1.0."""

    def test_default_weights_sum_to_one(self) -> None:
        merger = ScoreMerger()
        assert abs(merger.keyword_weight + merger.semantic_weight - 1.0) < 1e-9

    def test_default_values(self) -> None:
        merger = ScoreMerger()
        assert abs(merger.keyword_weight - 0.6) < 1e-9
        assert abs(merger.semantic_weight - 0.4) < 1e-9

    def test_custom_weights_normalized(self) -> None:
        config = RetrievalConfig(keyword_weight=3.0, semantic_weight=1.0)
        merger = ScoreMerger(config)
        assert abs(merger.keyword_weight - 0.75) < 1e-9
        assert abs(merger.semantic_weight - 0.25) < 1e-9
        assert abs(merger.keyword_weight + merger.semantic_weight - 1.0) < 1e-9

    def test_equal_weights(self) -> None:
        config = RetrievalConfig(keyword_weight=1.0, semantic_weight=1.0)
        merger = ScoreMerger(config)
        assert abs(merger.keyword_weight - 0.5) < 1e-9
        assert abs(merger.semantic_weight - 0.5) < 1e-9

    def test_zero_both_weights_defaults_to_keyword(self) -> None:
        config = RetrievalConfig(keyword_weight=0.0, semantic_weight=0.0)
        merger = ScoreMerger(config)
        assert merger.keyword_weight == 1.0
        assert merger.semantic_weight == 0.0

    def test_already_normalized(self) -> None:
        config = RetrievalConfig(keyword_weight=0.6, semantic_weight=0.4)
        merger = ScoreMerger(config)
        assert abs(merger.keyword_weight - 0.6) < 1e-9
        assert abs(merger.semantic_weight - 0.4) < 1e-9


# ── RecencyBoost computation tests ───────────────────────────────────────


class TestRecencyBoost:
    """RecencyBoost gives recent bullets a scoring advantage."""

    def test_recent_bullet_gets_boost(self) -> None:
        merger = ScoreMerger()  # default: 7 days, 1.2x
        created = _NOW - timedelta(days=3)
        boost = merger.compute_recency_boost(created, _NOW)
        assert boost == 1.2

    def test_old_bullet_no_boost(self) -> None:
        merger = ScoreMerger()
        created = _NOW - timedelta(days=30)
        boost = merger.compute_recency_boost(created, _NOW)
        assert boost == 1.0

    def test_exactly_at_boundary(self) -> None:
        """Bullet created exactly recency_boost_days ago should get boost."""
        merger = ScoreMerger()
        created = _NOW - timedelta(days=7)
        boost = merger.compute_recency_boost(created, _NOW)
        assert boost == 1.2

    def test_just_past_boundary(self) -> None:
        """Bullet created just past recency_boost_days should NOT get boost."""
        merger = ScoreMerger()
        created = _NOW - timedelta(days=7, seconds=1)
        boost = merger.compute_recency_boost(created, _NOW)
        assert boost == 1.0

    def test_none_created_at(self) -> None:
        """None created_at returns 1.0 (no boost)."""
        merger = ScoreMerger()
        assert merger.compute_recency_boost(None, _NOW) == 1.0

    def test_future_timestamp(self) -> None:
        """Future timestamps (clock skew) are treated as recent."""
        merger = ScoreMerger()
        future = _NOW + timedelta(days=5)
        boost = merger.compute_recency_boost(future, _NOW)
        assert boost == 1.2

    def test_custom_recency_config(self) -> None:
        config = RetrievalConfig(recency_boost_days=14, recency_boost_factor=1.5)
        merger = ScoreMerger(config)
        # 10 days ago — within 14-day window
        created = _NOW - timedelta(days=10)
        assert merger.compute_recency_boost(created, _NOW) == 1.5
        # 20 days ago — outside window
        old = _NOW - timedelta(days=20)
        assert merger.compute_recency_boost(old, _NOW) == 1.0


# ── Full mode merge tests ───────────────────────────────────────────────


class TestMergeFullMode:
    """Full mode: both keyword and semantic scores available."""

    def test_basic_merge(self) -> None:
        merger = ScoreMerger()  # kw=0.6, sem=0.4
        infos = {
            "b1": _make_info("b1", "content1", days_ago=30),
        }
        kw = {"b1": 35.0}  # max keyword score -> norm = 1.0
        sem = {"b1": 1.0}  # max semantic score
        results = merger.merge(kw, sem, infos)
        assert len(results) == 1
        r = results[0]
        assert r.bullet_id == "b1"
        # BlendedScore = 1.0*0.6 + 1.0*0.4 = 1.0
        # DecayWeight = 1.0, RecencyBoost = 1.0 (30 days > 7)
        assert abs(r.final_score - 1.0) < 1e-9

    def test_sorted_descending(self) -> None:
        merger = ScoreMerger()
        infos = {
            "b1": _make_info("b1", "low", days_ago=30),
            "b2": _make_info("b2", "high", days_ago=30),
        }
        kw = {"b1": 10.0, "b2": 30.0}
        sem = {"b1": 0.2, "b2": 0.9}
        results = merger.merge(kw, sem, infos)
        assert len(results) == 2
        assert results[0].bullet_id == "b2"
        assert results[1].bullet_id == "b1"
        assert results[0].final_score >= results[1].final_score

    def test_decay_weight_applied(self) -> None:
        merger = ScoreMerger()
        infos = {
            "b1": _make_info("b1", "full decay", days_ago=30, decay_weight=1.0),
            "b2": _make_info("b2", "half decay", days_ago=30, decay_weight=0.5),
        }
        kw = {"b1": 35.0, "b2": 35.0}
        sem = {"b1": 1.0, "b2": 1.0}
        results = merger.merge(kw, sem, infos)
        # Same blended score, but b2 has half decay weight
        assert abs(results[0].final_score - 2.0 * results[1].final_score) < 1e-9

    def test_recency_boost_applied(self) -> None:
        merger = ScoreMerger()
        infos = {
            "old": _make_info("old", "old bullet", days_ago=30),
            "new": _make_info("new", "new bullet", days_ago=2),
        }
        kw = {"old": 35.0, "new": 35.0}
        sem = {"old": 1.0, "new": 1.0}
        results = merger.merge(kw, sem, infos)
        # New bullet gets 1.2x recency boost
        new_r = next(r for r in results if r.bullet_id == "new")
        old_r = next(r for r in results if r.bullet_id == "old")
        assert new_r.recency_boost == 1.2
        assert old_r.recency_boost == 1.0
        assert abs(new_r.final_score - old_r.final_score * 1.2) < 1e-9

    def test_keyword_only_bullet(self) -> None:
        """Bullet only in keyword results (not in semantic) gets sem=0."""
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "kw only", days_ago=30)}
        kw = {"b1": 20.0}
        sem = {}  # b1 not in semantic results, but sem dict is non-empty overall
        # This is NOT degraded mode because semantic_results is provided (even if empty)
        # Actually, empty dict is falsy in Python, so it IS degraded mode
        results = merger.merge(kw, sem, infos)
        assert len(results) == 1
        # In degraded mode, kw_weight=1.0
        norm_kw = 20.0 / 35.0
        assert abs(results[0].final_score - norm_kw * 1.0 * 1.0) < 1e-9

    def test_semantic_only_bullet(self) -> None:
        """Bullet only in semantic results (not in keyword) gets kw=0."""
        merger = ScoreMerger()
        infos = {
            "b1": _make_info("b1", "sem only", days_ago=30),
            "b2": _make_info("b2", "has kw", days_ago=30),
        }
        kw = {"b2": 10.0}
        sem = {"b1": 0.8, "b2": 0.5}
        results = merger.merge(kw, sem, infos)
        b1 = next(r for r in results if r.bullet_id == "b1")
        assert b1.keyword_score == 0.0
        assert abs(b1.semantic_score - 0.8) < 1e-9

    def test_preserves_metadata(self) -> None:
        info = BulletInfo(
            bullet_id="b1",
            content="test",
            created_at=_NOW - timedelta(days=30),
            metadata={"source": "test", "tags": ["python"]},
        )
        merger = ScoreMerger()
        results = merger.merge({"b1": 10.0}, {"b1": 0.5}, {"b1": info})
        assert results[0].metadata == {"source": "test", "tags": ["python"]}

    def test_raw_scores_preserved(self) -> None:
        """ScoredBullet should contain the original raw scores."""
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "test", days_ago=30)}
        results = merger.merge({"b1": 22.5}, {"b1": 0.75}, {"b1": infos["b1"]})
        r = results[0]
        assert abs(r.keyword_score - 22.5) < 1e-9
        assert abs(r.semantic_score - 0.75) < 1e-9


# ── Degraded mode tests ─────────────────────────────────────────────────


class TestMergeDegradedMode:
    """Degraded mode: no semantic scores, keyword weight becomes 1.0."""

    def test_none_semantic(self) -> None:
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "kw only", days_ago=30)}
        results = merger.merge({"b1": 35.0}, None, infos)
        assert len(results) == 1
        r = results[0]
        # keyword_weight = 1.0 in degraded mode
        # NormKw = 35/35 = 1.0, FinalScore = 1.0 * 1.0 * 1.0 = 1.0
        assert abs(r.final_score - 1.0) < 1e-9
        assert r.semantic_score == 0.0

    def test_empty_semantic(self) -> None:
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "kw only", days_ago=30)}
        results = merger.merge({"b1": 17.5}, {}, infos)
        assert len(results) == 1
        # NormKw = 17.5/35 = 0.5, kw_weight=1.0, decay=1.0, recency=1.0
        assert abs(results[0].final_score - 0.5) < 1e-9

    def test_degraded_with_multiple_bullets(self) -> None:
        merger = ScoreMerger()
        infos = {
            "b1": _make_info("b1", "high", days_ago=30),
            "b2": _make_info("b2", "low", days_ago=30),
        }
        results = merger.merge({"b1": 30.0, "b2": 10.0}, None, infos)
        assert len(results) == 2
        assert results[0].bullet_id == "b1"
        assert results[0].final_score > results[1].final_score

    def test_degraded_recency_still_works(self) -> None:
        merger = ScoreMerger()
        infos = {
            "b1": _make_info("b1", "old", days_ago=30),
            "b2": _make_info("b2", "new", days_ago=2),
        }
        results = merger.merge({"b1": 35.0, "b2": 35.0}, None, infos)
        new_r = next(r for r in results if r.bullet_id == "b2")
        old_r = next(r for r in results if r.bullet_id == "b1")
        assert new_r.recency_boost == 1.2
        assert old_r.recency_boost == 1.0
        assert new_r.final_score > old_r.final_score


# ── Edge cases ───────────────────────────────────────────────────────────


class TestMergeEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_inputs(self) -> None:
        merger = ScoreMerger()
        results = merger.merge({}, None, {})
        assert results == []

    def test_missing_bullet_info_skipped(self) -> None:
        """Bullets not in bullet_infos are skipped with a warning."""
        merger = ScoreMerger()
        results = merger.merge({"unknown": 10.0}, None, {})
        assert results == []

    def test_keyword_score_clamped(self) -> None:
        """Keyword scores above MAX_KEYWORD_SCORE are clamped to 1.0 normalized."""
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "test", days_ago=30)}
        results = merger.merge({"b1": 100.0}, None, infos)
        # NormKw = min(100/35, 1.0) = 1.0
        assert abs(results[0].final_score - 1.0) < 1e-9

    def test_semantic_score_clamped(self) -> None:
        """Semantic scores outside [0, 1] are clamped."""
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "test", days_ago=30)}
        results = merger.merge({"b1": 0.0}, {"b1": 1.5}, infos)
        # Semantic clamped to 1.0
        r = results[0]
        expected = 0.0 * 0.6 + 1.0 * 0.4  # = 0.4
        assert abs(r.final_score - expected) < 1e-9

    def test_decay_weight_zero(self) -> None:
        """Zero decay weight should zero out final score."""
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "test", days_ago=30, decay_weight=0.0)}
        results = merger.merge({"b1": 35.0}, {"b1": 1.0}, infos)
        assert results[0].final_score == 0.0

    def test_zero_keyword_score(self) -> None:
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "test", days_ago=30)}
        results = merger.merge({"b1": 0.0}, {"b1": 0.8}, infos)
        # BlendedScore = 0.0*0.6 + 0.8*0.4 = 0.32
        assert abs(results[0].final_score - 0.32) < 1e-9


# ── Formula verification tests ──────────────────────────────────────────


class TestFormulaVerification:
    """Verify the exact scoring formula with known values."""

    def test_formula_full_mode(self) -> None:
        """FinalScore = (NormKw * kw_w + NormSem * sem_w) * decay * recency."""
        config = RetrievalConfig(
            keyword_weight=0.6,
            semantic_weight=0.4,
            recency_boost_days=7,
            recency_boost_factor=1.2,
        )
        merger = ScoreMerger(config)
        infos = {"b1": _make_info("b1", "test", days_ago=3, decay_weight=0.8)}
        kw = {"b1": 25.0}  # NormKw = 25/35
        sem = {"b1": 0.7}

        results = merger.merge(kw, sem, infos)
        r = results[0]

        norm_kw = 25.0 / 35.0
        blended = norm_kw * 0.6 + 0.7 * 0.4
        expected = blended * 0.8 * 1.2  # decay=0.8, recency=1.2 (3 days < 7)

        assert abs(r.final_score - expected) < 1e-9
        assert abs(r.keyword_score - 25.0) < 1e-9
        assert abs(r.semantic_score - 0.7) < 1e-9
        assert abs(r.decay_weight - 0.8) < 1e-9
        assert abs(r.recency_boost - 1.2) < 1e-9

    def test_formula_degraded_mode(self) -> None:
        """In degraded mode, kw_weight=1.0 and semantic is ignored."""
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "test", days_ago=3, decay_weight=0.9)}
        kw = {"b1": 20.0}

        results = merger.merge(kw, None, infos)
        r = results[0]

        norm_kw = 20.0 / 35.0
        expected = norm_kw * 1.0 * 0.9 * 1.2  # kw_w=1.0, decay=0.9, recency=1.2

        assert abs(r.final_score - expected) < 1e-9

    def test_max_keyword_score_constant(self) -> None:
        """MAX_KEYWORD_SCORE should be 35.0 (Exact:15 + Fuzzy:10 + Meta:10)."""
        assert MAX_KEYWORD_SCORE == 35.0


# ── Config property tests ───────────────────────────────────────────────


class TestScoreMergerConfig:
    """ScoreMerger config and property access."""

    def test_default_config(self) -> None:
        merger = ScoreMerger()
        assert merger.config.keyword_weight == 0.6
        assert merger.config.semantic_weight == 0.4
        assert merger.config.recency_boost_days == 7
        assert merger.config.recency_boost_factor == 1.2

    def test_custom_config(self) -> None:
        config = RetrievalConfig(
            keyword_weight=0.8,
            semantic_weight=0.2,
            recency_boost_days=14,
            recency_boost_factor=1.5,
        )
        merger = ScoreMerger(config)
        assert merger.config is config
        assert abs(merger.keyword_weight - 0.8) < 1e-9
        assert abs(merger.semantic_weight - 0.2) < 1e-9
