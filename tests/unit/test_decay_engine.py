"""Unit tests for memx.engines.decay — formulas, DecayEngine, sweep, reinforce."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from memx.config import DecayConfig
from memx.engines.decay.engine import (
    BulletDecayInfo,
    DecayEngine,
    DecayResult,
    DecaySweepResult,
)
from memx.engines.decay.formulas import boosted_weight, exponential_decay

# ---------------------------------------------------------------------------
# Helper: fixed reference time
# ---------------------------------------------------------------------------

NOW = datetime(2026, 2, 27, 12, 0, 0, tzinfo=timezone.utc)


def _created(days_ago: float) -> datetime:
    """Return a datetime *days_ago* before NOW."""
    return NOW - timedelta(days=days_ago)


# ===========================================================================
# Part 1: Pure formula tests
# ===========================================================================


class TestExponentialDecay:
    """Tests for formulas.exponential_decay()."""

    def test_age_zero_returns_one(self) -> None:
        """age_days=0 -> weight=1.0 (just created)."""
        assert exponential_decay(0.0, 30.0) == 1.0

    def test_negative_age_returns_one(self) -> None:
        """Negative age (clock skew) -> weight=1.0."""
        assert exponential_decay(-5.0, 30.0) == 1.0

    def test_one_half_life(self) -> None:
        """After exactly one half-life -> weight=0.5."""
        assert exponential_decay(30.0, 30.0) == pytest.approx(0.5)

    def test_two_half_lives(self) -> None:
        """After two half-lives -> weight=0.25."""
        assert exponential_decay(60.0, 30.0) == pytest.approx(0.25)

    def test_very_old_approaches_zero(self) -> None:
        """365 days with 30-day half-life -> very small."""
        result = exponential_decay(365.0, 30.0)
        assert result < 0.001
        assert result > 0.0

    def test_small_half_life(self) -> None:
        """Tiny half_life -> rapid decay but no error."""
        result = exponential_decay(1.0, 0.1)
        assert result < 0.01
        assert result >= 0.0

    def test_zero_half_life_guard(self) -> None:
        """half_life=0 -> returns 0.0 as guard."""
        assert exponential_decay(10.0, 0.0) == 0.0

    def test_negative_half_life_guard(self) -> None:
        """Negative half_life -> returns 0.0 as guard."""
        assert exponential_decay(10.0, -5.0) == 0.0


class TestBoostedWeight:
    """Tests for formulas.boosted_weight()."""

    def test_no_boost_no_recalls(self) -> None:
        """Zero recalls -> no boost applied."""
        assert boosted_weight(0.5, 0.1, 0) == pytest.approx(0.5)

    def test_boost_with_recalls(self) -> None:
        """Recall boost: 0.5 * (1 + 0.1*5) = 0.75."""
        assert boosted_weight(0.5, 0.1, 5) == pytest.approx(0.75)

    def test_clamp_to_max_one(self) -> None:
        """Large boost -> clamped to 1.0."""
        result = boosted_weight(0.8, 0.5, 10)
        assert result == 1.0

    def test_clamp_to_min_zero(self) -> None:
        """Base=0 -> result=0 regardless of boost."""
        assert boosted_weight(0.0, 0.1, 10) == 0.0

    def test_negative_recall_count_treated_as_zero(self) -> None:
        """Negative recall_count -> treated as 0."""
        assert boosted_weight(0.5, 0.1, -3) == pytest.approx(0.5)

    def test_zero_boost_factor(self) -> None:
        """boost_factor=0 -> recalls have no effect."""
        assert boosted_weight(0.5, 0.0, 100) == pytest.approx(0.5)


# ===========================================================================
# Part 2: DecayEngine tests
# ===========================================================================


class TestDecayEngineDefaults:
    """DecayEngine with default config."""

    def test_default_config(self) -> None:
        """Engine initializes with default DecayConfig."""
        engine = DecayEngine()
        assert engine.config.half_life_days == 30.0
        assert engine.config.boost_factor == 0.1
        assert engine.config.protection_days == 7
        assert engine.config.permanent_threshold == 15
        assert engine.config.archive_threshold == 0.02

    def test_custom_config(self) -> None:
        """Engine accepts custom DecayConfig."""
        cfg = DecayConfig(half_life_days=60.0, boost_factor=0.2)
        engine = DecayEngine(config=cfg)
        assert engine.config.half_life_days == 60.0
        assert engine.config.boost_factor == 0.2


class TestProtectionPeriod:
    """Memories within protection period should have weight=1.0."""

    def test_just_created(self) -> None:
        """Freshly created memory -> protected, weight=1.0."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=NOW, now=NOW)
        assert result.weight == 1.0
        assert result.is_protected is True
        assert result.should_archive is False
        assert result.is_permanent is False

    def test_within_protection(self) -> None:
        """3 days old (< 7 day protection) -> protected."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=_created(3), now=NOW)
        assert result.weight == 1.0
        assert result.is_protected is True

    def test_at_protection_boundary(self) -> None:
        """Exactly 7 days old (= protection_days) -> still protected."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=_created(7), now=NOW)
        assert result.weight == 1.0
        assert result.is_protected is True

    def test_past_protection(self) -> None:
        """8 days old (> 7 day protection) -> decay applies."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=_created(8), now=NOW)
        assert result.weight < 1.0
        assert result.is_protected is False

    def test_custom_protection_days(self) -> None:
        """Custom protection_days=14, 10 days old -> still protected."""
        cfg = DecayConfig(protection_days=14)
        engine = DecayEngine(config=cfg)
        result = engine.compute_weight(created_at=_created(10), now=NOW)
        assert result.weight == 1.0
        assert result.is_protected is True

    def test_zero_protection_days(self) -> None:
        """protection_days=0 -> no protection period, decay starts immediately."""
        cfg = DecayConfig(protection_days=0)
        engine = DecayEngine(config=cfg)
        result = engine.compute_weight(created_at=_created(1), now=NOW)
        assert result.weight < 1.0
        assert result.is_protected is False


class TestPermanentRetention:
    """Memories recalled enough times should be permanently retained."""

    def test_at_threshold(self) -> None:
        """recall_count == permanent_threshold (15) -> permanent."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(100), recall_count=15, now=NOW
        )
        assert result.weight == 1.0
        assert result.is_permanent is True
        assert result.should_archive is False

    def test_above_threshold(self) -> None:
        """recall_count >> permanent_threshold -> still permanent."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(365), recall_count=100, now=NOW
        )
        assert result.weight == 1.0
        assert result.is_permanent is True

    def test_below_threshold(self) -> None:
        """recall_count < permanent_threshold -> not permanent."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(100), recall_count=14, now=NOW
        )
        assert result.is_permanent is False

    def test_permanent_takes_priority_over_old_age(self) -> None:
        """Very old + high recall -> permanent (not archived)."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(1000), recall_count=20, now=NOW
        )
        assert result.weight == 1.0
        assert result.is_permanent is True
        assert result.should_archive is False


class TestArchivalThreshold:
    """Memories with very low weight should be flagged for archival."""

    def test_below_threshold_archives(self) -> None:
        """Very old memory with no recalls -> should_archive=True."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(300), recall_count=0, now=NOW
        )
        assert result.weight < 0.02
        assert result.should_archive is True

    def test_above_threshold_no_archive(self) -> None:
        """Moderately old memory -> should_archive=False."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(10), recall_count=0, now=NOW
        )
        assert result.weight >= 0.02
        assert result.should_archive is False

    def test_custom_archive_threshold(self) -> None:
        """Higher archive_threshold -> earlier archival."""
        cfg = DecayConfig(archive_threshold=0.5)
        engine = DecayEngine(config=cfg)
        result = engine.compute_weight(
            created_at=_created(31), recall_count=0, now=NOW
        )
        # After ~31 days with 30-day half_life, base ~= 0.49 < 0.5
        assert result.should_archive is True


class TestDecayFormula:
    """Verify the decay formula produces expected values."""

    def test_one_half_life(self) -> None:
        """After 30 days (= half_life), base ~= 0.5 (no recalls)."""
        # With protection_days=7, age=30 is past protection
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(30), recall_count=0, now=NOW
        )
        assert result.weight == pytest.approx(0.5, abs=0.01)

    def test_decay_with_recall_boost(self) -> None:
        """Recall boost increases weight: base * (1 + 0.1 * recall_count)."""
        engine = DecayEngine()
        # 30 days -> base=0.5, 5 recalls -> 0.5 * (1 + 0.1*5) = 0.75
        result = engine.compute_weight(
            created_at=_created(30), recall_count=5, now=NOW
        )
        assert result.weight == pytest.approx(0.75, abs=0.01)

    def test_weight_clamped_to_one(self) -> None:
        """High recall boost should not exceed 1.0."""
        engine = DecayEngine()
        # 10 days -> base ~= 0.794, 10 recalls -> 0.794 * (1+0.1*10) = 1.588 -> clamped
        result = engine.compute_weight(
            created_at=_created(10), recall_count=10, now=NOW
        )
        assert result.weight == 1.0


class TestEdgeCases:
    """Edge cases: None inputs, clock skew, extreme values."""

    def test_none_created_at(self) -> None:
        """created_at=None -> treated as now (fresh memory)."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=None, now=NOW)
        assert result.weight == 1.0
        assert result.is_protected is True

    def test_none_recall_count(self) -> None:
        """recall_count=None -> treated as 0."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(30), recall_count=None, now=NOW
        )
        assert result.weight == pytest.approx(0.5, abs=0.01)

    def test_none_now(self) -> None:
        """now=None -> uses current UTC time (should not raise)."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=datetime.now(timezone.utc))
        assert isinstance(result, DecayResult)
        assert result.weight == 1.0

    def test_future_created_at(self) -> None:
        """created_at in the future (clock skew) -> age=0, protected."""
        engine = DecayEngine()
        future = NOW + timedelta(days=5)
        result = engine.compute_weight(created_at=future, now=NOW)
        assert result.weight == 1.0
        assert result.is_protected is True

    def test_all_none_inputs(self) -> None:
        """All optional args None -> no crash, returns valid result."""
        engine = DecayEngine()
        result = engine.compute_weight()
        assert isinstance(result, DecayResult)
        assert 0.0 <= result.weight <= 1.0

    def test_last_recall_ignored_in_v1(self) -> None:
        """last_recall is accepted but does not change the result in v1."""
        engine = DecayEngine()
        without = engine.compute_weight(
            created_at=_created(30), recall_count=3, now=NOW
        )
        with_lr = engine.compute_weight(
            created_at=_created(30),
            recall_count=3,
            last_recall=_created(1),
            now=NOW,
        )
        assert without.weight == with_lr.weight

    def test_result_is_frozen_dataclass(self) -> None:
        """DecayResult should be immutable (frozen dataclass)."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=NOW, now=NOW)
        with pytest.raises(AttributeError):
            result.weight = 0.5  # type: ignore[misc]


# ===========================================================================
# Part 3: sweep() tests
# ===========================================================================


def _bullet(
    bid: str,
    days_ago: float,
    recall_count: int = 0,
    current_weight: float = 1.0,
    last_recall: datetime | None = None,
) -> BulletDecayInfo:
    """Helper to create a BulletDecayInfo with age relative to NOW."""
    return BulletDecayInfo(
        bullet_id=bid,
        created_at=_created(days_ago),
        recall_count=recall_count,
        current_weight=current_weight,
        last_recall=last_recall,
    )


class TestSweepBasic:
    """Basic sweep() behavior tests."""

    def test_empty_list_returns_zero_counts(self) -> None:
        """sweep([]) should return all-zero result without errors."""
        engine = DecayEngine()
        result = engine.sweep([], now=NOW)
        assert isinstance(result, DecaySweepResult)
        assert result.updated == 0
        assert result.archived == 0
        assert result.permanent == 0
        assert result.unchanged == 0
        assert result.errors == []
        assert result.details == {}

    def test_single_protected_bullet(self) -> None:
        """A recently created bullet is unchanged (still protected at weight=1.0)."""
        engine = DecayEngine()
        b = _bullet("b1", days_ago=3, current_weight=1.0)
        result = engine.sweep([b], now=NOW)
        assert result.unchanged == 1
        assert result.updated == 0
        assert "b1" in result.details
        assert result.details["b1"].is_protected is True

    def test_single_decayed_bullet(self) -> None:
        """A bullet past protection period should be counted as updated."""
        engine = DecayEngine()
        # 30 days old, current_weight=1.0 -> new weight ~0.5 -> updated
        b = _bullet("b1", days_ago=30, current_weight=1.0)
        result = engine.sweep([b], now=NOW)
        assert result.updated == 1
        assert result.details["b1"].weight == pytest.approx(0.5, abs=0.01)

    def test_single_archived_bullet(self) -> None:
        """A very old bullet with no recalls should be marked for archival."""
        engine = DecayEngine()
        b = _bullet("b1", days_ago=300, current_weight=0.05)
        result = engine.sweep([b], now=NOW)
        assert result.archived == 1
        assert result.details["b1"].should_archive is True

    def test_single_permanent_bullet(self) -> None:
        """A bullet with high recall_count should be counted as permanent."""
        engine = DecayEngine()
        b = _bullet("b1", days_ago=200, recall_count=20, current_weight=0.5)
        result = engine.sweep([b], now=NOW)
        assert result.permanent == 1
        assert result.details["b1"].is_permanent is True
        assert result.details["b1"].weight == 1.0


class TestSweepMixed:
    """Test sweep with mixed bullet categories."""

    def test_mixed_batch(self) -> None:
        """Batch with protected, decayed, archived, and permanent bullets."""
        engine = DecayEngine()
        bullets = [
            _bullet("protected", days_ago=2, current_weight=1.0),
            _bullet("decayed", days_ago=30, current_weight=1.0),
            _bullet("old", days_ago=300, current_weight=0.05),
            _bullet("permanent", days_ago=200, recall_count=20, current_weight=0.5),
        ]
        result = engine.sweep(bullets, now=NOW)
        assert result.unchanged == 1   # protected (weight stayed at 1.0)
        assert result.updated == 1     # decayed (weight changed)
        assert result.archived == 1    # old (should_archive)
        assert result.permanent == 1   # permanent (high recall)
        assert result.errors == []
        assert len(result.details) == 4

    def test_all_same_category(self) -> None:
        """All bullets in same state -> only one counter incremented."""
        engine = DecayEngine()
        bullets = [_bullet(f"b{i}", days_ago=1) for i in range(5)]
        result = engine.sweep(bullets, now=NOW)
        assert result.unchanged == 5
        assert result.updated == 0
        assert result.archived == 0
        assert result.permanent == 0


class TestSweepErrorIsolation:
    """A single bullet failure must not abort the entire sweep."""

    def test_bad_bullet_skipped_others_processed(self) -> None:
        """If compute_weight raises for one bullet, others still succeed."""
        engine = DecayEngine()
        good_bullet = _bullet("good", days_ago=3)
        # Use a non-datetime value that will cause a TypeError in _age_in_days
        bad_bullet = BulletDecayInfo(
            bullet_id="bad",
            created_at="not-a-datetime",  # type: ignore[arg-type]
            recall_count=0,
        )
        result = engine.sweep([bad_bullet, good_bullet], now=NOW)
        assert len(result.errors) == 1
        assert "bad" in result.errors[0]
        assert "good" in result.details
        assert result.unchanged == 1  # good bullet is protected

    def test_multiple_errors_collected(self) -> None:
        """Multiple bad bullets -> all errors collected."""
        engine = DecayEngine()
        bad1 = BulletDecayInfo(bullet_id="bad1", created_at="nope", recall_count=0)  # type: ignore[arg-type]
        bad2 = BulletDecayInfo(bullet_id="bad2", created_at="nope", recall_count=0)  # type: ignore[arg-type]
        good = _bullet("ok", days_ago=1)
        result = engine.sweep([bad1, bad2, good], now=NOW)
        assert len(result.errors) == 2
        assert result.unchanged == 1


class TestSweepUnchangedDetection:
    """Unchanged detection: weight stays the same as current_weight."""

    def test_same_weight_counted_as_unchanged(self) -> None:
        """If computed weight == current_weight, bullet is unchanged."""
        engine = DecayEngine()
        # Protected bullet with current_weight=1.0 -> computed=1.0 -> unchanged
        b = _bullet("b1", days_ago=1, current_weight=1.0)
        result = engine.sweep([b], now=NOW)
        assert result.unchanged == 1
        assert result.updated == 0

    def test_different_weight_counted_as_updated(self) -> None:
        """If computed weight != current_weight, bullet is updated."""
        engine = DecayEngine()
        # 30 days old with current_weight=1.0 -> computed ~0.5 -> updated
        b = _bullet("b1", days_ago=30, current_weight=1.0)
        result = engine.sweep([b], now=NOW)
        assert result.updated == 1
        assert result.unchanged == 0


class TestSweepPerformance:
    """Sanity check that sweep handles large batches without error."""

    def test_thousand_bullets(self) -> None:
        """Sweep 1000+ bullets without raising."""
        engine = DecayEngine()
        bullets = [_bullet(f"b{i}", days_ago=i % 365 + 1) for i in range(1200)]
        result = engine.sweep(bullets, now=NOW)
        total = result.updated + result.archived + result.permanent + result.unchanged
        assert total == 1200
        assert result.errors == []


# ===========================================================================
# Part 4: reinforce() tests
# ===========================================================================


class TestReinforceBasic:
    """Basic reinforce() behavior tests."""

    def test_empty_list_returns_zero(self) -> None:
        """reinforce([]) should return 0 without calling update_fn."""
        engine = DecayEngine()
        mock_fn = MagicMock()
        count = engine.reinforce([], mock_fn)
        assert count == 0
        mock_fn.assert_not_called()

    def test_single_bullet_reinforced(self) -> None:
        """reinforce with one ID -> calls update_fn once, returns 1."""
        engine = DecayEngine()
        mock_fn = MagicMock()
        count = engine.reinforce(["b1"], mock_fn)
        assert count == 1
        mock_fn.assert_called_once()
        # Verify payload structure
        call_args = mock_fn.call_args
        assert call_args[0][0] == "b1"
        payload = call_args[0][1]
        assert payload["recall_count_delta"] == 1
        assert isinstance(payload["last_recall"], datetime)

    def test_multiple_bullets_reinforced(self) -> None:
        """reinforce with multiple IDs -> all reinforced, returns correct count."""
        engine = DecayEngine()
        mock_fn = MagicMock()
        ids = ["b1", "b2", "b3"]
        count = engine.reinforce(ids, mock_fn)
        assert count == 3
        assert mock_fn.call_count == 3
        # Verify each ID was called
        called_ids = [call[0][0] for call in mock_fn.call_args_list]
        assert called_ids == ["b1", "b2", "b3"]


class TestReinforceErrorIsolation:
    """Callback failures should not abort remaining reinforcements."""

    def test_partial_failure(self) -> None:
        """If update_fn raises for one ID, others are still processed."""
        engine = DecayEngine()

        def flaky_fn(bid: str, payload: dict[str, object]) -> None:
            if bid == "b2":
                raise RuntimeError("storage failure")

        count = engine.reinforce(["b1", "b2", "b3"], flaky_fn)
        assert count == 2  # b1 and b3 succeed, b2 fails

    def test_all_failures(self) -> None:
        """If every callback fails, returns 0."""
        engine = DecayEngine()

        def always_fail(bid: str, payload: dict[str, object]) -> None:
            raise RuntimeError("always fail")

        count = engine.reinforce(["b1", "b2"], always_fail)
        assert count == 0

    def test_failure_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Failed reinforcement should produce a WARNING log."""
        engine = DecayEngine()

        def fail_fn(bid: str, payload: dict[str, object]) -> None:
            raise ValueError("test error")

        import logging

        with caplog.at_level(logging.WARNING):
            engine.reinforce(["b1"], fail_fn)
        assert "reinforce error" in caplog.text
        assert "b1" in caplog.text


class TestReinforcePayload:
    """Verify the payload structure passed to update_fn."""

    def test_payload_contains_required_keys(self) -> None:
        """Payload should have recall_count_delta and last_recall."""
        engine = DecayEngine()
        captured: list[tuple[str, dict[str, object]]] = []

        def capture_fn(bid: str, payload: dict[str, object]) -> None:
            captured.append((bid, payload))

        engine.reinforce(["b1"], capture_fn)
        assert len(captured) == 1
        bid, payload = captured[0]
        assert bid == "b1"
        assert "recall_count_delta" in payload
        assert "last_recall" in payload
        assert payload["recall_count_delta"] == 1
        assert isinstance(payload["last_recall"], datetime)
        # last_recall should be timezone-aware UTC
        lr = payload["last_recall"]
        assert isinstance(lr, datetime)
        assert lr.tzinfo is not None
