"""Supplementary decay tests for STORY-022 acceptance criteria.

Covers formula precision data points, protection period boundaries,
permanent/archive threshold edge cases, large-batch sweep, sweep
without explicit `now`, reinforce edge cases, and custom config
combinations.  Complements test_decay_engine.py (58 existing tests).
"""

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


# ===========================================================================
# 1. Formula precision — at least 5 known-input/known-output data points
# ===========================================================================


class TestExponentialDecayPrecision:
    """Verify exponential_decay matches 2^(-age/half_life) precisely."""

    @pytest.mark.parametrize(
        "age, half_life, expected",
        [
            (0, 30, 1.0),
            (30, 30, 0.5),
            (60, 30, 0.25),
            (90, 30, 0.125),
            (120, 30, 0.0625),
            (15, 30, 2 ** (-0.5)),  # sqrt(0.5)
        ],
        ids=[
            "age=0 -> 1.0",
            "age=1HL -> 0.5",
            "age=2HL -> 0.25",
            "age=3HL -> 0.125",
            "age=4HL -> 0.0625",
            "age=0.5HL -> sqrt(0.5)",
        ],
    )
    def test_known_data_point(self, age: float, half_life: float, expected: float) -> None:
        """Parametrized precision check for exponential_decay formula."""
        assert exponential_decay(age, half_life) == pytest.approx(expected, rel=1e-9)

    def test_large_half_life_slow_decay(self) -> None:
        """Very large half_life -> slow decay."""
        # half_life=365 (1 year), age=30 days -> 2^(-30/365)
        expected = 2 ** (-30.0 / 365.0)
        assert exponential_decay(30, 365) == pytest.approx(expected, rel=1e-9)


class TestBoostedWeightPrecision:
    """Verify boosted_weight precision with known inputs."""

    @pytest.mark.parametrize(
        "base, boost, recalls, expected",
        [
            (0.5, 0.1, 0, 0.5),
            (0.5, 0.1, 10, 1.0),  # 0.5*(1+1.0)=1.0
            (0.25, 0.1, 5, 0.375),  # 0.25*(1+0.5)=0.375
            (1.0, 0.0, 100, 1.0),
            (0.1, 0.2, 3, 0.16),  # 0.1*(1+0.6)=0.16
        ],
        ids=[
            "no recall",
            "high recall clamped to 1.0",
            "mid decay mid recall",
            "zero boost factor",
            "small base with boost",
        ],
    )
    def test_known_data_point(
        self, base: float, boost: float, recalls: int, expected: float
    ) -> None:
        """Parametrized precision check for boosted_weight formula."""
        assert boosted_weight(base, boost, recalls) == pytest.approx(expected, rel=1e-9)

    def test_story_data_point_half_life_30_recall_10(self) -> None:
        """Story spec: half_life=30, age=30, recall=10, boost=0.1 -> ~0.5*2.0 = 1.0 (clamped)."""
        base = exponential_decay(30.0, 30.0)
        result = boosted_weight(base, 0.1, 10)
        assert result == 1.0  # 0.5 * (1 + 0.1*10) = 0.5 * 2.0 = 1.0


# ===========================================================================
# 2. Protection period boundary — day 6 vs day 8 (default 7-day protection)
# ===========================================================================


class TestProtectionPeriodBoundary:
    """Acceptance criteria: day 6 (protected) vs day 8 (decaying)."""

    def test_day_6_protected(self) -> None:
        """Day 6 is within 7-day protection -> weight=1.0, is_protected=True."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=_created(6), now=NOW)
        assert result.weight == 1.0
        assert result.is_protected is True
        assert result.should_archive is False

    def test_day_8_decaying(self) -> None:
        """Day 8 is past 7-day protection -> weight<1.0, is_protected=False."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=_created(8), now=NOW)
        assert result.weight < 1.0
        assert result.is_protected is False

    def test_day_6_vs_day_8_weight_difference(self) -> None:
        """Day 6 should have higher weight than day 8."""
        engine = DecayEngine()
        r6 = engine.compute_weight(created_at=_created(6), now=NOW)
        r8 = engine.compute_weight(created_at=_created(8), now=NOW)
        assert r6.weight > r8.weight

    def test_protection_boundary_custom_10_day(self) -> None:
        """Custom 10-day protection: day 9 protected, day 11 decaying."""
        cfg = DecayConfig(protection_days=10)
        engine = DecayEngine(config=cfg)
        r9 = engine.compute_weight(created_at=_created(9), now=NOW)
        r11 = engine.compute_weight(created_at=_created(11), now=NOW)
        assert r9.is_protected is True
        assert r11.is_protected is False
        assert r9.weight == 1.0
        assert r11.weight < 1.0


# ===========================================================================
# 3. Permanent retention threshold — recall_count=14 vs recall_count=15
# ===========================================================================


class TestPermanentThresholdBoundary:
    """Acceptance criteria: recall_count=14 decays vs recall_count=15 permanent."""

    def test_recall_14_decays(self) -> None:
        """recall_count=14 (below threshold=15) -> decays normally."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(100), recall_count=14, now=NOW
        )
        assert result.is_permanent is False
        assert result.weight < 1.0

    def test_recall_15_permanent(self) -> None:
        """recall_count=15 (at threshold) -> permanent, weight=1.0."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(100), recall_count=15, now=NOW
        )
        assert result.is_permanent is True
        assert result.weight == 1.0
        assert result.should_archive is False

    def test_permanent_overrides_decay_for_very_old_memory(self) -> None:
        """Even 1000 days old, permanent threshold prevents archival."""
        engine = DecayEngine()
        result = engine.compute_weight(
            created_at=_created(1000), recall_count=15, now=NOW
        )
        assert result.is_permanent is True
        assert result.weight == 1.0
        assert result.should_archive is False

    def test_custom_permanent_threshold_5(self) -> None:
        """Custom permanent_threshold=5: recall=4 decays, recall=5 permanent."""
        cfg = DecayConfig(permanent_threshold=5)
        engine = DecayEngine(config=cfg)
        r4 = engine.compute_weight(created_at=_created(50), recall_count=4, now=NOW)
        r5 = engine.compute_weight(created_at=_created(50), recall_count=5, now=NOW)
        assert r4.is_permanent is False
        assert r5.is_permanent is True


# ===========================================================================
# 4. Archive threshold boundary — weight=0.021 vs weight=0.019
# ===========================================================================


class TestArchiveThresholdBoundary:
    """Acceptance criteria: weight~0.021 retained vs weight~0.019 archived.

    Default archive_threshold=0.02. Using age=169 (~0.0201) and age=170 (~0.0197).
    """

    def test_age_169_retained(self) -> None:
        """age=169 days -> weight ~0.0201, above 0.02 threshold -> NOT archived."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=_created(169), recall_count=0, now=NOW)
        assert result.weight > 0.02
        assert result.should_archive is False

    def test_age_170_archived(self) -> None:
        """age=170 days -> weight ~0.0197, below 0.02 threshold -> archived."""
        engine = DecayEngine()
        result = engine.compute_weight(created_at=_created(170), recall_count=0, now=NOW)
        assert result.weight < 0.02
        assert result.should_archive is True

    def test_archive_boundary_precision(self) -> None:
        """Verify exact formula values at the archive boundary."""
        engine = DecayEngine()
        r169 = engine.compute_weight(created_at=_created(169), recall_count=0, now=NOW)
        r170 = engine.compute_weight(created_at=_created(170), recall_count=0, now=NOW)
        # r169 should be ~0.0201 (just above), r170 should be ~0.0197 (just below)
        assert r169.weight == pytest.approx(2 ** (-169.0 / 30.0), rel=1e-6)
        assert r170.weight == pytest.approx(2 ** (-170.0 / 30.0), rel=1e-6)

    def test_custom_archive_threshold_0_1(self) -> None:
        """Custom archive_threshold=0.1: moderately old memory triggers archival earlier."""
        cfg = DecayConfig(archive_threshold=0.1)
        engine = DecayEngine(config=cfg)
        # age=100 days -> 2^(-100/30) ~= 0.0099, well below 0.1
        result = engine.compute_weight(created_at=_created(100), recall_count=0, now=NOW)
        assert result.should_archive is True
        # age=10 days -> 2^(-10/30) ~= 0.794, above 0.1
        result2 = engine.compute_weight(created_at=_created(10), recall_count=0, now=NOW)
        assert result2.should_archive is False


# ===========================================================================
# 5. Sweep batch — 100+ mixed-state memories
# ===========================================================================


class TestSweepLargeMixedBatch:
    """Acceptance criteria: sweep 100+ bullets with mixed states."""

    def test_sweep_100_plus_mixed_states(self) -> None:
        """Sweep 120 bullets across all four categories: protected/decayed/archived/permanent."""
        engine = DecayEngine()
        bullets = []

        # 30 protected bullets (age 1-6 days)
        for i in range(30):
            bullets.append(_bullet(f"protected_{i}", days_ago=i % 6 + 1))

        # 30 decaying bullets (age 10-60 days, weight changing)
        for i in range(30):
            bullets.append(_bullet(f"decayed_{i}", days_ago=10 + i, current_weight=1.0))

        # 30 archived bullets (age 200+ days, no recalls)
        for i in range(30):
            bullets.append(
                _bullet(f"archived_{i}", days_ago=200 + i * 5, current_weight=0.01)
            )

        # 30 permanent bullets (high recall count)
        for i in range(30):
            bullets.append(
                _bullet(f"permanent_{i}", days_ago=100 + i, recall_count=20, current_weight=0.5)
            )

        result = engine.sweep(bullets, now=NOW)

        assert result.errors == []
        assert len(result.details) == 120

        # Verify category counts
        total = result.updated + result.archived + result.permanent + result.unchanged
        assert total == 120

        # At least some in each category
        assert result.permanent == 30
        assert result.archived >= 25  # most of the old ones
        assert result.unchanged >= 25  # most protected ones

    def test_sweep_returns_details_for_each_bullet(self) -> None:
        """Every successfully processed bullet should have a detail entry."""
        engine = DecayEngine()
        bullets = [_bullet(f"b{i}", days_ago=i + 1) for i in range(50)]
        result = engine.sweep(bullets, now=NOW)
        assert len(result.details) == 50
        for i in range(50):
            assert f"b{i}" in result.details
            assert isinstance(result.details[f"b{i}"], DecayResult)


# ===========================================================================
# 6. Sweep without explicit `now` — covers line 192
# ===========================================================================


class TestSweepDefaultNow:
    """Cover the sweep() branch where now=None -> datetime.now(utc)."""

    def test_sweep_without_now_uses_current_time(self) -> None:
        """sweep() called without now arg should not raise and use current UTC."""
        engine = DecayEngine()
        # A recently created bullet (1 second ago) should be protected
        recent = BulletDecayInfo(
            bullet_id="recent",
            created_at=datetime.now(timezone.utc) - timedelta(seconds=1),
            recall_count=0,
            current_weight=1.0,
        )
        result = engine.sweep([recent])  # no now= argument
        assert result.errors == []
        assert "recent" in result.details
        assert result.details["recent"].is_protected is True

    def test_sweep_empty_without_now(self) -> None:
        """sweep([]) without now arg should return clean result."""
        engine = DecayEngine()
        result = engine.sweep([])  # no now= argument
        assert isinstance(result, DecaySweepResult)
        assert result.updated == 0


# ===========================================================================
# 7. Sweep error isolation — single failure does not affect others
# ===========================================================================


class TestSweepSingleFailureIsolation:
    """Acceptance criteria: one bad bullet doesn't abort the whole sweep."""

    def test_error_in_middle_of_batch(self) -> None:
        """Bad bullet in the middle; bullets before and after still processed."""
        engine = DecayEngine()
        bullets = [
            _bullet("before", days_ago=5),
            BulletDecayInfo(
                bullet_id="bad_middle",
                created_at="invalid",  # type: ignore[arg-type]
                recall_count=0,
            ),
            _bullet("after", days_ago=10),
        ]
        result = engine.sweep(bullets, now=NOW)
        assert len(result.errors) == 1
        assert "bad_middle" in result.errors[0]
        assert "before" in result.details
        assert "after" in result.details

    def test_sweep_error_logged_as_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Sweep errors should produce WARNING logs."""
        import logging

        engine = DecayEngine()
        bad = BulletDecayInfo(
            bullet_id="log_test", created_at="nope", recall_count=0  # type: ignore[arg-type]
        )
        with caplog.at_level(logging.WARNING):
            engine.sweep([bad], now=NOW)
        assert "sweep error" in caplog.text
        assert "log_test" in caplog.text


# ===========================================================================
# 8. Reinforce edge cases
# ===========================================================================


class TestReinforceEdgeCases:
    """Additional reinforce callback tests not in the existing 58."""

    def test_reinforce_duplicate_ids(self) -> None:
        """Duplicate IDs are reinforced separately (each call increments)."""
        engine = DecayEngine()
        mock_fn = MagicMock()
        count = engine.reinforce(["b1", "b1", "b1"], mock_fn)
        assert count == 3
        assert mock_fn.call_count == 3

    def test_reinforce_last_recall_is_utc(self) -> None:
        """Payload last_recall should be timezone-aware UTC."""
        engine = DecayEngine()
        captured_payloads: list[dict] = []

        def capture(bid: str, payload: dict) -> None:
            captured_payloads.append(payload)

        engine.reinforce(["b1"], capture)
        lr = captured_payloads[0]["last_recall"]
        assert lr.tzinfo is not None
        assert lr.tzinfo == timezone.utc

    def test_reinforce_mixed_success_failure_returns_correct_count(self) -> None:
        """5 IDs with 2 failures -> returns 3."""
        engine = DecayEngine()
        fail_ids = {"b2", "b4"}

        def selective_fail(bid: str, payload: dict) -> None:
            if bid in fail_ids:
                raise RuntimeError("fail")

        count = engine.reinforce(["b1", "b2", "b3", "b4", "b5"], selective_fail)
        assert count == 3


# ===========================================================================
# 9. Custom configuration combinations
# ===========================================================================


class TestCustomConfigCombinations:
    """Acceptance criteria: custom half_life, boost_factor, etc."""

    def test_custom_half_life_60(self) -> None:
        """With half_life=60, weight at 60 days should be ~0.5."""
        cfg = DecayConfig(half_life_days=60.0)
        engine = DecayEngine(config=cfg)
        result = engine.compute_weight(created_at=_created(60), recall_count=0, now=NOW)
        assert result.weight == pytest.approx(0.5, abs=0.01)

    def test_custom_boost_factor_0_2(self) -> None:
        """With boost_factor=0.2, recall_count=5: base * (1 + 0.2*5) = base * 2.0."""
        cfg = DecayConfig(boost_factor=0.2)
        engine = DecayEngine(config=cfg)
        # 30 days old -> base ~0.5, boosted = 0.5 * 2.0 = 1.0
        result = engine.compute_weight(
            created_at=_created(30), recall_count=5, now=NOW
        )
        assert result.weight == 1.0  # clamped

    def test_combined_custom_config(self) -> None:
        """Test with multiple custom parameters at once."""
        cfg = DecayConfig(
            half_life_days=10.0,
            boost_factor=0.05,
            protection_days=3,
            permanent_threshold=10,
            archive_threshold=0.1,
        )
        engine = DecayEngine(config=cfg)

        # Day 2: still within 3-day protection
        r2 = engine.compute_weight(created_at=_created(2), now=NOW)
        assert r2.is_protected is True
        assert r2.weight == 1.0

        # Day 4: past protection, half_life=10 -> 2^(-4/10)=0.758
        r4 = engine.compute_weight(created_at=_created(4), recall_count=0, now=NOW)
        assert r4.is_protected is False
        assert r4.weight == pytest.approx(2 ** (-4.0 / 10.0), rel=1e-6)

        # recall_count=10 -> permanent
        r10 = engine.compute_weight(created_at=_created(100), recall_count=10, now=NOW)
        assert r10.is_permanent is True

        # Day 40: 2^(-40/10)=0.0625 < archive_threshold(0.1) -> archive
        r40 = engine.compute_weight(created_at=_created(40), recall_count=0, now=NOW)
        assert r40.should_archive is True

    def test_very_short_half_life(self) -> None:
        """half_life=1 -> very rapid decay."""
        cfg = DecayConfig(half_life_days=1.0, protection_days=0)
        engine = DecayEngine(config=cfg)
        # Day 10: 2^(-10/1) = 2^(-10) ~= 0.000977
        result = engine.compute_weight(created_at=_created(10), recall_count=0, now=NOW)
        assert result.weight == pytest.approx(2 ** (-10.0), rel=1e-6)
        assert result.weight < 0.001

    def test_very_large_half_life(self) -> None:
        """half_life=3650 (10 years) -> nearly no decay in 30 days."""
        cfg = DecayConfig(half_life_days=3650.0, protection_days=0)
        engine = DecayEngine(config=cfg)
        result = engine.compute_weight(created_at=_created(30), recall_count=0, now=NOW)
        # 2^(-30/3650) ~= 0.9943
        assert result.weight > 0.99

    def test_sweep_on_session_end_flag(self) -> None:
        """Verify sweep_on_session_end config is accessible."""
        cfg = DecayConfig(sweep_on_session_end=False)
        engine = DecayEngine(config=cfg)
        assert engine.config.sweep_on_session_end is False


# ===========================================================================
# 10. DecayResult / DecaySweepResult dataclass behavior
# ===========================================================================


class TestDataclassBehavior:
    """Verify dataclass properties of result types."""

    def test_decay_sweep_result_defaults(self) -> None:
        """DecaySweepResult() should have all zeros and empty collections."""
        r = DecaySweepResult()
        assert r.updated == 0
        assert r.archived == 0
        assert r.permanent == 0
        assert r.unchanged == 0
        assert r.errors == []
        assert r.details == {}

    def test_bullet_decay_info_defaults(self) -> None:
        """BulletDecayInfo defaults: recall_count=0, current_weight=1.0, last_recall=None."""
        b = BulletDecayInfo(bullet_id="x", created_at=NOW)
        assert b.recall_count == 0
        assert b.current_weight == 1.0
        assert b.last_recall is None

    def test_decay_result_frozen(self) -> None:
        """DecayResult is a frozen dataclass; attribute assignment should raise."""
        r = DecayResult(weight=0.5, should_archive=False, is_permanent=False, is_protected=False)
        with pytest.raises(AttributeError):
            r.weight = 0.9  # type: ignore[misc]

    def test_decay_sweep_result_mutable(self) -> None:
        """DecaySweepResult is NOT frozen; counters can be updated."""
        r = DecaySweepResult()
        r.updated = 10
        r.archived = 5
        assert r.updated == 10
        assert r.archived == 5


# ===========================================================================
# 11. _age_in_days static method edge cases
# ===========================================================================


class TestAgeInDays:
    """Test DecayEngine._age_in_days edge cases."""

    def test_same_time_returns_zero(self) -> None:
        """Same created_at and now -> age=0."""
        assert DecayEngine._age_in_days(NOW, NOW) == 0.0

    def test_future_created_at_returns_zero(self) -> None:
        """Future created_at -> clamped to 0."""
        future = NOW + timedelta(days=10)
        assert DecayEngine._age_in_days(future, NOW) == 0.0

    def test_fractional_days(self) -> None:
        """12 hours = 0.5 days."""
        half_day_ago = NOW - timedelta(hours=12)
        assert DecayEngine._age_in_days(half_day_ago, NOW) == pytest.approx(0.5, rel=1e-6)

    def test_one_second(self) -> None:
        """1 second = 1/86400 days."""
        one_sec_ago = NOW - timedelta(seconds=1)
        expected = 1.0 / 86400.0
        assert DecayEngine._age_in_days(one_sec_ago, NOW) == pytest.approx(expected, rel=1e-6)
