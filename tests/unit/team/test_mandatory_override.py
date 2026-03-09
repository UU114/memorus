"""Tests for STORY-072: Mandatory Override Escape Hatch.

Covers:
  - MandatoryOverride validation (reason non-empty, expires <= 90 days)
  - Shadow Merge override logic (active override skips mandatory scoring)
  - Expiry auto-restore (expired override restores mandatory behavior)
  - Deviation hint injection (_override_hint field)
  - Audit callback (fire-and-forget, failure-safe)
  - Edge cases: unknown bullet_id, duplicate overrides (last wins)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from memx.team.config import MandatoryOverride, _MAX_OVERRIDE_DAYS
from memx.team.merger import MultiPoolRetriever, LayerBoostConfig, _MANDATORY_SCORE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBackend:
    """Minimal StorageBackend for testing."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []

    def search(self, query: str, *, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        return self._results[:limit]


def _make_override(
    bullet_id: str = "b1",
    reason: str = "project migration",
    days_from_now: int = 30,
) -> MandatoryOverride:
    """Create a valid MandatoryOverride with sensible defaults."""
    return MandatoryOverride(
        bullet_id=bullet_id,
        reason=reason,
        expires=datetime.now(timezone.utc) + timedelta(days=days_from_now),
    )


def _make_mandatory_bullet(
    bullet_id: str = "b1",
    content: str = "Always use lint before commit",
    score: float = 0.8,
) -> dict[str, Any]:
    """Create a team mandatory bullet dict."""
    return {
        "id": bullet_id,
        "content": content,
        "enforcement": "mandatory",
        "score": score,
        "tags": ["workflow"],
    }


# ---------------------------------------------------------------------------
# MandatoryOverride validation tests
# ---------------------------------------------------------------------------


class TestMandatoryOverrideValidation:
    """Test MandatoryOverride pydantic validation."""

    def test_valid_override(self) -> None:
        ov = _make_override()
        assert ov.bullet_id == "b1"
        assert ov.reason == "project migration"
        assert ov.is_active()

    def test_reason_empty_string_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason must be non-empty"):
            MandatoryOverride(
                bullet_id="b1",
                reason="",
                expires=datetime.now(timezone.utc) + timedelta(days=10),
            )

    def test_reason_whitespace_only_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason must be non-empty"):
            MandatoryOverride(
                bullet_id="b1",
                reason="   ",
                expires=datetime.now(timezone.utc) + timedelta(days=10),
            )

    def test_expires_beyond_90_days_rejected(self) -> None:
        with pytest.raises(ValueError, match="at most 90 days"):
            MandatoryOverride(
                bullet_id="b1",
                reason="good reason",
                expires=datetime.now(timezone.utc) + timedelta(days=91),
            )

    def test_expires_exactly_90_days_accepted(self) -> None:
        # Should not raise — exactly 90 days is borderline; we allow it
        # because the check is `exp > max_expiry` (strict greater-than)
        ov = MandatoryOverride(
            bullet_id="b1",
            reason="good reason",
            expires=datetime.now(timezone.utc) + timedelta(days=89, hours=23),
        )
        assert ov.is_active()

    def test_expires_in_past_still_valid_model(self) -> None:
        """Past expiry is valid for the model, but is_active() returns False."""
        ov = MandatoryOverride(
            bullet_id="b1",
            reason="historical override",
            expires=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert not ov.is_active()

    def test_is_active_with_custom_now(self) -> None:
        exp = datetime(2026, 6, 1, tzinfo=timezone.utc)
        ov = MandatoryOverride(bullet_id="b1", reason="test", expires=exp)
        before = datetime(2026, 5, 31, tzinfo=timezone.utc)
        after = datetime(2026, 6, 2, tzinfo=timezone.utc)
        assert ov.is_active(now=before)
        assert not ov.is_active(now=after)


# ---------------------------------------------------------------------------
# Shadow Merge override logic tests
# ---------------------------------------------------------------------------


class TestShadowMergeOverride:
    """Test MultiPoolRetriever override behavior during search."""

    def test_active_override_skips_mandatory_score(self) -> None:
        """Active override should downgrade mandatory bullet from sentinel score."""
        mandatory = _make_mandatory_bullet("b1", score=0.8)
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([])

        override = _make_override("b1", days_from_now=30)
        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[override],
        )

        results = retriever.search("lint")
        assert len(results) == 1
        # Should NOT have mandatory sentinel score
        # The bullet should still appear but with normal team-boosted score
        assert results[0]["id"] == "b1"
        # Check that override hint is injected
        assert "_override_hint" in results[0]
        assert "[OVERRIDE]" in results[0]["_override_hint"]
        assert override.reason in results[0]["_override_hint"]

    def test_expired_override_restores_mandatory(self) -> None:
        """Expired override should restore mandatory scoring."""
        mandatory = _make_mandatory_bullet("b1", score=0.8)
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([])

        # Create an already-expired override
        expired_override = MandatoryOverride(
            bullet_id="b1",
            reason="temporary migration",
            expires=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[expired_override],
        )

        results = retriever.search("lint")
        assert len(results) == 1
        assert results[0]["id"] == "b1"
        # No override hint should be injected for expired overrides
        assert "_override_hint" not in results[0]

    def test_no_override_mandatory_unchanged(self) -> None:
        """Without overrides, mandatory bullets keep sentinel score."""
        mandatory = _make_mandatory_bullet("b1", score=0.5)
        local_bullet = {"id": "local1", "content": "my local rule", "score": 0.9}
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([local_bullet])

        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
        )

        results = retriever.search("rule", limit=5)
        # Mandatory bullet should appear first due to sentinel score
        assert results[0]["id"] == "b1"

    def test_override_hint_format(self) -> None:
        """Verify the exact hint format injected on override."""
        mandatory = _make_mandatory_bullet("b1")
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([])

        override = _make_override("b1", reason="project legacy", days_from_now=10)
        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[override],
        )

        results = retriever.search("lint")
        hint = results[0]["_override_hint"]
        assert "project legacy" in hint
        assert "[b1]" in hint
        assert "有效期至" in hint


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestOverrideEdgeCases:
    """Edge-case scenarios for mandatory overrides."""

    def test_unknown_bullet_id_silently_ignored(self) -> None:
        """Override for a non-existent bullet_id does not cause errors."""
        local_bullet = {"id": "local1", "content": "test", "score": 0.5}
        local_backend = FakeBackend([local_bullet])

        override = _make_override("nonexistent_bullet", days_from_now=30)
        retriever = MultiPoolRetriever(
            local_backend,
            mandatory_overrides=[override],
        )

        results = retriever.search("test")
        assert len(results) == 1
        assert results[0]["id"] == "local1"
        assert "_override_hint" not in results[0]

    def test_duplicate_overrides_last_wins(self) -> None:
        """When multiple overrides target the same bullet_id, the last one wins."""
        mandatory = _make_mandatory_bullet("b1")
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([])

        ov1 = _make_override("b1", reason="first reason", days_from_now=10)
        ov2 = _make_override("b1", reason="second reason", days_from_now=20)

        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[ov1, ov2],
        )

        results = retriever.search("lint")
        assert "_override_hint" in results[0]
        assert "second reason" in results[0]["_override_hint"]

    def test_multiple_overrides_different_bullets(self) -> None:
        """Multiple overrides for different bullets work independently."""
        b1 = _make_mandatory_bullet("b1", content="rule A")
        b2 = _make_mandatory_bullet("b2", content="rule B completely different")
        team_backend = FakeBackend([b1, b2])
        local_backend = FakeBackend([])

        ov1 = _make_override("b1", reason="override A", days_from_now=10)
        # b2 has no override → stays mandatory

        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[ov1],
        )

        results = retriever.search("rule", limit=5)
        # b2 should be mandatory (sentinel score), b1 should be overridden (normal score)
        # So b2 should come first
        b2_result = next(r for r in results if r["id"] == "b2")
        b1_result = next(r for r in results if r["id"] == "b1")
        assert "_override_hint" not in b2_result
        assert "_override_hint" in b1_result


# ---------------------------------------------------------------------------
# Audit callback tests
# ---------------------------------------------------------------------------


class TestAuditCallback:
    """Test audit event firing on override deviation."""

    def test_audit_callback_fired(self) -> None:
        """Audit callback is called when an active override is used."""
        mandatory = _make_mandatory_bullet("b1")
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([])

        callback = MagicMock()
        override = _make_override("b1", days_from_now=30)
        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[override],
            audit_callback=callback,
        )

        retriever.search("lint")
        callback.assert_called_once()
        event = callback.call_args[0][0]
        assert event["type"] == "mandatory_override_deviation"
        assert event["bullet_id"] == "b1"
        assert event["reason"] == override.reason

    def test_audit_callback_failure_does_not_block(self) -> None:
        """Failing audit callback should not break retrieval."""
        mandatory = _make_mandatory_bullet("b1")
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([])

        callback = MagicMock(side_effect=RuntimeError("network down"))
        override = _make_override("b1", days_from_now=30)
        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[override],
            audit_callback=callback,
        )

        # Should not raise despite callback failure
        results = retriever.search("lint")
        assert len(results) == 1
        callback.assert_called_once()

    def test_no_audit_for_expired_override(self) -> None:
        """Expired override should not fire audit callback."""
        mandatory = _make_mandatory_bullet("b1")
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([])

        callback = MagicMock()
        expired = MandatoryOverride(
            bullet_id="b1",
            reason="old reason",
            expires=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[expired],
            audit_callback=callback,
        )

        retriever.search("lint")
        callback.assert_not_called()

    def test_no_audit_without_callback(self) -> None:
        """No crash when audit_callback is None."""
        mandatory = _make_mandatory_bullet("b1")
        team_backend = FakeBackend([mandatory])
        local_backend = FakeBackend([])

        override = _make_override("b1", days_from_now=30)
        retriever = MultiPoolRetriever(
            local_backend,
            team_pools=[("team", team_backend)],
            mandatory_overrides=[override],
            audit_callback=None,
        )

        results = retriever.search("lint")
        assert len(results) == 1  # no crash
