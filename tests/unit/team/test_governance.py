"""Tests for STORY-069: Three-tier governance logic.

Covers:
  - GovernanceClassifier tier assignment
  - Voting (upvote/downvote) score adjustments
  - Auto-approve weight multiplier
  - Backlog threshold monitoring
  - Timeout rejection
  - CLI vote commands
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from memx.team.cache_storage import (
    TeamCacheStorage,
    _AUTO_APPROVE_WEIGHT,
    _BACKLOG_MAX_PENDING_DAYS,
    _BACKLOG_MAX_STAGING,
    _DOWNVOTE_DELTA,
    _TIMEOUT_REJECT_DAYS,
    _UPVOTE_DELTA,
)
from memx.team.cli import team_group
from memx.team.config import TeamConfig
from memx.team.nominator import GovernanceClassifier
from memx.team.types import GovernanceTier, TeamBullet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier():
    return GovernanceClassifier()


@pytest.fixture
def team_config(tmp_path):
    return TeamConfig(
        enabled=True,
        team_id="test-gov",
        cache_max_bullets=100,
    )


@pytest.fixture
def cache(team_config, tmp_path, monkeypatch):
    """Create a TeamCacheStorage that writes to tmp_path instead of ~/.ace."""
    monkeypatch.setattr(
        "memx.team.cache_storage.TeamCacheStorage._load", lambda self: None
    )
    storage = TeamCacheStorage(team_config)
    storage._cache_dir = tmp_path
    storage._cache_file = tmp_path / "bullets.json"
    return storage


def _make_bullet(
    origin_id: str = "b1",
    score: float = 50.0,
    tags: list[str] | None = None,
    status: str = "staging",
    nominated_at: datetime | None = None,
    governance_tier: str = GovernanceTier.P2P_REVIEW.value,
) -> TeamBullet:
    """Helper to create a TeamBullet with test defaults."""
    return TeamBullet(
        content=f"test bullet {origin_id}",
        origin_id=origin_id,
        instructivity_score=score,
        tags=tags or [],
        status=status,
        nominated_at=nominated_at,
        governance_tier=governance_tier,
    )


# ---------------------------------------------------------------------------
# GovernanceClassifier tests
# ---------------------------------------------------------------------------


class TestGovernanceClassifier:
    """Test three-tier classification logic."""

    def test_auto_approve_high_score_no_sensitive_tags(self, classifier):
        tier = classifier.classify(95.0, ["python", "debugging"])
        assert tier == GovernanceTier.AUTO_APPROVE

    def test_auto_approve_exact_threshold(self, classifier):
        tier = classifier.classify(90.0, [])
        assert tier == GovernanceTier.AUTO_APPROVE

    def test_p2p_review_moderate_score(self, classifier):
        tier = classifier.classify(70.0, ["python"])
        assert tier == GovernanceTier.P2P_REVIEW

    def test_p2p_review_low_score_no_tags(self, classifier):
        tier = classifier.classify(30.0, [])
        assert tier == GovernanceTier.P2P_REVIEW

    def test_curator_required_security_tag(self, classifier):
        tier = classifier.classify(95.0, ["security"])
        assert tier == GovernanceTier.CURATOR_REQUIRED

    def test_curator_required_architecture_tag(self, classifier):
        tier = classifier.classify(50.0, ["architecture", "python"])
        assert tier == GovernanceTier.CURATOR_REQUIRED

    def test_curator_required_mandatory_tag(self, classifier):
        tier = classifier.classify(99.0, ["mandatory"])
        assert tier == GovernanceTier.CURATOR_REQUIRED

    def test_sensitive_tag_case_insensitive(self, classifier):
        tier = classifier.classify(95.0, ["Security"])
        assert tier == GovernanceTier.CURATOR_REQUIRED

    def test_curator_overrides_auto_approve(self, classifier):
        """Sensitive tags take priority even with high score."""
        tier = classifier.classify(100.0, ["architecture"])
        assert tier == GovernanceTier.CURATOR_REQUIRED

    def test_custom_sensitive_tags(self):
        custom = GovernanceClassifier(sensitive_tags=frozenset({"finance"}))
        assert custom.classify(95.0, ["finance"]) == GovernanceTier.CURATOR_REQUIRED
        assert custom.classify(95.0, ["security"]) == GovernanceTier.AUTO_APPROVE

    def test_custom_threshold(self):
        custom = GovernanceClassifier(auto_approve_threshold=80.0)
        assert custom.classify(85.0, []) == GovernanceTier.AUTO_APPROVE
        assert custom.classify(75.0, []) == GovernanceTier.P2P_REVIEW

    def test_zero_score(self, classifier):
        tier = classifier.classify(0.0, [])
        assert tier == GovernanceTier.P2P_REVIEW

    def test_empty_tags(self, classifier):
        tier = classifier.classify(95.0, [])
        assert tier == GovernanceTier.AUTO_APPROVE


# ---------------------------------------------------------------------------
# Voting tests
# ---------------------------------------------------------------------------


class TestVoting:
    """Test upvote/downvote score adjustments on TeamCacheStorage."""

    def test_upvote_increases_score(self, cache):
        bullet = _make_bullet("v1", score=50.0)
        cache._bullets["v1"] = bullet
        original_score = bullet.effective_score

        result = cache.vote_bullet("v1", upvote=True)

        assert result is not None
        assert result.upvotes == _UPVOTE_DELTA
        assert result.effective_score == original_score + _UPVOTE_DELTA

    def test_downvote_decreases_score(self, cache):
        bullet = _make_bullet("v2", score=50.0)
        cache._bullets["v2"] = bullet
        original_score = bullet.effective_score

        result = cache.vote_bullet("v2", upvote=False)

        assert result is not None
        assert result.downvotes == _DOWNVOTE_DELTA
        assert result.effective_score == original_score - _DOWNVOTE_DELTA

    def test_multiple_upvotes(self, cache):
        bullet = _make_bullet("v3", score=50.0)
        cache._bullets["v3"] = bullet

        cache.vote_bullet("v3", upvote=True)
        cache.vote_bullet("v3", upvote=True)
        result = cache.vote_bullet("v3", upvote=True)

        assert result is not None
        assert result.upvotes == 3 * _UPVOTE_DELTA
        assert result.effective_score == 50.0 + 3 * _UPVOTE_DELTA

    def test_vote_nonexistent_bullet(self, cache):
        result = cache.vote_bullet("nonexistent", upvote=True)
        assert result is None

    def test_effective_score_bounded(self, cache):
        """Score should not go below 0 or above 100."""
        bullet = _make_bullet("v4", score=5.0)
        cache._bullets["v4"] = bullet

        # Multiple downvotes to push below 0
        for _ in range(3):
            cache.vote_bullet("v4", upvote=False)

        assert cache._bullets["v4"].effective_score == 0.0

    def test_upvote_score_capped_at_100(self, cache):
        bullet = _make_bullet("v5", score=90.0)
        cache._bullets["v5"] = bullet

        # Two upvotes of +5 each = +10, 90+10=100 (capped)
        cache.vote_bullet("v5", upvote=True)
        cache.vote_bullet("v5", upvote=True)
        cache.vote_bullet("v5", upvote=True)

        assert cache._bullets["v5"].effective_score == 100.0


# ---------------------------------------------------------------------------
# Auto-approve weight tests
# ---------------------------------------------------------------------------


class TestAutoApproveWeight:
    """Test auto_approve effective_score multiplier."""

    def test_auto_approve_weight_applied(self, cache):
        bullet = _make_bullet(
            "a1",
            score=90.0,
            governance_tier=GovernanceTier.AUTO_APPROVE.value,
        )
        cache._bullets["a1"] = bullet
        original = bullet.effective_score

        result = cache.apply_auto_approve_weight("a1")

        assert result is not None
        # After weight, effective score should be approximately halved
        assert result.effective_score <= original * _AUTO_APPROVE_WEIGHT + 1

    def test_non_auto_approve_unchanged(self, cache):
        bullet = _make_bullet(
            "a2",
            score=70.0,
            governance_tier=GovernanceTier.P2P_REVIEW.value,
        )
        cache._bullets["a2"] = bullet
        original = bullet.effective_score

        result = cache.apply_auto_approve_weight("a2")

        assert result is not None
        assert result.effective_score == original

    def test_weight_nonexistent_bullet(self, cache):
        result = cache.apply_auto_approve_weight("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# Backlog monitoring tests
# ---------------------------------------------------------------------------


class TestBacklogMonitoring:
    """Test staging backlog threshold alerts."""

    def test_no_staging_no_alert(self, cache):
        bullet = _make_bullet("ok1", status="approved")
        cache._bullets["ok1"] = bullet

        info = cache.check_backlog()

        assert info["staging_count"] == 0
        assert not info["needs_attention"]

    def test_staging_overflow(self, cache):
        now = datetime.now(timezone.utc)
        for i in range(_BACKLOG_MAX_STAGING + 1):
            cache._bullets[f"s{i}"] = _make_bullet(
                f"s{i}", status="staging", nominated_at=now
            )

        info = cache.check_backlog()

        assert info["staging_count"] == _BACKLOG_MAX_STAGING + 1
        assert info["staging_overflow"]
        assert info["needs_attention"]

    def test_oldest_pending_overflow(self, cache):
        old_date = datetime.now(timezone.utc) - timedelta(
            days=_BACKLOG_MAX_PENDING_DAYS + 1
        )
        cache._bullets["old1"] = _make_bullet(
            "old1", status="staging", nominated_at=old_date
        )

        info = cache.check_backlog()

        assert info["oldest_pending_overflow"]
        assert info["needs_attention"]

    def test_within_thresholds(self, cache):
        now = datetime.now(timezone.utc)
        for i in range(5):
            cache._bullets[f"ok{i}"] = _make_bullet(
                f"ok{i}", status="staging", nominated_at=now
            )

        info = cache.check_backlog()

        assert not info["staging_overflow"]
        assert not info["oldest_pending_overflow"]
        assert not info["needs_attention"]

    def test_staging_without_nominated_at(self, cache):
        """Bullets without nominated_at should not trigger oldest overflow."""
        cache._bullets["nodate"] = _make_bullet("nodate", status="staging")

        info = cache.check_backlog()

        assert info["staging_count"] == 1
        assert info["oldest_pending_days"] == 0.0
        assert not info["oldest_pending_overflow"]


# ---------------------------------------------------------------------------
# Timeout rejection tests
# ---------------------------------------------------------------------------


class TestTimeoutRejection:
    """Test auto-rejection of timed-out staging bullets."""

    def test_reject_old_staging(self, cache):
        old = datetime.now(timezone.utc) - timedelta(days=_TIMEOUT_REJECT_DAYS + 1)
        cache._bullets["t1"] = _make_bullet("t1", status="staging", nominated_at=old)

        rejected = cache.reject_timed_out()

        assert "t1" in rejected
        assert cache._bullets["t1"].status == "rejected"

    def test_keep_recent_staging(self, cache):
        recent = datetime.now(timezone.utc) - timedelta(days=5)
        cache._bullets["t2"] = _make_bullet("t2", status="staging", nominated_at=recent)

        rejected = cache.reject_timed_out()

        assert len(rejected) == 0
        assert cache._bullets["t2"].status == "staging"

    def test_only_staging_affected(self, cache):
        old = datetime.now(timezone.utc) - timedelta(days=_TIMEOUT_REJECT_DAYS + 1)
        cache._bullets["t3"] = _make_bullet("t3", status="approved", nominated_at=old)

        rejected = cache.reject_timed_out()

        assert len(rejected) == 0
        assert cache._bullets["t3"].status == "approved"

    def test_custom_timeout(self, cache):
        age = datetime.now(timezone.utc) - timedelta(days=10)
        cache._bullets["t4"] = _make_bullet("t4", status="staging", nominated_at=age)

        rejected = cache.reject_timed_out(timeout_days=5)

        assert "t4" in rejected
        assert cache._bullets["t4"].status == "rejected"

    def test_no_nominated_at_not_rejected(self, cache):
        """Bullets without nominated_at should not be auto-rejected."""
        cache._bullets["t5"] = _make_bullet("t5", status="staging")

        rejected = cache.reject_timed_out()

        assert len(rejected) == 0
        assert cache._bullets["t5"].status == "staging"

    def test_mixed_bullets(self, cache):
        now = datetime.now(timezone.utc)
        old = now - timedelta(days=_TIMEOUT_REJECT_DAYS + 5)
        recent = now - timedelta(days=2)

        cache._bullets["old"] = _make_bullet("old", status="staging", nominated_at=old)
        cache._bullets["new"] = _make_bullet("new", status="staging", nominated_at=recent)
        cache._bullets["approved"] = _make_bullet("approved", status="approved", nominated_at=old)

        rejected = cache.reject_timed_out()

        assert rejected == ["old"]
        assert cache._bullets["old"].status == "rejected"
        assert cache._bullets["new"].status == "staging"
        assert cache._bullets["approved"].status == "approved"


# ---------------------------------------------------------------------------
# GovernanceTier enum tests
# ---------------------------------------------------------------------------


class TestGovernanceTier:
    """Test GovernanceTier enum values."""

    def test_enum_values(self):
        assert GovernanceTier.AUTO_APPROVE.value == "auto_approve"
        assert GovernanceTier.P2P_REVIEW.value == "p2p_review"
        assert GovernanceTier.CURATOR_REQUIRED.value == "curator_required"

    def test_team_bullet_default_tier(self):
        bullet = TeamBullet(content="test")
        assert bullet.governance_tier == GovernanceTier.P2P_REVIEW.value

    def test_team_bullet_custom_tier(self):
        bullet = TeamBullet(
            content="test",
            governance_tier=GovernanceTier.AUTO_APPROVE.value,
        )
        assert bullet.governance_tier == "auto_approve"

    def test_team_bullet_nominated_at(self):
        now = datetime.now(timezone.utc)
        bullet = TeamBullet(content="test", nominated_at=now)
        assert bullet.nominated_at == now


# ---------------------------------------------------------------------------
# CLI vote command tests
# ---------------------------------------------------------------------------


class TestCLIVoteCommands:
    """Test upvote/downvote CLI commands via click CliRunner."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_upvote_not_found(self, runner):
        """Upvote on nonexistent bullet shows error."""
        with patch("memx.team.cli._ensure_team_enabled") as mock_enabled:
            mock_enabled.return_value = (
                TeamConfig(enabled=True, team_id="test"),
                None,
            )
            with patch("memx.team.cache_storage.TeamCacheStorage._load"):
                result = runner.invoke(team_group, ["upvote", "nonexistent"])
                assert result.exit_code != 0

    def test_downvote_not_found(self, runner):
        """Downvote on nonexistent bullet shows error."""
        with patch("memx.team.cli._ensure_team_enabled") as mock_enabled:
            mock_enabled.return_value = (
                TeamConfig(enabled=True, team_id="test"),
                None,
            )
            with patch("memx.team.cache_storage.TeamCacheStorage._load"):
                result = runner.invoke(team_group, ["downvote", "nonexistent"])
                assert result.exit_code != 0

    def test_upvote_team_not_enabled(self, runner):
        """Error when team features disabled."""
        with patch("memx.team.cli._ensure_team_enabled") as mock_enabled:
            mock_enabled.return_value = (None, "Team not enabled")
            result = runner.invoke(team_group, ["upvote", "some-id"])
            assert result.exit_code != 0

    def test_upvote_json_output(self, runner):
        """JSON output for upvote on missing bullet."""
        with patch("memx.team.cli._ensure_team_enabled") as mock_enabled:
            mock_enabled.return_value = (
                TeamConfig(enabled=True, team_id="test"),
                None,
            )
            with patch("memx.team.cache_storage.TeamCacheStorage._load"):
                result = runner.invoke(
                    team_group, ["upvote", "nonexistent", "--json"]
                )
                data = json.loads(result.output)
                assert "error" in data

    def test_backlog_command(self, runner):
        """Backlog command runs without error when team enabled."""
        with patch("memx.team.cli._ensure_team_enabled") as mock_enabled:
            mock_enabled.return_value = (
                TeamConfig(enabled=True, team_id="test"),
                None,
            )
            with patch("memx.team.cache_storage.TeamCacheStorage._load"):
                result = runner.invoke(team_group, ["backlog"])
                assert "Staging bullets:" in result.output

    def test_backlog_json(self, runner):
        """Backlog JSON output contains expected fields."""
        with patch("memx.team.cli._ensure_team_enabled") as mock_enabled:
            mock_enabled.return_value = (
                TeamConfig(enabled=True, team_id="test"),
                None,
            )
            with patch("memx.team.cache_storage.TeamCacheStorage._load"):
                result = runner.invoke(team_group, ["backlog", "--json"])
                data = json.loads(result.output)
                assert "staging_count" in data
                assert "needs_attention" in data
