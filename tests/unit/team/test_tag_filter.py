"""Tests for TagSubscriptionManager — tag-based subscription filtering."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from memx.team.config import TeamConfig
from memx.team.tag_filter import TagSubscriptionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(
    tags: list[str] | None = None,
    team_id: str | None = None,
) -> TagSubscriptionManager:
    """Create a TagSubscriptionManager with given tags."""
    cfg = TeamConfig(
        subscribed_tags=tags or [],
        team_id=team_id,
    )
    return TagSubscriptionManager(cfg)


def _bullet(tags: list[str] | None = None, **extra) -> dict:
    """Create a minimal bullet dict."""
    b: dict = {"tags": tags or []}
    b.update(extra)
    return b


# ---------------------------------------------------------------------------
# get_sync_tags
# ---------------------------------------------------------------------------


class TestGetSyncTags:
    """get_sync_tags returns tags when subscribed, None otherwise."""

    def test_returns_tags_when_subscribed(self):
        mgr = _make_manager(["frontend", "react"])
        result = mgr.get_sync_tags()
        assert result == ["frontend", "react"]

    def test_returns_none_when_no_subscription(self):
        mgr = _make_manager([])
        assert mgr.get_sync_tags() is None

    def test_returns_none_for_default(self):
        mgr = _make_manager()
        assert mgr.get_sync_tags() is None


# ---------------------------------------------------------------------------
# has_tags_changed
# ---------------------------------------------------------------------------


class TestHasTagsChanged:
    """Detects additions, removals, and order-insensitive equality."""

    def test_detects_addition(self):
        mgr = _make_manager(["frontend", "react"])
        assert mgr.has_tags_changed(["frontend"]) is True

    def test_detects_removal(self):
        mgr = _make_manager(["frontend"])
        assert mgr.has_tags_changed(["frontend", "react"]) is True

    def test_same_tags_different_order(self):
        mgr = _make_manager(["react", "frontend"])
        assert mgr.has_tags_changed(["frontend", "react"]) is False

    def test_case_insensitive(self):
        mgr = _make_manager(["Frontend", "REACT"])
        assert mgr.has_tags_changed(["frontend", "react"]) is False

    def test_none_previous_with_current_tags(self):
        mgr = _make_manager(["frontend"])
        assert mgr.has_tags_changed(None) is True

    def test_none_previous_no_current_tags(self):
        mgr = _make_manager([])
        assert mgr.has_tags_changed(None) is False


# ---------------------------------------------------------------------------
# filter_bullets
# ---------------------------------------------------------------------------


class TestFilterBullets:
    """Client-side filtering by subscribed tags."""

    def test_keeps_matching_bullets(self):
        mgr = _make_manager(["frontend"])
        bullets = [
            _bullet(["frontend", "css"]),
            _bullet(["backend"]),
            _bullet(["frontend"]),
        ]
        result = mgr.filter_bullets(bullets)
        assert len(result) == 2
        assert all("frontend" in b["tags"] for b in result)

    def test_removes_non_matching(self):
        mgr = _make_manager(["frontend"])
        bullets = [_bullet(["backend"]), _bullet(["devops"])]
        result = mgr.filter_bullets(bullets)
        assert result == []

    def test_passes_all_when_no_subscription(self):
        mgr = _make_manager([])
        bullets = [_bullet(["backend"]), _bullet(["devops"])]
        result = mgr.filter_bullets(bullets)
        assert len(result) == 2

    def test_case_insensitive_matching(self):
        mgr = _make_manager(["Frontend"])
        bullets = [_bullet(["frontend"]), _bullet(["FRONTEND"])]
        result = mgr.filter_bullets(bullets)
        assert len(result) == 2

    def test_empty_bullet_tags(self):
        mgr = _make_manager(["frontend"])
        bullets = [_bullet([]), _bullet()]
        result = mgr.filter_bullets(bullets)
        assert result == []


# ---------------------------------------------------------------------------
# get_stale_bullets
# ---------------------------------------------------------------------------


class TestGetStaleBullets:
    """Finds cached bullets that no longer match new tag subscription."""

    def test_finds_old_tag_bullets(self):
        mgr = _make_manager(["frontend"])
        cache = {
            "b1": _bullet(["frontend"]),
            "b2": _bullet(["backend"]),
            "b3": _bullet(["frontend", "react"]),
        }
        stale = mgr.get_stale_bullets(cache, ["frontend"])
        assert stale == ["b2"]

    def test_returns_empty_when_no_new_tags(self):
        mgr = _make_manager([])
        cache = {
            "b1": _bullet(["frontend"]),
            "b2": _bullet(["backend"]),
        }
        stale = mgr.get_stale_bullets(cache, [])
        assert stale == []

    def test_supports_object_with_tags_attribute(self):
        mgr = _make_manager(["frontend"])
        cache = {
            "b1": SimpleNamespace(tags=["frontend"]),
            "b2": SimpleNamespace(tags=["backend"]),
        }
        stale = mgr.get_stale_bullets(cache, ["frontend"])
        assert stale == ["b2"]

    def test_case_insensitive_stale_detection(self):
        mgr = _make_manager(["frontend"])
        cache = {
            "b1": _bullet(["FRONTEND"]),
            "b2": _bullet(["Backend"]),
        }
        stale = mgr.get_stale_bullets(cache, ["frontend"])
        assert stale == ["b2"]


# ---------------------------------------------------------------------------
# Tag normalization
# ---------------------------------------------------------------------------


class TestTagNormalization:
    """Tags are normalized to lowercase with whitespace stripped."""

    def test_whitespace_stripped(self):
        mgr = _make_manager(["  frontend  ", " react "])
        assert mgr.get_sync_tags() == ["frontend", "react"]

    def test_case_normalized(self):
        mgr = _make_manager(["FRONTEND", "React"])
        assert mgr.get_sync_tags() == ["frontend", "react"]

    def test_mixed_normalization(self):
        mgr = _make_manager(["  FrontEnd  "])
        assert mgr.get_sync_tags() == ["frontend"]
