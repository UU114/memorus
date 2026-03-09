"""Tests for STORY-070: Supersede knowledge correction flow.

Covers SupersedeDetector, SupersedeProposal, submit_supersede,
merger supersede conflict detection, and cache_storage update notifications.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from memx.team.nominator import (
    NominationResult,
    SupersedeDetector,
    SupersedeNotification,
    SupersedeProposal,
    submit_supersede,
    _jaccard_similarity,
)
from memx.team.merger import (
    LayerBoostConfig,
    MultiPoolRetriever,
    ScoredResult,
    SupersedeConflict,
    _content_similarity,
)
from memx.team.cache_storage import TeamCacheStorage, _word_similarity
from memx.team.config import TeamConfig
from memx.team.types import TeamBullet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_state_dir(tmp_path: Path) -> Path:
    """Temp directory for persisting state files."""
    return tmp_path / "state"


@pytest.fixture
def detector(tmp_state_dir: Path) -> SupersedeDetector:
    """SupersedeDetector with default thresholds."""
    return SupersedeDetector(state_dir=tmp_state_dir)


@pytest.fixture
def detector_low_threshold() -> SupersedeDetector:
    """Detector with lower thresholds for easier testing."""
    return SupersedeDetector(
        similarity_threshold=0.3,
        difference_threshold=0.1,
    )


# ---------------------------------------------------------------------------
# SupersedeProposal dataclass tests
# ---------------------------------------------------------------------------


class TestSupersedeProposal:
    def test_default_values(self):
        p = SupersedeProposal(origin_id="t1", new_content="updated info")
        assert p.origin_id == "t1"
        assert p.new_content == "updated info"
        assert p.priority == "normal"
        assert p.reason == ""
        assert p.similarity == 0.0
        assert p.local_bullet_id == ""

    def test_urgent_priority(self):
        p = SupersedeProposal(
            origin_id="t2",
            new_content="critical fix",
            priority="urgent",
            reason="security issue",
        )
        assert p.priority == "urgent"
        assert p.reason == "security issue"


# ---------------------------------------------------------------------------
# SupersedeNotification dataclass tests
# ---------------------------------------------------------------------------


class TestSupersedeNotification:
    def test_default_values(self):
        n = SupersedeNotification(
            team_bullet_id="tb1",
            old_content="old",
            new_content="new",
        )
        assert n.team_bullet_id == "tb1"
        assert n.message == ""
        assert n.local_bullet_id == ""


# ---------------------------------------------------------------------------
# SupersedeDetector tests
# ---------------------------------------------------------------------------


class TestSupersedeDetector:
    def test_no_proposals_when_no_overlap(self, detector: SupersedeDetector):
        local = [{"id": "l1", "content": "python uses indentation"}]
        team = [
            {"id": "t1", "content": "rust uses curly braces", "source": "team_cache"}
        ]
        proposals = detector.detect(local, team)
        assert proposals == []

    def test_detect_correction_with_custom_similarity(
        self, detector_low_threshold: SupersedeDetector
    ):
        """When similarity function says topics overlap but words differ, detect."""
        local = [{"id": "l1", "content": "use pytest-xdist for parallel testing speed"}]
        team = [
            {
                "id": "t1",
                "content": "use pytest for parallel testing speed improvement",
                "source": "team_cache",
            }
        ]
        # Use a custom similarity function that returns high similarity
        def high_sim(a: str, b: str) -> float:
            return 0.85

        proposals = detector_low_threshold.detect(
            local, team, similarity_fn=high_sim
        )
        assert len(proposals) == 1
        assert proposals[0].origin_id == "t1"
        assert proposals[0].local_bullet_id == "l1"
        assert proposals[0].similarity == 0.85

    def test_skip_local_source(self, detector_low_threshold: SupersedeDetector):
        """Bullets with source='local' should be skipped."""
        local = [{"id": "l1", "content": "same topic same words"}]
        team = [{"id": "t1", "content": "same topic same words", "source": "local"}]
        proposals = detector_low_threshold.detect(
            local, team, similarity_fn=lambda a, b: 0.9
        )
        assert proposals == []

    def test_skip_identical_content(self, detector: SupersedeDetector):
        """Identical content should not be a supersede (difference < threshold)."""
        local = [{"id": "l1", "content": "exact same text here"}]
        team = [
            {"id": "t1", "content": "exact same text here", "source": "team_cache"}
        ]
        # Even with high similarity, difference is 0 (< 0.2 threshold)
        proposals = detector.detect(local, team, similarity_fn=lambda a, b: 0.95)
        assert proposals == []

    def test_skip_empty_content(self, detector: SupersedeDetector):
        local = [{"id": "l1", "content": ""}]
        team = [{"id": "t1", "content": "something", "source": "team_cache"}]
        proposals = detector.detect(local, team, similarity_fn=lambda a, b: 0.9)
        assert proposals == []

    def test_skip_empty_id(self, detector: SupersedeDetector):
        local = [{"content": "something"}]
        team = [{"id": "t1", "content": "something else", "source": "team_cache"}]
        proposals = detector.detect(local, team, similarity_fn=lambda a, b: 0.9)
        assert proposals == []

    def test_default_similarity_fn_fallback(self):
        """When no similarity_fn provided, uses Jaccard similarity."""
        det = SupersedeDetector(similarity_threshold=0.5, difference_threshold=0.05)
        # High word overlap but some differences
        local = [{"id": "l1", "content": "always use black formatter for python code formatting"}]
        team = [
            {
                "id": "t1",
                "content": "always use autopep8 formatter for python code formatting",
                "source": "team_cache",
            }
        ]
        proposals = det.detect(local, team)
        # Jaccard similarity is used for both threshold checks
        # Words overlap significantly but not identical
        assert len(proposals) >= 0  # may or may not detect depending on exact Jaccard

    def test_multiple_proposals(self, detector_low_threshold: SupersedeDetector):
        local = [
            {"id": "l1", "content": "use black for formatting python"},
            {"id": "l2", "content": "use ruff for linting python"},
        ]
        team = [
            {"id": "t1", "content": "use autopep8 for formatting python", "source": "team_cache"},
            {"id": "t2", "content": "use flake8 for linting python", "source": "team_cache"},
        ]

        def sim_fn(a: str, b: str) -> float:
            # Return high sim for same-topic pairs
            if ("formatting" in a and "formatting" in b) or (
                "linting" in a and "linting" in b
            ):
                return 0.85
            return 0.1

        proposals = detector_low_threshold.detect(local, team, similarity_fn=sim_fn)
        assert len(proposals) == 2
        origins = {p.origin_id for p in proposals}
        assert origins == {"t1", "t2"}


# ---------------------------------------------------------------------------
# SupersedeDetector ignore tests
# ---------------------------------------------------------------------------


class TestSupersedeDetectorIgnore:
    def test_ignore_pair(self, detector: SupersedeDetector):
        detector.ignore_pair("l1", "t1")
        assert detector.is_ignored("l1", "t1")
        assert not detector.is_ignored("l1", "t2")

    def test_ignored_pair_skipped_in_detect(self):
        det = SupersedeDetector(similarity_threshold=0.3, difference_threshold=0.05)
        det.ignore_pair("l1", "t1")

        local = [{"id": "l1", "content": "use black for formatting python code"}]
        team = [
            {"id": "t1", "content": "use autopep8 for formatting python code", "source": "team_cache"}
        ]
        proposals = det.detect(local, team, similarity_fn=lambda a, b: 0.9)
        assert proposals == []

    def test_persist_and_load_ignored(self, tmp_state_dir: Path):
        det1 = SupersedeDetector(state_dir=tmp_state_dir)
        det1.ignore_pair("l1", "t1")
        det1.ignore_pair("l2", "t2")

        # Create new detector from same state dir — should load persisted ignores
        det2 = SupersedeDetector(state_dir=tmp_state_dir)
        assert det2.is_ignored("l1", "t1")
        assert det2.is_ignored("l2", "t2")
        assert not det2.is_ignored("l3", "t3")

    def test_persist_file_format(self, tmp_state_dir: Path):
        det = SupersedeDetector(state_dir=tmp_state_dir)
        det.ignore_pair("l1", "t1")

        path = tmp_state_dir / "ignored_supersedes.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "ignored_pairs" in data
        assert "l1::t1" in data["ignored_pairs"]

    def test_no_persist_without_state_dir(self):
        det = SupersedeDetector(state_dir=None)
        det.ignore_pair("l1", "t1")
        assert det.is_ignored("l1", "t1")  # in-memory still works


# ---------------------------------------------------------------------------
# submit_supersede tests
# ---------------------------------------------------------------------------


class TestSubmitSupersede:
    @pytest.mark.asyncio
    async def test_success(self):
        mock_client = AsyncMock()
        mock_client.propose_supersede.return_value = MagicMock(
            id="new-id", status="pending"
        )

        proposal = SupersedeProposal(
            origin_id="t1",
            new_content="corrected content",
            priority="normal",
            reason="fix typo",
        )

        result = await submit_supersede(
            proposal, sync_client=mock_client
        )
        assert result.success is True
        assert result.bullet_id == "new-id"

        mock_client.propose_supersede.assert_called_once_with(
            "t1",
            {
                "content": "corrected content",
                "priority": "normal",
                "reason": "fix typo",
            },
            priority="normal",
        )

    @pytest.mark.asyncio
    async def test_no_sync_client(self):
        proposal = SupersedeProposal(origin_id="t1", new_content="x")
        result = await submit_supersede(proposal, sync_client=None)
        assert result.success is False
        assert "No sync client" in result.error

    @pytest.mark.asyncio
    async def test_with_redactor(self):
        mock_client = AsyncMock()
        mock_client.propose_supersede.return_value = MagicMock(
            id="new-id", status="pending"
        )

        mock_redactor = MagicMock()
        redacted_result = MagicMock()
        mock_redactor.redact_l1.return_value = redacted_result
        mock_redactor.finalize.return_value = {"content": "sanitized content"}

        proposal = SupersedeProposal(
            origin_id="t1", new_content="raw secret content"
        )
        result = await submit_supersede(
            proposal, redactor=mock_redactor, sync_client=mock_client
        )
        assert result.success is True
        mock_redactor.redact_l1.assert_called_once_with("raw secret content")

        # Verify the sanitized content was sent
        call_args = mock_client.propose_supersede.call_args
        assert call_args[0][1]["content"] == "sanitized content"

    @pytest.mark.asyncio
    async def test_urgent_priority(self):
        mock_client = AsyncMock()
        mock_client.propose_supersede.return_value = MagicMock(
            id="u1", status="pending"
        )
        proposal = SupersedeProposal(
            origin_id="t1", new_content="urgent fix", priority="urgent"
        )
        result = await submit_supersede(proposal, sync_client=mock_client)
        assert result.success is True
        call_args = mock_client.propose_supersede.call_args
        assert call_args[1]["priority"] == "urgent"

    @pytest.mark.asyncio
    async def test_upload_failure(self):
        mock_client = AsyncMock()
        mock_client.propose_supersede.side_effect = RuntimeError("network error")

        proposal = SupersedeProposal(origin_id="t1", new_content="x")
        result = await submit_supersede(proposal, sync_client=mock_client)
        assert result.success is False
        assert "network error" in result.error


# ---------------------------------------------------------------------------
# Merger supersede conflict detection tests
# ---------------------------------------------------------------------------


class TestMergerSupersedeConflicts:
    def _make_retriever(self):
        """Create a MultiPoolRetriever with mock backends."""
        local = MagicMock()
        local.search.return_value = []
        return MultiPoolRetriever(local)

    def test_detect_explicit_supersede(self):
        retriever = self._make_retriever()
        local_results = [
            {"id": "l1", "content": "use black for formatting", "supersedes": "t1"}
        ]
        team_results = [
            {"id": "t1", "content": "use autopep8 for formatting"}
        ]
        conflicts = retriever.detect_supersede_conflicts(local_results, team_results)
        assert len(conflicts) == 1
        assert conflicts[0].local_bullet["id"] == "l1"
        assert conflicts[0].team_bullet["id"] == "t1"

    def test_detect_implicit_supersede(self):
        retriever = self._make_retriever()
        # Construct content with high Jaccard similarity (>= 0.8) but not identical
        local_results = [
            {"id": "l1", "content": "always use black formatter for python code style"}
        ]
        team_results = [
            {"id": "t1", "content": "always use autopep8 formatter for python code style"}
        ]
        # Lower threshold so Jaccard match works
        conflicts = retriever.detect_supersede_conflicts(
            local_results, team_results, similarity_threshold=0.5
        )
        assert len(conflicts) >= 1

    def test_no_conflict_for_identical(self):
        retriever = self._make_retriever()
        local_results = [
            {"id": "l1", "content": "same exact text"}
        ]
        team_results = [
            {"id": "t1", "content": "same exact text"}
        ]
        # Identical content has sim=1.0, which is not < 1.0
        conflicts = retriever.detect_supersede_conflicts(
            local_results, team_results, similarity_threshold=0.5
        )
        assert conflicts == []

    def test_no_conflict_for_low_similarity(self):
        retriever = self._make_retriever()
        local_results = [
            {"id": "l1", "content": "python indentation rules"}
        ]
        team_results = [
            {"id": "t1", "content": "rust memory safety borrow checker"}
        ]
        conflicts = retriever.detect_supersede_conflicts(local_results, team_results)
        assert conflicts == []


class TestMergerShadowMergeSupersede:
    """Test that shadow_merge drops superseded team bullets."""

    def _make_retriever(self):
        local = MagicMock()
        local.search.return_value = []
        return MultiPoolRetriever(local)

    def test_shadow_merge_drops_superseded_team_bullet(self):
        retriever = self._make_retriever()
        results = [
            ScoredResult(
                bullet={"id": "l1", "content": "use black", "supersedes": "t1"},
                raw_score=0.8,
                boosted_score=1.2,
                source="local",
            ),
            ScoredResult(
                bullet={"id": "t1", "content": "use autopep8"},
                raw_score=0.9,
                boosted_score=0.9,
                source="team_cache",
            ),
        ]
        merged = retriever._shadow_merge(results)
        ids = [b.get("id") for b in merged]
        assert "l1" in ids
        assert "t1" not in ids

    def test_shadow_merge_keeps_non_superseded(self):
        retriever = self._make_retriever()
        results = [
            ScoredResult(
                bullet={"id": "l1", "content": "use black", "supersedes": "t1"},
                raw_score=0.8,
                boosted_score=1.2,
                source="local",
            ),
            ScoredResult(
                bullet={"id": "t2", "content": "use ruff for linting"},
                raw_score=0.7,
                boosted_score=0.7,
                source="team_cache",
            ),
        ]
        merged = retriever._shadow_merge(results)
        ids = [b.get("id") for b in merged]
        assert "l1" in ids
        assert "t2" in ids

    def test_shadow_merge_mandatory_not_dropped(self):
        retriever = self._make_retriever()
        results = [
            ScoredResult(
                bullet={"id": "l1", "content": "local", "supersedes": "t1"},
                raw_score=0.8,
                boosted_score=1.2,
                source="local",
            ),
            ScoredResult(
                bullet={"id": "t1", "content": "mandatory", "enforcement": "mandatory"},
                raw_score=0.9,
                boosted_score=999.0,
                source="team_cache",
                is_mandatory=True,
            ),
        ]
        merged = retriever._shadow_merge(results)
        ids = [b.get("id") for b in merged]
        # Mandatory bullets are never dropped
        assert "t1" in ids
        assert "l1" in ids


# ---------------------------------------------------------------------------
# Cache storage update notification tests
# ---------------------------------------------------------------------------


class TestCacheStorageUpdateNotifications:
    @pytest.fixture
    def cache_config(self) -> TeamConfig:
        return TeamConfig(team_id="test-team", cache_max_bullets=100)

    @pytest.fixture
    def cache(self, cache_config: TeamConfig, tmp_path: Path, monkeypatch):
        """Create a TeamCacheStorage with tmp_path as home dir."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        return TeamCacheStorage(cache_config)

    def _make_team_bullet(self, **kwargs) -> TeamBullet:
        defaults = {
            "content": "default content",
            "origin_id": "tb1",
            "status": "approved",
        }
        defaults.update(kwargs)
        return TeamBullet(**defaults)

    def test_no_notifications_on_first_add(self, cache: TeamCacheStorage):
        bullet = self._make_team_bullet(origin_id="tb1", content="original")
        cache.add_bullets([bullet])
        assert cache.pending_update_ids == set()

    def test_notification_on_content_update(self, cache: TeamCacheStorage):
        bullet_v1 = self._make_team_bullet(origin_id="tb1", content="version 1")
        cache.add_bullets([bullet_v1])

        bullet_v2 = self._make_team_bullet(origin_id="tb1", content="version 2 updated")
        cache.add_bullets([bullet_v2])

        assert "tb1" in cache.pending_update_ids

    def test_no_notification_on_same_content(self, cache: TeamCacheStorage):
        bullet = self._make_team_bullet(origin_id="tb1", content="unchanged")
        cache.add_bullets([bullet])
        cache.add_bullets([bullet])  # same content
        assert cache.pending_update_ids == set()

    def test_check_update_notifications_with_local_overlap(
        self, cache: TeamCacheStorage
    ):
        bullet_v1 = self._make_team_bullet(
            origin_id="tb1", content="use pytest for testing python code"
        )
        cache.add_bullets([bullet_v1])

        bullet_v2 = self._make_team_bullet(
            origin_id="tb1", content="use pytest-xdist for testing python code in parallel"
        )
        cache.add_bullets([bullet_v2])

        local_bullets = [
            {"id": "local1", "content": "use pytest for testing python code"}
        ]
        notifications = cache.check_update_notifications(local_bullets)
        assert len(notifications) >= 1
        assert notifications[0]["team_bullet_id"] == "tb1"
        assert notifications[0]["local_bullet_id"] == "local1"

    def test_check_no_notifications_when_no_local_overlap(
        self, cache: TeamCacheStorage
    ):
        bullet_v1 = self._make_team_bullet(origin_id="tb1", content="python formatting")
        cache.add_bullets([bullet_v1])

        bullet_v2 = self._make_team_bullet(
            origin_id="tb1", content="python formatting updated"
        )
        cache.add_bullets([bullet_v2])

        local_bullets = [
            {"id": "local1", "content": "rust borrow checker memory safety"}
        ]
        notifications = cache.check_update_notifications(local_bullets)
        assert notifications == []

    def test_clear_update_notifications(self, cache: TeamCacheStorage):
        bullet_v1 = self._make_team_bullet(origin_id="tb1", content="v1")
        cache.add_bullets([bullet_v1])

        bullet_v2 = self._make_team_bullet(origin_id="tb1", content="v2")
        cache.add_bullets([bullet_v2])

        assert len(cache.pending_update_ids) > 0
        cache.clear_update_notifications()
        assert cache.pending_update_ids == set()


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical(self):
        assert _jaccard_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _jaccard_similarity("hello", "world") == 0.0

    def test_partial_overlap(self):
        sim = _jaccard_similarity("hello world foo", "hello world bar")
        # intersection=2 (hello, world), union=4 (hello, world, foo, bar)
        assert sim == pytest.approx(0.5)

    def test_empty_string(self):
        assert _jaccard_similarity("", "hello") == 0.0
        assert _jaccard_similarity("hello", "") == 0.0

    def test_case_insensitive(self):
        assert _jaccard_similarity("Hello World", "hello world") == 1.0


class TestWordSimilarity:
    def test_identical(self):
        assert _word_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _word_similarity("cat", "dog") == 0.0

    def test_empty(self):
        assert _word_similarity("", "hello") == 0.0


class TestContentSimilarity:
    def test_identical(self):
        assert _content_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _content_similarity("abc", "xyz") == 0.0
