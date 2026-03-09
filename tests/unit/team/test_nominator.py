"""Unit tests for memx.team.nominator — automatic nomination pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from memx.team.config import AutoNominateConfig
from memx.team.nominator import (
    NominationResult,
    NominationSummary,
    Nominator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bullet(
    bid: str = "b1",
    content: str = "Use pytest fixtures",
    recall_count: int = 5,
    score: float = 80.0,
) -> dict:
    """Build a minimal bullet dict for testing."""
    return {
        "id": bid,
        "content": content,
        "recall_count": recall_count,
        "instructivity_score": score,
    }


def _mock_redactor() -> MagicMock:
    """Return a mock Redactor with L1 / apply_user_edits / finalize stubs."""
    redactor = MagicMock()

    # redact_l1 returns a mock RedactedResult
    redacted = MagicMock()
    redacted.clean_content = "redacted content"
    redactor.redact_l1.return_value = redacted

    # apply_user_edits returns a new mock
    edited = MagicMock()
    edited.clean_content = "user edited content"
    redactor.apply_user_edits.return_value = edited

    # finalize returns a plain dict
    redactor.finalize.return_value = {"content": "redacted content", "was_redacted": True}

    return redactor


def _mock_sync_client(server_id: str = "srv-1") -> MagicMock:
    """Return a mock AceSyncClient whose nominate_bullet resolves."""
    client = MagicMock()
    response = MagicMock()
    response.id = server_id
    response.status = "pending"
    client.nominate_bullet = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> AutoNominateConfig:
    return AutoNominateConfig(min_recall_count=3, min_score=70.0)


@pytest.fixture
def silent_config() -> AutoNominateConfig:
    return AutoNominateConfig(silent=True)


@pytest.fixture
def nominator(config: AutoNominateConfig) -> Nominator:
    return Nominator(config=config)


# ---------------------------------------------------------------------------
# 1. scan_candidates — threshold filtering
# ---------------------------------------------------------------------------


class TestScanCandidates:
    """Tests for scan_candidates filtering logic."""

    def test_filters_by_recall_and_score(self, config: AutoNominateConfig) -> None:
        """Only bullets meeting both thresholds pass."""
        nom = Nominator(config=config)
        bullets = [
            _make_bullet("b1", recall_count=5, score=80.0),  # pass
            _make_bullet("b2", recall_count=2, score=90.0),  # recall too low
            _make_bullet("b3", recall_count=5, score=60.0),  # score too low
            _make_bullet("b4", recall_count=1, score=50.0),  # both too low
        ]
        result = nom.scan_candidates(bullets)
        assert len(result) == 1
        assert result[0]["id"] == "b1"

    def test_excludes_already_nominated(self, config: AutoNominateConfig) -> None:
        """Bullets that were previously nominated are excluded."""
        nom = Nominator(config=config)
        nom._nominated_ids.add("b1")
        bullets = [
            _make_bullet("b1", recall_count=5, score=80.0),
            _make_bullet("b2", recall_count=5, score=80.0),
        ]
        result = nom.scan_candidates(bullets)
        assert len(result) == 1
        assert result[0]["id"] == "b2"

    def test_excludes_permanently_ignored(self, config: AutoNominateConfig) -> None:
        """Permanently ignored bullets are excluded."""
        nom = Nominator(config=config)
        nom._ignored_ids.add("b1")
        bullets = [_make_bullet("b1", recall_count=5, score=80.0)]
        result = nom.scan_candidates(bullets)
        assert len(result) == 0

    def test_excludes_session_skipped(self, config: AutoNominateConfig) -> None:
        """Session-skipped bullets are excluded."""
        nom = Nominator(config=config)
        nom.skip_bullet("b1")
        bullets = [_make_bullet("b1", recall_count=5, score=80.0)]
        result = nom.scan_candidates(bullets)
        assert len(result) == 0

    def test_uses_origin_id_fallback(self, config: AutoNominateConfig) -> None:
        """Falls back to origin_id when id is absent."""
        nom = Nominator(config=config)
        bullet = {"origin_id": "o1", "content": "x", "recall_count": 5, "instructivity_score": 80.0}
        result = nom.scan_candidates([bullet])
        assert len(result) == 1

    def test_boundary_values_included(self, config: AutoNominateConfig) -> None:
        """Bullets exactly at threshold are included (>= comparison)."""
        nom = Nominator(config=config)
        bullet = _make_bullet("b1", recall_count=3, score=70.0)
        result = nom.scan_candidates([bullet])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# 2. should_prompt — rate limiting and silent mode
# ---------------------------------------------------------------------------


class TestShouldPrompt:
    """Tests for prompt gating logic."""

    def test_rate_limit_respected(self, config: AutoNominateConfig) -> None:
        """After max_prompts_per_session, should_prompt returns False."""
        nom = Nominator(config=config)
        nom._candidates = [_make_bullet()]
        assert nom.should_prompt() is True

        nom._session_prompt_count = config.max_prompts_per_session
        assert nom.should_prompt() is False

    def test_silent_mode_no_prompt(self, silent_config: AutoNominateConfig) -> None:
        """Silent mode always returns False."""
        nom = Nominator(config=silent_config)
        nom._candidates = [_make_bullet()]
        assert nom.should_prompt() is False

    def test_no_candidates_no_prompt(self, config: AutoNominateConfig) -> None:
        """No candidates means no prompt."""
        nom = Nominator(config=config)
        assert nom.should_prompt() is False


# ---------------------------------------------------------------------------
# 3. nominate — pipeline tests
# ---------------------------------------------------------------------------


class TestNominate:
    """Tests for the full nomination pipeline."""

    @pytest.mark.asyncio
    async def test_nominate_pipeline(self, config: AutoNominateConfig) -> None:
        """Full pipeline: redact -> finalize -> upload."""
        redactor = _mock_redactor()
        client = _mock_sync_client("srv-42")
        nom = Nominator(config=config, redactor=redactor, sync_client=client)

        bullet = _make_bullet("b1")
        result = await nom.nominate(bullet)

        assert result.success is True
        assert result.bullet_id == "srv-42"
        redactor.redact_l1.assert_called_once_with("Use pytest fixtures")
        client.nominate_bullet.assert_awaited_once()
        assert "b1" in nom._nominated_ids

    @pytest.mark.asyncio
    async def test_nominate_with_user_edits(self, config: AutoNominateConfig) -> None:
        """User edits are applied via Redactor.apply_user_edits."""
        redactor = _mock_redactor()
        client = _mock_sync_client()
        nom = Nominator(config=config, redactor=redactor, sync_client=client)

        bullet = _make_bullet("b1")
        result = await nom.nominate(bullet, user_approved_content="my edit")

        assert result.success is True
        redactor.apply_user_edits.assert_called_once()

    @pytest.mark.asyncio
    async def test_nominate_network_error(self, config: AutoNominateConfig) -> None:
        """Network error returns NominationResult with error message."""
        redactor = _mock_redactor()
        client = MagicMock()
        client.nominate_bullet = AsyncMock(side_effect=ConnectionError("timeout"))
        nom = Nominator(config=config, redactor=redactor, sync_client=client)

        bullet = _make_bullet("b1")
        result = await nom.nominate(bullet)

        assert result.success is False
        assert "timeout" in result.error
        assert "b1" not in nom._nominated_ids

    @pytest.mark.asyncio
    async def test_nominate_no_sync_client(self, config: AutoNominateConfig) -> None:
        """Returns error when sync_client is None."""
        nom = Nominator(config=config, redactor=_mock_redactor())
        result = await nom.nominate(_make_bullet())
        assert result.success is False
        assert "sync client" in result.error.lower()

    @pytest.mark.asyncio
    async def test_nominate_no_redactor(self, config: AutoNominateConfig) -> None:
        """Returns error when redactor is None."""
        nom = Nominator(config=config, sync_client=_mock_sync_client())
        result = await nom.nominate(_make_bullet())
        assert result.success is False
        assert "redactor" in result.error.lower()

    @pytest.mark.asyncio
    async def test_nominate_increments_prompt_count(
        self, config: AutoNominateConfig
    ) -> None:
        """Each successful nomination increments session prompt count."""
        nom = Nominator(
            config=config, redactor=_mock_redactor(), sync_client=_mock_sync_client()
        )
        assert nom._session_prompt_count == 0
        await nom.nominate(_make_bullet("b1"))
        assert nom._session_prompt_count == 1


# ---------------------------------------------------------------------------
# 4. ignore / skip
# ---------------------------------------------------------------------------


class TestIgnoreSkip:
    """Tests for ignore_bullet and skip_bullet."""

    def test_ignore_persisted(self, config: AutoNominateConfig, tmp_path: Path) -> None:
        """ignore_bullet persists to state file."""
        nom = Nominator(config=config, state_dir=tmp_path)
        nom.ignore_bullet("b99")
        assert "b99" in nom._ignored_ids

        # Verify state file written
        state_file = tmp_path / "nomination_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        assert "b99" in data["ignored_ids"]

    def test_skip_session_only(self, config: AutoNominateConfig, tmp_path: Path) -> None:
        """skip_bullet is session-only, not written to state file."""
        nom = Nominator(config=config, state_dir=tmp_path)
        nom.skip_bullet("b55")
        assert "b55" in nom._session_skipped_ids

        # State file should not contain skipped IDs
        state_file = tmp_path / "nomination_state.json"
        if state_file.exists():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            assert "b55" not in data.get("nominated_ids", [])
            assert "b55" not in data.get("ignored_ids", [])


# ---------------------------------------------------------------------------
# 5. get_pending_nominations
# ---------------------------------------------------------------------------


class TestGetPending:
    """Tests for get_pending_nominations."""

    def test_excludes_nominated_and_skipped(self, config: AutoNominateConfig) -> None:
        """Pending list excludes nominated and session-skipped bullets."""
        nom = Nominator(config=config)
        nom._candidates = [
            _make_bullet("b1"),
            _make_bullet("b2"),
            _make_bullet("b3"),
        ]
        nom._nominated_ids.add("b1")
        nom._session_skipped_ids.add("b2")

        pending = nom.get_pending_nominations()
        ids = [p["id"] for p in pending]
        assert ids == ["b3"]


# ---------------------------------------------------------------------------
# 6. session_summary
# ---------------------------------------------------------------------------


class TestSessionSummary:
    """Tests for session_summary."""

    def test_summary_correctness(self, config: AutoNominateConfig) -> None:
        """Summary counts reflect actual state."""
        nom = Nominator(config=config)
        nom._candidates = [
            _make_bullet("b1"),
            _make_bullet("b2"),
            _make_bullet("b3"),
        ]
        nom._nominated_ids.add("b1")
        nom._ignored_ids.add("x1")
        nom._session_skipped_ids.add("b2")

        summary = nom.session_summary()

        assert summary.total_candidates == 3
        assert summary.nominated_count == 1
        assert summary.ignored_count == 1
        assert summary.pending_count == 1
        assert len(summary.pending_bullets) == 1
        assert summary.pending_bullets[0]["id"] == "b3"


# ---------------------------------------------------------------------------
# 7. State persistence round-trip
# ---------------------------------------------------------------------------


class TestStatePersistence:
    """Tests for _load_state / _save_state round-trip."""

    def test_roundtrip(self, config: AutoNominateConfig, tmp_path: Path) -> None:
        """Nominated and ignored IDs survive save/load cycle."""
        nom1 = Nominator(config=config, state_dir=tmp_path)
        nom1._nominated_ids = {"n1", "n2"}
        nom1._ignored_ids = {"i1"}
        nom1._save_state()

        # New instance loads persisted state
        nom2 = Nominator(config=config, state_dir=tmp_path)
        assert nom2._nominated_ids == {"n1", "n2"}
        assert nom2._ignored_ids == {"i1"}
        # Session state is NOT persisted
        assert nom2._session_skipped_ids == set()

    def test_no_state_dir_no_error(self, config: AutoNominateConfig) -> None:
        """Without state_dir, save/load are no-ops (no error)."""
        nom = Nominator(config=config)
        nom._save_state()  # should not raise
        nom._load_state()  # should not raise

    def test_corrupt_state_file(
        self, config: AutoNominateConfig, tmp_path: Path
    ) -> None:
        """Corrupt state file doesn't crash, just logs warning."""
        state_file = tmp_path / "nomination_state.json"
        state_file.write_text("not valid json!", encoding="utf-8")

        nom = Nominator(config=config, state_dir=tmp_path)
        assert nom._nominated_ids == set()
        assert nom._ignored_ids == set()
