"""Unit tests for memx.team.cli — team status, sync, and nomination CLI commands."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memx.team.cli import nominate_group, team_group
from memx.team.config import AutoNominateConfig, TeamConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture()
def enabled_config() -> TeamConfig:
    """TeamConfig with team features enabled and a server URL."""
    return TeamConfig(
        enabled=True,
        server_url="https://sync.example.com",
        team_id="test-team",
        subscribed_tags=["python", "devops"],
        cache_max_bullets=500,
        cache_ttl_minutes=30,
    )


@pytest.fixture()
def enabled_config_no_server() -> TeamConfig:
    """TeamConfig with team enabled but no server_url (git-fallback)."""
    return TeamConfig(
        enabled=True,
        server_url=None,
        team_id="local-team",
        subscribed_tags=[],
        cache_max_bullets=1000,
        cache_ttl_minutes=60,
    )


# ---------------------------------------------------------------------------
# Helper: patch _ensure_team_enabled
# ---------------------------------------------------------------------------

_ENSURE = "memx.team.cli._ensure_team_enabled"


# ---------------------------------------------------------------------------
# team status
# ---------------------------------------------------------------------------


class TestTeamStatus:
    """Tests for `ace team status`."""

    def test_status_team_not_enabled(self, runner: CliRunner) -> None:
        """When team is not enabled, show friendly error and exit 1."""
        with patch(_ENSURE, return_value=(None, "Team features not enabled. Set 'enabled: true' in team_config.yaml or MEMX_TEAM_ENABLED=true")):
            result = runner.invoke(team_group, ["status"])
        assert result.exit_code != 0
        assert "Team features not enabled" in result.output or "Team features not enabled" in (result.stderr_bytes or b"").decode("utf-8", errors="replace")

    def test_status_team_not_enabled_json(self, runner: CliRunner) -> None:
        """JSON output when team not enabled contains error key."""
        with patch(_ENSURE, return_value=(None, "Team features not enabled.")):
            result = runner.invoke(team_group, ["status", "--json"])
        # CliRunner captures stdout; error JSON goes to stdout with --json
        data = json.loads(result.output)
        assert "error" in data
        assert "not enabled" in data["error"]

    def test_status_displays_info(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """Human-readable status shows mode, team id, server, tags."""
        mock_cache = MagicMock()
        mock_cache.bullet_count = 42
        mock_cache.last_sync_time = datetime(2026, 3, 8, 12, 0, 0, tzinfo=timezone.utc)

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch(
                "memx.team.cache_storage.TeamCacheStorage",
                return_value=mock_cache,
            ),
        ):
            result = runner.invoke(team_group, ["status"])

        assert result.exit_code == 0
        assert "federation" in result.output
        assert "test-team" in result.output
        assert "https://sync.example.com" in result.output

    def test_status_git_fallback_mode(
        self, runner: CliRunner, enabled_config_no_server: TeamConfig
    ) -> None:
        """When no server_url, mode should be git-fallback."""
        mock_cache = MagicMock()
        mock_cache.bullet_count = 0
        mock_cache.last_sync_time = None

        with (
            patch(_ENSURE, return_value=(enabled_config_no_server, None)),
            patch(
                "memx.team.cache_storage.TeamCacheStorage",
                return_value=mock_cache,
            ),
        ):
            result = runner.invoke(team_group, ["status"])

        assert result.exit_code == 0
        assert "git-fallback" in result.output

    def test_status_json_output(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """--json outputs valid JSON with expected keys."""
        mock_cache = MagicMock()
        mock_cache.bullet_count = 10
        mock_cache.last_sync_time = None

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch(
                "memx.team.cache_storage.TeamCacheStorage",
                return_value=mock_cache,
            ),
        ):
            result = runner.invoke(team_group, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["mode"] == "federation"
        assert data["team_id"] == "test-team"
        assert data["cache_max_bullets"] == 500
        assert data["subscribed_tags"] == ["python", "devops"]

    def test_status_cache_error_shows_na(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """When cache cannot be loaded, cached_bullets and last_sync show N/A."""
        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch(
                "memx.team.cache_storage.TeamCacheStorage",
                side_effect=RuntimeError("cache broken"),
            ),
        ):
            result = runner.invoke(team_group, ["status", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["cached_bullets"] == "N/A"
        assert data["last_sync"] == "N/A"

    def test_status_no_tags_shows_all(
        self, runner: CliRunner, enabled_config_no_server: TeamConfig
    ) -> None:
        """When subscribed_tags is empty, display 'all'."""
        mock_cache = MagicMock()
        mock_cache.bullet_count = 0
        mock_cache.last_sync_time = None

        with (
            patch(_ENSURE, return_value=(enabled_config_no_server, None)),
            patch(
                "memx.team.cache_storage.TeamCacheStorage",
                return_value=mock_cache,
            ),
        ):
            result = runner.invoke(team_group, ["status"])

        assert result.exit_code == 0
        assert "Subscribed Tags: all" in result.output


# ---------------------------------------------------------------------------
# team sync
# ---------------------------------------------------------------------------


class TestTeamSync:
    """Tests for `ace team sync`."""

    def test_sync_team_not_enabled(self, runner: CliRunner) -> None:
        """Error when team not enabled."""
        with patch(_ENSURE, return_value=(None, "Team features not enabled.")):
            result = runner.invoke(team_group, ["sync"])
        assert result.exit_code != 0

    def test_sync_no_server_url(
        self, runner: CliRunner, enabled_config_no_server: TeamConfig
    ) -> None:
        """Error when no server_url configured."""
        with patch(_ENSURE, return_value=(enabled_config_no_server, None)):
            result = runner.invoke(team_group, ["sync"])
        assert result.exit_code != 0
        assert "server_url" in result.output or "server_url" in (result.stderr_bytes or b"").decode("utf-8", errors="replace")

    def test_sync_no_server_url_json(
        self, runner: CliRunner, enabled_config_no_server: TeamConfig
    ) -> None:
        """JSON error when no server_url."""
        with patch(_ENSURE, return_value=(enabled_config_no_server, None)):
            result = runner.invoke(team_group, ["sync", "--json"])
        data = json.loads(result.output)
        assert "error" in data
        assert "server_url" in data["error"]

    def test_sync_success(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """Successful sync displays result."""
        mock_cache = MagicMock()
        mock_cache.bullet_count = 15

        mock_manager = MagicMock()
        mock_manager.last_sync_status = "success"
        mock_manager.sync_count = 1

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch("memx.team.cache_storage.TeamCacheStorage", return_value=mock_cache),
            patch("memx.team.sync_client.AceSyncClient"),
            patch("memx.team.sync_manager.SyncManager", return_value=mock_manager),
        ):
            result = runner.invoke(team_group, ["sync"])

        assert result.exit_code == 0
        assert "Sync success" in result.output
        mock_manager.sync_now.assert_called_once()

    def test_sync_full_resets_timestamp(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """--full flag resets _last_sync_timestamp to None."""
        mock_cache = MagicMock()
        mock_cache.bullet_count = 5

        mock_manager = MagicMock()
        mock_manager.last_sync_status = "success"
        mock_manager.sync_count = 1
        mock_manager._last_sync_timestamp = datetime(2026, 3, 1, tzinfo=timezone.utc)

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch("memx.team.cache_storage.TeamCacheStorage", return_value=mock_cache),
            patch("memx.team.sync_client.AceSyncClient"),
            patch("memx.team.sync_manager.SyncManager", return_value=mock_manager),
        ):
            result = runner.invoke(team_group, ["sync", "--full"])

        assert result.exit_code == 0
        # Verify timestamp was reset
        assert mock_manager._last_sync_timestamp is None

    def test_sync_json_output(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """--json outputs valid JSON result."""
        mock_cache = MagicMock()
        mock_cache.bullet_count = 20

        mock_manager = MagicMock()
        mock_manager.last_sync_status = "success"
        mock_manager.sync_count = 3

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch("memx.team.cache_storage.TeamCacheStorage", return_value=mock_cache),
            patch("memx.team.sync_client.AceSyncClient"),
            patch("memx.team.sync_manager.SyncManager", return_value=mock_manager),
        ):
            result = runner.invoke(team_group, ["sync", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "success"
        assert data["cached_bullets"] == 20
        assert data["sync_count"] == 3

    def test_sync_exception_json(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """Exception during sync shows JSON error."""
        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch(
                "memx.team.cache_storage.TeamCacheStorage",
                side_effect=RuntimeError("connection lost"),
            ),
        ):
            result = runner.invoke(team_group, ["sync", "--json"])
        data = json.loads(result.output)
        assert "error" in data
        assert "connection lost" in data["error"]


# ---------------------------------------------------------------------------
# nominate list
# ---------------------------------------------------------------------------


class TestNominateList:
    """Tests for `ace nominate list`."""

    def test_list_team_not_enabled(self, runner: CliRunner) -> None:
        """Error when team not enabled."""
        with patch(_ENSURE, return_value=(None, "Team features not enabled.")):
            result = runner.invoke(nominate_group, ["list"])
        assert result.exit_code != 0

    def test_list_no_candidates(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """Empty candidate list shows message."""
        mock_nominator = MagicMock()
        mock_nominator.get_pending_nominations.return_value = []

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch("memx.team.nominator.Nominator", return_value=mock_nominator),
        ):
            result = runner.invoke(nominate_group, ["list"])

        assert result.exit_code == 0
        assert "No candidates found" in result.output

    def test_list_shows_candidates(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """Candidates displayed in table format."""
        candidates = [
            {
                "id": "bullet-001",
                "instructivity_score": 85.0,
                "content": "Always use type hints in Python function signatures",
            },
            {
                "id": "bullet-002",
                "instructivity_score": 72.5,
                "content": "Run linters before committing code changes",
            },
        ]
        mock_nominator = MagicMock()
        mock_nominator.get_pending_nominations.return_value = candidates

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch("memx.team.nominator.Nominator", return_value=mock_nominator),
        ):
            result = runner.invoke(nominate_group, ["list"])

        assert result.exit_code == 0
        assert "bullet-001" in result.output
        assert "85.0" in result.output
        assert "bullet-002" in result.output

    def test_list_json_output(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """--json outputs valid JSON with candidates key."""
        candidates = [
            {"id": "b1", "instructivity_score": 90.0, "content": "test rule"},
        ]
        mock_nominator = MagicMock()
        mock_nominator.get_pending_nominations.return_value = candidates

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch("memx.team.nominator.Nominator", return_value=mock_nominator),
        ):
            result = runner.invoke(nominate_group, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "candidates" in data
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["id"] == "b1"

    def test_list_json_empty(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """--json with no candidates outputs empty list."""
        mock_nominator = MagicMock()
        mock_nominator.get_pending_nominations.return_value = []

        with (
            patch(_ENSURE, return_value=(enabled_config, None)),
            patch("memx.team.nominator.Nominator", return_value=mock_nominator),
        ):
            result = runner.invoke(nominate_group, ["list", "--json"])

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["candidates"] == []


# ---------------------------------------------------------------------------
# nominate submit
# ---------------------------------------------------------------------------


class TestNominateSubmit:
    """Tests for `ace nominate submit <id>`."""

    def test_submit_team_not_enabled(self, runner: CliRunner) -> None:
        """Error when team not enabled."""
        with patch(_ENSURE, return_value=(None, "Team features not enabled.")):
            result = runner.invoke(nominate_group, ["submit", "bullet-123"])
        assert result.exit_code != 0

    def test_submit_team_not_enabled_json(self, runner: CliRunner) -> None:
        """JSON error when team not enabled."""
        with patch(_ENSURE, return_value=(None, "Team features not enabled.")):
            result = runner.invoke(nominate_group, ["submit", "bullet-123", "--json"])
        data = json.loads(result.output)
        assert "error" in data

    def test_submit_bullet_not_found(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """Placeholder: bullet not found error."""
        with patch(_ENSURE, return_value=(enabled_config, None)):
            result = runner.invoke(nominate_group, ["submit", "nonexistent-id"])
        assert result.exit_code != 0
        assert "not found" in result.output or "not found" in (result.stderr_bytes or b"").decode("utf-8", errors="replace")

    def test_submit_bullet_not_found_json(
        self, runner: CliRunner, enabled_config: TeamConfig
    ) -> None:
        """JSON output for bullet not found."""
        with patch(_ENSURE, return_value=(enabled_config, None)):
            result = runner.invoke(
                nominate_group, ["submit", "my-bullet-id", "--json"]
            )
        data = json.loads(result.output)
        assert data["error"] == "Bullet not found"
        assert data["bullet_id"] == "my-bullet-id"


# ---------------------------------------------------------------------------
# _ensure_team_enabled
# ---------------------------------------------------------------------------


class TestEnsureTeamEnabled:
    """Tests for the _ensure_team_enabled helper."""

    def test_returns_config_when_enabled(self) -> None:
        """Returns (config, None) when team is enabled."""
        from memx.team.cli import _ensure_team_enabled

        tc = TeamConfig(enabled=True, server_url="https://x.com")
        with patch("memx.team.config.load_team_config", return_value=tc):
            config, err = _ensure_team_enabled()

        assert config is not None
        assert config.enabled is True
        assert err is None

    def test_returns_error_when_disabled(self) -> None:
        """Returns (None, error_msg) when team is disabled."""
        from memx.team.cli import _ensure_team_enabled

        tc = TeamConfig(enabled=False)
        with patch("memx.team.config.load_team_config", return_value=tc):
            config, err = _ensure_team_enabled()

        assert config is None
        assert err is not None
        assert "not enabled" in err

    def test_returns_error_on_exception(self) -> None:
        """Returns (None, error_msg) when config loading fails."""
        from memx.team.cli import _ensure_team_enabled

        with patch(
            "memx.team.config.load_team_config",
            side_effect=RuntimeError("bad yaml"),
        ):
            config, err = _ensure_team_enabled()

        assert config is None
        assert err is not None
        assert "bad yaml" in err
