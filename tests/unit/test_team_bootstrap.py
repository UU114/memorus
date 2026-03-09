"""Unit tests for memx.ext.team_bootstrap — conditional Team Layer injection.

Covers all branches:
  - Team package not installed (ImportError)
  - Team installed but not enabled/configured
  - Team enabled with Git Fallback
  - Team enabled with federation server_url
  - Bootstrap failure does not affect Core
  - _detect_git_fallback logic
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_memory(**kwargs: Any) -> SimpleNamespace:
    """Create a minimal mock Memory object for bootstrap tests."""
    mem = SimpleNamespace(
        _retrieval_pipeline=kwargs.get("pipeline", MagicMock()),
        _storage_backend=kwargs.get("storage", MagicMock()),
    )
    return mem


def _make_team_config(enabled: bool = False, server_url: str | None = None) -> Any:
    """Create a minimal TeamConfig-like object."""
    from memx.team.config import TeamConfig

    return TeamConfig(enabled=enabled, server_url=server_url)


# ---------------------------------------------------------------------------
# Tests: try_bootstrap_team
# ---------------------------------------------------------------------------

class TestTryBootstrapTeam:
    """Tests for the top-level try_bootstrap_team function."""

    def test_team_not_installed_returns_false(self) -> None:
        """When memx.team.config cannot be imported, returns False silently."""
        from memx.ext.team_bootstrap import try_bootstrap_team

        memory = _make_mock_memory()

        # Temporarily hide memx.team.config from the import system
        saved_config = sys.modules.get("memx.team.config")
        saved_bootstrap = sys.modules.get("memx.ext.team_bootstrap")
        sys.modules["memx.team.config"] = None  # type: ignore[assignment]
        # Remove cached bootstrap module so reload picks up the blocked import
        sys.modules.pop("memx.ext.team_bootstrap", None)
        try:
            import memx.ext.team_bootstrap as tb_mod

            result = tb_mod.try_bootstrap_team(memory)
            assert result is False
        finally:
            if saved_config is not None:
                sys.modules["memx.team.config"] = saved_config
            else:
                sys.modules.pop("memx.team.config", None)
            # Restore bootstrap module
            sys.modules.pop("memx.ext.team_bootstrap", None)
            if saved_bootstrap is not None:
                sys.modules["memx.ext.team_bootstrap"] = saved_bootstrap

    def test_team_not_enabled_no_git_fallback_returns_false(self) -> None:
        """When Team is installed but not enabled and no Git Fallback, returns False."""
        from memx.ext import team_bootstrap

        memory = _make_mock_memory()

        with patch.object(
            team_bootstrap,
            "_detect_git_fallback",
            return_value=False,
        ):
            with patch(
                "memx.team.config.load_team_config",
                return_value=_make_team_config(enabled=False),
            ):
                result = team_bootstrap.try_bootstrap_team(memory)

        assert result is False

    def test_team_enabled_with_git_fallback(self) -> None:
        """When Git Fallback is detected, Team is bootstrapped."""
        from memx.ext import team_bootstrap

        memory = _make_mock_memory()

        with patch.object(
            team_bootstrap,
            "_detect_git_fallback",
            return_value=True,
        ):
            with patch.object(
                team_bootstrap,
                "_build_multi_pool_retriever",
                return_value=MagicMock(),
            ) as mock_build:
                result = team_bootstrap.try_bootstrap_team(memory)

        assert result is True
        mock_build.assert_called_once()

    def test_team_enabled_explicit_config(self) -> None:
        """When team_config.enabled=True, bootstraps Team."""
        from memx.ext import team_bootstrap

        memory = _make_mock_memory()

        with patch.object(
            team_bootstrap,
            "_detect_git_fallback",
            return_value=False,
        ):
            with patch.object(
                team_bootstrap,
                "_build_multi_pool_retriever",
                return_value=MagicMock(),
            ) as mock_build:
                enabled_cfg = _make_team_config(enabled=True)
                with patch(
                    "memx.team.config.load_team_config",
                    return_value=enabled_cfg,
                ):
                    result = team_bootstrap.try_bootstrap_team(memory)

        assert result is True

    def test_bootstrap_exception_returns_false(self) -> None:
        """When bootstrap throws any exception, returns False (Core unaffected)."""
        from memx.ext import team_bootstrap

        memory = _make_mock_memory()

        with patch.object(
            team_bootstrap,
            "_detect_git_fallback",
            side_effect=RuntimeError("disk error"),
        ):
            result = team_bootstrap.try_bootstrap_team(memory)

        assert result is False

    def test_retriever_injected_into_pipeline(self) -> None:
        """When bootstrap succeeds, retriever is set on pipeline."""
        from memx.ext import team_bootstrap

        pipeline = MagicMock()
        memory = _make_mock_memory(pipeline=pipeline)

        mock_retriever = MagicMock()
        with patch.object(
            team_bootstrap,
            "_detect_git_fallback",
            return_value=True,
        ):
            with patch.object(
                team_bootstrap,
                "_build_multi_pool_retriever",
                return_value=mock_retriever,
            ):
                result = team_bootstrap.try_bootstrap_team(memory)

        assert result is True
        assert memory._retrieval_pipeline._team_retriever == mock_retriever

    def test_no_pipeline_stores_retriever_for_deferred(self) -> None:
        """When pipeline is None, retriever is stored on memory for later."""
        from memx.ext import team_bootstrap

        memory = _make_mock_memory(pipeline=None)

        mock_retriever = MagicMock()
        with patch.object(
            team_bootstrap,
            "_detect_git_fallback",
            return_value=True,
        ):
            with patch.object(
                team_bootstrap,
                "_build_multi_pool_retriever",
                return_value=mock_retriever,
            ):
                result = team_bootstrap.try_bootstrap_team(memory)

        assert result is True
        assert memory._team_retriever == mock_retriever

    def test_build_returns_none_still_returns_true(self) -> None:
        """When _build returns None (no pools), still returns True (Team detected)."""
        from memx.ext import team_bootstrap

        memory = _make_mock_memory()

        with patch.object(
            team_bootstrap,
            "_detect_git_fallback",
            return_value=True,
        ):
            with patch.object(
                team_bootstrap,
                "_build_multi_pool_retriever",
                return_value=None,
            ):
                result = team_bootstrap.try_bootstrap_team(memory)

        # Returns True because Team was detected, even though retriever is None
        assert result is True

    def test_custom_config_path_forwarded(self) -> None:
        """When config_path is provided, it is forwarded to load_team_config."""
        from memx.ext import team_bootstrap

        memory = _make_mock_memory()

        with patch.object(
            team_bootstrap,
            "_detect_git_fallback",
            return_value=False,
        ):
            with patch(
                "memx.team.config.load_team_config",
                return_value=_make_team_config(enabled=False),
            ) as mock_load:
                team_bootstrap.try_bootstrap_team(
                    memory, config_path="/custom/path.yaml"
                )

        mock_load.assert_called_once_with(Path("/custom/path.yaml"))


# ---------------------------------------------------------------------------
# Tests: _detect_git_fallback
# ---------------------------------------------------------------------------

class TestDetectGitFallback:
    """Tests for the _detect_git_fallback helper."""

    def test_playbook_exists_returns_true(self, tmp_path: Path) -> None:
        """Returns True when .ace/playbook.jsonl exists."""
        from memx.ext.team_bootstrap import _detect_git_fallback

        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        (ace_dir / "playbook.jsonl").write_text("{}")
        # Also create .git to stop traversal
        (tmp_path / ".git").mkdir()

        with patch("memx.ext.team_bootstrap.Path.cwd", return_value=tmp_path):
            assert _detect_git_fallback() is True

    def test_no_playbook_returns_false(self, tmp_path: Path) -> None:
        """Returns False when no .ace/playbook.jsonl exists."""
        from memx.ext.team_bootstrap import _detect_git_fallback

        (tmp_path / ".git").mkdir()

        with patch("memx.ext.team_bootstrap.Path.cwd", return_value=tmp_path):
            assert _detect_git_fallback() is False

    def test_stops_at_git_root(self, tmp_path: Path) -> None:
        """Stops walking up when .git is found."""
        from memx.ext.team_bootstrap import _detect_git_fallback

        sub = tmp_path / "project"
        sub.mkdir()
        (sub / ".git").mkdir()

        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        (ace_dir / "playbook.jsonl").write_text("{}")

        with patch("memx.ext.team_bootstrap.Path.cwd", return_value=sub):
            assert _detect_git_fallback() is False

    def test_playbook_in_parent_found(self, tmp_path: Path) -> None:
        """Playbook in parent directory (before .git root) is found."""
        from memx.ext.team_bootstrap import _detect_git_fallback

        sub = tmp_path / "subdir"
        sub.mkdir()

        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        (ace_dir / "playbook.jsonl").write_text("{}")
        (tmp_path / ".git").mkdir()

        with patch("memx.ext.team_bootstrap.Path.cwd", return_value=sub):
            assert _detect_git_fallback() is True


# ---------------------------------------------------------------------------
# Tests: _build_multi_pool_retriever
# ---------------------------------------------------------------------------

class TestBuildMultiPoolRetriever:
    """Tests for the _build_multi_pool_retriever helper."""

    def test_no_pools_returns_none(self) -> None:
        """When neither git_fallback nor server_url, returns None."""
        from memx.ext.team_bootstrap import _build_multi_pool_retriever

        memory = _make_mock_memory()
        config = _make_team_config(enabled=True, server_url=None)

        result = _build_multi_pool_retriever(memory, config, git_fallback=False)
        assert result is None

    def test_git_fallback_pool_created(self) -> None:
        """When git_fallback=True, creates a pool with GitFallbackStorage."""
        from memx.ext.team_bootstrap import _build_multi_pool_retriever

        memory = _make_mock_memory()
        config = _make_team_config(enabled=True)

        # GitFallbackStorage exists, so it should create a pool
        result = _build_multi_pool_retriever(memory, config, git_fallback=True)

        assert result is not None
        # Since MultiPoolRetriever doesn't exist, expect raw pools list
        if isinstance(result, list):
            assert len(result) == 1
            assert result[0][0] == "git_fallback"

    def test_federation_pool_created_when_available(self) -> None:
        """When server_url is set and TeamCacheStorage is available, creates retriever."""
        from memx.ext.team_bootstrap import _build_multi_pool_retriever

        memory = _make_mock_memory()
        config = _make_team_config(enabled=True, server_url="http://example.com")

        # TeamCacheStorage is implemented (Sprint 6), so federation pool is created
        result = _build_multi_pool_retriever(memory, config, git_fallback=False)
        assert result is not None

    def test_both_pools_created(self) -> None:
        """When both git_fallback and server_url available, creates both pools."""
        from memx.ext import team_bootstrap
        from memx.ext.team_bootstrap import _build_multi_pool_retriever

        memory = _make_mock_memory()
        config = _make_team_config(
            enabled=True, server_url="http://example.com"
        )

        # Mock TeamCacheStorage to be importable
        mock_cache = MagicMock()
        mock_module = MagicMock()
        mock_module.TeamCacheStorage = mock_cache

        with patch.dict(sys.modules, {"memx.team.cache_storage": mock_module}):
            result = _build_multi_pool_retriever(
                memory, config, git_fallback=True
            )

        assert result is not None
        if isinstance(result, list):
            pool_names = [name for name, _ in result]
            assert "git_fallback" in pool_names
            assert "federation" in pool_names


# ---------------------------------------------------------------------------
# Tests: Memory integration
# ---------------------------------------------------------------------------

class TestMemoryIntegration:
    """Verify that Memory._try_team_bootstrap works correctly."""

    def test_try_team_bootstrap_returns_bool(self) -> None:
        """Memory._try_team_bootstrap returns a bool."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._config.privacy.always_sanitize = False
        mem._config.daemon.enabled = False
        mem._mem0 = None
        mem._mem0_init_error = None
        mem._ingest_pipeline = None
        mem._retrieval_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        with patch(
            "memx.ext.team_bootstrap.try_bootstrap_team",
            return_value=False,
        ):
            result = mem._try_team_bootstrap()

        assert isinstance(result, bool)
        assert result is False

    def test_try_team_bootstrap_catches_import_error(self) -> None:
        """If ext module itself cannot be imported, returns False."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)

        with patch(
            "builtins.__import__",
            side_effect=ImportError("no ext"),
        ):
            result = mem._try_team_bootstrap()

        assert result is False
