"""Unit tests for daemon graceful degradation (STORY-039).

Tests cover:
- DaemonFallbackManager standalone behavior
- Degradation when daemon is unavailable
- Recovery detection when daemon comes back
- Log output (WARNING on degradation, INFO on recovery)
- Counter-based recovery checks with cooldown
- Integration with Memory class (search/add fallback)
- Transparent return values in both modes
- Edge cases (mid-request crash, frequent flapping, config disabled)

All daemon connections are mocked -- no real IPC is used.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.core.config import DaemonConfig, MemXConfig
from memx.core.daemon.client import DaemonClient
from memx.core.daemon.fallback import (
    DEFAULT_RECOVERY_COOLDOWN,
    DEFAULT_RECOVERY_INTERVAL,
    DaemonFallbackManager,
)
from memx.core.exceptions import DaemonUnavailableError
from memx.core.memory import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fallback(
    ping_result: bool = True,
    recovery_interval: int = DEFAULT_RECOVERY_INTERVAL,
    recovery_cooldown: float = DEFAULT_RECOVERY_COOLDOWN,
) -> DaemonFallbackManager:
    """Create a DaemonFallbackManager with a mocked DaemonClient.ping()."""
    config = DaemonConfig(enabled=True)
    mgr = DaemonFallbackManager(
        config=config,
        recovery_interval=recovery_interval,
        recovery_cooldown=recovery_cooldown,
    )
    mgr._client = MagicMock(spec=DaemonClient)
    mgr._client.ping = AsyncMock(return_value=ping_result)
    mgr._client.recall = AsyncMock(
        return_value=[{"memory": "test bullet", "score": 0.9}]
    )
    mgr._client.curate = AsyncMock(
        return_value={"results": [], "ace_ingest": {"bullets_added": 1}}
    )
    return mgr


def _make_memory_with_daemon(
    daemon_available: bool = True,
) -> Memory:
    """Create a Memory instance with daemon fallback manager wired up."""
    m = Memory.__new__(Memory)
    m._config = MemXConfig(daemon=DaemonConfig(enabled=True))
    m._mem0 = MagicMock()
    m._mem0_init_error = None
    m._mem0.add.return_value = {"results": [{"id": "1", "memory": "direct-add"}]}
    m._mem0.search.return_value = {"results": [{"id": "2", "memory": "direct-search"}]}
    m._ingest_pipeline = None
    m._retrieval_pipeline = None
    m._sanitizer = None

    # Set up daemon fallback
    mgr = _make_fallback(ping_result=daemon_available)
    mgr._available = daemon_available
    m._daemon_fallback = mgr
    return m


# ---------------------------------------------------------------------------
# DaemonFallbackManager -- initialization
# ---------------------------------------------------------------------------


class TestFallbackManagerInit:
    """Tests for DaemonFallbackManager initialization."""

    def test_default_attributes(self) -> None:
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config)
        assert mgr.is_available is False
        assert mgr.op_counter == 0
        assert mgr._degraded_logged is False
        assert mgr._recovery_interval == DEFAULT_RECOVERY_INTERVAL
        assert mgr._recovery_cooldown == DEFAULT_RECOVERY_COOLDOWN

    def test_custom_recovery_settings(self) -> None:
        mgr = DaemonFallbackManager(
            recovery_interval=5, recovery_cooldown=10.0
        )
        assert mgr._recovery_interval == 5
        assert mgr._recovery_cooldown == 10.0

    def test_client_created(self) -> None:
        config = DaemonConfig(enabled=True, socket_path="/tmp/test.sock")
        mgr = DaemonFallbackManager(config=config)
        assert mgr.client is not None
        assert isinstance(mgr.client, DaemonClient)


# ---------------------------------------------------------------------------
# DaemonFallbackManager -- check_initial_availability
# ---------------------------------------------------------------------------


class TestInitialAvailability:
    """Tests for check_initial_availability."""

    def test_daemon_reachable_sets_available(self) -> None:
        mgr = _make_fallback(ping_result=True)
        result = mgr.check_initial_availability()
        assert result is True
        assert mgr.is_available is True

    def test_daemon_unreachable_sets_unavailable(self) -> None:
        mgr = _make_fallback(ping_result=False)
        result = mgr.check_initial_availability()
        assert result is False
        assert mgr.is_available is False

    def test_degradation_warning_logged_on_unavailable(self, caplog: Any) -> None:
        mgr = _make_fallback(ping_result=False)
        with caplog.at_level(logging.WARNING, logger="memx.core.daemon.fallback"):
            mgr.check_initial_availability()
        assert any(
            "Daemon unavailable, falling back to direct mode" in r.message
            for r in caplog.records
        )

    def test_no_warning_when_available(self, caplog: Any) -> None:
        mgr = _make_fallback(ping_result=True)
        with caplog.at_level(logging.WARNING, logger="memx.core.daemon.fallback"):
            mgr.check_initial_availability()
        assert not any(
            "Daemon unavailable" in r.message for r in caplog.records
        )

    def test_ping_exception_treated_as_unavailable(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._client.ping = AsyncMock(side_effect=RuntimeError("loop issue"))
        # check_initial_availability uses asyncio.run which may raise
        # The method should handle exceptions gracefully
        result = mgr.check_initial_availability()
        assert result is False
        assert mgr.is_available is False


# ---------------------------------------------------------------------------
# DaemonFallbackManager -- try_recall
# ---------------------------------------------------------------------------


class TestTryRecall:
    """Tests for try_recall."""

    async def test_recall_when_available(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        result = await mgr.try_recall("test query", user_id="u1", limit=5)
        assert result is not None
        assert len(result) == 1
        assert result[0]["memory"] == "test bullet"
        mgr._client.recall.assert_awaited_once_with(
            "test query", user_id="u1", limit=5
        )

    async def test_recall_returns_none_when_unavailable(self) -> None:
        mgr = _make_fallback(ping_result=False)
        mgr._available = False
        result = await mgr.try_recall("test query")
        assert result is None
        mgr._client.recall.assert_not_awaited()

    async def test_recall_catches_daemon_unavailable_error(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        mgr._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("connection lost")
        )
        result = await mgr.try_recall("test query")
        assert result is None
        assert mgr.is_available is False

    async def test_recall_increments_op_counter(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        assert mgr.op_counter == 0
        await mgr.try_recall("q1")
        assert mgr.op_counter == 1
        await mgr.try_recall("q2")
        assert mgr.op_counter == 2


# ---------------------------------------------------------------------------
# DaemonFallbackManager -- try_curate
# ---------------------------------------------------------------------------


class TestTryCurate:
    """Tests for try_curate."""

    async def test_curate_when_available(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        result = await mgr.try_curate("User prefers dark mode", user_id="u1")
        assert result is not None
        assert result["ace_ingest"]["bullets_added"] == 1
        mgr._client.curate.assert_awaited_once_with(
            "User prefers dark mode", user_id="u1"
        )

    async def test_curate_returns_none_when_unavailable(self) -> None:
        mgr = _make_fallback(ping_result=False)
        mgr._available = False
        result = await mgr.try_curate("msg")
        assert result is None
        mgr._client.curate.assert_not_awaited()

    async def test_curate_catches_daemon_unavailable_error(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        mgr._client.curate = AsyncMock(
            side_effect=DaemonUnavailableError("daemon crashed")
        )
        result = await mgr.try_curate("msg", user_id="u1")
        assert result is None
        assert mgr.is_available is False

    async def test_curate_increments_op_counter(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        await mgr.try_curate("m1")
        await mgr.try_curate("m2")
        assert mgr.op_counter == 2


# ---------------------------------------------------------------------------
# DaemonFallbackManager -- degradation logging
# ---------------------------------------------------------------------------


class TestDegradationLogging:
    """Tests for degradation log messages."""

    async def test_warning_logged_on_first_degradation(self, caplog: Any) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        mgr._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("gone")
        )
        with caplog.at_level(logging.WARNING, logger="memx.core.daemon.fallback"):
            await mgr.try_recall("q")
        warnings = [
            r for r in caplog.records
            if "Daemon unavailable, falling back to direct mode" in r.message
        ]
        assert len(warnings) == 1

    async def test_no_duplicate_warning_on_repeated_failure(self, caplog: Any) -> None:
        mgr = _make_fallback(ping_result=True, recovery_cooldown=9999.0)
        mgr._available = True
        mgr._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("gone")
        )
        with caplog.at_level(logging.WARNING, logger="memx.core.daemon.fallback"):
            await mgr.try_recall("q1")
            # Daemon is now marked unavailable; next calls won't even try IPC
            # But let's force another degradation scenario
            mgr._available = True
            mgr._degraded_logged = True  # Already logged
            await mgr.try_recall("q2")

        warnings = [
            r for r in caplog.records
            if "Daemon unavailable, falling back to direct mode" in r.message
        ]
        # Only 1 warning, because _degraded_logged was True for second call
        assert len(warnings) == 1


# ---------------------------------------------------------------------------
# DaemonFallbackManager -- recovery
# ---------------------------------------------------------------------------


class TestRecovery:
    """Tests for daemon recovery detection."""

    async def test_check_recovery_when_daemon_comes_back(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = False
        mgr._last_failure_time = 0.0  # Ensure cooldown elapsed
        result = await mgr.check_recovery()
        assert result is True
        assert mgr.is_available is True

    async def test_check_recovery_still_down(self) -> None:
        mgr = _make_fallback(ping_result=False)
        mgr._available = False
        mgr._last_failure_time = 0.0
        result = await mgr.check_recovery()
        assert result is False
        assert mgr.is_available is False

    async def test_recovery_info_logged(self, caplog: Any) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = False
        mgr._last_failure_time = 0.0
        with caplog.at_level(logging.INFO, logger="memx.core.daemon.fallback"):
            await mgr.check_recovery()
        info_msgs = [
            r for r in caplog.records
            if "Daemon reconnected, switching to IPC mode" in r.message
        ]
        assert len(info_msgs) == 1

    async def test_recovery_resets_degraded_logged(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = False
        mgr._degraded_logged = True
        mgr._last_failure_time = 0.0
        await mgr.check_recovery()
        assert mgr._degraded_logged is False

    async def test_already_available_returns_true(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        result = await mgr.check_recovery()
        assert result is True

    async def test_recovery_respects_cooldown(self) -> None:
        mgr = _make_fallback(ping_result=True, recovery_cooldown=9999.0)
        mgr._available = False
        mgr._last_failure_time = time.monotonic()  # Just failed
        result = await mgr.check_recovery()
        assert result is False
        # Ping should not have been called due to cooldown
        mgr._client.ping.assert_not_awaited()


# ---------------------------------------------------------------------------
# DaemonFallbackManager -- counter-based recovery
# ---------------------------------------------------------------------------


class TestCounterBasedRecovery:
    """Tests for automatic recovery checks triggered by operation counter."""

    def test_recovery_triggered_at_interval(self) -> None:
        mgr = _make_fallback(
            ping_result=True,
            recovery_interval=5,
            recovery_cooldown=0.0,
        )
        mgr._available = False
        mgr._last_failure_time = 0.0

        # Tick 5 times to trigger recovery check
        for _ in range(5):
            mgr._tick_op_counter()

        assert mgr.is_available is True

    def test_no_recovery_before_interval(self) -> None:
        mgr = _make_fallback(
            ping_result=True,
            recovery_interval=10,
            recovery_cooldown=0.0,
        )
        mgr._available = False
        mgr._last_failure_time = 0.0

        # Tick 9 times (not enough)
        for _ in range(9):
            mgr._tick_op_counter()

        assert mgr.is_available is False

    def test_no_recovery_check_when_available(self) -> None:
        mgr = _make_fallback(
            ping_result=True,
            recovery_interval=1,
            recovery_cooldown=0.0,
        )
        mgr._available = True
        mgr._tick_op_counter()
        # Should not call ping since already available
        mgr._client.ping.assert_not_called()

    def test_recovery_check_respects_cooldown(self) -> None:
        mgr = _make_fallback(
            ping_result=True,
            recovery_interval=1,
            recovery_cooldown=9999.0,
        )
        mgr._available = False
        mgr._last_failure_time = time.monotonic()
        mgr._tick_op_counter()
        # Should not have called ping due to cooldown
        mgr._client.ping.assert_not_called()
        assert mgr.is_available is False

    def test_failed_recovery_updates_failure_time(self) -> None:
        mgr = _make_fallback(
            ping_result=False,
            recovery_interval=1,
            recovery_cooldown=0.0,
        )
        mgr._available = False
        mgr._last_failure_time = 0.0
        before = time.monotonic()
        mgr._tick_op_counter()
        assert mgr._last_failure_time >= before
        assert mgr.is_available is False


# ---------------------------------------------------------------------------
# DaemonFallbackManager -- degradation/recovery cycle
# ---------------------------------------------------------------------------


class TestDegradationRecoveryCycle:
    """Tests for full degradation-then-recovery cycle."""

    async def test_full_cycle(self, caplog: Any) -> None:
        """Daemon available -> crash -> degraded -> recover -> available."""
        mgr = _make_fallback(
            ping_result=True,
            recovery_interval=3,
            recovery_cooldown=0.0,
        )
        mgr._available = True

        # Step 1: Daemon crashes during recall
        mgr._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("crash")
        )
        with caplog.at_level(logging.WARNING, logger="memx.core.daemon.fallback"):
            result = await mgr.try_recall("q")
        assert result is None
        assert mgr.is_available is False

        # Step 2: Subsequent calls return None immediately (no IPC attempt)
        mgr._client.recall.reset_mock()
        result2 = await mgr.try_recall("q2")
        assert result2 is None
        mgr._client.recall.assert_not_awaited()

        # Step 3: Recovery after interval ticks
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._client.recall = AsyncMock(
            return_value=[{"memory": "recovered", "score": 1.0}]
        )

        # Need enough ticks to trigger recovery (interval=3, we already have 2 ticks)
        with caplog.at_level(logging.INFO, logger="memx.core.daemon.fallback"):
            result3 = await mgr.try_recall("q3")

        # After tick 3, recovery check triggered => daemon back
        assert mgr.is_available is True

        # Step 4: Next call should use daemon
        result4 = await mgr.try_recall("q4")
        assert result4 is not None
        assert result4[0]["memory"] == "recovered"


# ---------------------------------------------------------------------------
# Memory integration -- daemon enabled and available
# ---------------------------------------------------------------------------


class TestMemoryDaemonAvailable:
    """Tests for Memory with daemon available (IPC path)."""

    def test_search_uses_daemon(self) -> None:
        m = _make_memory_with_daemon(daemon_available=True)
        result = m.search("test query", user_id="u1")
        assert "results" in result
        m._daemon_fallback._client.recall.assert_called()
        # Should NOT have called mem0.search
        m._mem0.search.assert_not_called()

    def test_add_uses_daemon(self) -> None:
        m = _make_memory_with_daemon(daemon_available=True)
        result = m.add("User likes Python", user_id="u1")
        assert result is not None
        m._daemon_fallback._client.curate.assert_called()
        m._mem0.add.assert_not_called()


# ---------------------------------------------------------------------------
# Memory integration -- daemon unavailable (direct mode)
# ---------------------------------------------------------------------------


class TestMemoryDaemonUnavailable:
    """Tests for Memory with daemon unavailable (direct fallback)."""

    def test_search_falls_back_to_mem0(self) -> None:
        m = _make_memory_with_daemon(daemon_available=False)
        result = m.search("test query", user_id="u1")
        # Should have used mem0 directly
        m._mem0.search.assert_called_once()
        assert result == {"results": [{"id": "2", "memory": "direct-search"}]}

    def test_add_falls_back_to_mem0(self) -> None:
        m = _make_memory_with_daemon(daemon_available=False)
        result = m.add("test message", user_id="u1")
        m._mem0.add.assert_called_once()
        assert result == {"results": [{"id": "1", "memory": "direct-add"}]}


# ---------------------------------------------------------------------------
# Memory integration -- daemon crashes mid-request
# ---------------------------------------------------------------------------


class TestMemoryDaemonMidCrash:
    """Tests for daemon crashing during a Memory operation."""

    def test_search_falls_back_on_mid_crash(self) -> None:
        m = _make_memory_with_daemon(daemon_available=True)
        m._daemon_fallback._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("mid-crash")
        )
        result = m.search("test query", user_id="u1")
        # Should have fallen back to mem0
        m._mem0.search.assert_called_once()
        assert result == {"results": [{"id": "2", "memory": "direct-search"}]}
        # Daemon should be marked unavailable
        assert m._daemon_fallback.is_available is False

    def test_add_falls_back_on_mid_crash(self) -> None:
        m = _make_memory_with_daemon(daemon_available=True)
        m._daemon_fallback._client.curate = AsyncMock(
            side_effect=DaemonUnavailableError("mid-crash")
        )
        result = m.add("test message", user_id="u1")
        m._mem0.add.assert_called_once()
        assert result == {"results": [{"id": "1", "memory": "direct-add"}]}
        assert m._daemon_fallback.is_available is False


# ---------------------------------------------------------------------------
# Memory integration -- daemon disabled
# ---------------------------------------------------------------------------


class TestMemoryDaemonDisabled:
    """Tests for Memory with daemon.enabled=False."""

    def test_no_fallback_manager_when_disabled(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=False))
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._mem0.search.return_value = {"results": []}
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None
        m._daemon_fallback = None
        m._init_daemon_fallback()
        assert m._daemon_fallback is None

    def test_search_uses_mem0_when_disabled(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=False))
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._mem0.search.return_value = {"results": []}
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None
        m._daemon_fallback = None
        result = m.search("query")
        m._mem0.search.assert_called_once()

    def test_add_uses_mem0_when_disabled(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=False))
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._mem0.add.return_value = {"results": []}
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None
        m._daemon_fallback = None
        result = m.add("msg")
        m._mem0.add.assert_called_once()


# ---------------------------------------------------------------------------
# Memory -- daemon_available property
# ---------------------------------------------------------------------------


class TestDaemonAvailableProperty:
    """Tests for Memory.daemon_available property."""

    def test_returns_true_when_daemon_up(self) -> None:
        m = _make_memory_with_daemon(daemon_available=True)
        assert m.daemon_available is True

    def test_returns_false_when_daemon_down(self) -> None:
        m = _make_memory_with_daemon(daemon_available=False)
        assert m.daemon_available is False

    def test_returns_false_when_no_fallback(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._daemon_fallback = None
        assert m.daemon_available is False


# ---------------------------------------------------------------------------
# Transparent return values
# ---------------------------------------------------------------------------


class TestTransparentReturnValues:
    """Tests that return values are consistent regardless of mode."""

    def test_search_returns_dict_with_results_key_via_daemon(self) -> None:
        m = _make_memory_with_daemon(daemon_available=True)
        result = m.search("query")
        assert isinstance(result, dict)
        assert "results" in result

    def test_search_returns_dict_with_results_key_via_mem0(self) -> None:
        m = _make_memory_with_daemon(daemon_available=False)
        result = m.search("query")
        assert isinstance(result, dict)
        assert "results" in result

    def test_add_returns_dict_via_daemon(self) -> None:
        m = _make_memory_with_daemon(daemon_available=True)
        result = m.add("msg")
        assert isinstance(result, dict)

    def test_add_returns_dict_via_mem0(self) -> None:
        m = _make_memory_with_daemon(daemon_available=False)
        result = m.add("msg")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Log spam prevention
# ---------------------------------------------------------------------------


class TestLogSpamPrevention:
    """Tests that degradation/recovery logs are not spammed."""

    async def test_repeated_degradation_only_logs_once(self, caplog: Any) -> None:
        mgr = _make_fallback(ping_result=False, recovery_cooldown=9999.0)
        with caplog.at_level(logging.WARNING, logger="memx.core.daemon.fallback"):
            # First degradation
            mgr._mark_unavailable()
            # Second degradation (simulate another failure after forced available)
            mgr._mark_unavailable()

        warnings = [
            r for r in caplog.records
            if "Daemon unavailable, falling back to direct mode" in r.message
        ]
        # Only one warning because _degraded_logged prevents duplicates
        assert len(warnings) == 1

    async def test_recovery_then_degradation_logs_both(self, caplog: Any) -> None:
        mgr = _make_fallback(
            ping_result=True,
            recovery_cooldown=0.0,
        )
        mgr._available = False
        mgr._degraded_logged = True
        mgr._last_failure_time = 0.0

        with caplog.at_level(logging.DEBUG, logger="memx.core.daemon.fallback"):
            # Recovery
            await mgr.check_recovery()
            assert mgr.is_available is True

            # Simulate another crash
            mgr._client.recall = AsyncMock(
                side_effect=DaemonUnavailableError("crash again")
            )
            await mgr.try_recall("q")

        info_msgs = [
            r for r in caplog.records
            if "Daemon reconnected" in r.message
        ]
        warn_msgs = [
            r for r in caplog.records
            if "Daemon unavailable, falling back to direct mode" in r.message
        ]
        assert len(info_msgs) == 1
        assert len(warn_msgs) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    async def test_try_recall_default_params(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        await mgr.try_recall("query")
        mgr._client.recall.assert_awaited_once_with(
            "query", user_id="default", limit=5
        )

    async def test_try_curate_default_params(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = True
        await mgr.try_curate("msg")
        mgr._client.curate.assert_awaited_once_with("msg", user_id="default")

    def test_cooldown_elapsed_when_never_failed(self) -> None:
        mgr = _make_fallback(ping_result=True)
        assert mgr._is_cooldown_elapsed() is True

    def test_cooldown_not_elapsed_immediately_after_failure(self) -> None:
        mgr = _make_fallback(ping_result=True, recovery_cooldown=60.0)
        mgr._last_failure_time = time.monotonic()
        assert mgr._is_cooldown_elapsed() is False

    def test_cooldown_elapsed_after_sufficient_time(self) -> None:
        mgr = _make_fallback(ping_result=True, recovery_cooldown=0.01)
        mgr._last_failure_time = time.monotonic() - 1.0
        assert mgr._is_cooldown_elapsed() is True

    async def test_ping_exception_during_recovery(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._available = False
        mgr._last_failure_time = 0.0
        mgr._client.ping = AsyncMock(side_effect=Exception("unexpected"))
        result = await mgr.check_recovery()
        assert result is False
        assert mgr.is_available is False

    def test_memory_init_daemon_fallback_exception_handled(self) -> None:
        """If DaemonFallbackManager init raises, Memory should still work."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=True))
        m._daemon_fallback = None

        with patch(
            "memx.core.daemon.fallback.DaemonFallbackManager",
            side_effect=ImportError("no module"),
        ):
            # Should not raise -- just logs warning
            m._init_daemon_fallback()

        assert m._daemon_fallback is None

    def test_mark_available_resets_degraded_flag(self) -> None:
        mgr = _make_fallback(ping_result=True)
        mgr._degraded_logged = True
        mgr._mark_available()
        assert mgr._degraded_logged is False

    def test_mark_unavailable_sets_failure_time(self) -> None:
        mgr = _make_fallback(ping_result=True)
        before = time.monotonic()
        mgr._mark_unavailable()
        assert mgr._last_failure_time >= before
        assert mgr.is_available is False


# ---------------------------------------------------------------------------
# Memory._init_daemon_fallback integration
# ---------------------------------------------------------------------------


class TestMemoryInitDaemonFallback:
    """Tests for Memory._init_daemon_fallback method."""

    def test_skips_when_daemon_disabled(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=False))
        m._daemon_fallback = None
        m._init_daemon_fallback()
        assert m._daemon_fallback is None

    def test_creates_fallback_when_enabled(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=True))
        m._daemon_fallback = None

        # Mock the DaemonClient.ping to avoid real IPC
        with patch.object(
            DaemonClient, "ping", new_callable=AsyncMock, return_value=True
        ):
            m._init_daemon_fallback()

        assert m._daemon_fallback is not None
        assert isinstance(m._daemon_fallback, DaemonFallbackManager)

    def test_handles_init_failure_gracefully(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=True))
        m._daemon_fallback = None

        # Patch the class itself to simulate import/init failure
        with patch(
            "memx.core.daemon.fallback.DaemonFallbackManager",
            side_effect=Exception("init failed"),
        ):
            m._init_daemon_fallback()

        assert m._daemon_fallback is None
