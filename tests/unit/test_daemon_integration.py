"""Integration-level and edge-case tests for the Daemon subsystem (STORY-040).

These tests fill coverage gaps left by the per-module unit tests:
  - Cross-module interactions (server <-> client <-> fallback <-> Memory)
  - Edge cases in request/response handling
  - Config propagation across layers
  - Session lifecycle and idle timeout scenarios
  - PID management edge cases
  - IPC command coverage (all 6 commands)
  - Concurrent operations
  - Platform-specific branching

All IPC is mocked -- no real daemon or socket connections.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.core.config import DaemonConfig, MemXConfig
from memx.core.daemon import (
    DaemonClient,
    DaemonFallbackManager,
    DaemonRequest,
    DaemonResponse,
    IPCTransport,
    MemXDaemon,
    NamedPipeTransport,
    UnixSocketTransport,
    get_transport,
)
from memx.core.daemon.server import (
    DEFAULT_IDLE_TIMEOUT,
    MAX_REQUEST_SIZE,
    PID_PATH,
    PIPE_NAME,
    SOCKET_PATH,
    _is_process_alive,
)
from memx.core.exceptions import DaemonError, DaemonUnavailableError
from memx.core.memory import Memory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockTransport(IPCTransport):
    """In-memory transport that records sent data and returns canned responses."""

    def __init__(self, responses: list[DaemonResponse] | None = None) -> None:
        self._responses = list(responses or [DaemonResponse(status="ok")])
        self._sent: list[bytes] = []
        self.connected = False
        self.closed = False

    async def connect(self) -> None:
        self.connected = True

    async def send(self, data: bytes) -> None:
        self._sent.append(data)

    async def recv(self) -> bytes:
        resp = self._responses.pop(0)
        return resp.to_json().encode("utf-8")

    async def close(self) -> None:
        self.closed = True


def _make_daemon(
    idle_timeout: int = 2,
    socket_path: str | None = None,
) -> MemXDaemon:
    """Create a MemXDaemon with a short idle timeout for testing."""
    config = DaemonConfig(
        enabled=True,
        idle_timeout_seconds=idle_timeout,
        socket_path=socket_path,
    )
    return MemXDaemon(config=config)


def _make_memory_with_daemon(daemon_available: bool = True) -> Memory:
    """Create a Memory instance with daemon fallback wired up (mocked)."""
    m = Memory.__new__(Memory)
    m._config = MemXConfig(daemon=DaemonConfig(enabled=True))
    m._mem0 = MagicMock()
    m._mem0_init_error = None
    m._mem0.add.return_value = {"results": [{"id": "d1", "memory": "direct-add"}]}
    m._mem0.search.return_value = {"results": [{"id": "d2", "memory": "direct-search"}]}
    m._ingest_pipeline = None
    m._retrieval_pipeline = None
    m._sanitizer = None

    # Build a fallback manager with mocked client
    config = DaemonConfig(enabled=True)
    mgr = DaemonFallbackManager(config=config, recovery_interval=5, recovery_cooldown=0.0)
    mgr._client = MagicMock(spec=DaemonClient)
    mgr._client.ping = AsyncMock(return_value=daemon_available)
    mgr._client.recall = AsyncMock(
        return_value=[{"memory": "daemon-result", "score": 0.95}]
    )
    mgr._client.curate = AsyncMock(
        return_value={"results": [], "ace_ingest": {"bullets_added": 2}}
    )
    mgr._available = daemon_available
    m._daemon_fallback = mgr
    return m


# ===========================================================================
# 1. Cross-module: Server request routing (all 6 commands)
# ===========================================================================


class TestAllSixCommands:
    """Verify that all 6 IPC commands produce correct responses through handle_request."""

    async def test_ping_command(self) -> None:
        daemon = _make_daemon()
        daemon._start_time = datetime.now()
        resp = await daemon.handle_request(DaemonRequest(cmd="ping"))
        assert resp.status == "ok"
        assert "version" in resp.data
        assert "sessions" in resp.data
        assert "pid" in resp.data
        assert "uptime_seconds" in resp.data

    async def test_recall_command_with_memory(self) -> None:
        daemon = _make_daemon()
        daemon._memory = MagicMock()
        daemon._memory.search.return_value = {"results": [{"m": "found"}]}
        resp = await daemon.handle_request(
            DaemonRequest(cmd="recall", data={"query": "test", "user_id": "u1", "limit": 3})
        )
        assert resp.status == "ok"
        daemon._memory.search.assert_called_once_with(query="test", user_id="u1", limit=3)

    async def test_curate_command_with_memory(self) -> None:
        daemon = _make_daemon()
        daemon._memory = MagicMock()
        daemon._memory.add.return_value = {"added": 1}
        resp = await daemon.handle_request(
            DaemonRequest(cmd="curate", data={"messages": "hello", "user_id": "u1"})
        )
        assert resp.status == "ok"
        daemon._memory.add.assert_called_once_with(messages="hello", user_id="u1")

    async def test_session_register_command(self) -> None:
        daemon = _make_daemon()
        resp = await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s1"})
        )
        assert resp.status == "ok"
        assert daemon.session_count == 1
        assert "s1" in daemon._sessions

    async def test_session_unregister_command(self) -> None:
        daemon = _make_daemon()
        daemon._sessions["s1"] = datetime.now()
        resp = await daemon.handle_request(
            DaemonRequest(cmd="session_unregister", data={"session_id": "s1"})
        )
        assert resp.status == "ok"
        assert daemon.session_count == 0
        # Clean up timer
        if daemon._idle_timer and not daemon._idle_timer.done():
            daemon._idle_timer.cancel()

    async def test_shutdown_command(self) -> None:
        daemon = _make_daemon()
        daemon._shutdown_event = asyncio.Event()
        resp = await daemon.handle_request(DaemonRequest(cmd="shutdown"))
        assert resp.status == "ok"
        assert daemon._shutdown_event.is_set()


# ===========================================================================
# 2. Cross-module: Server <-> Client serialization round-trip
# ===========================================================================


class TestServerClientRoundTrip:
    """Verify request/response serialization compatibility between server and client."""

    async def test_ping_round_trip(self) -> None:
        """Client sends ping -> Server processes -> Client parses response."""
        daemon = _make_daemon()
        daemon._start_time = datetime.now()

        # Simulate client-side request creation
        from dataclasses import asdict

        req = DaemonRequest(cmd="ping")
        payload = json.dumps(asdict(req))

        # Server-side parsing
        server_req = DaemonRequest.from_json(payload)
        assert server_req.cmd == "ping"

        # Server-side handling
        server_resp = await daemon.handle_request(server_req)

        # Client-side response parsing
        resp_json = server_resp.to_json()
        parsed = json.loads(resp_json)
        client_resp = DaemonResponse(**parsed)
        assert client_resp.status == "ok"
        assert client_resp.data["version"] == "1.0.0"

    async def test_recall_round_trip(self) -> None:
        """Recall request/response serializes correctly across client-server boundary."""
        daemon = _make_daemon()
        daemon._memory = MagicMock()
        daemon._memory.search.return_value = {
            "results": [{"memory": "bullet-1", "score": 0.85}]
        }

        from dataclasses import asdict

        req = DaemonRequest(
            cmd="recall", data={"query": "patterns", "user_id": "u1", "limit": 3}
        )
        payload = json.dumps(asdict(req))

        server_req = DaemonRequest.from_json(payload)
        server_resp = await daemon.handle_request(server_req)
        resp_json = server_resp.to_json()
        parsed = json.loads(resp_json)
        client_resp = DaemonResponse(**parsed)

        assert client_resp.status == "ok"
        assert "results" in client_resp.data

    async def test_error_round_trip(self) -> None:
        """Error responses serialize correctly."""
        daemon = _make_daemon()
        daemon._memory = None  # Will cause recall to fail
        req = DaemonRequest(cmd="recall", data={"query": "test"})
        resp = await daemon.handle_request(req)
        assert resp.status == "error"

        # Parse back through JSON
        parsed = json.loads(resp.to_json())
        reconstituted = DaemonResponse(**parsed)
        assert reconstituted.status == "error"
        assert "Memory" in (reconstituted.error or "")


# ===========================================================================
# 3. Session lifecycle: register 3, unregister 2, verify 1 remains
# ===========================================================================


class TestSessionLifecycleIntegration:
    """Test multi-session lifecycle matching acceptance criteria."""

    async def test_register_three_unregister_two_one_remains(self) -> None:
        """AC: Register 3 sessions, unregister 2, verify 1 remains active."""
        daemon = _make_daemon()

        # Register 3 sessions
        for sid in ("alpha", "beta", "gamma"):
            resp = await daemon.handle_request(
                DaemonRequest(cmd="session_register", data={"session_id": sid})
            )
            assert resp.status == "ok"
        assert daemon.session_count == 3

        # Unregister 2
        for sid in ("alpha", "gamma"):
            resp = await daemon.handle_request(
                DaemonRequest(cmd="session_unregister", data={"session_id": sid})
            )
            assert resp.status == "ok"

        assert daemon.session_count == 1
        assert "beta" in daemon._sessions
        assert "alpha" not in daemon._sessions
        assert "gamma" not in daemon._sessions

        # No idle timer since 1 session still active
        assert daemon._idle_timer is None

    async def test_duplicate_session_registration(self) -> None:
        """Registering the same session ID twice should overwrite the timestamp."""
        daemon = _make_daemon()
        await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s1"})
        )
        ts1 = daemon._sessions["s1"]

        # Small delay to ensure different timestamp
        await asyncio.sleep(0.01)
        await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s1"})
        )
        ts2 = daemon._sessions["s1"]

        # Session count should still be 1 (overwritten)
        assert daemon.session_count == 1
        assert ts2 >= ts1

    async def test_unregister_all_sessions_starts_idle_timer(self) -> None:
        """After last session unregisters, idle timer must start."""
        daemon = _make_daemon()
        for sid in ("s1", "s2"):
            await daemon.handle_request(
                DaemonRequest(cmd="session_register", data={"session_id": sid})
            )

        # Unregister s1 -- s2 still active, no timer
        await daemon.handle_request(
            DaemonRequest(cmd="session_unregister", data={"session_id": "s1"})
        )
        assert daemon._idle_timer is None

        # Unregister s2 -- no sessions left, timer starts
        await daemon.handle_request(
            DaemonRequest(cmd="session_unregister", data={"session_id": "s2"})
        )
        assert daemon._idle_timer is not None
        assert not daemon._idle_timer.done()

        # Cleanup
        daemon._idle_timer.cancel()

    async def test_register_cancels_active_idle_timer(self) -> None:
        """Registering a new session while idle timer is running should cancel it."""
        daemon = _make_daemon()
        # Manually start idle timer
        daemon._start_idle_timer()
        assert daemon._idle_timer is not None

        # Register session -- timer should be cancelled
        await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s1"})
        )
        assert daemon._idle_timer is None


# ===========================================================================
# 4. Idle timeout -> auto shutdown integration
# ===========================================================================


class TestIdleTimeoutIntegration:
    """Test idle timeout triggers auto-shutdown."""

    async def test_idle_timeout_triggers_shutdown_event(self) -> None:
        """AC: Last session unregisters -> wait timeout -> daemon auto-exits."""
        daemon = _make_daemon(idle_timeout=1)
        daemon._shutdown_event = asyncio.Event()

        # Register and unregister to trigger idle timer
        await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s1"})
        )
        await daemon.handle_request(
            DaemonRequest(cmd="session_unregister", data={"session_id": "s1"})
        )

        # Wait for idle timeout
        await asyncio.sleep(1.5)
        assert daemon._shutdown_event.is_set()

    async def test_new_session_prevents_auto_shutdown(self) -> None:
        """New session registration before timeout prevents shutdown."""
        daemon = _make_daemon(idle_timeout=1)
        daemon._shutdown_event = asyncio.Event()

        # Trigger idle timer
        await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s1"})
        )
        await daemon.handle_request(
            DaemonRequest(cmd="session_unregister", data={"session_id": "s1"})
        )

        # Quickly register new session before timeout
        await asyncio.sleep(0.3)
        await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s2"})
        )

        # Wait past timeout -- should NOT trigger shutdown
        await asyncio.sleep(1.0)
        assert not daemon._shutdown_event.is_set()

    async def test_rapid_idle_timer_start_cancel_cycles(self) -> None:
        """Rapid register/unregister cycles should not leak timers."""
        daemon = _make_daemon(idle_timeout=10)

        for i in range(5):
            await daemon.handle_request(
                DaemonRequest(cmd="session_register", data={"session_id": f"s{i}"})
            )
            await daemon.handle_request(
                DaemonRequest(cmd="session_unregister", data={"session_id": f"s{i}"})
            )

        # Only one idle timer should be active (the last one)
        assert daemon._idle_timer is not None
        assert not daemon._idle_timer.done()
        daemon._idle_timer.cancel()


# ===========================================================================
# 5. PID file management edge cases
# ===========================================================================


class TestPIDEdgeCases:
    """Edge cases for PID file management."""

    def test_write_pid_creates_parent_directories(self, tmp_path: Path) -> None:
        """PID file creation should create parent directories if missing."""
        pid_path = tmp_path / "deep" / "nested" / "daemon.pid"
        daemon = _make_daemon()
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._write_pid()
            assert pid_path.exists()
            assert int(pid_path.read_text()) == os.getpid()

    def test_check_pid_with_empty_file(self, tmp_path: Path) -> None:
        """Empty PID file should be treated as corrupt and cleaned up."""
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("")
        daemon = _make_daemon()
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._check_pid()  # Should not raise
            assert not pid_path.exists()

    def test_check_pid_with_whitespace_only(self, tmp_path: Path) -> None:
        """Whitespace-only PID file should be treated as corrupt."""
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("   \n  ")
        daemon = _make_daemon()
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._check_pid()
            assert not pid_path.exists()

    def test_check_pid_with_negative_number(self, tmp_path: Path) -> None:
        """Negative PID should be treated as not alive."""
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("-1")
        daemon = _make_daemon()
        with (
            patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)),
            patch("memx.core.daemon.server._is_process_alive", return_value=False),
        ):
            daemon._check_pid()
            assert not pid_path.exists()

    def test_remove_pid_handles_permission_error(self, tmp_path: Path) -> None:
        """Permission error during PID removal should be handled gracefully."""
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("12345")
        daemon = _make_daemon()
        original_unlink = Path.unlink

        def failing_unlink(self: Path, missing_ok: bool = False) -> None:
            raise OSError("Permission denied")

        with (
            patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)),
            patch.object(Path, "unlink", failing_unlink),
        ):
            daemon._remove_pid()  # Should log warning but not raise

    def test_is_daemon_running_with_corrupt_pid(self, tmp_path: Path) -> None:
        """Class-level is_daemon_running handles corrupt PID file."""
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("corrupt_data")
        with patch("memx.core.daemon.server.PID_PATH", pid_path):
            assert MemXDaemon.is_daemon_running() is False

    def test_read_pid_with_extra_whitespace(self, tmp_path: Path) -> None:
        """Read PID handles leading/trailing whitespace."""
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("  42  \n")
        with patch("memx.core.daemon.server.PID_PATH", pid_path):
            assert MemXDaemon.read_pid() == 42


# ===========================================================================
# 6. DaemonRequest edge cases
# ===========================================================================


class TestDaemonRequestEdgeCases:
    """Additional edge cases for DaemonRequest parsing."""

    def test_from_json_cmd_is_integer_raises(self) -> None:
        """Non-string cmd field should raise ValueError."""
        with pytest.raises(ValueError, match="cmd"):
            DaemonRequest.from_json('{"cmd": 123}')

    def test_from_json_cmd_is_empty_string_raises(self) -> None:
        """Empty string cmd should raise ValueError."""
        with pytest.raises(ValueError, match="cmd"):
            DaemonRequest.from_json('{"cmd": ""}')

    def test_from_json_array_raises(self) -> None:
        """JSON array (not object) should raise ValueError."""
        with pytest.raises(ValueError, match="JSON object"):
            DaemonRequest.from_json("[1, 2, 3]")

    def test_from_json_null_raises(self) -> None:
        """JSON null should raise ValueError."""
        with pytest.raises(ValueError, match="JSON object"):
            DaemonRequest.from_json("null")

    def test_from_json_preserves_nested_data(self) -> None:
        """Complex nested data should be preserved."""
        raw = json.dumps({
            "cmd": "curate",
            "data": {
                "messages": [{"role": "user", "content": "nested"}],
                "user_id": "u1",
            },
        })
        req = DaemonRequest.from_json(raw)
        assert req.data["messages"][0]["content"] == "nested"

    def test_from_json_extra_fields_ignored(self) -> None:
        """Extra fields in JSON should be ignored."""
        raw = '{"cmd": "ping", "extra": "ignored", "data": {"foo": "bar"}}'
        req = DaemonRequest.from_json(raw)
        assert req.cmd == "ping"
        assert req.data == {"foo": "bar"}

    def test_dataclass_equality(self) -> None:
        """Two DaemonRequest instances with same data should be equal."""
        r1 = DaemonRequest(cmd="ping", data={"a": 1})
        r2 = DaemonRequest(cmd="ping", data={"a": 1})
        assert r1 == r2


# ===========================================================================
# 7. DaemonResponse edge cases
# ===========================================================================


class TestDaemonResponseEdgeCases:
    """Additional edge cases for DaemonResponse serialization."""

    def test_response_with_none_error(self) -> None:
        """Response with error=None should not include 'error' key in JSON."""
        resp = DaemonResponse(status="ok", error=None)
        parsed = json.loads(resp.to_json())
        assert "error" not in parsed

    def test_response_with_explicit_empty_string_error(self) -> None:
        """Response with error='' (empty string) should include 'error' key."""
        resp = DaemonResponse(status="error", error="")
        parsed = json.loads(resp.to_json())
        assert "error" in parsed
        assert parsed["error"] == ""

    def test_response_with_large_data(self) -> None:
        """Response with large data payload serializes correctly."""
        large_data = {"items": [{"id": i, "content": f"item-{i}"} for i in range(100)]}
        resp = DaemonResponse(status="ok", data=large_data)
        parsed = json.loads(resp.to_json())
        assert len(parsed["data"]["items"]) == 100

    def test_response_status_field_always_present(self) -> None:
        """Status field is always present in serialized JSON."""
        for status in ("ok", "error", "custom"):
            resp = DaemonResponse(status=status)
            parsed = json.loads(resp.to_json())
            assert parsed["status"] == status


# ===========================================================================
# 8. Config propagation through layers
# ===========================================================================


class TestConfigPropagation:
    """Verify DaemonConfig settings propagate through server, client, transport."""

    def test_custom_socket_path_in_daemon(self) -> None:
        """Custom socket_path in DaemonConfig flows to daemon IPC address."""
        daemon = _make_daemon(socket_path="/custom/daemon.sock")
        assert daemon.ipc_address == "/custom/daemon.sock"

    def test_custom_idle_timeout(self) -> None:
        """Custom idle_timeout_seconds flows to daemon config."""
        daemon = _make_daemon(idle_timeout=42)
        assert daemon._config.idle_timeout_seconds == 42

    def test_default_idle_timeout_constant(self) -> None:
        """DEFAULT_IDLE_TIMEOUT matches DaemonConfig default."""
        assert DEFAULT_IDLE_TIMEOUT == 300
        config = DaemonConfig()
        assert config.idle_timeout_seconds == DEFAULT_IDLE_TIMEOUT

    def test_client_inherits_config(self) -> None:
        """DaemonClient receives and stores the provided DaemonConfig."""
        config = DaemonConfig(enabled=True, socket_path="/tmp/test.sock")
        client = DaemonClient(config=config)
        assert client._config.socket_path == "/tmp/test.sock"
        assert client._config.enabled is True

    def test_transport_factory_with_custom_config(self) -> None:
        """get_transport passes socket_path from config to the transport."""
        config = DaemonConfig(socket_path="/tmp/custom.sock")
        with patch("memx.core.daemon.ipc.sys") as mock_sys:
            mock_sys.platform = "linux"
            transport = get_transport(config)
            assert isinstance(transport, UnixSocketTransport)
            assert transport._socket_path == "/tmp/custom.sock"

    def test_transport_factory_windows_custom_pipe(self) -> None:
        """get_transport passes custom pipe name on Windows."""
        config = DaemonConfig(socket_path=r"\\.\pipe\custom")
        with patch("memx.core.daemon.ipc.sys") as mock_sys:
            mock_sys.platform = "win32"
            transport = get_transport(config)
            assert isinstance(transport, NamedPipeTransport)
            assert transport._pipe_name == r"\\.\pipe\custom"

    def test_fallback_manager_uses_config(self) -> None:
        """FallbackManager passes config to its internal DaemonClient."""
        config = DaemonConfig(enabled=True, socket_path="/tmp/special.sock")
        mgr = DaemonFallbackManager(config=config)
        assert mgr.client._config.socket_path == "/tmp/special.sock"


# ===========================================================================
# 9. Client <-> Fallback degradation integration
# ===========================================================================


class TestClientFallbackIntegration:
    """Integration tests for DaemonClient and DaemonFallbackManager."""

    async def test_fallback_try_recall_uses_client_recall(self) -> None:
        """try_recall delegates to DaemonClient.recall when available."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config)
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._client.recall = AsyncMock(return_value=[{"m": "r", "s": 0.9}])
        mgr._available = True

        result = await mgr.try_recall("test query", user_id="u1", limit=3)
        assert result is not None
        assert result[0]["m"] == "r"
        mgr._client.recall.assert_awaited_once_with("test query", user_id="u1", limit=3)

    async def test_fallback_try_curate_uses_client_curate(self) -> None:
        """try_curate delegates to DaemonClient.curate when available."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config)
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._client.curate = AsyncMock(return_value={"added": 1})
        mgr._available = True

        result = await mgr.try_curate("msg", user_id="u1")
        assert result == {"added": 1}
        mgr._client.curate.assert_awaited_once_with("msg", user_id="u1")

    async def test_client_failure_degrades_fallback(self) -> None:
        """DaemonUnavailableError from client marks fallback as degraded."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config, recovery_cooldown=0.0)
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=False)
        mgr._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("connection lost")
        )
        mgr._available = True

        result = await mgr.try_recall("query")
        assert result is None
        assert mgr.is_available is False

    async def test_client_recovery_restores_fallback(self) -> None:
        """Successful ping during recovery check restores availability."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config, recovery_cooldown=0.0)
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._available = False
        mgr._last_failure_time = 0.0

        result = await mgr.check_recovery()
        assert result is True
        assert mgr.is_available is True

    async def test_async_tick_counter_triggers_recovery(self) -> None:
        """Async tick counter triggers recovery ping at the correct interval."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(
            config=config, recovery_interval=3, recovery_cooldown=0.0
        )
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._client.recall = AsyncMock(return_value=[])
        mgr._available = False
        mgr._last_failure_time = 0.0

        # Tick 3 times via try_recall (each call ticks once)
        for _ in range(3):
            await mgr.try_recall("q")

        # After 3 ticks, recovery should have been triggered
        assert mgr.is_available is True

    async def test_async_tick_counter_failed_recovery_updates_time(self) -> None:
        """Failed recovery ping updates the last_failure_time."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(
            config=config, recovery_interval=1, recovery_cooldown=0.0
        )
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=False)
        mgr._available = False
        mgr._last_failure_time = 0.0

        before = time.monotonic()
        await mgr._async_tick_op_counter()
        assert mgr._last_failure_time >= before
        assert mgr.is_available is False


# ===========================================================================
# 10. Memory <-> Daemon fallback integration
# ===========================================================================


class TestMemoryDaemonIntegration:
    """Integration tests for Memory class with daemon fallback."""

    def test_search_via_daemon_wraps_in_results(self) -> None:
        """Memory.search wraps daemon recall results in {results: [...]}."""
        m = _make_memory_with_daemon(daemon_available=True)
        result = m.search("query", user_id="u1")
        assert "results" in result
        assert result["results"][0]["memory"] == "daemon-result"
        m._mem0.search.assert_not_called()

    def test_add_via_daemon_returns_dict(self) -> None:
        """Memory.add returns daemon curate result."""
        m = _make_memory_with_daemon(daemon_available=True)
        result = m.add("msg", user_id="u1")
        assert result["ace_ingest"]["bullets_added"] == 2
        m._mem0.add.assert_not_called()

    def test_search_fallback_to_mem0_when_unavailable(self) -> None:
        """Memory.search falls back to mem0 when daemon unavailable."""
        m = _make_memory_with_daemon(daemon_available=False)
        result = m.search("query", user_id="u1")
        m._mem0.search.assert_called_once()
        assert result["results"][0]["memory"] == "direct-search"

    def test_add_fallback_to_mem0_when_unavailable(self) -> None:
        """Memory.add falls back to mem0 when daemon unavailable."""
        m = _make_memory_with_daemon(daemon_available=False)
        result = m.add("msg", user_id="u1")
        m._mem0.add.assert_called_once()
        assert result["results"][0]["memory"] == "direct-add"

    def test_search_fallback_on_mid_request_crash(self) -> None:
        """Daemon crash during search triggers fallback to mem0."""
        m = _make_memory_with_daemon(daemon_available=True)
        m._daemon_fallback._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("crash")
        )
        result = m.search("query", user_id="u1")
        m._mem0.search.assert_called_once()
        assert m._daemon_fallback.is_available is False

    def test_add_fallback_on_mid_request_crash(self) -> None:
        """Daemon crash during add triggers fallback to mem0."""
        m = _make_memory_with_daemon(daemon_available=True)
        m._daemon_fallback._client.curate = AsyncMock(
            side_effect=DaemonUnavailableError("crash")
        )
        result = m.add("msg", user_id="u1")
        m._mem0.add.assert_called_once()
        assert m._daemon_fallback.is_available is False

    def test_daemon_available_property_true(self) -> None:
        """Memory.daemon_available returns True when daemon is up."""
        m = _make_memory_with_daemon(daemon_available=True)
        assert m.daemon_available is True

    def test_daemon_available_property_false_after_crash(self) -> None:
        """Memory.daemon_available becomes False after daemon crash."""
        m = _make_memory_with_daemon(daemon_available=True)
        m._daemon_fallback._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("crash")
        )
        m.search("query")  # Triggers crash
        assert m.daemon_available is False

    def test_daemon_available_property_no_fallback(self) -> None:
        """Memory.daemon_available returns False when no fallback manager."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._daemon_fallback = None
        assert m.daemon_available is False


# ===========================================================================
# 11. Server lifecycle edge cases
# ===========================================================================


class TestServerLifecycleEdgeCases:
    """Edge cases in server start/stop lifecycle."""

    async def test_stop_idempotent(self, tmp_path: Path) -> None:
        """Calling stop() twice should be safe."""
        daemon = _make_daemon()
        pid_path = tmp_path / "daemon.pid"
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._running = True
            daemon._server = MagicMock()
            daemon._server.close = MagicMock()
            daemon._server.wait_closed = AsyncMock()
            await daemon.stop()
            assert daemon.is_running is False

            # Second stop should be safe
            await daemon.stop()
            assert daemon.is_running is False

    async def test_stop_cleans_sessions(self, tmp_path: Path) -> None:
        """Stop clears all sessions."""
        daemon = _make_daemon()
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text(str(os.getpid()))
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._running = True
            daemon._server = MagicMock()
            daemon._server.close = MagicMock()
            daemon._server.wait_closed = AsyncMock()
            daemon._sessions = {
                "s1": datetime.now(),
                "s2": datetime.now(),
                "s3": datetime.now(),
            }
            await daemon.stop()
            assert daemon.session_count == 0

    async def test_stop_cancels_idle_timer(self, tmp_path: Path) -> None:
        """Stop cancels any running idle timer."""
        daemon = _make_daemon()
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text(str(os.getpid()))
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._running = True
            daemon._server = MagicMock()
            daemon._server.close = MagicMock()
            daemon._server.wait_closed = AsyncMock()
            daemon._idle_timer = asyncio.get_event_loop().create_task(asyncio.sleep(999))

            await daemon.stop()
            assert daemon._idle_timer is None or daemon._idle_timer.cancelled()

    async def test_start_with_alive_pid_raises(self, tmp_path: Path) -> None:
        """Start fails with DaemonError when another daemon is alive."""
        daemon = _make_daemon()
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("12345")
        with (
            patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)),
            patch("memx.core.daemon.server._is_process_alive", return_value=True),
        ):
            with pytest.raises(DaemonError, match="already running"):
                await daemon.start()

    async def test_shutdown_during_active_sessions(self) -> None:
        """Shutdown command while sessions are active should still work."""
        daemon = _make_daemon()
        daemon._shutdown_event = asyncio.Event()
        daemon._sessions["s1"] = datetime.now()
        daemon._sessions["s2"] = datetime.now()

        resp = await daemon.handle_request(DaemonRequest(cmd="shutdown"))
        assert resp.status == "ok"
        assert daemon._shutdown_event.is_set()
        # Sessions are not cleared by shutdown command itself (that happens in stop())
        assert daemon.session_count == 2

    async def test_shutdown_without_event(self) -> None:
        """Shutdown command when _shutdown_event is None should still return ok."""
        daemon = _make_daemon()
        daemon._shutdown_event = None
        resp = await daemon.handle_request(DaemonRequest(cmd="shutdown"))
        assert resp.status == "ok"

    def test_uptime_seconds_before_start(self) -> None:
        """_uptime_seconds returns 0 before daemon starts."""
        daemon = _make_daemon()
        assert daemon._uptime_seconds() == 0.0

    def test_uptime_seconds_after_start(self) -> None:
        """_uptime_seconds returns positive after _start_time is set."""
        daemon = _make_daemon()
        daemon._start_time = datetime.now()
        # Sleep briefly
        import time as _time

        _time.sleep(0.05)
        assert daemon._uptime_seconds() > 0.0


# ===========================================================================
# 12. IPC connection handler edge cases
# ===========================================================================


class TestConnectionHandlerEdgeCases:
    """Edge cases for _handle_connection."""

    async def test_handle_connection_writer_drain_error(self) -> None:
        """Error during writer.drain should be handled gracefully."""
        daemon = _make_daemon()
        daemon._start_time = datetime.now()
        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)

        reader.read = AsyncMock(
            return_value=json.dumps({"cmd": "ping"}).encode("utf-8")
        )
        writer.write = MagicMock()
        writer.drain = AsyncMock(side_effect=ConnectionError("pipe broken"))
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        # Should not raise -- handled in the outer try/except
        await daemon._handle_connection(reader, writer)

    async def test_handle_connection_reader_exception(self) -> None:
        """Exception during reader.read should be handled gracefully."""
        daemon = _make_daemon()
        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)

        reader.read = AsyncMock(side_effect=ConnectionResetError("connection reset"))
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon._handle_connection(reader, writer)
        # Writer should be closed in finally
        writer.close.assert_called()

    async def test_handle_connection_writer_close_error(self) -> None:
        """Error during writer.close in finally should not propagate."""
        daemon = _make_daemon()
        daemon._start_time = datetime.now()
        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)

        reader.read = AsyncMock(
            return_value=json.dumps({"cmd": "ping"}).encode("utf-8")
        )
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock(side_effect=OSError("close failed"))
        writer.wait_closed = AsyncMock()

        # Should not raise
        await daemon._handle_connection(reader, writer)

    async def test_handle_connection_processes_all_commands(self) -> None:
        """Verify connection handler routes requests to handle_request properly."""
        daemon = _make_daemon()
        daemon._start_time = datetime.now()

        for cmd in ("ping", "session_register", "session_unregister"):
            reader = AsyncMock(spec=asyncio.StreamReader)
            writer = AsyncMock(spec=asyncio.StreamWriter)

            data: dict[str, Any] = {"cmd": cmd}
            if "session" in cmd:
                data["data"] = {"session_id": f"test-{cmd}"}

            reader.read = AsyncMock(return_value=json.dumps(data).encode("utf-8"))
            writer.write = MagicMock()
            writer.drain = AsyncMock()
            writer.close = MagicMock()
            writer.wait_closed = AsyncMock()

            await daemon._handle_connection(reader, writer)
            writer.write.assert_called_once()
            raw = writer.write.call_args[0][0]
            resp = json.loads(raw.decode("utf-8"))
            assert resp["status"] == "ok"


# ===========================================================================
# 13. Error handling in command handlers
# ===========================================================================


class TestCommandErrorHandling:
    """Test that exceptions in command handlers return error responses."""

    async def test_recall_with_search_raising_unexpected_error(self) -> None:
        """Unexpected error in Memory.search returns error response."""
        daemon = _make_daemon()
        daemon._memory = MagicMock()
        daemon._memory.search.side_effect = TypeError("unexpected type")
        resp = await daemon.handle_request(
            DaemonRequest(cmd="recall", data={"query": "test"})
        )
        assert resp.status == "error"
        assert "unexpected type" in (resp.error or "")

    async def test_curate_with_add_raising_unexpected_error(self) -> None:
        """Unexpected error in Memory.add returns error response."""
        daemon = _make_daemon()
        daemon._memory = MagicMock()
        daemon._memory.add.side_effect = IOError("disk full")
        resp = await daemon.handle_request(
            DaemonRequest(cmd="curate", data={"messages": "test"})
        )
        assert resp.status == "error"
        assert "disk full" in (resp.error or "")

    async def test_recall_missing_user_id_defaults_to_none(self) -> None:
        """Recall without user_id passes None."""
        daemon = _make_daemon()
        daemon._memory = MagicMock()
        daemon._memory.search.return_value = {"results": []}
        resp = await daemon.handle_request(
            DaemonRequest(cmd="recall", data={"query": "test"})
        )
        assert resp.status == "ok"
        daemon._memory.search.assert_called_once_with(query="test", user_id=None, limit=5)

    async def test_curate_missing_user_id_defaults_to_none(self) -> None:
        """Curate without user_id passes None."""
        daemon = _make_daemon()
        daemon._memory = MagicMock()
        daemon._memory.add.return_value = {}
        resp = await daemon.handle_request(
            DaemonRequest(cmd="curate", data={"messages": "test"})
        )
        assert resp.status == "ok"
        daemon._memory.add.assert_called_once_with(messages="test", user_id=None)


# ===========================================================================
# 14. __init__.py exports
# ===========================================================================


class TestDaemonExports:
    """Verify daemon __init__.py exports all expected symbols."""

    def test_all_exports_present(self) -> None:
        """All expected public classes/functions are exported from memx.core.daemon."""
        from memx.core import daemon

        expected = [
            "DaemonClient",
            "DaemonFallbackManager",
            "DaemonRequest",
            "DaemonResponse",
            "IPCTransport",
            "MemXDaemon",
            "NamedPipeTransport",
            "UnixSocketTransport",
            "get_transport",
        ]
        for name in expected:
            assert hasattr(daemon, name), f"Missing export: {name}"

    def test_all_list_matches_exports(self) -> None:
        """__all__ contains exactly the expected public names."""
        from memx.core.daemon import __all__

        assert set(__all__) == {
            "DaemonClient",
            "DaemonFallbackManager",
            "DaemonRequest",
            "DaemonResponse",
            "IPCTransport",
            "MemXDaemon",
            "NamedPipeTransport",
            "UnixSocketTransport",
            "get_transport",
        }


# ===========================================================================
# 15. Exception hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    """Verify exception class relationships."""

    def test_daemon_error_is_memx_error(self) -> None:
        from memx.core.exceptions import MemXError

        assert issubclass(DaemonError, MemXError)

    def test_daemon_unavailable_is_daemon_error(self) -> None:
        assert issubclass(DaemonUnavailableError, DaemonError)

    def test_daemon_unavailable_is_connection_error(self) -> None:
        assert issubclass(DaemonUnavailableError, ConnectionError)

    def test_daemon_unavailable_catchable_as_memx_error(self) -> None:
        from memx.core.exceptions import MemXError

        with pytest.raises(MemXError):
            raise DaemonUnavailableError("test")

    def test_daemon_error_message_preserved(self) -> None:
        err = DaemonError("custom message here")
        assert str(err) == "custom message here"


# ===========================================================================
# 16. Platform-specific branching
# ===========================================================================


class TestPlatformBranching:
    """Tests for platform-specific behavior."""

    def test_ipc_address_windows_default(self) -> None:
        """On Windows without custom path, returns PIPE_NAME."""
        daemon = _make_daemon()
        daemon._config = DaemonConfig()
        with patch("memx.core.daemon.server.sys") as mock_sys:
            mock_sys.platform = "win32"
            assert daemon.ipc_address == PIPE_NAME

    def test_ipc_address_linux_default(self) -> None:
        """On Linux without custom path, returns SOCKET_PATH."""
        daemon = _make_daemon()
        daemon._config = DaemonConfig()
        with patch("memx.core.daemon.server.sys") as mock_sys:
            mock_sys.platform = "linux"
            assert daemon.ipc_address == str(SOCKET_PATH)

    def test_ipc_address_custom_overrides_platform(self) -> None:
        """Custom socket_path overrides platform default."""
        daemon = _make_daemon(socket_path="/custom/path.sock")
        assert daemon.ipc_address == "/custom/path.sock"

    def test_get_transport_darwin(self) -> None:
        """macOS uses UnixSocketTransport."""
        with patch("memx.core.daemon.ipc.sys") as mock_sys:
            mock_sys.platform = "darwin"
            transport = get_transport()
            assert isinstance(transport, UnixSocketTransport)


# ===========================================================================
# 17. DaemonConfig validation
# ===========================================================================


class TestDaemonConfigValidation:
    """Tests for DaemonConfig validation and defaults."""

    def test_default_values(self) -> None:
        config = DaemonConfig()
        assert config.enabled is False
        assert config.idle_timeout_seconds == 300
        assert config.socket_path is None

    def test_custom_values(self) -> None:
        config = DaemonConfig(enabled=True, idle_timeout_seconds=60, socket_path="/tmp/s.sock")
        assert config.enabled is True
        assert config.idle_timeout_seconds == 60
        assert config.socket_path == "/tmp/s.sock"

    def test_idle_timeout_must_be_positive(self) -> None:
        """idle_timeout_seconds must be > 0."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            DaemonConfig(idle_timeout_seconds=0)

    def test_memx_config_includes_daemon(self) -> None:
        """MemXConfig.daemon defaults to DaemonConfig."""
        config = MemXConfig()
        assert isinstance(config.daemon, DaemonConfig)
        assert config.daemon.enabled is False

    def test_memx_config_from_dict_with_daemon(self) -> None:
        """DaemonConfig can be set through MemXConfig.from_dict."""
        config = MemXConfig.from_dict({"daemon": {"enabled": True, "idle_timeout_seconds": 120}})
        assert config.daemon.enabled is True
        assert config.daemon.idle_timeout_seconds == 120


# ===========================================================================
# 18. Server constants
# ===========================================================================


class TestServerConstants:
    """Verify server-level constants are defined correctly."""

    def test_pipe_name(self) -> None:
        assert PIPE_NAME == r"\\.\pipe\memx-daemon"

    def test_socket_path_in_home(self) -> None:
        assert "daemon.sock" in str(SOCKET_PATH)

    def test_pid_path_in_home(self) -> None:
        assert "daemon.pid" in str(PID_PATH)

    def test_max_request_size(self) -> None:
        assert MAX_REQUEST_SIZE == 1 * 1024 * 1024

    def test_default_idle_timeout(self) -> None:
        assert DEFAULT_IDLE_TIMEOUT == 300


# ===========================================================================
# 19. Concurrent operations
# ===========================================================================


class TestConcurrentOperations:
    """Test concurrent command handling on the daemon."""

    async def test_concurrent_session_registrations(self) -> None:
        """Multiple concurrent session registrations should all succeed."""
        daemon = _make_daemon()

        async def register(sid: str) -> DaemonResponse:
            return await daemon.handle_request(
                DaemonRequest(cmd="session_register", data={"session_id": sid})
            )

        results = await asyncio.gather(
            register("s1"),
            register("s2"),
            register("s3"),
            register("s4"),
            register("s5"),
        )

        assert all(r.status == "ok" for r in results)
        assert daemon.session_count == 5

    async def test_concurrent_recall_requests(self) -> None:
        """Multiple concurrent recall requests should all get responses."""
        daemon = _make_daemon()
        daemon._memory = MagicMock()
        daemon._memory.search.return_value = {"results": []}

        async def recall(query: str) -> DaemonResponse:
            return await daemon.handle_request(
                DaemonRequest(cmd="recall", data={"query": query})
            )

        results = await asyncio.gather(
            recall("q1"), recall("q2"), recall("q3")
        )
        assert all(r.status == "ok" for r in results)
        assert daemon._memory.search.call_count == 3

    async def test_concurrent_mixed_commands(self) -> None:
        """Mixed concurrent commands should all be handled."""
        daemon = _make_daemon()
        daemon._start_time = datetime.now()
        daemon._memory = MagicMock()
        daemon._memory.search.return_value = {"results": []}

        results = await asyncio.gather(
            daemon.handle_request(DaemonRequest(cmd="ping")),
            daemon.handle_request(
                DaemonRequest(cmd="session_register", data={"session_id": "s1"})
            ),
            daemon.handle_request(
                DaemonRequest(cmd="recall", data={"query": "test"})
            ),
        )
        assert all(r.status == "ok" for r in results)


# ===========================================================================
# 20. Fallback manager sync tick counter
# ===========================================================================


class TestSyncTickCounter:
    """Tests for the synchronous _tick_op_counter used from Memory."""

    def test_sync_tick_increments_counter(self) -> None:
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config, recovery_interval=10)
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._available = True

        mgr._tick_op_counter()
        assert mgr.op_counter == 1
        mgr._tick_op_counter()
        assert mgr.op_counter == 2

    def test_sync_tick_no_recovery_when_available(self) -> None:
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config, recovery_interval=1, recovery_cooldown=0.0)
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._available = True

        mgr._tick_op_counter()
        # Should not call ping since already available
        mgr._client.ping.assert_not_called()

    def test_sync_tick_recovery_on_interval(self) -> None:
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config, recovery_interval=2, recovery_cooldown=0.0)
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._available = False
        mgr._last_failure_time = 0.0

        mgr._tick_op_counter()  # op 1, not at interval
        assert mgr.is_available is False

        mgr._tick_op_counter()  # op 2, at interval -> triggers recovery
        assert mgr.is_available is True

    def test_sync_tick_exception_handled(self) -> None:
        """Exception during sync ping should not crash."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(config=config, recovery_interval=1, recovery_cooldown=0.0)
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.ping = AsyncMock(side_effect=Exception("boom"))
        mgr._available = False
        mgr._last_failure_time = 0.0

        mgr._tick_op_counter()  # Should not raise
        assert mgr.is_available is False


# ===========================================================================
# 21. Full degradation -> recovery -> re-degradation cycle
# ===========================================================================


class TestFullDegradationCycle:
    """Full integration test for degradation and recovery lifecycle."""

    async def test_available_crash_degrade_recover_available(self) -> None:
        """Full cycle: available -> crash -> degraded -> recover -> available again."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(
            config=config, recovery_interval=2, recovery_cooldown=0.0
        )
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._client.recall = AsyncMock(return_value=[{"memory": "ok", "score": 1.0}])
        mgr._client.curate = AsyncMock(return_value={"added": 1})
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._available = True

        # Phase 1: Normal operation
        result = await mgr.try_recall("q1")
        assert result is not None
        assert mgr.is_available is True

        # Phase 2: Daemon crashes
        mgr._client.recall = AsyncMock(
            side_effect=DaemonUnavailableError("crash")
        )
        result = await mgr.try_recall("q2")
        assert result is None
        assert mgr.is_available is False

        # Phase 3: Degraded -- operations return None
        result = await mgr.try_curate("msg")
        assert result is None

        # Phase 4: Recovery triggered (we need 2 more ticks to reach interval=2)
        mgr._client.ping = AsyncMock(return_value=True)
        mgr._client.recall = AsyncMock(return_value=[{"memory": "back", "score": 0.9}])
        # op_counter is now 3 (from q1, q2, msg), next tick at 4 = interval multiple of 2
        result = await mgr.try_recall("q3")
        # At op 4, recovery check should be triggered
        assert mgr.is_available is True

        # Phase 5: Normal operation restored
        result = await mgr.try_recall("q4")
        assert result is not None
        assert result[0]["memory"] == "back"

    async def test_flapping_daemon(self) -> None:
        """Daemon that repeatedly goes up and down."""
        config = DaemonConfig(enabled=True)
        mgr = DaemonFallbackManager(
            config=config, recovery_interval=1, recovery_cooldown=0.0
        )
        mgr._client = MagicMock(spec=DaemonClient)
        mgr._available = True

        for i in range(3):
            # Crash
            mgr._client.recall = AsyncMock(
                side_effect=DaemonUnavailableError(f"crash-{i}")
            )
            result = await mgr.try_recall(f"q-crash-{i}")
            assert result is None
            assert mgr.is_available is False

            # Recover
            mgr._client.ping = AsyncMock(return_value=True)
            mgr._client.recall = AsyncMock(return_value=[{"m": f"ok-{i}"}])
            mgr._last_failure_time = 0.0  # Reset cooldown
            result = await mgr.try_recall(f"q-recover-{i}")
            # Should trigger recovery at interval=1
            assert mgr.is_available is True


# ===========================================================================
# 22. _is_process_alive helper
# ===========================================================================


class TestIsProcessAliveHelper:
    """Additional tests for _is_process_alive."""

    def test_own_process_alive(self) -> None:
        assert _is_process_alive(os.getpid()) is True

    def test_impossible_pid(self) -> None:
        # PID that almost certainly doesn't exist
        assert _is_process_alive(2_000_000_000) is False

    def test_zero_pid(self) -> None:
        """PID 0 may behave differently per platform (special process)."""
        # Just verify it doesn't crash
        result = _is_process_alive(0)
        assert isinstance(result, bool)


# ===========================================================================
# 23. Transport abstract base class
# ===========================================================================


class TestIPCTransportABC:
    """Test that IPCTransport ABC cannot be instantiated directly."""

    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            IPCTransport()  # type: ignore[abstract]

    def test_mock_transport_is_valid_subclass(self) -> None:
        transport = MockTransport()
        assert isinstance(transport, IPCTransport)


# ===========================================================================
# 24. Memory _init_daemon_fallback edge cases
# ===========================================================================


class TestInitDaemonFallbackEdge:
    """Edge cases for Memory._init_daemon_fallback."""

    def test_init_when_daemon_disabled(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=False))
        m._daemon_fallback = None
        m._init_daemon_fallback()
        assert m._daemon_fallback is None

    def test_init_when_daemon_enabled_but_unreachable(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=True))
        m._daemon_fallback = None
        with patch.object(DaemonClient, "ping", new_callable=AsyncMock, return_value=False):
            m._init_daemon_fallback()
        assert m._daemon_fallback is not None
        assert m._daemon_fallback.is_available is False

    def test_init_daemon_fallback_import_error(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=True))
        m._daemon_fallback = None
        with patch(
            "memx.core.daemon.fallback.DaemonFallbackManager",
            side_effect=ImportError("no module"),
        ):
            m._init_daemon_fallback()
        assert m._daemon_fallback is None

    def test_init_daemon_fallback_runtime_error(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(daemon=DaemonConfig(enabled=True))
        m._daemon_fallback = None
        with patch(
            "memx.core.daemon.fallback.DaemonFallbackManager",
            side_effect=RuntimeError("broken"),
        ):
            m._init_daemon_fallback()
        assert m._daemon_fallback is None
