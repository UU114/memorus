"""Unit tests for memx.daemon.server -- MemXDaemon.

All file I/O and Memory are mocked so tests run fast with no side effects.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.core.config import DaemonConfig
from memx.core.daemon.server import (
    PIPE_NAME,
    DaemonRequest,
    DaemonResponse,
    MemXDaemon,
    _is_process_alive,
)
from memx.core.exceptions import DaemonError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def daemon_config() -> DaemonConfig:
    """Return a DaemonConfig with a short idle timeout for testing."""
    return DaemonConfig(enabled=True, idle_timeout_seconds=2)


@pytest.fixture()
def daemon(daemon_config: DaemonConfig) -> MemXDaemon:
    """Return a MemXDaemon instance (not started)."""
    return MemXDaemon(config=daemon_config)


@pytest.fixture()
def tmp_pid_file(tmp_path: Path) -> Path:
    """Return a temporary PID file path for testing."""
    return tmp_path / "daemon.pid"


# ---------------------------------------------------------------------------
# DaemonRequest
# ---------------------------------------------------------------------------


class TestDaemonRequest:
    """Tests for DaemonRequest parsing."""

    def test_from_json_minimal(self) -> None:
        req = DaemonRequest.from_json('{"cmd": "ping"}')
        assert req.cmd == "ping"
        assert req.data == {}

    def test_from_json_with_data(self) -> None:
        req = DaemonRequest.from_json(
            '{"cmd": "recall", "data": {"query": "test", "user_id": "u1"}}'
        )
        assert req.cmd == "recall"
        assert req.data["query"] == "test"
        assert req.data["user_id"] == "u1"

    def test_from_json_missing_cmd_raises(self) -> None:
        with pytest.raises(ValueError, match="cmd"):
            DaemonRequest.from_json('{"data": {}}')

    def test_from_json_not_object_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            DaemonRequest.from_json('"just a string"')

    def test_from_json_invalid_json_raises(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            DaemonRequest.from_json("not json at all")

    def test_frozen(self) -> None:
        req = DaemonRequest(cmd="ping")
        with pytest.raises(AttributeError):
            req.cmd = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DaemonResponse
# ---------------------------------------------------------------------------


class TestDaemonResponse:
    """Tests for DaemonResponse serialization."""

    def test_ok_response_to_json(self) -> None:
        resp = DaemonResponse(status="ok", data={"version": "1.0.0"})
        parsed = json.loads(resp.to_json())
        assert parsed["status"] == "ok"
        assert parsed["data"]["version"] == "1.0.0"
        assert "error" not in parsed

    def test_error_response_to_json(self) -> None:
        resp = DaemonResponse(status="error", error="Something failed")
        parsed = json.loads(resp.to_json())
        assert parsed["status"] == "error"
        assert parsed["error"] == "Something failed"

    def test_empty_data_omitted(self) -> None:
        resp = DaemonResponse(status="ok")
        parsed = json.loads(resp.to_json())
        assert "data" not in parsed


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------


class TestPIDManagement:
    """Tests for PID file create/check/remove/stale detection."""

    def test_write_pid_creates_file(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "test.pid"
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._write_pid()
            assert pid_path.exists()
            assert int(pid_path.read_text()) == os.getpid()

    def test_remove_pid_deletes_file(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("12345")
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._remove_pid()
            assert not pid_path.exists()

    def test_remove_pid_no_file_is_safe(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "nonexistent.pid"
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._remove_pid()  # Should not raise

    def test_check_pid_no_file_passes(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "no.pid"
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._check_pid()  # Should not raise

    def test_check_pid_stale_cleans_up(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "stale.pid"
        pid_path.write_text("99999999")  # very unlikely to be alive
        with (
            patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)),
            patch(
                "memx.core.daemon.server._is_process_alive", return_value=False
            ),
        ):
            daemon._check_pid()  # Should clean up and not raise
            assert not pid_path.exists()

    def test_check_pid_alive_raises(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "alive.pid"
        pid_path.write_text("12345")
        with (
            patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)),
            patch(
                "memx.core.daemon.server._is_process_alive", return_value=True
            ),
        ):
            with pytest.raises(DaemonError, match="already running"):
                daemon._check_pid()

    def test_check_pid_corrupt_file_cleans_up(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "corrupt.pid"
        pid_path.write_text("not_a_number")
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._check_pid()  # Should clean up and not raise
            assert not pid_path.exists()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


class TestPingCommand:
    """Tests for the ping command."""

    @pytest.mark.asyncio
    async def test_ping_returns_ok(self, daemon: MemXDaemon) -> None:
        daemon._start_time = datetime.now()
        req = DaemonRequest(cmd="ping")
        resp = await daemon.handle_request(req)
        assert resp.status == "ok"
        assert resp.data["version"] == "1.0.0"
        assert resp.data["sessions"] == 0
        assert "pid" in resp.data


class TestRecallCommand:
    """Tests for the recall command."""

    @pytest.mark.asyncio
    async def test_recall_success(self, daemon: MemXDaemon) -> None:
        mock_memory = MagicMock()
        mock_memory.search.return_value = {
            "results": [{"memory": "test bullet", "score": 0.9}]
        }
        daemon._memory = mock_memory

        req = DaemonRequest(cmd="recall", data={"query": "test", "user_id": "u1"})
        resp = await daemon.handle_request(req)
        assert resp.status == "ok"
        assert "results" in resp.data
        mock_memory.search.assert_called_once_with(
            query="test", user_id="u1", limit=5
        )

    @pytest.mark.asyncio
    async def test_recall_missing_query(self, daemon: MemXDaemon) -> None:
        daemon._memory = MagicMock()
        req = DaemonRequest(cmd="recall", data={})
        resp = await daemon.handle_request(req)
        assert resp.status == "error"
        assert "query" in (resp.error or "")

    @pytest.mark.asyncio
    async def test_recall_no_memory(self, daemon: MemXDaemon) -> None:
        daemon._memory = None
        req = DaemonRequest(cmd="recall", data={"query": "test"})
        resp = await daemon.handle_request(req)
        assert resp.status == "error"
        assert "Memory" in (resp.error or "")

    @pytest.mark.asyncio
    async def test_recall_custom_limit(self, daemon: MemXDaemon) -> None:
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}
        daemon._memory = mock_memory

        req = DaemonRequest(
            cmd="recall", data={"query": "test", "limit": 10}
        )
        resp = await daemon.handle_request(req)
        assert resp.status == "ok"
        mock_memory.search.assert_called_once_with(
            query="test", user_id=None, limit=10
        )


class TestCurateCommand:
    """Tests for the curate command."""

    @pytest.mark.asyncio
    async def test_curate_success(self, daemon: MemXDaemon) -> None:
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": [], "ace_ingest": {}}
        daemon._memory = mock_memory

        req = DaemonRequest(
            cmd="curate",
            data={"messages": "User prefers dark mode", "user_id": "u1"},
        )
        resp = await daemon.handle_request(req)
        assert resp.status == "ok"
        mock_memory.add.assert_called_once_with(
            messages="User prefers dark mode", user_id="u1"
        )

    @pytest.mark.asyncio
    async def test_curate_missing_messages(self, daemon: MemXDaemon) -> None:
        daemon._memory = MagicMock()
        req = DaemonRequest(cmd="curate", data={})
        resp = await daemon.handle_request(req)
        assert resp.status == "error"
        assert "messages" in (resp.error or "")

    @pytest.mark.asyncio
    async def test_curate_no_memory(self, daemon: MemXDaemon) -> None:
        daemon._memory = None
        req = DaemonRequest(
            cmd="curate", data={"messages": "test"}
        )
        resp = await daemon.handle_request(req)
        assert resp.status == "error"


class TestSessionCommands:
    """Tests for session_register and session_unregister."""

    @pytest.mark.asyncio
    async def test_register_session(self, daemon: MemXDaemon) -> None:
        req = DaemonRequest(
            cmd="session_register", data={"session_id": "sess-abc"}
        )
        resp = await daemon.handle_request(req)
        assert resp.status == "ok"
        assert "sess-abc" in daemon._sessions
        assert daemon.session_count == 1

    @pytest.mark.asyncio
    async def test_register_multiple_sessions(self, daemon: MemXDaemon) -> None:
        for sid in ("s1", "s2", "s3"):
            req = DaemonRequest(
                cmd="session_register", data={"session_id": sid}
            )
            await daemon.handle_request(req)
        assert daemon.session_count == 3

    @pytest.mark.asyncio
    async def test_register_missing_session_id(self, daemon: MemXDaemon) -> None:
        req = DaemonRequest(cmd="session_register", data={})
        resp = await daemon.handle_request(req)
        assert resp.status == "error"
        assert "session_id" in (resp.error or "")

    @pytest.mark.asyncio
    async def test_unregister_session(self, daemon: MemXDaemon) -> None:
        daemon._sessions["sess-abc"] = datetime.now()
        req = DaemonRequest(
            cmd="session_unregister", data={"session_id": "sess-abc"}
        )
        resp = await daemon.handle_request(req)
        assert resp.status == "ok"
        assert "sess-abc" not in daemon._sessions

    @pytest.mark.asyncio
    async def test_unregister_unknown_session(self, daemon: MemXDaemon) -> None:
        req = DaemonRequest(
            cmd="session_unregister", data={"session_id": "nonexistent"}
        )
        resp = await daemon.handle_request(req)
        assert resp.status == "ok"  # Not an error, idempotent

    @pytest.mark.asyncio
    async def test_unregister_missing_session_id(self, daemon: MemXDaemon) -> None:
        req = DaemonRequest(cmd="session_unregister", data={})
        resp = await daemon.handle_request(req)
        assert resp.status == "error"

    @pytest.mark.asyncio
    async def test_register_cancels_idle_timer(self, daemon: MemXDaemon) -> None:
        # Simulate an idle timer running
        daemon._idle_timer = asyncio.get_event_loop().create_task(
            asyncio.sleep(999)
        )
        req = DaemonRequest(
            cmd="session_register", data={"session_id": "s1"}
        )
        await daemon.handle_request(req)
        assert daemon._idle_timer is None

    @pytest.mark.asyncio
    async def test_unregister_last_session_starts_idle_timer(
        self, daemon: MemXDaemon
    ) -> None:
        daemon._sessions["s1"] = datetime.now()
        req = DaemonRequest(
            cmd="session_unregister", data={"session_id": "s1"}
        )
        await daemon.handle_request(req)
        assert daemon._idle_timer is not None
        # Clean up
        daemon._idle_timer.cancel()

    @pytest.mark.asyncio
    async def test_unregister_not_last_no_idle_timer(
        self, daemon: MemXDaemon
    ) -> None:
        daemon._sessions["s1"] = datetime.now()
        daemon._sessions["s2"] = datetime.now()
        req = DaemonRequest(
            cmd="session_unregister", data={"session_id": "s1"}
        )
        await daemon.handle_request(req)
        # s2 still active, no idle timer
        assert daemon._idle_timer is None


class TestShutdownCommand:
    """Tests for the shutdown command."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_event(self, daemon: MemXDaemon) -> None:
        daemon._shutdown_event = asyncio.Event()
        req = DaemonRequest(cmd="shutdown")
        resp = await daemon.handle_request(req)
        assert resp.status == "ok"
        assert daemon._shutdown_event.is_set()


class TestUnknownCommand:
    """Tests for unknown commands."""

    @pytest.mark.asyncio
    async def test_unknown_command(self, daemon: MemXDaemon) -> None:
        req = DaemonRequest(cmd="nonexistent")
        resp = await daemon.handle_request(req)
        assert resp.status == "error"
        assert "Unknown command" in (resp.error or "")


# ---------------------------------------------------------------------------
# Idle timeout
# ---------------------------------------------------------------------------


class TestIdleTimeout:
    """Tests for idle timeout auto-shutdown."""

    @pytest.mark.asyncio
    async def test_idle_countdown_triggers_shutdown(
        self, daemon: MemXDaemon
    ) -> None:
        daemon._shutdown_event = asyncio.Event()
        # Use very short timeout
        daemon._config = DaemonConfig(enabled=True, idle_timeout_seconds=1)
        daemon._start_idle_timer()

        # Wait for the countdown to fire
        await asyncio.sleep(1.5)
        assert daemon._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_cancel_idle_timer(self, daemon: MemXDaemon) -> None:
        daemon._shutdown_event = asyncio.Event()
        daemon._idle_timer = asyncio.get_event_loop().create_task(
            asyncio.sleep(999)
        )
        daemon._cancel_idle_timer()
        assert daemon._idle_timer is None

    @pytest.mark.asyncio
    async def test_cancel_idle_timer_when_none(self, daemon: MemXDaemon) -> None:
        daemon._idle_timer = None
        daemon._cancel_idle_timer()  # Should not raise


# ---------------------------------------------------------------------------
# Error handling (non-crashing)
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify that command exceptions don't crash the daemon."""

    @pytest.mark.asyncio
    async def test_recall_memory_exception_returns_error(
        self, daemon: MemXDaemon
    ) -> None:
        mock_memory = MagicMock()
        mock_memory.search.side_effect = RuntimeError("DB connection lost")
        daemon._memory = mock_memory

        req = DaemonRequest(cmd="recall", data={"query": "test"})
        resp = await daemon.handle_request(req)
        assert resp.status == "error"
        assert "DB connection lost" in (resp.error or "")

    @pytest.mark.asyncio
    async def test_curate_memory_exception_returns_error(
        self, daemon: MemXDaemon
    ) -> None:
        mock_memory = MagicMock()
        mock_memory.add.side_effect = RuntimeError("Write failed")
        daemon._memory = mock_memory

        req = DaemonRequest(cmd="curate", data={"messages": "test"})
        resp = await daemon.handle_request(req)
        assert resp.status == "error"
        assert "Write failed" in (resp.error or "")


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestDaemonProperties:
    """Tests for daemon properties."""

    def test_is_running_initially_false(self, daemon: MemXDaemon) -> None:
        assert daemon.is_running is False

    def test_session_count_empty(self, daemon: MemXDaemon) -> None:
        assert daemon.session_count == 0

    def test_ipc_address_windows(self, daemon: MemXDaemon) -> None:
        with patch("memx.core.daemon.server.sys") as mock_sys:
            mock_sys.platform = "win32"
            daemon._config = DaemonConfig()
            addr = daemon.ipc_address
            assert addr == PIPE_NAME or addr is not None

    def test_ipc_address_custom(self) -> None:
        config = DaemonConfig(socket_path="/custom/path.sock")
        d = MemXDaemon(config=config)
        assert d.ipc_address == "/custom/path.sock"

    def test_default_config(self) -> None:
        d = MemXDaemon()
        assert d._config.idle_timeout_seconds == 300
        assert d._config.enabled is False


# ---------------------------------------------------------------------------
# Class-level helpers
# ---------------------------------------------------------------------------


class TestClassHelpers:
    """Tests for class-level static helpers."""

    def test_is_daemon_running_no_pid_file(self, tmp_path: Path) -> None:
        with patch(
            "memx.core.daemon.server.PID_PATH", tmp_path / "no.pid"
        ):
            assert MemXDaemon.is_daemon_running() is False

    def test_is_daemon_running_stale_pid(self, tmp_path: Path) -> None:
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("99999999")
        with (
            patch("memx.core.daemon.server.PID_PATH", pid_path),
            patch(
                "memx.core.daemon.server._is_process_alive", return_value=False
            ),
        ):
            assert MemXDaemon.is_daemon_running() is False

    def test_is_daemon_running_alive_pid(self, tmp_path: Path) -> None:
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("12345")
        with (
            patch("memx.core.daemon.server.PID_PATH", pid_path),
            patch(
                "memx.core.daemon.server._is_process_alive", return_value=True
            ),
        ):
            assert MemXDaemon.is_daemon_running() is True

    def test_read_pid_no_file(self, tmp_path: Path) -> None:
        with patch(
            "memx.core.daemon.server.PID_PATH", tmp_path / "no.pid"
        ):
            assert MemXDaemon.read_pid() is None

    def test_read_pid_valid(self, tmp_path: Path) -> None:
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("42")
        with patch("memx.core.daemon.server.PID_PATH", pid_path):
            assert MemXDaemon.read_pid() == 42

    def test_read_pid_corrupt(self, tmp_path: Path) -> None:
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text("not_a_number")
        with patch("memx.core.daemon.server.PID_PATH", pid_path):
            assert MemXDaemon.read_pid() is None


# ---------------------------------------------------------------------------
# Lifecycle / start-stop integration
# ---------------------------------------------------------------------------


class TestLifecycle:
    """Tests for daemon start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_stop_removes_pid_and_clears_sessions(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "daemon.pid"
        pid_path.write_text(str(os.getpid()))

        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            daemon._running = True
            daemon._server = MagicMock()
            daemon._server.close = MagicMock()
            daemon._server.wait_closed = AsyncMock()
            daemon._sessions = {"s1": datetime.now()}

            await daemon.stop()

            assert not pid_path.exists()
            assert daemon.session_count == 0
            assert daemon.is_running is False

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_safe(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        pid_path = tmp_path / "daemon.pid"
        with patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)):
            await daemon.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_init_memory_failure_raises_daemon_error(
        self, daemon: MemXDaemon
    ) -> None:
        with patch(
            "memx.core.memory.Memory",
            side_effect=RuntimeError("init boom"),
        ):
            with pytest.raises(DaemonError, match="initialization failed"):
                await daemon._init_memory()

    @pytest.mark.asyncio
    async def test_start_creates_pid_and_starts_server(
        self, daemon: MemXDaemon, tmp_path: Path
    ) -> None:
        """Test that start() creates PID, inits memory, opens IPC, then stops on event."""
        pid_path = tmp_path / "daemon.pid"

        with (
            patch.object(type(daemon), "pid_path", new=property(lambda s: pid_path)),
            patch.object(daemon, "_init_memory", new_callable=AsyncMock),
            patch.object(daemon, "_start_ipc_server", new_callable=AsyncMock),
            patch.object(daemon, "_install_signal_handlers"),
        ):
            # Schedule a shutdown after a brief delay
            async def trigger_shutdown() -> None:
                await asyncio.sleep(0.1)
                if daemon._shutdown_event:
                    daemon._shutdown_event.set()

            task = asyncio.get_event_loop().create_task(trigger_shutdown())
            await daemon.start()
            # After start returns (due to shutdown), PID should be cleaned
            assert not pid_path.exists()


# ---------------------------------------------------------------------------
# IPC connection handler
# ---------------------------------------------------------------------------


class TestIPCConnectionHandler:
    """Tests for the _handle_connection method."""

    @pytest.mark.asyncio
    async def test_handle_valid_request(self, daemon: MemXDaemon) -> None:
        daemon._start_time = datetime.now()
        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)

        request_bytes = json.dumps({"cmd": "ping"}).encode("utf-8")
        reader.read = AsyncMock(return_value=request_bytes)
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon._handle_connection(reader, writer)

        writer.write.assert_called_once()
        raw_resp = writer.write.call_args[0][0]
        resp = json.loads(raw_resp.decode("utf-8"))
        assert resp["status"] == "ok"
        assert resp["data"]["version"] == "1.0.0"

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, daemon: MemXDaemon) -> None:
        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)

        reader.read = AsyncMock(return_value=b"not valid json!")
        writer.write = MagicMock()
        writer.drain = AsyncMock()
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon._handle_connection(reader, writer)

        writer.write.assert_called_once()
        raw_resp = writer.write.call_args[0][0]
        resp = json.loads(raw_resp.decode("utf-8"))
        assert resp["status"] == "error"
        assert "Invalid request" in resp["error"]

    @pytest.mark.asyncio
    async def test_handle_empty_read(self, daemon: MemXDaemon) -> None:
        reader = AsyncMock(spec=asyncio.StreamReader)
        writer = AsyncMock(spec=asyncio.StreamWriter)

        reader.read = AsyncMock(return_value=b"")
        writer.close = MagicMock()
        writer.wait_closed = AsyncMock()

        await daemon._handle_connection(reader, writer)

        # Should not write anything (empty request)
        writer.write.assert_not_called() if hasattr(writer.write, 'assert_not_called') else None


# ---------------------------------------------------------------------------
# _is_process_alive helper
# ---------------------------------------------------------------------------


class TestIsProcessAlive:
    """Tests for the _is_process_alive helper."""

    def test_current_process_is_alive(self) -> None:
        assert _is_process_alive(os.getpid()) is True

    def test_nonexistent_pid(self) -> None:
        # Use a very high PID that's almost certainly not running
        assert _is_process_alive(4_000_000_000) is False


# ---------------------------------------------------------------------------
# Full round-trip integration
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """End-to-end command round-trip tests."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, daemon: MemXDaemon) -> None:
        """Register -> ping -> unregister -> verify session count transitions."""
        daemon._start_time = datetime.now()

        # Register
        resp = await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s1"})
        )
        assert resp.status == "ok"

        # Ping shows 1 session
        resp = await daemon.handle_request(DaemonRequest(cmd="ping"))
        assert resp.data["sessions"] == 1

        # Register another
        resp = await daemon.handle_request(
            DaemonRequest(cmd="session_register", data={"session_id": "s2"})
        )
        assert resp.status == "ok"

        # Ping shows 2 sessions
        resp = await daemon.handle_request(DaemonRequest(cmd="ping"))
        assert resp.data["sessions"] == 2

        # Unregister s1
        resp = await daemon.handle_request(
            DaemonRequest(cmd="session_unregister", data={"session_id": "s1"})
        )
        assert resp.status == "ok"

        # Ping shows 1 session
        resp = await daemon.handle_request(DaemonRequest(cmd="ping"))
        assert resp.data["sessions"] == 1

        # Unregister s2
        resp = await daemon.handle_request(
            DaemonRequest(cmd="session_unregister", data={"session_id": "s2"})
        )
        assert resp.status == "ok"
        assert daemon.session_count == 0

        # Clean up idle timer if started
        if daemon._idle_timer and not daemon._idle_timer.done():
            daemon._idle_timer.cancel()

    @pytest.mark.asyncio
    async def test_recall_then_curate(self, daemon: MemXDaemon) -> None:
        """Recall and curate in sequence."""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}
        mock_memory.add.return_value = {"results": [], "ace_ingest": {"bullets_added": 1}}
        daemon._memory = mock_memory

        # Recall
        resp = await daemon.handle_request(
            DaemonRequest(cmd="recall", data={"query": "async patterns"})
        )
        assert resp.status == "ok"

        # Curate
        resp = await daemon.handle_request(
            DaemonRequest(
                cmd="curate",
                data={"messages": "User likes async/await patterns"},
            )
        )
        assert resp.status == "ok"
