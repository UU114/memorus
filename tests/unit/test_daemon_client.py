"""Unit tests for memx.daemon.client and memx.daemon.ipc.

All IPC connections are mocked so tests run fast with no real daemon.
"""

from __future__ import annotations

import asyncio
import json
import struct
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.core.config import DaemonConfig
from memx.core.daemon.client import DaemonClient
from memx.core.daemon.ipc import (
    IPCTransport,
    NamedPipeTransport,
    UnixSocketTransport,
    _HEADER_FMT,
    _HEADER_SIZE,
    MAX_MESSAGE_SIZE,
    get_transport,
)
from memx.core.daemon.server import DaemonResponse
from memx.core.exceptions import DaemonUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response_bytes(
    status: str = "ok",
    data: dict[str, Any] | None = None,
    error: str | None = None,
) -> bytes:
    """Build a length-prefixed response payload."""
    resp = DaemonResponse(status=status, data=data or {}, error=error)
    body = resp.to_json().encode("utf-8")
    header = struct.pack(_HEADER_FMT, len(body))
    return header + body


class FakeTransport(IPCTransport):
    """In-memory transport for testing DaemonClient without real IPC."""

    def __init__(self, response: DaemonResponse | None = None) -> None:
        self._response = response or DaemonResponse(status="ok")
        self.connected = False
        self.closed = False
        self.sent_data: bytes | None = None

    async def connect(self) -> None:
        self.connected = True

    async def send(self, data: bytes) -> None:
        self.sent_data = data

    async def recv(self) -> bytes:
        return self._response.to_json().encode("utf-8")

    async def close(self) -> None:
        self.closed = True


class FailConnectTransport(IPCTransport):
    """Transport that raises ConnectionError on connect."""

    async def connect(self) -> None:
        raise ConnectionError("Connection refused")

    async def send(self, data: bytes) -> None:
        raise ConnectionError("Not connected")

    async def recv(self) -> bytes:
        raise ConnectionError("Not connected")

    async def close(self) -> None:
        pass


class TimeoutTransport(IPCTransport):
    """Transport that times out on connect."""

    async def connect(self) -> None:
        await asyncio.sleep(999)

    async def send(self, data: bytes) -> None:
        pass

    async def recv(self) -> bytes:
        return b""

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# IPCTransport -- get_transport factory
# ---------------------------------------------------------------------------


class TestGetTransport:
    """Tests for the get_transport factory function."""

    def test_windows_returns_named_pipe(self) -> None:
        with patch("memx.core.daemon.ipc.sys") as mock_sys:
            mock_sys.platform = "win32"
            transport = get_transport()
            assert isinstance(transport, NamedPipeTransport)

    def test_linux_returns_unix_socket(self) -> None:
        with patch("memx.core.daemon.ipc.sys") as mock_sys:
            mock_sys.platform = "linux"
            transport = get_transport()
            assert isinstance(transport, UnixSocketTransport)

    def test_custom_socket_path_windows(self) -> None:
        config = DaemonConfig(socket_path=r"\\.\pipe\custom-pipe")
        with patch("memx.core.daemon.ipc.sys") as mock_sys:
            mock_sys.platform = "win32"
            transport = get_transport(config)
            assert isinstance(transport, NamedPipeTransport)
            assert transport._pipe_name == r"\\.\pipe\custom-pipe"

    def test_custom_socket_path_linux(self) -> None:
        config = DaemonConfig(socket_path="/tmp/custom.sock")
        with patch("memx.core.daemon.ipc.sys") as mock_sys:
            mock_sys.platform = "linux"
            transport = get_transport(config)
            assert isinstance(transport, UnixSocketTransport)
            assert transport._socket_path == "/tmp/custom.sock"

    def test_default_config_used_when_none(self) -> None:
        with patch("memx.core.daemon.ipc.sys") as mock_sys:
            mock_sys.platform = "win32"
            transport = get_transport(None)
            assert isinstance(transport, NamedPipeTransport)


# ---------------------------------------------------------------------------
# NamedPipeTransport
# ---------------------------------------------------------------------------


class TestNamedPipeTransport:
    """Tests for NamedPipeTransport."""

    def test_default_pipe_name(self) -> None:
        t = NamedPipeTransport()
        assert t._pipe_name == r"\\.\pipe\memx-daemon"

    def test_custom_pipe_name(self) -> None:
        t = NamedPipeTransport(pipe_name=r"\\.\pipe\custom")
        assert t._pipe_name == r"\\.\pipe\custom"

    async def test_send_not_connected_raises(self) -> None:
        t = NamedPipeTransport()
        with pytest.raises(ConnectionError, match="not connected"):
            await t.send(b"hello")

    async def test_recv_not_connected_raises(self) -> None:
        t = NamedPipeTransport()
        with pytest.raises(ConnectionError, match="not connected"):
            await t.recv()

    async def test_close_when_not_connected(self) -> None:
        t = NamedPipeTransport()
        await t.close()  # Should not raise

    async def test_send_writes_header_and_data(self) -> None:
        t = NamedPipeTransport()
        mock_writer = AsyncMock()
        mock_writer.write = MagicMock()
        mock_writer.drain = AsyncMock()
        t._writer = mock_writer

        data = b'{"cmd": "ping"}'
        await t.send(data)

        expected_header = struct.pack(_HEADER_FMT, len(data))
        mock_writer.write.assert_called_once_with(expected_header + data)
        mock_writer.drain.assert_awaited_once()

    async def test_recv_reads_header_then_body(self) -> None:
        t = NamedPipeTransport()
        body = b'{"status": "ok"}'
        header = struct.pack(_HEADER_FMT, len(body))
        mock_reader = AsyncMock()
        mock_reader.readexactly = AsyncMock(side_effect=[header, body])
        t._reader = mock_reader

        result = await t.recv()
        assert result == body
        assert mock_reader.readexactly.call_count == 2

    async def test_recv_too_large_raises(self) -> None:
        t = NamedPipeTransport()
        header = struct.pack(_HEADER_FMT, MAX_MESSAGE_SIZE + 1)
        mock_reader = AsyncMock()
        mock_reader.readexactly = AsyncMock(return_value=header)
        t._reader = mock_reader

        with pytest.raises(ValueError, match="too large"):
            await t.recv()

    async def test_close_clears_reader_writer(self) -> None:
        t = NamedPipeTransport()
        mock_writer = AsyncMock()
        mock_writer.close = MagicMock()
        mock_writer.wait_closed = AsyncMock()
        t._writer = mock_writer
        t._reader = AsyncMock()

        await t.close()
        assert t._writer is None
        assert t._reader is None


# ---------------------------------------------------------------------------
# UnixSocketTransport
# ---------------------------------------------------------------------------


class TestUnixSocketTransport:
    """Tests for UnixSocketTransport."""

    def test_default_socket_path(self) -> None:
        t = UnixSocketTransport()
        assert "daemon.sock" in t._socket_path

    def test_custom_socket_path(self) -> None:
        t = UnixSocketTransport(socket_path="/tmp/test.sock")
        assert t._socket_path == "/tmp/test.sock"

    async def test_send_not_connected_raises(self) -> None:
        t = UnixSocketTransport()
        with pytest.raises(ConnectionError, match="not connected"):
            await t.send(b"hello")

    async def test_recv_not_connected_raises(self) -> None:
        t = UnixSocketTransport()
        with pytest.raises(ConnectionError, match="not connected"):
            await t.recv()

    async def test_close_when_not_connected(self) -> None:
        t = UnixSocketTransport()
        await t.close()  # Should not raise


# ---------------------------------------------------------------------------
# DaemonClient -- ping
# ---------------------------------------------------------------------------


class TestPing:
    """Tests for DaemonClient.ping()."""

    async def test_ping_returns_true_when_daemon_alive(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok", data={"version": "1.0.0"}))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            result = await client.ping()
            assert result is True

    async def test_ping_returns_false_on_connection_error(self) -> None:
        with patch("memx.core.daemon.client.get_transport", return_value=FailConnectTransport()):
            client = DaemonClient()
            result = await client.ping()
            assert result is False

    async def test_ping_returns_false_on_timeout(self) -> None:
        with patch("memx.core.daemon.client.get_transport", return_value=TimeoutTransport()):
            client = DaemonClient(connect_timeout=0.05)
            result = await client.ping()
            assert result is False

    async def test_ping_returns_false_on_error_status(self) -> None:
        fake = FakeTransport(DaemonResponse(status="error", error="bad"))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            result = await client.ping()
            assert result is False

    async def test_is_running_delegates_to_ping(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok"))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            result = await client.is_running()
            assert result is True


# ---------------------------------------------------------------------------
# DaemonClient -- recall
# ---------------------------------------------------------------------------


class TestRecall:
    """Tests for DaemonClient.recall()."""

    async def test_recall_returns_results(self) -> None:
        results = [{"memory": "test bullet", "score": 0.9}]
        fake = FakeTransport(DaemonResponse(status="ok", data={"results": results}))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            out = await client.recall("async patterns", user_id="u1")
            assert out == results

    async def test_recall_sends_correct_request(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok", data={"results": []}))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.recall("test query", user_id="user1", limit=10)
            assert fake.sent_data is not None
            req = json.loads(fake.sent_data.decode("utf-8"))
            assert req["cmd"] == "recall"
            assert req["data"]["query"] == "test query"
            assert req["data"]["user_id"] == "user1"
            assert req["data"]["limit"] == 10

    async def test_recall_empty_results(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok", data={"results": []}))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            out = await client.recall("nothing here")
            assert out == []

    async def test_recall_default_user_id(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok", data={"results": []}))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.recall("query")
            req = json.loads(fake.sent_data.decode("utf-8"))  # type: ignore[union-attr]
            assert req["data"]["user_id"] == "default"

    async def test_recall_raises_on_connection_error(self) -> None:
        with patch("memx.core.daemon.client.get_transport", return_value=FailConnectTransport()):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError, match="unavailable"):
                await client.recall("test")

    async def test_recall_raises_on_error_response(self) -> None:
        fake = FakeTransport(
            DaemonResponse(status="error", error="Memory not initialized")
        )
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError, match="recall failed"):
                await client.recall("test")

    async def test_recall_missing_results_key(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok", data={}))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            out = await client.recall("test")
            assert out == []


# ---------------------------------------------------------------------------
# DaemonClient -- curate
# ---------------------------------------------------------------------------


class TestCurate:
    """Tests for DaemonClient.curate()."""

    async def test_curate_returns_data(self) -> None:
        result_data = {"results": [], "ace_ingest": {"bullets_added": 1}}
        fake = FakeTransport(DaemonResponse(status="ok", data=result_data))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            out = await client.curate("User prefers dark mode", user_id="u1")
            assert out == result_data

    async def test_curate_sends_correct_request(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok", data={}))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.curate(
                messages=[{"role": "user", "content": "I like cats"}],
                user_id="u2",
            )
            req = json.loads(fake.sent_data.decode("utf-8"))  # type: ignore[union-attr]
            assert req["cmd"] == "curate"
            assert req["data"]["messages"] == [{"role": "user", "content": "I like cats"}]
            assert req["data"]["user_id"] == "u2"

    async def test_curate_raises_on_connection_error(self) -> None:
        with patch("memx.core.daemon.client.get_transport", return_value=FailConnectTransport()):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError):
                await client.curate("msg", user_id="u1")

    async def test_curate_raises_on_error_response(self) -> None:
        fake = FakeTransport(
            DaemonResponse(status="error", error="Missing 'messages'")
        )
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError, match="curate failed"):
                await client.curate("test", user_id="u1")


# ---------------------------------------------------------------------------
# DaemonClient -- register_session / unregister_session
# ---------------------------------------------------------------------------


class TestSessionManagement:
    """Tests for register_session and unregister_session."""

    async def test_register_session_success(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok"))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.register_session("sess-abc")
            req = json.loads(fake.sent_data.decode("utf-8"))  # type: ignore[union-attr]
            assert req["cmd"] == "session_register"
            assert req["data"]["session_id"] == "sess-abc"

    async def test_unregister_session_success(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok"))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.unregister_session("sess-abc")
            req = json.loads(fake.sent_data.decode("utf-8"))  # type: ignore[union-attr]
            assert req["cmd"] == "session_unregister"
            assert req["data"]["session_id"] == "sess-abc"

    async def test_register_raises_on_connection_error(self) -> None:
        with patch("memx.core.daemon.client.get_transport", return_value=FailConnectTransport()):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError):
                await client.register_session("sess-abc")

    async def test_unregister_raises_on_connection_error(self) -> None:
        with patch("memx.core.daemon.client.get_transport", return_value=FailConnectTransport()):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError):
                await client.unregister_session("sess-abc")

    async def test_register_raises_on_error_response(self) -> None:
        fake = FakeTransport(
            DaemonResponse(status="error", error="Missing 'session_id'")
        )
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError, match="session_register"):
                await client.register_session("")

    async def test_unregister_raises_on_error_response(self) -> None:
        fake = FakeTransport(
            DaemonResponse(status="error", error="Missing 'session_id'")
        )
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError, match="session_unregister"):
                await client.unregister_session("")


# ---------------------------------------------------------------------------
# DaemonClient -- shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    """Tests for DaemonClient.shutdown()."""

    async def test_shutdown_success(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok"))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.shutdown()
            req = json.loads(fake.sent_data.decode("utf-8"))  # type: ignore[union-attr]
            assert req["cmd"] == "shutdown"

    async def test_shutdown_daemon_already_gone(self) -> None:
        """Shutdown should not raise even if daemon is already gone."""
        with patch("memx.core.daemon.client.get_transport", return_value=FailConnectTransport()):
            client = DaemonClient()
            await client.shutdown()  # Should not raise

    async def test_shutdown_error_response_logged(self) -> None:
        fake = FakeTransport(
            DaemonResponse(status="error", error="some issue")
        )
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.shutdown()  # Should not raise, just log


# ---------------------------------------------------------------------------
# DaemonClient -- transport lifecycle
# ---------------------------------------------------------------------------


class TestTransportLifecycle:
    """Tests that transport is properly opened and closed."""

    async def test_transport_closed_after_success(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok"))
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.ping()
            assert fake.closed is True

    async def test_transport_closed_after_failure(self) -> None:
        fake = FakeTransport(DaemonResponse(status="ok"))

        async def failing_connect() -> None:
            raise ConnectionError("refused")

        fake.connect = failing_connect  # type: ignore[assignment]
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            await client.ping()  # Returns False, should not raise
            assert fake.closed is True

    async def test_invalid_json_response_raises(self) -> None:
        """If the daemon returns garbage, DaemonUnavailableError is raised."""
        fake = FakeTransport(DaemonResponse(status="ok"))

        async def bad_recv() -> bytes:
            return b"not json at all"

        fake.recv = bad_recv  # type: ignore[assignment]
        with patch("memx.core.daemon.client.get_transport", return_value=fake):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError, match="Invalid response"):
                await client.recall("test")


# ---------------------------------------------------------------------------
# DaemonClient -- configuration
# ---------------------------------------------------------------------------


class TestClientConfiguration:
    """Tests for DaemonClient configuration."""

    def test_default_config(self) -> None:
        client = DaemonClient()
        assert client._connect_timeout == 2.0
        assert client._request_timeout == 10.0

    def test_custom_timeouts(self) -> None:
        client = DaemonClient(connect_timeout=5.0, request_timeout=30.0)
        assert client._connect_timeout == 5.0
        assert client._request_timeout == 30.0

    def test_custom_daemon_config(self) -> None:
        config = DaemonConfig(enabled=True, socket_path="/tmp/test.sock")
        client = DaemonClient(config=config)
        assert client._config.socket_path == "/tmp/test.sock"
        assert client._config.enabled is True


# ---------------------------------------------------------------------------
# DaemonUnavailableError
# ---------------------------------------------------------------------------


class TestDaemonUnavailableError:
    """Tests for the DaemonUnavailableError exception."""

    def test_inherits_from_connection_error(self) -> None:
        err = DaemonUnavailableError("test")
        assert isinstance(err, ConnectionError)

    def test_inherits_from_daemon_error(self) -> None:
        from memx.core.exceptions import DaemonError

        err = DaemonUnavailableError("test")
        assert isinstance(err, DaemonError)

    def test_inherits_from_memx_error(self) -> None:
        from memx.core.exceptions import MemXError

        err = DaemonUnavailableError("test")
        assert isinstance(err, MemXError)

    def test_catchable_as_connection_error(self) -> None:
        with pytest.raises(ConnectionError):
            raise DaemonUnavailableError("daemon is gone")

    def test_message(self) -> None:
        err = DaemonUnavailableError("custom message")
        assert str(err) == "custom message"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error scenarios."""

    async def test_recv_timeout_during_request(self) -> None:
        """Simulate daemon crashing mid-request (recv times out)."""

        class SlowRecvTransport(IPCTransport):
            async def connect(self) -> None:
                pass

            async def send(self, data: bytes) -> None:
                pass

            async def recv(self) -> bytes:
                await asyncio.sleep(999)
                return b""

            async def close(self) -> None:
                pass

        with patch("memx.core.daemon.client.get_transport", return_value=SlowRecvTransport()):
            client = DaemonClient(request_timeout=0.05)
            with pytest.raises(DaemonUnavailableError, match="unavailable"):
                await client.recall("test")

    async def test_os_error_on_connect(self) -> None:
        """OSError during connect is caught and wrapped."""

        class OsErrorTransport(IPCTransport):
            async def connect(self) -> None:
                raise OSError("Permission denied")

            async def send(self, data: bytes) -> None:
                pass

            async def recv(self) -> bytes:
                return b""

            async def close(self) -> None:
                pass

        with patch("memx.core.daemon.client.get_transport", return_value=OsErrorTransport()):
            client = DaemonClient()
            with pytest.raises(DaemonUnavailableError, match="unavailable"):
                await client.register_session("s1")

    async def test_concurrent_clients(self) -> None:
        """Multiple concurrent requests each get their own transport."""
        call_count = 0

        class CountingTransport(IPCTransport):
            async def connect(self) -> None:
                nonlocal call_count
                call_count += 1

            async def send(self, data: bytes) -> None:
                pass

            async def recv(self) -> bytes:
                resp = DaemonResponse(status="ok", data={"results": []})
                return resp.to_json().encode("utf-8")

            async def close(self) -> None:
                pass

        def make_transport(_: Any = None) -> CountingTransport:
            return CountingTransport()

        with patch("memx.core.daemon.client.get_transport", side_effect=make_transport):
            client = DaemonClient()
            tasks = [client.recall("q") for _ in range(5)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 5
            assert call_count == 5  # Each request opened its own connection
