"""IPC transport abstraction for cross-platform daemon communication.

Provides Named Pipe transport on Windows and Unix Socket transport on
Linux/macOS.  Each transport implements a simple length-prefixed framing
protocol: a 4-byte big-endian uint32 length header followed by the
JSON payload bytes.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import sys
from abc import ABC, abstractmethod
from typing import Optional

from memx.core.config import DaemonConfig
from memx.core.daemon.server import PIPE_NAME, SOCKET_PATH

logger = logging.getLogger(__name__)

# Length-prefix format: 4-byte unsigned big-endian integer
_HEADER_FMT = "!I"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)

# Maximum single message size (same as server limit)
MAX_MESSAGE_SIZE = 1 * 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class IPCTransport(ABC):
    """Abstract IPC transport for daemon communication."""

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the daemon."""
        ...

    @abstractmethod
    async def send(self, data: bytes) -> None:
        """Send a length-prefixed message."""
        ...

    @abstractmethod
    async def recv(self) -> bytes:
        """Receive a length-prefixed message."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the transport connection."""
        ...


# ---------------------------------------------------------------------------
# Named Pipe transport (Windows)
# ---------------------------------------------------------------------------


class NamedPipeTransport(IPCTransport):
    """Windows Named Pipe transport using asyncio streams.

    On Windows with ProactorEventLoop, ``asyncio.open_connection`` accepts
    a named pipe path as the first positional argument.
    """

    def __init__(self, pipe_name: str = PIPE_NAME) -> None:
        self._pipe_name = pipe_name
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        """Open a stream connection to the named pipe."""
        logger.debug("Connecting to Named Pipe: %s", self._pipe_name)
        self._reader, self._writer = await asyncio.open_connection(
            self._pipe_name,
        )
        logger.debug("Connected to Named Pipe: %s", self._pipe_name)

    async def send(self, data: bytes) -> None:
        """Send *data* with a 4-byte length header."""
        if self._writer is None:
            raise ConnectionError("Transport not connected")
        header = struct.pack(_HEADER_FMT, len(data))
        self._writer.write(header + data)
        await self._writer.drain()

    async def recv(self) -> bytes:
        """Read a length-prefixed response from the pipe."""
        if self._reader is None:
            raise ConnectionError("Transport not connected")
        raw_header = await self._reader.readexactly(_HEADER_SIZE)
        (length,) = struct.unpack(_HEADER_FMT, raw_header)
        if length > MAX_MESSAGE_SIZE:
            raise ValueError(
                f"Message too large: {length} bytes (max {MAX_MESSAGE_SIZE})"
            )
        return await self._reader.readexactly(length)

    async def close(self) -> None:
        """Close the pipe connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            finally:
                self._writer = None
                self._reader = None


# ---------------------------------------------------------------------------
# Unix Socket transport (Linux / macOS)
# ---------------------------------------------------------------------------


class UnixSocketTransport(IPCTransport):
    """Unix domain socket transport using asyncio streams."""

    def __init__(self, socket_path: str = str(SOCKET_PATH)) -> None:
        self._socket_path = socket_path
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        """Open a stream connection to the Unix socket."""
        logger.debug("Connecting to Unix Socket: %s", self._socket_path)
        self._reader, self._writer = await asyncio.open_unix_connection(
            self._socket_path,
        )
        logger.debug("Connected to Unix Socket: %s", self._socket_path)

    async def send(self, data: bytes) -> None:
        """Send *data* with a 4-byte length header."""
        if self._writer is None:
            raise ConnectionError("Transport not connected")
        header = struct.pack(_HEADER_FMT, len(data))
        self._writer.write(header + data)
        await self._writer.drain()

    async def recv(self) -> bytes:
        """Read a length-prefixed response from the socket."""
        if self._reader is None:
            raise ConnectionError("Transport not connected")
        raw_header = await self._reader.readexactly(_HEADER_SIZE)
        (length,) = struct.unpack(_HEADER_FMT, raw_header)
        if length > MAX_MESSAGE_SIZE:
            raise ValueError(
                f"Message too large: {length} bytes (max {MAX_MESSAGE_SIZE})"
            )
        return await self._reader.readexactly(length)

    async def close(self) -> None:
        """Close the socket connection."""
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
            finally:
                self._writer = None
                self._reader = None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_transport(config: Optional[DaemonConfig] = None) -> IPCTransport:
    """Create the appropriate IPC transport for the current platform.

    On Windows, returns a :class:`NamedPipeTransport`.
    On Linux/macOS, returns a :class:`UnixSocketTransport`.
    """
    cfg = config or DaemonConfig()
    if sys.platform == "win32":
        pipe = cfg.socket_path or PIPE_NAME
        return NamedPipeTransport(pipe_name=pipe)
    else:
        sock = cfg.socket_path or str(SOCKET_PATH)
        return UnixSocketTransport(socket_path=sock)
