"""MemX Daemon -- background server for zero-latency Hook calls."""

from memx.core.daemon.client import DaemonClient
from memx.core.daemon.fallback import DaemonFallbackManager
from memx.core.daemon.ipc import (
    IPCTransport,
    NamedPipeTransport,
    UnixSocketTransport,
    get_transport,
)
from memx.core.daemon.server import (
    DaemonRequest,
    DaemonResponse,
    MemXDaemon,
)

__all__ = [
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
