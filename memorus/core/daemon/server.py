"""MemorusDaemon -- background server that keeps models loaded in memory.

Provides IPC access (Named Pipe on Windows, Unix Socket on Linux/Mac) to
avoid cold-start latency.  Multiple CLI sessions share the same Daemon
instance.  Auto-exits after an idle timeout with zero active sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from memorus import __version__
from memorus.core.config import DaemonConfig
from memorus.core.exceptions import DaemonError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PIPE_NAME = r"\\.\pipe\memorus-daemon"
SOCKET_PATH = Path("~/.memorus/daemon.sock").expanduser()
PID_PATH = Path("~/.memorus/daemon.pid").expanduser()
MAX_REQUEST_SIZE = 1 * 1024 * 1024  # 1 MB
DEFAULT_IDLE_TIMEOUT = 300  # 5 minutes


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DaemonRequest:
    """Incoming IPC request."""

    cmd: str
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, raw: str) -> DaemonRequest:
        """Parse a JSON string into a DaemonRequest."""
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("Request must be a JSON object")
        cmd = parsed.get("cmd")
        if not cmd or not isinstance(cmd, str):
            raise ValueError("Request must include a 'cmd' string")
        return cls(cmd=cmd, data=parsed.get("data", {}))


@dataclass
class DaemonResponse:
    """Outgoing IPC response."""

    status: str  # "ok" | "error"
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to JSON string."""
        payload: dict[str, Any] = {"status": self.status}
        if self.data:
            payload["data"] = self.data
        if self.error is not None:
            payload["error"] = self.error
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# PID file helpers
# ---------------------------------------------------------------------------


def _is_process_alive(pid: int) -> bool:
    """Check whether a process with *pid* is currently running."""
    if sys.platform == "win32":
        # On Windows, use ctypes OpenProcess to probe without killing
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[union-attr]
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
        )
        if handle:
            ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[union-attr]
            return True
        return False
    else:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# MemorusDaemon
# ---------------------------------------------------------------------------


class MemorusDaemon:
    """Background daemon that holds Memory and serves IPC requests.

    Usage::

        daemon = MemorusDaemon(config)
        await daemon.start()   # blocks until shutdown
    """

    def __init__(self, config: Optional[DaemonConfig] = None) -> None:
        self._config = config or DaemonConfig()
        self._memory: Any = None  # memorus.memory.Memory (lazy)
        self._sessions: dict[str, datetime] = {}
        self._idle_timer: Optional[asyncio.Task[None]] = None
        self._server: Any = None  # asyncio server handle
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        # IdleOrchestrator (STORY-R094)
        self._orchestrator: Any = None
        self._orchestrator_task: Optional[asyncio.Task[None]] = None

    # -- Properties ---------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Return True when the daemon event loop is active."""
        return self._running

    @property
    def session_count(self) -> int:
        """Number of currently registered sessions."""
        return len(self._sessions)

    @property
    def pid_path(self) -> Path:
        """Path to the PID file."""
        return PID_PATH

    @property
    def ipc_address(self) -> str:
        """IPC address (pipe name or socket path) used by this daemon."""
        if self._config.socket_path:
            return self._config.socket_path
        if sys.platform == "win32":
            return PIPE_NAME
        return str(SOCKET_PATH)

    # -- PID file management ------------------------------------------------

    def _check_pid(self) -> None:
        """Ensure no other daemon is already running.

        If a stale PID file exists (process dead), clean it up.
        """
        if not self.pid_path.exists():
            return
        try:
            pid = int(self.pid_path.read_text().strip())
        except (ValueError, OSError) as exc:
            logger.warning("Corrupt PID file, removing: %s", exc)
            self._remove_pid()
            return

        if _is_process_alive(pid):
            raise DaemonError(
                f"Another MemorusDaemon is already running (PID {pid}). "
                f"Stop it first or delete {self.pid_path} if stale."
            )
        # Stale PID file -- previous daemon crashed
        logger.warning("Stale PID file (PID %d not alive), cleaning up.", pid)
        self._remove_pid()

    def _write_pid(self) -> None:
        """Write the current process PID to the PID file."""
        self.pid_path.parent.mkdir(parents=True, exist_ok=True)
        self.pid_path.write_text(str(os.getpid()))
        logger.info("PID file written: %s (PID %d)", self.pid_path, os.getpid())

    def _remove_pid(self) -> None:
        """Remove the PID file if it exists."""
        try:
            self.pid_path.unlink(missing_ok=True)
            logger.info("PID file removed: %s", self.pid_path)
        except OSError as exc:
            logger.warning("Failed to remove PID file: %s", exc)

    # -- IPC server ---------------------------------------------------------

    async def _start_ipc_server(self) -> None:
        """Start the appropriate IPC listener for this platform."""
        if sys.platform == "win32":
            # On Windows, use a Named Pipe via proactor event loop.
            # asyncio on Windows (proactor) supports start_server with named pipe path.
            addr = self._config.socket_path or PIPE_NAME
            self._server = await asyncio.start_server(
                self._handle_connection,
                path=addr,
            )
            logger.info("IPC server listening on Named Pipe: %s", addr)
        else:
            addr = self._config.socket_path or str(SOCKET_PATH)
            # Clean up stale socket file
            sock_path = Path(addr)
            if sock_path.exists():
                sock_path.unlink()
                logger.info("Removed stale socket file: %s", addr)
            sock_path.parent.mkdir(parents=True, exist_ok=True)
            self._server = await asyncio.start_unix_server(
                self._handle_connection,
                path=addr,
            )
            logger.info("IPC server listening on Unix Socket: %s", addr)

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single IPC connection (one request-response cycle)."""
        peer = "unknown"
        try:
            raw = await reader.read(MAX_REQUEST_SIZE)
            if not raw:
                return
            text = raw.decode("utf-8", errors="replace")
            logger.debug("IPC request received (%d bytes)", len(raw))

            try:
                request = DaemonRequest.from_json(text)
            except (json.JSONDecodeError, ValueError) as exc:
                response = DaemonResponse(
                    status="error",
                    error=f"Invalid request: {exc}",
                )
                writer.write(response.to_json().encode("utf-8"))
                await writer.drain()
                return

            response = await self.handle_request(request)
            writer.write(response.to_json().encode("utf-8"))
            await writer.drain()

        except Exception as exc:
            logger.error("Error handling IPC connection: %s", exc, exc_info=True)
            try:
                err_resp = DaemonResponse(status="error", error=str(exc))
                writer.write(err_resp.to_json().encode("utf-8"))
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # -- Request routing ----------------------------------------------------

    async def handle_request(self, request: DaemonRequest) -> DaemonResponse:
        """Route an IPC request to the appropriate handler."""
        logger.info("Handling command: %s", request.cmd)
        try:
            match request.cmd:
                case "ping":
                    return self._handle_ping()
                case "recall":
                    return self._handle_recall(request.data)
                case "curate":
                    return self._handle_curate(request.data)
                case "session_register":
                    return self._handle_session_register(request.data)
                case "session_unregister":
                    return self._handle_session_unregister(request.data)
                case "consolidate_now":
                    return await self._handle_consolidate_now()
                case "shutdown":
                    return await self._handle_shutdown()
                case _:
                    return DaemonResponse(
                        status="error",
                        error=f"Unknown command: {request.cmd}",
                    )
        except Exception as exc:
            logger.error(
                "Error processing command '%s': %s",
                request.cmd, exc, exc_info=True,
            )
            return DaemonResponse(status="error", error=str(exc))

    # -- Command handlers ---------------------------------------------------

    def _handle_ping(self) -> DaemonResponse:
        """Health check -- returns version and session count."""
        return DaemonResponse(
            status="ok",
            data={
                "version": __version__,
                "sessions": len(self._sessions),
                "pid": os.getpid(),
                "uptime_seconds": self._uptime_seconds(),
            },
        )

    def _handle_recall(self, data: dict[str, Any]) -> DaemonResponse:
        """Search memories using the loaded Memory instance."""
        if self._memory is None:
            return DaemonResponse(
                status="error", error="Memory not initialized"
            )
        query = data.get("query")
        if not query:
            return DaemonResponse(
                status="error", error="Missing 'query' in data"
            )
        user_id = data.get("user_id")
        limit = data.get("limit", 5)
        results = self._memory.search(
            query=query,
            user_id=user_id,
            limit=limit,
        )
        return DaemonResponse(status="ok", data={"results": results})

    def _handle_curate(self, data: dict[str, Any]) -> DaemonResponse:
        """Ingest messages via Memory.add()."""
        if self._memory is None:
            return DaemonResponse(
                status="error", error="Memory not initialized"
            )
        messages = data.get("messages")
        if messages is None:
            return DaemonResponse(
                status="error", error="Missing 'messages' in data"
            )
        user_id = data.get("user_id")
        result = self._memory.add(messages=messages, user_id=user_id)
        return DaemonResponse(status="ok", data=result)

    def _handle_session_register(self, data: dict[str, Any]) -> DaemonResponse:
        """Register a new active session."""
        session_id = data.get("session_id")
        if not session_id:
            return DaemonResponse(
                status="error", error="Missing 'session_id' in data"
            )
        self._sessions[session_id] = datetime.now()
        self._cancel_idle_timer()
        logger.info(
            "Session registered: %s (total: %d)",
            session_id, len(self._sessions),
        )
        return DaemonResponse(status="ok")

    def _handle_session_unregister(self, data: dict[str, Any]) -> DaemonResponse:
        """Remove a session from the active list."""
        session_id = data.get("session_id")
        if not session_id:
            return DaemonResponse(
                status="error", error="Missing 'session_id' in data"
            )
        removed = self._sessions.pop(session_id, None)
        if removed is None:
            logger.warning("Session not found for unregister: %s", session_id)
        else:
            logger.info(
                "Session unregistered: %s (remaining: %d)",
                session_id, len(self._sessions),
            )
        if not self._sessions:
            self._start_idle_timer()
        return DaemonResponse(status="ok")

    async def _handle_consolidate_now(self) -> DaemonResponse:
        """Force-run a consolidate pass via the IdleOrchestrator (STORY-R094)."""
        if self._orchestrator is None:
            return DaemonResponse(
                status="error", error="IdleOrchestrator not initialized"
            )
        try:
            report = await self._orchestrator.run_once(force=True)
        except Exception as e:
            logger.error("consolidate_now failed: %s", e, exc_info=True)
            return DaemonResponse(status="error", error=str(e))
        if report is None:
            return DaemonResponse(
                status="error", error="consolidate did not produce a report"
            )
        return DaemonResponse(
            status="ok",
            data={
                "merged_groups": report.merged_groups,
                "merged_bullets_total": report.merged_bullets_total,
                "soft_deleted": report.soft_deleted,
                "auto_supersede": report.auto_supersede,
                "queued_for_review": report.queued_for_review,
                "marked_conflict": report.marked_conflict,
                "duration_seconds": report.duration_seconds,
                "errors": list(report.errors),
            },
        )

    async def _handle_shutdown(self) -> DaemonResponse:
        """Initiate graceful shutdown."""
        logger.info("Shutdown command received.")
        # Schedule stop after response is sent
        if self._shutdown_event:
            self._shutdown_event.set()
        return DaemonResponse(status="ok")

    # -- Idle timeout -------------------------------------------------------

    def _start_idle_timer(self) -> None:
        """Start a countdown to auto-shutdown when no sessions are active."""
        self._cancel_idle_timer()
        timeout = self._config.idle_timeout_seconds
        logger.info(
            "No active sessions. Starting idle timer (%d seconds).", timeout,
        )
        loop = asyncio.get_event_loop()
        self._idle_timer = loop.create_task(self._idle_countdown(timeout))

    def _cancel_idle_timer(self) -> None:
        """Cancel the idle auto-shutdown timer if running."""
        if self._idle_timer is not None and not self._idle_timer.done():
            self._idle_timer.cancel()
            self._idle_timer = None
            logger.debug("Idle timer cancelled.")

    async def _idle_countdown(self, timeout: int) -> None:
        """Wait *timeout* seconds then trigger auto-shutdown."""
        try:
            await asyncio.sleep(timeout)
            logger.info(
                "Idle timeout reached (%d seconds). Auto-shutting down.", timeout,
            )
            if self._shutdown_event:
                self._shutdown_event.set()
        except asyncio.CancelledError:
            logger.debug("Idle countdown cancelled.")

    # -- Lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        """Start the daemon: check PID, init Memory, open IPC, wait for shutdown.

        This coroutine blocks until ``stop()`` or a shutdown command.
        """
        logger.info("MemorusDaemon starting (PID %d)...", os.getpid())
        self._check_pid()
        self._write_pid()
        self._start_time = datetime.now()
        self._shutdown_event = asyncio.Event()

        try:
            await self._init_memory()
            await self._start_ipc_server()
            self._running = True
            self._install_signal_handlers()

            # Start idle timer immediately (no sessions yet)
            self._start_idle_timer()

            # Spawn IdleOrchestrator (STORY-R094)
            self._spawn_orchestrator()

            logger.info("MemorusDaemon ready and serving.")
            # Block until shutdown
            await self._shutdown_event.wait()

        except Exception as exc:
            logger.error("MemorusDaemon failed to start: %s", exc, exc_info=True)
            raise
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Gracefully stop the daemon: close server, flush, remove PID."""
        if not self._running and self._server is None:
            # Already stopped or never started (only cleanup PID if needed)
            self._remove_pid()
            return

        logger.info("MemorusDaemon shutting down...")
        self._running = False

        # Stop orchestrator (STORY-R094)
        if self._orchestrator is not None:
            try:
                self._orchestrator.stop()
            except Exception as e:
                logger.debug("Orchestrator stop: %s", e)
        if self._orchestrator_task is not None and not self._orchestrator_task.done():
            self._orchestrator_task.cancel()
            try:
                await self._orchestrator_task
            except (asyncio.CancelledError, Exception):
                pass
            self._orchestrator_task = None

        # Cancel idle timer
        self._cancel_idle_timer()

        # Close IPC server
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None
            logger.info("IPC server closed.")

        # Clean up socket file on Unix
        if sys.platform != "win32":
            addr = self._config.socket_path or str(SOCKET_PATH)
            sock_path = Path(addr)
            if sock_path.exists():
                try:
                    sock_path.unlink()
                except OSError:
                    pass

        # Remove PID file
        self._remove_pid()

        # Clear sessions
        self._sessions.clear()

        logger.info("MemorusDaemon stopped.")

    async def _init_memory(self) -> None:
        """Initialize the Memory instance with ACE enabled."""
        logger.info("Initializing Memory...")
        try:
            from memorus.core.memory import Memory
            self._memory = Memory(config={"ace_enabled": True})
            logger.info("Memory initialized successfully.")
        except Exception as exc:
            logger.error("Failed to initialize Memory: %s", exc)
            raise DaemonError(f"Memory initialization failed: {exc}") from exc

    # -- IdleOrchestrator (STORY-R094) --------------------------------------

    def _spawn_orchestrator(self) -> None:
        """Construct and launch the IdleOrchestrator background task."""
        try:
            from memorus.core.config import ConsolidateConfig, CuratorConfig, MemorusConfig
            from memorus.core.daemon.orchestrator import (
                IdleOrchestrator,
                Mem0MemoryAdapter,
            )

            # Pull config from the Memory's MemorusConfig if available.
            mcfg = getattr(self._memory, "_config", None) if self._memory else None
            if isinstance(mcfg, MemorusConfig):
                consolidate_cfg = mcfg.consolidate
                curator_cfg = mcfg.curator
            else:
                consolidate_cfg = ConsolidateConfig()
                curator_cfg = CuratorConfig()

            adapter = Mem0MemoryAdapter(self._memory)
            orch = IdleOrchestrator(
                config=consolidate_cfg,
                curator_config=curator_cfg,
                adapter=adapter,
                load_bullets=self._load_bullets_for_consolidate,
            )
            self._orchestrator = orch

            sessions_empty_fn = self._make_sessions_empty_fn()
            loop = asyncio.get_event_loop()
            self._orchestrator_task = loop.create_task(
                orch.run_forever(sessions_empty_fn)
            )
            logger.info("IdleOrchestrator spawned.")
        except Exception as e:
            logger.error("Failed to spawn IdleOrchestrator: %s", e, exc_info=True)

    def _make_sessions_empty_fn(self) -> Any:
        async def _fn() -> bool:
            return len(self._sessions) == 0
        return _fn

    def _load_bullets_for_consolidate(self) -> list[Any]:
        """Load all live bullets as :class:`ExistingBullet` for consolidation."""
        from memorus.core.engines.curator.engine import ExistingBullet

        if self._memory is None:
            return []
        try:
            raw = self._memory.get_all()
        except Exception as e:
            logger.warning("orchestrator: get_all failed: %s", e)
            return []

        memories = raw.get("results") if isinstance(raw, dict) else None
        if not memories:
            memories = raw.get("memories", []) if isinstance(raw, dict) else []

        bullets: list[ExistingBullet] = []
        for mem in memories or []:
            if not isinstance(mem, dict):
                continue
            meta = mem.get("metadata") or {}
            if isinstance(meta, dict) and meta.get("memorus_deleted_at"):
                continue
            bid = mem.get("id") or ""
            content = mem.get("memory") or mem.get("content") or ""
            scope = meta.get("memorus_scope", "global") if isinstance(meta, dict) else "global"
            bullets.append(
                ExistingBullet(
                    bullet_id=bid,
                    content=content,
                    scope=scope,
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )
        return bullets

    def _install_signal_handlers(self) -> None:
        """Install SIGTERM/SIGINT handlers for clean shutdown."""
        loop = asyncio.get_event_loop()

        def _signal_handler() -> None:
            logger.info("Signal received, initiating shutdown...")
            if self._shutdown_event:
                self._shutdown_event.set()

        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, _signal_handler)
        else:
            # On Windows, signal handlers work differently.
            # asyncio proactor loop does not support add_signal_handler.
            # We rely on KeyboardInterrupt for Ctrl+C.
            pass

    def _uptime_seconds(self) -> float:
        """Return seconds since daemon started."""
        if hasattr(self, "_start_time"):
            return (datetime.now() - self._start_time).total_seconds()
        return 0.0

    # -- Class-level helpers ------------------------------------------------

    @classmethod
    def is_daemon_running(cls) -> bool:
        """Check if a daemon process is currently alive by inspecting the PID file."""
        if not PID_PATH.exists():
            return False
        try:
            pid = int(PID_PATH.read_text().strip())
            return _is_process_alive(pid)
        except (ValueError, OSError):
            return False

    @classmethod
    def read_pid(cls) -> Optional[int]:
        """Read the PID from the PID file, or return None if not present."""
        if not PID_PATH.exists():
            return None
        try:
            return int(PID_PATH.read_text().strip())
        except (ValueError, OSError):
            return None
