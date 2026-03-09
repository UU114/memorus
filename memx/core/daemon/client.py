"""DaemonClient -- typed wrapper around IPC calls to MemXDaemon.

Usage::

    client = DaemonClient()
    if await client.ping():
        results = await client.recall("async patterns", user_id="default")
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Optional

from memx.core.config import DaemonConfig
from memx.core.daemon.ipc import IPCTransport, get_transport
from memx.core.daemon.server import DaemonRequest, DaemonResponse
from memx.core.exceptions import DaemonUnavailableError

logger = logging.getLogger(__name__)

# Default timeouts (seconds)
DEFAULT_CONNECT_TIMEOUT = 2.0
DEFAULT_REQUEST_TIMEOUT = 10.0


class DaemonClient:
    """Client for communicating with MemXDaemon via IPC.

    Each public method creates a fresh transport connection, sends a
    :class:`DaemonRequest`, and parses the :class:`DaemonResponse`.
    If the daemon is not reachable, :class:`DaemonUnavailableError` is raised
    (except for :meth:`ping` which returns ``False``).

    Parameters
    ----------
    config:
        Optional :class:`DaemonConfig`.  Uses defaults when *None*.
    connect_timeout:
        Seconds to wait for the IPC connection to be established.
    request_timeout:
        Seconds to wait for a response after sending a request.
    """

    def __init__(
        self,
        config: Optional[DaemonConfig] = None,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._config = config or DaemonConfig()
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout

    # -- Public API ---------------------------------------------------------

    async def ping(self) -> bool:
        """Check whether the daemon is alive.

        Returns ``True`` if the daemon responded with status ``"ok"``,
        ``False`` on any connection or timeout error.
        """
        try:
            resp = await self._request(DaemonRequest(cmd="ping"))
            return resp.status == "ok"
        except (DaemonUnavailableError, ConnectionError, TimeoutError):
            return False

    async def is_running(self) -> bool:
        """Alias for :meth:`ping` -- convenient readability."""
        return await self.ping()

    async def recall(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search memories through the daemon.

        Returns a list of result dicts.  Raises
        :class:`DaemonUnavailableError` when the daemon is unreachable.
        """
        resp = await self._request(
            DaemonRequest(
                cmd="recall",
                data={"query": query, "user_id": user_id, "limit": limit},
            )
        )
        if resp.status == "error":
            raise DaemonUnavailableError(
                f"Daemon recall failed: {resp.error}"
            )
        return resp.data.get("results", [])

    async def curate(
        self,
        messages: Any,
        user_id: str = "default",
    ) -> dict[str, Any]:
        """Ingest messages via the daemon.

        Returns the daemon's response data dict.  Raises
        :class:`DaemonUnavailableError` when the daemon is unreachable.
        """
        resp = await self._request(
            DaemonRequest(
                cmd="curate",
                data={"messages": messages, "user_id": user_id},
            )
        )
        if resp.status == "error":
            raise DaemonUnavailableError(
                f"Daemon curate failed: {resp.error}"
            )
        return resp.data

    async def register_session(self, session_id: str) -> None:
        """Register a CLI session with the daemon."""
        resp = await self._request(
            DaemonRequest(
                cmd="session_register",
                data={"session_id": session_id},
            )
        )
        if resp.status == "error":
            raise DaemonUnavailableError(
                f"Daemon session_register failed: {resp.error}"
            )

    async def unregister_session(self, session_id: str) -> None:
        """Unregister a CLI session from the daemon."""
        resp = await self._request(
            DaemonRequest(
                cmd="session_unregister",
                data={"session_id": session_id},
            )
        )
        if resp.status == "error":
            raise DaemonUnavailableError(
                f"Daemon session_unregister failed: {resp.error}"
            )

    async def shutdown(self) -> None:
        """Request the daemon to shut down gracefully."""
        try:
            resp = await self._request(DaemonRequest(cmd="shutdown"))
            if resp.status == "error":
                logger.warning("Daemon shutdown returned error: %s", resp.error)
        except DaemonUnavailableError:
            # If the daemon is already gone, shutdown is effectively done
            logger.debug("Daemon already unreachable during shutdown request.")

    # -- Internal -----------------------------------------------------------

    async def _request(self, req: DaemonRequest) -> DaemonResponse:
        """Send a request to the daemon and return the parsed response.

        Creates a fresh transport for each request (no connection pooling).
        Raises :class:`DaemonUnavailableError` on connection/timeout/parse
        failures.
        """
        transport: IPCTransport = get_transport(self._config)
        try:
            await asyncio.wait_for(
                transport.connect(),
                timeout=self._connect_timeout,
            )
            payload = json.dumps(asdict(req)).encode("utf-8")
            await transport.send(payload)
            raw = await asyncio.wait_for(
                transport.recv(),
                timeout=self._request_timeout,
            )
            return DaemonResponse(**json.loads(raw.decode("utf-8")))
        except (ConnectionError, TimeoutError, OSError) as exc:
            raise DaemonUnavailableError(
                f"Daemon unavailable: {exc}"
            ) from exc
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            raise DaemonUnavailableError(
                f"Invalid response from daemon: {exc}"
            ) from exc
        finally:
            await transport.close()
