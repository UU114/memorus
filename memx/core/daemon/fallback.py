"""DaemonFallbackManager -- graceful degradation when Daemon is unavailable.

When the Daemon is not running, has crashed, or becomes unresponsive,
MemX transparently falls back to direct in-process Memory operations.
Degradation and recovery are automatic and only logged at key transitions.

Usage::

    manager = DaemonFallbackManager(daemon_config)
    if manager.is_available:
        results = await manager.try_recall(query, user_id=user_id)
    else:
        # Direct mode -- call Memory.search() directly
        results = memory.search(query, user_id=user_id)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from memx.core.config import DaemonConfig
from memx.core.daemon.client import DaemonClient
from memx.core.exceptions import DaemonUnavailableError

logger = logging.getLogger(__name__)

# Default number of operations between recovery checks
DEFAULT_RECOVERY_INTERVAL = 10

# Minimum seconds between recovery ping attempts
DEFAULT_RECOVERY_COOLDOWN = 30.0


class DaemonFallbackManager:
    """Manages daemon availability with automatic degradation and recovery.

    This class wraps a :class:`DaemonClient` and adds transparent fallback
    logic.  When the daemon becomes unreachable, the manager flags itself as
    degraded.  Periodic recovery checks (counter-based with time cooldown)
    detect when the daemon comes back online.

    Parameters
    ----------
    config:
        :class:`DaemonConfig` controlling daemon connection settings.
    recovery_interval:
        Number of operations between automatic recovery ping attempts.
    recovery_cooldown:
        Minimum seconds to wait after a failure before retrying a ping.
    """

    def __init__(
        self,
        config: Optional[DaemonConfig] = None,
        recovery_interval: int = DEFAULT_RECOVERY_INTERVAL,
        recovery_cooldown: float = DEFAULT_RECOVERY_COOLDOWN,
    ) -> None:
        self._config = config or DaemonConfig()
        self._client = DaemonClient(config=self._config)
        self._available: bool = False
        self._degraded_logged: bool = False
        self._recovery_logged: bool = False
        self._op_counter: int = 0
        self._recovery_interval = recovery_interval
        self._recovery_cooldown = recovery_cooldown
        self._last_failure_time: float = 0.0

    # -- Properties ---------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Whether the daemon is currently considered available."""
        return self._available

    @property
    def client(self) -> DaemonClient:
        """The underlying DaemonClient instance."""
        return self._client

    @property
    def op_counter(self) -> int:
        """Number of operations processed since last recovery check."""
        return self._op_counter

    # -- Initial availability check -----------------------------------------

    def check_initial_availability(self) -> bool:
        """Ping the daemon once to determine initial availability.

        Called during Memory initialization.  Returns True if daemon is
        reachable, False otherwise.  Logs a WARNING on first degradation.
        """
        try:
            available = asyncio.run(self._client.ping())
        except RuntimeError:
            # If there is already an event loop running, use it
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Cannot use asyncio.run inside a running loop
                    available = False
                else:
                    available = loop.run_until_complete(self._client.ping())
            except Exception:
                available = False

        self._available = available
        if not available:
            self._log_degradation()
        else:
            self._degraded_logged = False
            self._recovery_logged = False
        return available

    # -- Operation wrappers -------------------------------------------------

    async def try_recall(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
    ) -> Optional[list[dict[str, Any]]]:
        """Attempt recall via daemon.  Returns None if unavailable.

        When the daemon is available, sends the recall request through IPC.
        On :class:`DaemonUnavailableError`, marks daemon as unavailable and
        returns ``None`` so the caller can fall back to direct mode.
        """
        await self._async_tick_op_counter()
        if not self._available:
            return None
        try:
            return await self._client.recall(query, user_id=user_id, limit=limit)
        except DaemonUnavailableError:
            self._mark_unavailable()
            return None

    async def try_curate(
        self,
        messages: Any,
        user_id: str = "default",
    ) -> Optional[dict[str, Any]]:
        """Attempt curate via daemon.  Returns None if unavailable.

        When the daemon is available, sends the curate request through IPC.
        On :class:`DaemonUnavailableError`, marks daemon as unavailable and
        returns ``None`` so the caller can fall back to direct mode.
        """
        await self._async_tick_op_counter()
        if not self._available:
            return None
        try:
            return await self._client.curate(messages, user_id=user_id)
        except DaemonUnavailableError:
            self._mark_unavailable()
            return None

    # -- Recovery check -----------------------------------------------------

    async def check_recovery(self) -> bool:
        """Explicitly check if the daemon has recovered.

        Returns True if daemon is now available.  This is called
        automatically via the operation counter, but can also be called
        manually.
        """
        if self._available:
            return True
        if not self._is_cooldown_elapsed():
            return False

        try:
            alive = await self._client.ping()
        except Exception:
            alive = False

        if alive:
            self._mark_available()
            return True
        return False

    # -- Internal -----------------------------------------------------------

    async def _async_tick_op_counter(self) -> None:
        """Async version: increment counter and trigger recovery check if needed.

        Used from async methods (try_recall, try_curate) where we can
        directly await the ping coroutine.
        """
        self._op_counter += 1
        if (
            not self._available
            and self._op_counter % self._recovery_interval == 0
            and self._is_cooldown_elapsed()
        ):
            try:
                recovered = await self._client.ping()
            except Exception:
                recovered = False

            if recovered:
                self._mark_available()
            else:
                self._last_failure_time = time.monotonic()

    def _tick_op_counter(self) -> None:
        """Sync version: increment counter and trigger recovery check if needed.

        Used from synchronous contexts where asyncio.run() is available
        (e.g., called from Memory.search/add via asyncio.run at top level).
        """
        self._op_counter += 1
        if (
            not self._available
            and self._op_counter % self._recovery_interval == 0
            and self._is_cooldown_elapsed()
        ):
            try:
                recovered = asyncio.run(self._client.ping())
            except RuntimeError:
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        recovered = False
                    else:
                        recovered = loop.run_until_complete(self._client.ping())
                except Exception:
                    recovered = False
            except Exception:
                recovered = False

            if recovered:
                self._mark_available()
            else:
                self._last_failure_time = time.monotonic()

    def _mark_unavailable(self) -> None:
        """Mark daemon as unavailable and log degradation warning."""
        self._available = False
        self._last_failure_time = time.monotonic()
        self._log_degradation()

    def _mark_available(self) -> None:
        """Mark daemon as available and log recovery info."""
        self._available = True
        self._degraded_logged = False
        if not self._recovery_logged:
            logger.info("Daemon reconnected, switching to IPC mode")
            self._recovery_logged = True
        # Reset recovery_logged after a successful reconnection cycle
        # so next disconnection+reconnection can log again
        self._recovery_logged = False

    def _log_degradation(self) -> None:
        """Log degradation warning (only on first occurrence per episode)."""
        if not self._degraded_logged:
            logger.warning(
                "Daemon unavailable, falling back to direct mode"
            )
            self._degraded_logged = True

    def _is_cooldown_elapsed(self) -> bool:
        """Check if enough time has passed since last failure to retry."""
        if self._last_failure_time == 0.0:
            return True
        return (time.monotonic() - self._last_failure_time) >= self._recovery_cooldown
