"""ReflectorRouter — split incoming turns between realtime and inbox paths.

Decision tree (STORY-R095):

* if ``event.metadata["is_correction"] == True``  -> realtime
* if reflector ``mode == "rules"``                -> realtime
* if ``batch.batch_enabled == False``             -> realtime
* otherwise                                        -> enqueue to inbox

The router never runs the distillation pipeline itself; it returns a
:class:`RouteDecision` and (for inbox routes) enqueues an ``InboxEntry``.
The caller (``ReflectorEngine``) then either runs the realtime pipeline or
returns ``AckPending``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from memorus.core.config import ReflectorConfig
from memorus.core.engines.reflector.inbox import Inbox, InboxEntry, make_entry
from memorus.core.types import InteractionEvent

logger = logging.getLogger(__name__)


# Heuristic regex — matches user utterances that look like a correction.
# Router prefers an explicit ``event.metadata["is_correction"]`` flag when set;
# the regex is only a fallback so that plain sync callers can still surface
# "oops that's wrong" without plumbing metadata through every layer.
_CORRECTION_RE = re.compile(
    r"\b(?:no,?\s+|actually|correction|wrong|mistake|that'?s\s+(?:not|wrong|incorrect)|don'?t\b|not\s+that|oops|sorry,?\s+(?:i\s+)?meant)\b",
    re.IGNORECASE,
)


class RouteKind:
    REALTIME = "realtime"
    INBOX = "inbox"


@dataclass
class RouteDecision:
    kind: str  # "realtime" | "inbox"
    reason: str  # short machine-readable code
    entry: Optional[InboxEntry] = None


class ReflectorRouter:
    """Decide where each InteractionEvent should flow."""

    def __init__(self, config: ReflectorConfig, inbox: Optional[Inbox] = None) -> None:
        self._config = config
        self._inbox = inbox or Inbox(
            path=config.batch.inbox_path,
            consumed_retention_seconds=config.batch.consumed_retention_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def inbox(self) -> Inbox:
        return self._inbox

    def route(self, event: InteractionEvent) -> RouteDecision:
        """Decide between realtime and inbox."""
        if event is None:
            return RouteDecision(kind=RouteKind.REALTIME, reason="null_event")

        is_correction = self._is_correction(event)

        if is_correction:
            logger.debug("Router: correction detected -> realtime")
            return RouteDecision(kind=RouteKind.REALTIME, reason="correction")

        if self._config.mode == "rules":
            logger.debug("Router: rules mode -> realtime (no LLM)")
            return RouteDecision(kind=RouteKind.REALTIME, reason="rules_mode")

        if not self._config.batch.batch_enabled:
            logger.debug("Router: batch disabled -> realtime")
            return RouteDecision(kind=RouteKind.REALTIME, reason="batch_disabled")

        entry = self._build_entry(event, correction=False)
        try:
            self._inbox.append(entry)
        except OSError as e:
            logger.error(
                "Router: inbox append failed (%s), falling back to realtime", e,
            )
            return RouteDecision(kind=RouteKind.REALTIME, reason="inbox_write_failed")

        logger.debug(
            "Router: enqueued to inbox id=%s conv=%s offset=%d",
            entry.id, entry.conversation_id, entry.turn_offset,
        )
        return RouteDecision(kind=RouteKind.INBOX, reason="enqueued", entry=entry)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_correction(event: InteractionEvent) -> bool:
        """Return True iff this turn should be treated as a correction."""
        meta = event.metadata or {}
        flag = meta.get("is_correction")
        if isinstance(flag, bool):
            return flag
        # Only fall back to regex if no explicit flag was set. This keeps the
        # rule transparent: callers who know better always win.
        return bool(_CORRECTION_RE.search(event.user_message or ""))

    def _build_entry(self, event: InteractionEvent, *, correction: bool) -> InboxEntry:
        meta = event.metadata or {}
        conv_id = str(
            meta.get("conversation_id") or event.session_id or event.user_id or "",
        )
        turn_offset = int(meta.get("turn_offset", 0))
        return make_entry(
            user_message=event.user_message,
            assistant_message=event.assistant_message,
            conversation_id=conv_id,
            turn_offset=turn_offset,
            role=str(meta.get("role", "user")),
            correction_detected=correction,
            timestamp=event.timestamp,
            metadata={
                k: v
                for k, v in meta.items()
                if k not in {"is_correction", "conversation_id", "turn_offset", "role"}
            },
        )
