"""Extensible pattern rules for the Reflector PatternDetector.

Each rule is a self-contained class that implements :class:`PatternRule`.
The detector iterates over a list of rules, calling ``check()`` on each one.
To add a new detection heuristic, create a subclass and register it in
``PatternDetector._default_rules()``.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from memx.core.types import DetectedPattern, InteractionEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class PatternRule(ABC):
    """Base class for all pattern detection rules."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique rule name used as ``pattern_type``."""
        ...

    @abstractmethod
    def check(
        self, event: InteractionEvent, history: Sequence[InteractionEvent]
    ) -> Optional[DetectedPattern]:
        """Return a :class:`DetectedPattern` if *event* matches, else ``None``."""
        ...


# ---------------------------------------------------------------------------
# Rule 1 — Error-fix pattern (migrated from detector.py)
# ---------------------------------------------------------------------------


class ErrorFixRule(PatternRule):
    """Detect error -> fix pattern.

    Fires when the user message contains error-related language and the
    assistant message contains actionable fix keywords with substantive length.
    """

    ERROR_KEYWORDS = frozenset({
        "error", "fail", "exception", "traceback", "crash", "bug",
        "broken", "doesn't work", "not working", "can't", "cannot",
        "issue", "problem", "wrong",
    })

    FIX_KEYWORDS = frozenset({
        "try", "use", "run", "change", "fix", "solution", "instead",
        "should", "replace", "update", "install", "set", "add",
    })

    @property
    def name(self) -> str:
        return "error_fix"

    def check(
        self, event: InteractionEvent, history: Sequence[InteractionEvent]
    ) -> Optional[DetectedPattern]:
        user_lower = event.user_message.lower()
        has_error_context = any(kw in user_lower for kw in self.ERROR_KEYWORDS)

        # Also honour metadata.error_msg
        has_error_meta = bool(event.metadata.get("error_msg"))

        logger.debug(
            "ErrorFixRule: has_error=%s has_meta=%s asst_len=%d",
            has_error_context, has_error_meta, len(event.assistant_message.strip()),
        )

        if not has_error_context and not has_error_meta:
            return None

        # Assistant must provide a substantive response
        if len(event.assistant_message.strip()) < 20:
            logger.debug("ErrorFixRule: assistant response too short (<20)")
            return None

        assistant_lower = event.assistant_message.lower()
        has_fix = any(kw in assistant_lower for kw in self.FIX_KEYWORDS)
        if not has_fix:
            logger.debug("ErrorFixRule: no fix keywords found in assistant response")
            return None

        content = f"Error: {event.user_message[:200]}\nFix: {event.assistant_message[:300]}"
        logger.debug("ErrorFixRule: MATCH -> content_len=%d", len(content))
        return DetectedPattern(
            pattern_type=self.name,
            content=content,
            confidence=0.8,
            source_event=event,
            metadata={"detection_rule": self.name},
        )


# ---------------------------------------------------------------------------
# Rule 2 — Retry-success pattern (migrated from detector.py)
# ---------------------------------------------------------------------------


class RetrySuccessRule(PatternRule):
    """Detect retry-after-failure pattern.

    Fires when a previous event had an error on the same topic and the
    current event's assistant response indicates resolution.
    """

    ERROR_KEYWORDS = frozenset({"error", "fail", "broken", "not working"})
    SUCCESS_KEYWORDS = frozenset({"works", "success", "done", "fixed", "resolved"})

    @property
    def name(self) -> str:
        return "retry_success"

    def check(
        self, event: InteractionEvent, history: Sequence[InteractionEvent]
    ) -> Optional[DetectedPattern]:
        if len(history) < 1:
            logger.debug("RetrySuccessRule: no history, skipping")
            return None

        current_keywords = set(re.findall(r"\b\w{4,}\b", event.user_message.lower()))
        if not current_keywords:
            return None

        for prev_event in reversed(history):
            prev_keywords = set(
                re.findall(r"\b\w{4,}\b", prev_event.user_message.lower())
            )
            overlap = current_keywords & prev_keywords
            if len(overlap) < 2:
                continue

            prev_had_error = any(
                kw in prev_event.user_message.lower() for kw in self.ERROR_KEYWORDS
            )
            current_is_success = any(
                kw in event.assistant_message.lower() for kw in self.SUCCESS_KEYWORDS
            )
            logger.debug(
                "RetrySuccessRule: overlap=%d prev_error=%s curr_success=%s",
                len(overlap), prev_had_error, current_is_success,
            )
            if prev_had_error and current_is_success:
                content = (
                    f"Previous issue: {prev_event.user_message[:200]}\n"
                    f"Resolution: {event.assistant_message[:300]}"
                )
                return DetectedPattern(
                    pattern_type=self.name,
                    content=content,
                    confidence=0.7,
                    source_event=event,
                    metadata={
                        "detection_rule": self.name,
                        "related_keywords": list(overlap)[:5],
                    },
                )
        return None


# ---------------------------------------------------------------------------
# Rule 3 — Configuration change pattern (NEW)
# ---------------------------------------------------------------------------


class ConfigChangeRule(PatternRule):
    """Detect configuration change patterns.

    Fires when the conversation involves configuration keywords or references
    config-related file extensions, and the assistant gives a substantive reply.
    """

    CONFIG_KEYWORDS = frozenset({
        "config", "setting", "env", "configure", "setup", "environment",
        "variable", "option", "parameter",
    })
    CONFIG_EXTENSIONS = frozenset({
        ".env", ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf",
    })

    @property
    def name(self) -> str:
        return "config_change"

    def check(
        self, event: InteractionEvent, history: Sequence[InteractionEvent]
    ) -> Optional[DetectedPattern]:
        combined = f"{event.user_message} {event.assistant_message}".lower()
        has_config_keyword = any(kw in combined for kw in self.CONFIG_KEYWORDS)
        has_config_file = any(ext in combined for ext in self.CONFIG_EXTENSIONS)

        logger.debug(
            "ConfigChangeRule: has_keyword=%s has_file=%s",
            has_config_keyword, has_config_file,
        )

        if not has_config_keyword and not has_config_file:
            return None

        # Need substantive assistant response
        if len(event.assistant_message.strip()) < 20:
            logger.debug("ConfigChangeRule: assistant response too short")
            return None

        logger.debug("ConfigChangeRule: MATCH")
        return DetectedPattern(
            pattern_type=self.name,
            content=f"Config: {event.assistant_message[:400]}",
            confidence=0.6,
            source_event=event,
            metadata={"detection_rule": self.name},
        )


# ---------------------------------------------------------------------------
# Rule 4 — New tool / first-time command usage (NEW)
# ---------------------------------------------------------------------------


class NewToolRule(PatternRule):
    """Detect first-time tool or command usage.

    Fires when the conversation contains package-install commands (e.g.
    ``pip install``, ``npm install``) or when ``metadata["tool_name"]``
    references a tool not seen in history.
    """

    TOOL_PATTERNS = re.compile(
        r"\b(?:pip install|npm install|cargo add|brew install|apt install|"
        r"conda install|gem install|go get)\s+\S+",
        re.IGNORECASE,
    )

    @property
    def name(self) -> str:
        return "new_tool"

    def check(
        self, event: InteractionEvent, history: Sequence[InteractionEvent]
    ) -> Optional[DetectedPattern]:
        combined = f"{event.user_message} {event.assistant_message}"
        tool_matches: list[str] = self.TOOL_PATTERNS.findall(combined)

        logger.debug("NewToolRule: regex matches=%s", tool_matches[:3])

        if not tool_matches:
            # Fallback: check metadata for tool_name
            tool_name = event.metadata.get("tool_name", "")
            if not tool_name:
                return None
            # Only fire if this tool has not been seen in history
            seen_tools = {e.metadata.get("tool_name", "") for e in history}
            if tool_name in seen_tools:
                logger.debug("NewToolRule: tool %r already seen in history", tool_name)
                return None
            tool_matches = [tool_name]

        if not tool_matches:
            return None

        # Check assistant has substantive response
        if len(event.assistant_message.strip()) < 20:
            logger.debug("NewToolRule: assistant response too short")
            return None

        logger.debug("NewToolRule: MATCH tools=%s", tool_matches[:3])
        return DetectedPattern(
            pattern_type=self.name,
            content=f"Tool: {', '.join(tool_matches[:3])}\n{event.assistant_message[:300]}",
            confidence=0.65,
            source_event=event,
            metadata={"detection_rule": self.name, "tools": tool_matches[:3]},
        )


# ---------------------------------------------------------------------------
# Rule 5 — Repetitive operations (NEW)
# ---------------------------------------------------------------------------


class RepetitiveOpRule(PatternRule):
    """Detect repetitive operations (same topic 3+ times).

    Fires when the current event's user message shares significant keyword
    overlap with at least ``MIN_OCCURRENCES - 1`` previous events.
    """

    MIN_OCCURRENCES: int = 3

    @property
    def name(self) -> str:
        return "repetitive_op"

    def check(
        self, event: InteractionEvent, history: Sequence[InteractionEvent]
    ) -> Optional[DetectedPattern]:
        if len(history) < self.MIN_OCCURRENCES - 1:
            logger.debug("RepetitiveOpRule: history too short (%d < %d)", len(history), self.MIN_OCCURRENCES - 1)
            return None

        # Extract keywords (words with 4+ chars) from current event
        current_words = set(re.findall(r"\b\w{4,}\b", event.user_message.lower()))
        if len(current_words) < 2:
            return None

        # Count past events with significant overlap
        similar_count = 0
        for prev in history:
            prev_words = set(re.findall(r"\b\w{4,}\b", prev.user_message.lower()))
            overlap = current_words & prev_words
            if len(overlap) >= 2:
                similar_count += 1

        logger.debug("RepetitiveOpRule: similar_count=%d (need>=%d)", similar_count, self.MIN_OCCURRENCES - 1)
        if similar_count < self.MIN_OCCURRENCES - 1:
            return None

        logger.debug("RepetitiveOpRule: MATCH (occurrences=%d)", similar_count + 1)
        return DetectedPattern(
            pattern_type=self.name,
            content=f"Repeated pattern ({similar_count + 1}x): {event.user_message[:300]}",
            confidence=0.5 + min(0.3, similar_count * 0.1),
            source_event=event,
            metadata={"detection_rule": self.name, "occurrences": similar_count + 1},
        )
