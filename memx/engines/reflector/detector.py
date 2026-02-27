"""PatternDetector — Stage 1 of the Reflector knowledge distillation pipeline.

Uses an extensible list of :class:`PatternRule` instances to detect learnable
patterns from AI interactions.  Rules can be swapped, extended, or overridden
by passing a custom list to the constructor.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

from memx.engines.reflector.patterns import PatternRule
from memx.types import DetectedPattern, InteractionEvent

logger = logging.getLogger(__name__)


class PatternDetector:
    """Detect learnable patterns from AI interactions. Never raises from detect()."""

    def __init__(
        self,
        rules: Optional[list[PatternRule]] = None,
        max_history: int = 20,
    ):
        self._rules: list[PatternRule] = (
            rules if rules is not None else self._default_rules()
        )
        self._history: deque[InteractionEvent] = deque(maxlen=max_history)

    # ------------------------------------------------------------------
    # Default rule set
    # ------------------------------------------------------------------

    @staticmethod
    def _default_rules() -> list[PatternRule]:
        """Return the built-in set of pattern detection rules."""
        from memx.engines.reflector.patterns import (
            ConfigChangeRule,
            ErrorFixRule,
            NewToolRule,
            RepetitiveOpRule,
            RetrySuccessRule,
        )

        return [
            ErrorFixRule(),
            RetrySuccessRule(),
            ConfigChangeRule(),
            NewToolRule(),
            RepetitiveOpRule(),
        ]

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------

    def detect(self, event: InteractionEvent) -> list[DetectedPattern]:
        """Detect patterns from interaction event. Returns empty list on failure."""
        try:
            # Filter: reject code-heavy content
            combined = f"{event.user_message}\n{event.assistant_message}"
            if self._is_code_heavy(combined):
                self._history.append(event)
                return []

            patterns: list[DetectedPattern] = []
            for rule in self._rules:
                try:
                    if p := rule.check(event, list(self._history)):
                        patterns.append(p)
                except Exception as e:
                    logger.warning("Rule %s failed: %s", rule.name, e)

            self._history.append(event)
            return patterns
        except Exception as e:
            logger.warning("PatternDetector.detect failed: %s", e)
            return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _is_code_heavy(content: str, threshold: float = 0.6) -> bool:
        """Return True if content is predominantly code (> threshold ratio)."""
        lines = content.strip().splitlines()
        if not lines:
            return False
        if len(lines) < 3:
            return False

        code_indicators = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Indented lines
            if line != line.lstrip() and len(stripped) > 0:
                code_indicators += 1
            # Lines with code-like characters
            elif any(ch in stripped for ch in {"{", "}", ";", "()", "[]", "==", "!=", "->"}):
                code_indicators += 1
            # Import/def/class statements
            elif stripped.startswith(("import ", "from ", "def ", "class ", "return ", "if ", "for ")):
                code_indicators += 1
            # Short lines (< 3 words) that look like code
            elif len(stripped.split()) < 3 and any(c in stripped for c in "=(){};"):
                code_indicators += 1

        non_empty = sum(1 for line in lines if line.strip())
        if non_empty == 0:
            return False
        return code_indicators / non_empty > threshold

    def clear_history(self) -> None:
        """Clear event history (call on session end)."""
        self._history.clear()
