"""PrivacySanitizer -- hardcoded privacy safety net for MemX."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from memx.privacy.patterns import BUILTIN_PATTERNS

logger = logging.getLogger(__name__)


@dataclass
class FilteredItem:
    """Record of a single redacted item."""

    pattern_name: str
    snippet: str  # Truncated: first 8 + "..." + last 4 chars
    position: int


@dataclass
class SanitizeResult:
    """Result of sanitization pass."""

    clean_content: str
    filtered_items: list[FilteredItem] = field(default_factory=list)
    was_modified: bool = False


class PrivacySanitizer:
    """Hardcoded privacy safety net. Core patterns cannot be disabled."""

    def __init__(self, custom_patterns: list[str] | None = None) -> None:
        # Builtin patterns are ALWAYS active
        self._patterns: list[tuple[str, re.Pattern[str], str]] = []
        for name, regex, replacement in BUILTIN_PATTERNS:
            self._patterns.append((name, re.compile(regex), replacement))

        # Add user custom patterns (appended, never replacing builtins)
        if custom_patterns:
            for i, pat in enumerate(custom_patterns):
                try:
                    self._patterns.append(
                        (f"custom_{i}", re.compile(pat), "<REDACTED>")
                    )
                except re.error as e:
                    logger.warning("Invalid custom pattern '%s': %s", pat, e)

    def sanitize(self, content: str) -> SanitizeResult:
        """Sanitize content, replacing sensitive data. Never raises."""
        if not content:
            return SanitizeResult(clean_content=content, was_modified=False)

        filtered: list[FilteredItem] = []
        clean = content

        for name, pattern, replacement in self._patterns:
            # Find all matches first (for reporting)
            for match in pattern.finditer(clean):
                snippet = self._truncate_match(match.group())
                filtered.append(
                    FilteredItem(
                        pattern_name=name,
                        snippet=snippet,
                        position=match.start(),
                    )
                )
            # Then replace
            clean = pattern.sub(replacement, clean)

        return SanitizeResult(
            clean_content=clean,
            filtered_items=filtered,
            was_modified=len(filtered) > 0,
        )

    @staticmethod
    def _truncate_match(text: str) -> str:
        """Truncate matched text for logging (hide most of the secret)."""
        if len(text) <= 12:
            return text[:4] + "..."
        return text[:8] + "..." + text[-4:]
