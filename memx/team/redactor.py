"""Redactor — three-layer sanitization engine for team knowledge sharing.

L1: Deterministic sanitization (PrivacySanitizer + team-specific patterns)
L2: User review (diff + warnings, NOT skippable)
L3: Optional LLM generalization (enabled via config)
"""

from __future__ import annotations

import difflib
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from memx.core.privacy.sanitizer import FilteredItem, PrivacySanitizer
from memx.team.config import RedactorConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team-specific patterns (additional to Core PrivacySanitizer builtins)
# ---------------------------------------------------------------------------

# Each pattern: (name, regex, replacement)
TEAM_PATTERNS: list[tuple[str, str, str]] = [
    # Internal IPs (RFC 1918 private ranges)
    (
        "internal_ip_10",
        r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
        "[INTERNAL_IP]",
    ),
    (
        "internal_ip_172",
        r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b",
        "[INTERNAL_IP]",
    ),
    (
        "internal_ip_192",
        r"\b192\.168\.\d{1,3}\.\d{1,3}\b",
        "[INTERNAL_IP]",
    ),
    # Internal URLs (common internal hostnames)
    (
        "internal_url",
        r"https?://(?:internal|intranet|private|staging|dev|local)\.[^\s,;\"']+",
        "[INTERNAL_URL]",
    ),
    # Project paths (Unix home dirs not caught by core, and project-style paths)
    # Also matches after core sanitization where /home/user becomes <USER_PATH>
    (
        "project_path_unix",
        r"(?:/(?:home|Users)/[^\s/]+|<USER_PATH>)/(?:projects|workspace|repos|src)/[^\s,;\"']*",
        "[PROJECT_PATH]",
    ),
    # Database connection strings (full URI form)
    (
        "db_connection_string",
        r"(?:postgres|postgresql|mysql|mongodb|redis|amqp)(?:ql)?://[^\s,;\"']+",
        "[DB_CONNECTION]",
    ),
    # AWS/Cloud ARNs
    (
        "cloud_arn",
        r"arn:aws:[a-zA-Z0-9\-]+:[a-zA-Z0-9\-]*:\d{0,12}:[^\s,;\"']+",
        "[CLOUD_RESOURCE]",
    ),
]


# ---------------------------------------------------------------------------
# LLM Generalizer protocol
# ---------------------------------------------------------------------------


class LLMGeneralizer(ABC):
    """Abstract interface for LLM-based content generalization.

    Actual implementations live in the ext/ layer — this defines the contract.
    """

    @abstractmethod
    async def generalize(self, content: str) -> str:
        """Generalize content to remove implicit sensitive context.

        Args:
            content: Already-sanitized content (L1 output).

        Returns:
            Generalized content string.
        """
        ...


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class RedactedResult:
    """Result of a redaction pass."""

    original_content: str
    clean_content: str
    filtered_items: list[FilteredItem] = field(default_factory=list)
    was_modified: bool = False
    context_summary: str | None = None

    @property
    def is_fully_redacted(self) -> bool:
        """True if content is entirely redacted (empty or only placeholders)."""
        if not self.clean_content or not self.clean_content.strip():
            return True
        # Check if content consists only of placeholder tokens
        stripped = re.sub(r"\[[\w_]+\]", "", self.clean_content)
        stripped = re.sub(r"<[\w_]+>", "", stripped)
        return not stripped.strip()


@dataclass
class ReviewPayload:
    """Formatted payload for user review (L2)."""

    original_content: str
    redacted_content: str
    diff_lines: list[str]  # unified diff format
    filtered_items: list[FilteredItem] = field(default_factory=list)
    is_fully_redacted: bool = False
    warning: str | None = None  # e.g., "Content fully redacted"


# ---------------------------------------------------------------------------
# Redactor engine
# ---------------------------------------------------------------------------


class Redactor:
    """Three-layer sanitization engine for team knowledge sharing.

    L1: Deterministic pattern matching (PrivacySanitizer + team patterns)
    L2: User review formatting (diff, warnings)
    L3: Optional LLM generalization
    """

    def __init__(
        self,
        config: RedactorConfig,
        sanitizer: PrivacySanitizer | None = None,
        llm_generalizer: LLMGeneralizer | None = None,
    ) -> None:
        self._config = config
        self._llm_generalizer = llm_generalizer

        # Build combined custom patterns: team patterns + user custom patterns
        combined_custom: list[str] = []

        # Add team-specific patterns as regex strings
        # We compile them into a PrivacySanitizer-compatible format below
        self._team_patterns: list[tuple[str, re.Pattern[str], str]] = []
        for name, regex, replacement in TEAM_PATTERNS:
            try:
                self._team_patterns.append((name, re.compile(regex), replacement))
            except re.error as e:
                logger.warning("Invalid team pattern '%s': %s", name, e)

        # Build PrivacySanitizer with user custom_patterns only
        # (team patterns are applied separately to preserve named replacements)
        if sanitizer is not None:
            self._sanitizer = sanitizer
        else:
            self._sanitizer = PrivacySanitizer(
                custom_patterns=config.custom_patterns if config.custom_patterns else None
            )

    def redact_l1(self, content: str) -> RedactedResult:
        """L1: Deterministic sanitization using PrivacySanitizer + team patterns.

        Applies core builtin patterns, user custom patterns, then team-specific
        patterns in sequence.

        Args:
            content: Raw content to sanitize.

        Returns:
            RedactedResult with clean content and filtered items list.
        """
        if not content:
            return RedactedResult(
                original_content=content,
                clean_content=content,
                was_modified=False,
            )

        # Step 1: Core sanitizer (builtin + user custom patterns)
        core_result = self._sanitizer.sanitize(content)
        clean = core_result.clean_content
        all_filtered: list[FilteredItem] = list(core_result.filtered_items)

        # Step 2: Team-specific patterns on the already-sanitized output
        for name, pattern, replacement in self._team_patterns:
            matches = list(pattern.finditer(clean))
            for match in matches:
                snippet = self._truncate_match(match.group())
                all_filtered.append(
                    FilteredItem(
                        pattern_name=name,
                        snippet=snippet,
                        position=match.start(),
                    )
                )
            if matches:
                logger.debug(
                    "Redactor team pattern '%s' matched %d time(s)",
                    name,
                    len(matches),
                )
            clean = pattern.sub(replacement, clean)

        was_modified = core_result.was_modified or len(all_filtered) > len(
            core_result.filtered_items
        )

        return RedactedResult(
            original_content=content,
            clean_content=clean,
            filtered_items=all_filtered,
            was_modified=was_modified,
        )

    def prepare_for_review(self, result: RedactedResult) -> ReviewPayload:
        """L2: Format sanitized content for user review.

        Generates a unified diff and warnings for the user to inspect
        before approving team sharing. This step is NOT skippable.

        Args:
            result: RedactedResult from L1 (or L3).

        Returns:
            ReviewPayload with diff and warnings.
        """
        # Generate unified diff
        original_lines = result.original_content.splitlines(keepends=True)
        redacted_lines = result.clean_content.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                original_lines,
                redacted_lines,
                fromfile="original",
                tofile="redacted",
                lineterm="",
            )
        )

        warning: str | None = None
        if result.is_fully_redacted:
            warning = "Content fully redacted, consider not nominating"

        return ReviewPayload(
            original_content=result.original_content,
            redacted_content=result.clean_content,
            diff_lines=diff_lines,
            filtered_items=result.filtered_items,
            is_fully_redacted=result.is_fully_redacted,
            warning=warning,
        )

    def apply_user_edits(
        self, result: RedactedResult, user_edited_content: str
    ) -> RedactedResult:
        """Apply user edits, then re-run L1 to catch any new sensitive data.

        The user may modify the redacted content during review. We must
        re-scan the edited version to ensure no new sensitive data was
        accidentally introduced.

        Args:
            result: Original RedactedResult (for tracking original_content).
            user_edited_content: Content after user edits.

        Returns:
            New RedactedResult with re-scanned content.
        """
        # Re-run L1 on user-edited content
        re_scanned = self.redact_l1(user_edited_content)

        # Preserve original_content from the initial result
        return RedactedResult(
            original_content=result.original_content,
            clean_content=re_scanned.clean_content,
            filtered_items=re_scanned.filtered_items,
            was_modified=re_scanned.was_modified or result.was_modified,
            context_summary=result.context_summary,
        )

    async def redact_l3(self, result: RedactedResult) -> RedactedResult:
        """L3: Optional LLM generalization. No-op if not configured.

        When enabled, sends L1-sanitized content to an LLM to generalize
        it further (e.g., remove implicit context clues). Falls back to
        L1 result on any error.

        Args:
            result: RedactedResult from L1.

        Returns:
            RedactedResult with LLM-generalized content, or original on failure.
        """
        if not self._config.llm_generalize:
            logger.debug("Redactor L3 skipped: llm_generalize is disabled")
            return result

        if self._llm_generalizer is None:
            logger.warning(
                "Redactor L3 requested but no LLMGeneralizer provided; "
                "falling back to L1 result"
            )
            return result

        try:
            generalized = await self._llm_generalizer.generalize(
                result.clean_content
            )
            logger.debug(
                "Redactor L3 generalized content: %d -> %d chars",
                len(result.clean_content),
                len(generalized),
            )
            return RedactedResult(
                original_content=result.original_content,
                clean_content=generalized,
                filtered_items=result.filtered_items,
                was_modified=True,
                context_summary=result.context_summary,
            )
        except Exception:
            logger.warning(
                "Redactor L3 LLM generalization failed; falling back to L1 result",
                exc_info=True,
            )
            return result

    def finalize(
        self,
        result: RedactedResult,
        *,
        context_summary: str | None = None,
    ) -> dict[str, Any]:
        """Produce final sanitized bullet dict ready for team sharing.

        Attaches context_summary if provided, and returns a dictionary
        suitable for constructing a TeamBullet.

        Args:
            result: Final RedactedResult (after L1/L2/L3).
            context_summary: Optional context summary to attach.

        Returns:
            Dict with sanitized content and metadata.
        """
        summary = context_summary or result.context_summary
        return {
            "content": result.clean_content,
            "context_summary": summary,
            "was_redacted": result.was_modified,
            "redacted_items_count": len(result.filtered_items),
        }

    @staticmethod
    def _truncate_match(text: str) -> str:
        """Truncate matched text for logging (hide most of the secret)."""
        if len(text) <= 12:
            return text[:4] + "..."
        return text[:8] + "..." + text[-4:]
