"""ConflictDetector -- detects contradictory memories in the knowledge base.

Scans pairs of existing memories for potential contradictions by checking:
1. Text similarity in a configurable [min, max] range (likely same topic)
2. Negation asymmetry (one affirms, the other negates)
3. Opposing keyword pairs (always/never, enable/disable, etc.)
4. Version conflicts (same tool/library but different version references)

Grouped by section for performance on large datasets.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memx.core.config import CuratorConfig

from memx.core.engines.curator.engine import CuratorEngine, ExistingBullet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Conflict:
    """A pair of memories that may contradict each other."""

    memory_a_id: str
    memory_b_id: str
    memory_a_content: str
    memory_b_content: str
    similarity: float
    reason: str


@dataclass
class ConflictResult:
    """Result of conflict detection scan."""

    conflicts: list[Conflict] = field(default_factory=list)
    total_pairs_checked: int = 0
    scan_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# ConflictDetector
# ---------------------------------------------------------------------------


class ConflictDetector:
    """Detects contradictory memories based on similarity and negation analysis.

    Memories are compared pairwise within the same section. A pair is flagged
    as a conflict when:
    - Their text similarity falls within [conflict_min, conflict_max]
      (indicating they discuss the same topic but are not duplicates)
    - A contradiction signal is found (opposing keywords or negation asymmetry)
    """

    NEGATION_WORDS: dict[str, set[str]] = {
        "en": {
            "not", "never", "don't", "dont", "shouldn't", "avoid",
            "disable", "without", "no",
        },
        "zh": {"不要", "禁止", "避免", "不可", "别", "勿", "不能", "不应"},
    }

    OPPOSING_PAIRS: list[tuple[str, str]] = [
        ("always", "never"),
        ("enable", "disable"),
        ("use", "avoid"),
        ("with", "without"),
        ("add", "remove"),
        ("true", "false"),
        ("yes", "no"),
        ("allow", "deny"),
        ("open", "close"),
    ]

    # Patterns that capture (tool_or_lib, version_string) from text.
    # Order matters: more specific patterns first.
    VERSION_PATTERNS: list[re.Pattern[str]] = [
        # "react@18.2.0", "lodash@4", "express@5.0"
        re.compile(r"([a-zA-Z][\w.-]*)@(\d+(?:\.\d+){0,2}(?:-[\w.]+)?)"),
        # "react 18.2.0", "python 3.11", "node 20.x"
        re.compile(r"([a-zA-Z][\w.-]*)\s+(v?\d+(?:\.\d+){0,2}(?:-[\w.]+)?(?:\.x)?)"),
        # "react v18", "vue v3.4"
        re.compile(r"([a-zA-Z][\w.-]*)\s+v(\d+(?:\.\d+){0,2}(?:-[\w.]+)?)"),
    ]

    # Words that are NOT tool/library names (avoid false positives).
    _VERSION_STOPWORDS: set[str] = {
        "version", "v", "step", "part", "chapter", "section", "level",
        "tier", "phase", "stage", "day", "year", "http", "port",
        "error", "code", "status", "line", "page", "item", "rule",
    }

    def __init__(self, config: CuratorConfig | None = None) -> None:
        if config is not None:
            self._min_sim = config.conflict_min_similarity
            self._max_sim = config.conflict_max_similarity
        else:
            self._min_sim = 0.5
            self._max_sim = 0.8

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, memories: list[ExistingBullet]) -> ConflictResult:
        """Scan all memory pairs for potential contradictions.

        Groups memories by scope for efficiency: only pairs in the same
        scope are compared. Returns a ConflictResult with all detected
        conflicts and scan statistics.
        """
        start = time.monotonic()
        result = ConflictResult()

        logger.debug(
            "ConflictDetector.detect: memories=%d sim_range=[%.2f, %.2f]",
            len(memories), self._min_sim, self._max_sim,
        )

        if len(memories) < 2:
            logger.debug("ConflictDetector: <2 memories, nothing to check")
            result.scan_time_ms = (time.monotonic() - start) * 1000
            return result

        # Group by scope for optimisation
        groups: dict[str, list[ExistingBullet]] = {}
        for mem in memories:
            groups.setdefault(mem.scope, []).append(mem)
        logger.debug("ConflictDetector: %d scope group(s): %s",
                      len(groups), {k: len(v) for k, v in groups.items()})

        for group in groups.values():
            if len(group) < 2:
                continue
            for a, b in combinations(group, 2):
                result.total_pairs_checked += 1
                sim = CuratorEngine.text_similarity(a.content, b.content)

                if sim < self._min_sim or sim > self._max_sim:
                    logger.debug(
                        "ConflictDetector: pair (%s, %s) sim=%.3f OUT of range",
                        a.bullet_id, b.bullet_id, sim,
                    )
                    continue

                logger.debug(
                    "ConflictDetector: pair (%s, %s) sim=%.3f IN range, checking contradiction",
                    a.bullet_id, b.bullet_id, sim,
                )
                reason = self._check_contradiction(a.content, b.content)
                if reason is not None:
                    logger.debug(
                        "ConflictDetector: CONFLICT found (%s, %s) reason=%s",
                        a.bullet_id, b.bullet_id, reason,
                    )
                    result.conflicts.append(
                        Conflict(
                            memory_a_id=a.bullet_id,
                            memory_b_id=b.bullet_id,
                            memory_a_content=a.content,
                            memory_b_content=b.content,
                            similarity=sim,
                            reason=reason,
                        )
                    )
                else:
                    logger.debug(
                        "ConflictDetector: pair (%s, %s) no contradiction signals",
                        a.bullet_id, b.bullet_id,
                    )

        result.scan_time_ms = (time.monotonic() - start) * 1000
        logger.debug(
            "ConflictDetector.detect done: %d pairs checked, %d conflicts, %.1fms",
            result.total_pairs_checked, len(result.conflicts), result.scan_time_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Contradiction checks
    # ------------------------------------------------------------------

    def _check_contradiction(self, text_a: str, text_b: str) -> str | None:
        """Check whether two texts contradict each other.

        Returns a human-readable reason string if a contradiction is found,
        or None otherwise.

        Checks performed:
        1. Opposing keyword pairs (e.g. one says "always", the other "never")
        2. Negation asymmetry (one contains negation words, the other does not)
        3. Version conflict (same tool/library but different version numbers)
        """
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())

        # Check 1: Opposing pairs
        for word_a, word_b in self.OPPOSING_PAIRS:
            if word_a in tokens_a and word_b in tokens_b:
                logger.debug("ConflictDetector._check: opposing pair '%s' vs '%s'", word_a, word_b)
                return f"opposing_pair: {word_a} vs {word_b}"
            if word_b in tokens_a and word_a in tokens_b:
                logger.debug("ConflictDetector._check: opposing pair '%s' vs '%s'", word_b, word_a)
                return f"opposing_pair: {word_b} vs {word_a}"

        # Check 2: Negation asymmetry (check both EN and ZH)
        all_negations: set[str] = set()
        for lang_words in self.NEGATION_WORDS.values():
            all_negations.update(lang_words)

        # For Chinese negation, also check character-level containment
        neg_a = bool(tokens_a & all_negations) or self._has_zh_negation(text_a)
        neg_b = bool(tokens_b & all_negations) or self._has_zh_negation(text_b)
        logger.debug("ConflictDetector._check: neg_a=%s neg_b=%s", neg_a, neg_b)

        if neg_a != neg_b:
            return "negation_asymmetry"

        # Check 3: Version conflict
        version_reason = self._check_version_conflict(text_a, text_b)
        if version_reason is not None:
            return version_reason

        return None

    def _extract_versions(self, text: str) -> dict[str, str]:
        """Extract {tool_name: version} pairs from text.

        Applies all VERSION_PATTERNS and returns the first version found
        per tool name (lowercased). Filters out stopwords.
        """
        results: dict[str, str] = {}
        for pattern in self.VERSION_PATTERNS:
            for match in pattern.finditer(text):
                tool = match.group(1).lower().rstrip(".")
                ver = match.group(2).lstrip("v")
                if tool in self._VERSION_STOPWORDS or len(tool) < 2:
                    continue
                if tool not in results:
                    results[tool] = ver
        return results

    def _check_version_conflict(self, text_a: str, text_b: str) -> str | None:
        """Detect version conflict: same tool referenced with different versions.

        Returns a reason string like 'version_conflict: react 17.0 vs 18.2'
        or None if no version conflict is found.
        """
        versions_a = self._extract_versions(text_a)
        versions_b = self._extract_versions(text_b)

        if not versions_a or not versions_b:
            return None

        for tool, ver_a in versions_a.items():
            ver_b = versions_b.get(tool)
            if ver_b is not None and ver_a != ver_b:
                logger.debug(
                    "ConflictDetector._check_version: %s %s vs %s",
                    tool, ver_a, ver_b,
                )
                return f"version_conflict: {tool} {ver_a} vs {ver_b}"

        return None

    def _has_zh_negation(self, text: str) -> bool:
        """Check if text contains Chinese negation characters/words."""
        return any(word in text for word in self.NEGATION_WORDS["zh"])
