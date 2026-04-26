"""ConflictDetector -- detects contradictory memories in the knowledge base.

Scans pairs of existing memories for potential contradictions by checking:
1. Text similarity in a configurable [min, max] range (likely same topic)
2. Negation asymmetry (one affirms, the other negates)
3. Opposing keyword pairs (always/never, enable/disable, etc.)
4. Version conflicts (same tool/library but different version references)
5. Anchor mismatch: two bullets anchored at the same file region, but one is
   Verified and the other is Stale (STORY-R104).

Grouped by section for performance on large datasets.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memorus.core.config import CuratorConfig

from memorus.core.engines.curator.engine import CuratorEngine, ExistingBullet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class ConflictType(str, Enum):
    """Categorises the kind of conflict flagged between two bullets.

    The wire form is the lowercase snake_case string. Must stay aligned
    with ``memorus_ace::curator::ConflictType`` on the Rust side.
    """

    OPPOSING_PAIR = "opposing_pair"
    NEGATION_ASYMMETRY = "negation_asymmetry"
    VERSION_CONFLICT = "version_conflict"
    # STORY-R104: two bullets anchored at the same file region where one has
    # verified_status == "verified" and the other has verified_status == "stale".
    ANCHOR_MISMATCH = "anchor_mismatch"


@dataclass
class Conflict:
    """A pair of memories that may contradict each other."""

    memory_a_id: str
    memory_b_id: str
    memory_a_content: str
    memory_b_content: str
    similarity: float
    reason: str
    # STORY-R104: optional type classifier. Defaults to ``None`` to preserve
    # backward compatibility with callers that only read ``reason``.
    conflict_type: ConflictType | None = None


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

                # STORY-R104: anchor-mismatch pass runs regardless of text
                # similarity — the signal is the shared anchor region, not
                # lexical overlap.
                if self._detect_anchor_mismatch(a, b):
                    logger.debug(
                        "ConflictDetector: ANCHOR_MISMATCH (%s, %s)",
                        a.bullet_id, b.bullet_id,
                    )
                    result.conflicts.append(
                        Conflict(
                            memory_a_id=a.bullet_id,
                            memory_b_id=b.bullet_id,
                            memory_a_content=a.content,
                            memory_b_content=b.content,
                            similarity=sim,
                            reason=ConflictType.ANCHOR_MISMATCH.value,
                            conflict_type=ConflictType.ANCHOR_MISMATCH,
                        )
                    )
                    # A single classification per pair is enough — fall through
                    # to the similarity-gated contradiction checks only when
                    # anchor-mismatch did NOT fire, to avoid double-counting.
                    continue

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
                            conflict_type=_classify_reason(reason),
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

    # ------------------------------------------------------------------
    # Anchor-mismatch detection (STORY-R104)
    # ------------------------------------------------------------------

    def _detect_anchor_mismatch(
        self, bullet_a: ExistingBullet, bullet_b: ExistingBullet
    ) -> bool:
        """Return True iff the bullet pair qualifies as an anchor-mismatch.

        An anchor mismatch fires when ALL of the following hold:

        * both bullets carry at least one anchor dict
        * some anchor pair (a_i, b_j) shares ``file_path`` AND their
          ``anchor_text`` values have Jaccard token-set overlap ≥ 0.5
        * exactly one of the two bullets has ``verified_status == "verified"``
          and the other has ``verified_status == "stale"`` — any other
          combination (both verified, both stale, verified↔not_applicable,
          missing status, etc.) does NOT trigger.
        """
        status_a = _extract_verified_status(bullet_a.metadata)
        status_b = _extract_verified_status(bullet_b.metadata)
        if not _is_verified_stale_pair(status_a, status_b):
            return False

        anchors_a = _extract_anchors(bullet_a.metadata)
        anchors_b = _extract_anchors(bullet_b.metadata)
        if not anchors_a or not anchors_b:
            return False

        for a_anchor in anchors_a:
            for b_anchor in anchors_b:
                if a_anchor.get("file_path") != b_anchor.get("file_path"):
                    continue
                if not a_anchor.get("file_path"):
                    continue
                if overlap_ratio(
                    a_anchor.get("anchor_text", ""),
                    b_anchor.get("anchor_text", ""),
                ) >= 0.5:
                    return True
        return False


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def overlap_ratio(a: str, b: str) -> float:
    """Jaccard overlap ratio over whitespace-tokenized word sets.

    Returns ``|A ∩ B| / |A ∪ B|``. Matches the contract described in
    STORY-R104 §Technical Notes — caller is responsible for supplying
    two strings to compare (typically two ``Anchor.anchor_text`` values).
    """
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


def _extract_verified_status(metadata: dict[str, object]) -> str | None:
    """Return ``verified_status`` from metadata, if present and valid."""
    if not isinstance(metadata, dict):
        return None
    # Prefer the top-level key but fall back to nested BulletMetadata dict
    # to stay robust against both storage shapes.
    raw = metadata.get("verified_status")
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    # Enum-like objects expose ``.value``; duck-type check.
    value = getattr(raw, "value", None)
    if isinstance(value, str):
        return value
    return None


def _extract_anchors(metadata: dict[str, object]) -> list[dict[str, object]]:
    """Return a list of anchor dicts from metadata, robust to shape."""
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("anchors")
    if not isinstance(raw, list):
        return []
    result: list[dict[str, object]] = []
    for entry in raw:
        if isinstance(entry, dict):
            result.append(entry)
            continue
        # Support pydantic models that expose model_dump or dict()
        dumper = getattr(entry, "model_dump", None) or getattr(entry, "dict", None)
        if callable(dumper):
            try:
                dumped = dumper()
            except Exception:
                continue
            if isinstance(dumped, dict):
                result.append(dumped)
    return result


def _is_verified_stale_pair(a: str | None, b: str | None) -> bool:
    """Return True iff one side is ``"verified"`` and the other is ``"stale"``."""
    if a is None or b is None:
        return False
    return {a, b} == {"verified", "stale"}


def _classify_reason(reason: str) -> ConflictType | None:
    """Map a textual reason back to its ``ConflictType`` enum value."""
    if reason.startswith("opposing_pair"):
        return ConflictType.OPPOSING_PAIR
    if reason == "negation_asymmetry":
        return ConflictType.NEGATION_ASYMMETRY
    if reason.startswith("version_conflict"):
        return ConflictType.VERSION_CONFLICT
    if reason == ConflictType.ANCHOR_MISMATCH.value:
        return ConflictType.ANCHOR_MISMATCH
    return None
