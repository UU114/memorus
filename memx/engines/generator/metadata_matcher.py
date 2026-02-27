"""MetadataMatcher (L3) — metadata-based relevance matching for Generator search.

Matches query tokens against structured metadata fields:
  - related_tools:  prefix matching (e.g. "git" matches "git-rebase")
  - key_entities:   prefix matching (e.g. "React" matches "ReactDOM")
  - tags:           exact matching  (e.g. "python" matches "python")

All comparisons are case-insensitive.  Score range: 0.0 -- 10.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetadataInfo:
    """Metadata fields for a bullet, used in matching."""

    related_tools: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MatchResult:
    """Result of a metadata match operation.

    Attributes:
        score:            Total metadata match score, clamped to [0.0, 10.0].
        tools_score:      Score contribution from related_tools matches.
        entities_score:   Score contribution from key_entities matches.
        tags_score:       Score contribution from tags matches.
        matched_tools:    Tools that matched the query.
        matched_entities: Entities that matched the query.
        matched_tags:     Tags that matched the query.
    """

    score: float
    tools_score: float
    entities_score: float
    tags_score: float
    matched_tools: list[str] = field(default_factory=list)
    matched_entities: list[str] = field(default_factory=list)
    matched_tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MetadataMatcher
# ---------------------------------------------------------------------------


class MetadataMatcher:
    """L3 metadata-based matcher for the Generator search engine.

    Scoring breakdown (configurable via constructor):
      - related_tools hit:  up to ``tools_score``   (default 4.0)
      - key_entities hit:   up to ``entities_score`` (default 3.0)
      - tags hit:           up to ``tags_score``     (default 3.0)
      - Maximum total:      10.0

    Usage::

        matcher = MetadataMatcher()
        result = matcher.match("git rebase tips", metadata_info)
        print(result.score, result.matched_tools)
    """

    def __init__(
        self,
        tools_score: float = 4.0,
        entities_score: float = 3.0,
        tags_score: float = 3.0,
    ) -> None:
        self._tools_score = tools_score
        self._entities_score = entities_score
        self._tags_score = tags_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def match(self, query: str, metadata: MetadataInfo) -> MatchResult:
        """Match *query* against *metadata* fields and return scored result.

        Args:
            query:    Free-text search query.
            metadata: Structured metadata of a bullet.

        Returns:
            MatchResult with per-field scores and matched items.
        """
        tokens = self._tokenize(query)

        # Empty query -> zero score
        if not tokens:
            return self._empty_result()

        # Per-field matching
        tools_hits = self._prefix_match(tokens, metadata.related_tools)
        entities_hits = self._prefix_match(tokens, metadata.key_entities)
        tags_hits = self._exact_match(tokens, metadata.tags)

        # Compute per-field scores (full score if any hit, proportional otherwise)
        t_score = self._tools_score if tools_hits else 0.0
        e_score = self._entities_score if entities_hits else 0.0
        g_score = self._tags_score if tags_hits else 0.0

        total = min(10.0, t_score + e_score + g_score)

        return MatchResult(
            score=total,
            tools_score=t_score,
            entities_score=e_score,
            tags_score=g_score,
            matched_tools=tools_hits,
            matched_entities=entities_hits,
            matched_tags=tags_hits,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_score(self) -> float:
        """Theoretical maximum score (sum of all field weights)."""
        return self._tools_score + self._entities_score + self._tags_score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(query: str) -> list[str]:
        """Split query into lowercased tokens.

        Splits on whitespace and common punctuation so that queries like
        ``"git, react"`` produce ``["git", "react"]``.
        """
        # Replace common separators with spaces, then split
        normalized = query.lower()
        for ch in (",", ";", ":", "|", "/", "\\"):
            normalized = normalized.replace(ch, " ")
        return [tok for tok in normalized.split() if tok]

    @staticmethod
    def _prefix_match(tokens: list[str], candidates: list[str]) -> list[str]:
        """Return candidates where any token is a prefix (case-insensitive).

        A token ``t`` matches candidate ``c`` when ``c_lower.startswith(t)``.
        """
        if not candidates:
            return []
        matched: list[str] = []
        for candidate in candidates:
            c_lower = candidate.lower()
            for token in tokens:
                if c_lower.startswith(token):
                    matched.append(candidate)
                    break
        return matched

    @staticmethod
    def _exact_match(tokens: list[str], candidates: list[str]) -> list[str]:
        """Return candidates that exactly equal any token (case-insensitive)."""
        if not candidates:
            return []
        token_set = set(tokens)
        return [c for c in candidates if c.lower() in token_set]

    def _empty_result(self) -> MatchResult:
        """Return a zero-score MatchResult."""
        return MatchResult(
            score=0.0,
            tools_score=0.0,
            entities_score=0.0,
            tags_score=0.0,
        )
