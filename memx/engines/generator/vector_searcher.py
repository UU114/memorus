"""VectorSearcher — L4 semantic search adapter wrapping mem0 VectorStore.

Provides meaning-based retrieval with normalized similarity scores [0, 1].
Gracefully degrades to empty results when embeddings are unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class VectorMatch:
    """A single semantic search result with normalized score."""

    bullet_id: str
    score: float  # [0.0, 1.0] normalized similarity
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a numeric value to [lo, hi] range."""
    return max(lo, min(hi, value))


def _normalize_score(raw_score: float) -> float:
    """Normalize a raw similarity/distance score to [0.0, 1.0].

    mem0 backends may return cosine similarity (already in [-1, 1]) or
    distance values.  This function handles common cases:

    - Cosine similarity in [-1, 1]: mapped to [0, 1] via (s + 1) / 2
    - Scores already in [0, 1]: passed through
    - Scores > 1.0 (e.g. raw dot-product): clamped to 1.0
    - Negative distances: clamped to 0.0
    """
    if raw_score < 0.0:
        # Likely cosine similarity in [-1, 1] range
        return _clamp((raw_score + 1.0) / 2.0)
    if raw_score > 1.0:
        # Likely a raw distance or dot-product; clamp
        return 1.0
    return _clamp(raw_score)


class VectorSearcher:
    """L4 semantic search adapter for the Generator engine.

    Wraps a search callback (injected by the Memory layer) that calls
    mem0's VectorStore.search().  When the callback is ``None`` or
    raises an exception, the searcher silently returns empty results
    to support graceful degradation.
    """

    def __init__(self, search_fn: Callable[..., Any] | None = None) -> None:
        self._search_fn = search_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorMatch]:
        """Run a semantic search and return normalized results.

        Returns an empty list when the backend is unavailable or an
        error occurs (no exception is raised to the caller).
        """
        if not self.available:
            return []

        try:
            assert self._search_fn is not None  # guaranteed by `available`
            raw_results = self._search_fn(query=query, limit=limit, filters=filters)
            return self._parse_results(raw_results)
        except Exception as exc:
            logger.warning("VectorSearcher.search() failed: %s", exc)
            return []

    @property
    def available(self) -> bool:
        """Whether the vector search backend is available."""
        return self._search_fn is not None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_results(self, raw: Any) -> list[VectorMatch]:
        """Parse raw mem0 search output into a list of VectorMatch.

        Supports two common mem0 return formats:
        1. A dict with a ``"results"`` key containing a list of dicts.
        2. A plain list of result dicts.

        Each result dict is expected to have at least an ``"id"`` field.
        Score is read from ``"score"``, ``"similarity"``, or ``"distance"``.
        Content is read from ``"memory"`` or ``"content"``.
        """
        results_list: list[dict[str, Any]]

        if isinstance(raw, dict):
            inner = raw.get("results", raw.get("memories", []))
            results_list = inner if isinstance(inner, list) else []
        elif isinstance(raw, list):
            results_list = raw
        else:
            logger.warning(
                "VectorSearcher: unexpected result type %s, returning empty",
                type(raw).__name__,
            )
            return []

        matches: list[VectorMatch] = []
        for item in results_list:
            if not isinstance(item, dict):
                continue
            match = self._parse_single(item)
            if match is not None:
                matches.append(match)

        return matches

    @staticmethod
    def _parse_single(item: dict[str, Any]) -> VectorMatch | None:
        """Parse a single result dict into a VectorMatch, or None on failure."""
        bullet_id = str(item.get("id", item.get("bullet_id", "")))
        if not bullet_id:
            return None

        # Extract raw score from various possible keys
        raw_score: float = 0.0
        for key in ("score", "similarity", "distance"):
            val = item.get(key)
            if val is not None:
                try:
                    raw_score = float(val)
                except (TypeError, ValueError):
                    continue
                break

        score = _normalize_score(raw_score)

        # Extract content
        content = str(item.get("memory", item.get("content", "")))

        # Extract metadata (exclude known top-level keys)
        metadata: dict[str, Any] = {}
        raw_meta = item.get("metadata")
        if isinstance(raw_meta, dict):
            metadata = raw_meta
        else:
            # Collect extra keys as metadata
            known_keys = {"id", "bullet_id", "score", "similarity", "distance",
                          "memory", "content", "metadata"}
            metadata = {k: v for k, v in item.items() if k not in known_keys}

        return VectorMatch(
            bullet_id=bullet_id,
            score=score,
            content=content,
            metadata=metadata,
        )
