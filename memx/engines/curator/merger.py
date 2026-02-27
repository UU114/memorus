"""MergeStrategy implementations for the Curator engine.

Provides two strategies for combining a CandidateBullet with an ExistingBullet:
- KeepBestStrategy: retains whichever has the higher instructivity_score
- MergeContentStrategy: concatenates content and unions metadata fields

Strategy selection is driven by ``CuratorConfig.merge_strategy``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memx.engines.curator.engine import ExistingBullet
from memx.types import CandidateBullet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MergeResult:
    """Result of merging a candidate with an existing bullet."""

    merged_content: str
    merged_metadata: dict[str, Any] = field(default_factory=dict)
    source_id: str = ""  # existing bullet ID to update
    strategy_used: str = ""  # "keep_best" | "merge_content"


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class MergeStrategy(ABC):
    """Abstract base for merge strategies."""

    @abstractmethod
    def merge(
        self,
        candidate: CandidateBullet,
        existing: ExistingBullet,
    ) -> MergeResult:
        """Merge *candidate* into *existing* and return the result."""
        ...


# ---------------------------------------------------------------------------
# KeepBestStrategy
# ---------------------------------------------------------------------------


def _union_list(a: list[str], b: list[str]) -> list[str]:
    """Return the deduplicated union of two string lists, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in [*a, *b]:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


class KeepBestStrategy(MergeStrategy):
    """Retain the bullet with the higher instructivity_score.

    Tie-breaking: when scores are equal, keep the one with longer content.
    If content lengths are also equal, keep the existing bullet.
    """

    def merge(
        self,
        candidate: CandidateBullet,
        existing: ExistingBullet,
    ) -> MergeResult:
        candidate_score = candidate.instructivity_score
        existing_score = existing.metadata.get("instructivity_score", 0.0)

        candidate_empty = not candidate.content or not candidate.content.strip()
        existing_empty = not existing.content or not existing.content.strip()

        # Edge case: if one side is empty, keep the non-empty one
        if candidate_empty and not existing_empty:
            keep_candidate = False
        elif existing_empty and not candidate_empty:
            keep_candidate = True
        else:
            # Determine winner: higher score wins; tie-break on content length
            keep_candidate = False
            if candidate_score > existing_score:
                keep_candidate = True
            elif candidate_score == existing_score:
                if len(candidate.content) > len(existing.content):
                    keep_candidate = True

        if keep_candidate:
            winner_content = candidate.content
            winner_score = candidate_score
            winner_recall = existing.metadata.get("recall_count", 0)
            winner_tools = _union_list(
                candidate.related_tools,
                existing.metadata.get("related_tools", []),
            )
            winner_entities = _union_list(
                candidate.key_entities,
                existing.metadata.get("key_entities", []),
            )
            winner_tags = _union_list(
                candidate.tags,
                existing.metadata.get("tags", []),
            )
        else:
            winner_content = existing.content
            winner_score = existing_score
            winner_recall = existing.metadata.get("recall_count", 0)
            winner_tools = _union_list(
                existing.metadata.get("related_tools", []),
                candidate.related_tools,
            )
            winner_entities = _union_list(
                existing.metadata.get("key_entities", []),
                candidate.key_entities,
            )
            winner_tags = _union_list(
                existing.metadata.get("tags", []),
                candidate.tags,
            )

        return MergeResult(
            merged_content=winner_content,
            merged_metadata={
                "instructivity_score": winner_score,
                "recall_count": winner_recall,
                "related_tools": winner_tools,
                "key_entities": winner_entities,
                "tags": winner_tags,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            source_id=existing.bullet_id,
            strategy_used="keep_best",
        )


# ---------------------------------------------------------------------------
# MergeContentStrategy
# ---------------------------------------------------------------------------


class MergeContentStrategy(MergeStrategy):
    """Concatenate the content of both bullets, union metadata fields.

    Content is deduplicated by sentence-level comparison to avoid
    verbatim repetition.  Keeps the higher instructivity_score and
    recall_count from either source.
    """

    def merge(
        self,
        candidate: CandidateBullet,
        existing: ExistingBullet,
    ) -> MergeResult:
        merged_content = self._merge_content(candidate.content, existing.content)

        candidate_score = candidate.instructivity_score
        existing_score = existing.metadata.get("instructivity_score", 0.0)

        existing_recall: int = existing.metadata.get("recall_count", 0)

        merged_tools = _union_list(
            existing.metadata.get("related_tools", []),
            candidate.related_tools,
        )
        merged_entities = _union_list(
            existing.metadata.get("key_entities", []),
            candidate.key_entities,
        )
        merged_tags = _union_list(
            existing.metadata.get("tags", []),
            candidate.tags,
        )

        return MergeResult(
            merged_content=merged_content,
            merged_metadata={
                "instructivity_score": max(candidate_score, existing_score),
                "recall_count": existing_recall,
                "related_tools": merged_tools,
                "key_entities": merged_entities,
                "tags": merged_tags,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            source_id=existing.bullet_id,
            strategy_used="merge_content",
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _merge_content(candidate_text: str, existing_text: str) -> str:
        """Concatenate two content strings with sentence-level deduplication.

        If one side is empty, return the other as-is.
        """
        # Handle empty content on either side
        if not candidate_text or not candidate_text.strip():
            return existing_text.strip()
        if not existing_text or not existing_text.strip():
            return candidate_text.strip()

        # If content is identical, no need to concatenate
        if candidate_text.strip() == existing_text.strip():
            return existing_text.strip()

        # Split into sentences (simple heuristic: split on ". " or newlines)
        existing_sentences = _split_sentences(existing_text)
        candidate_sentences = _split_sentences(candidate_text)

        # Collect unique sentences preserving order (existing first)
        seen: set[str] = set()
        merged: list[str] = []
        for s in [*existing_sentences, *candidate_sentences]:
            normalised = s.strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                merged.append(s.strip())

        return " ".join(merged)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like segments."""
    # Replace newlines with period-space for uniform splitting
    normalised = text.replace("\n", ". ")
    parts = normalised.split(". ")
    # Strip trailing periods from each part for consistent dedup
    return [p.rstrip(".").strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGIES: dict[str, type[MergeStrategy]] = {
    "keep_best": KeepBestStrategy,
    "merge_content": MergeContentStrategy,
}


def get_merge_strategy(name: str) -> MergeStrategy:
    """Return a MergeStrategy instance for *name*.

    Falls back to ``KeepBestStrategy`` with a WARNING log if *name* is
    unknown, ensuring the system never crashes on misconfiguration.
    """
    cls = _STRATEGIES.get(name)
    if cls is None:
        logger.warning(
            "Unknown merge strategy %r; falling back to 'keep_best'",
            name,
        )
        cls = KeepBestStrategy
    return cls()
