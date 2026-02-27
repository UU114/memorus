"""GeneratorEngine — top-level orchestrator for MemX hybrid retrieval.

Coordinates L1 (ExactMatcher), L2 (FuzzyMatcher), L3 (MetadataMatcher),
L4 (VectorSearcher), and ScoreMerger into a complete search pipeline.

Automatically detects embedding availability and switches between
"full" mode (all four layers) and "degraded" mode (L1-L3 only).
Each matcher is independently error-isolated so a single failure
does not affect the rest of the pipeline.

Usage::

    engine = GeneratorEngine(config=retrieval_config, vector_searcher=vs)
    results = engine.search("git rebase", bullets, limit=20)
    print(engine.mode)  # "full" or "degraded"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from memx.config import RetrievalConfig
from memx.engines.generator.exact_matcher import ExactMatcher
from memx.engines.generator.fuzzy_matcher import FuzzyMatcher
from memx.engines.generator.metadata_matcher import MetadataInfo, MetadataMatcher
from memx.engines.generator.score_merger import BulletInfo, ScoredBullet, ScoreMerger
from memx.engines.generator.vector_searcher import VectorSearcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Input data container
# ---------------------------------------------------------------------------


@dataclass
class BulletForSearch:
    """Input data for a bullet to be searched against.

    Combines text content, structured metadata, and scoring context
    needed by all four matcher layers and the ScoreMerger.

    Attributes:
        bullet_id:    Unique identifier for the bullet.
        content:      Full text content for L1/L2 matching.
        metadata:     Structured metadata for L3 matching.
        created_at:   Creation timestamp for recency boost.
        decay_weight: Pre-computed decay weight from DecayEngine [0.0, 1.0].
        extra:        Additional pass-through metadata for the ScoredBullet.
    """

    bullet_id: str
    content: str = ""
    metadata: MetadataInfo = field(default_factory=MetadataInfo)
    created_at: datetime | None = None
    decay_weight: float = 1.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# GeneratorEngine
# ---------------------------------------------------------------------------


class GeneratorEngine:
    """Top-level hybrid retrieval engine orchestrating L1-L4 + ScoreMerger.

    Supports two operating modes:
      - **full**: All four matcher layers (L1-L4) feed into ScoreMerger.
      - **degraded**: L4 (VectorSearcher) is skipped; ScoreMerger uses
        keyword-only scoring with semantic_weight automatically set to 0.

    The mode is determined dynamically on each ``search()`` call by
    checking ``VectorSearcher.available``, enabling automatic recovery
    when embeddings become available again.
    """

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        vector_searcher: VectorSearcher | None = None,
    ) -> None:
        self._config = config or RetrievalConfig()
        self._exact_matcher = ExactMatcher()
        self._fuzzy_matcher = FuzzyMatcher()
        self._metadata_matcher = MetadataMatcher()
        self._vector_searcher = vector_searcher or VectorSearcher()
        self._score_merger = ScoreMerger(self._config)
        # Track whether we have already logged a degradation warning
        self._degraded_logged = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        bullets: list[BulletForSearch],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredBullet]:
        """Run the full hybrid retrieval pipeline.

        Orchestrates L1->L2->L3->(L4)->ScoreMerger and returns
        results sorted by final_score descending.

        Args:
            query:   Free-text search query.
            bullets: List of bullets to search against.
            limit:   Maximum number of results to return.
            filters: Optional filters passed to VectorSearcher.

        Returns:
            List of ScoredBullet sorted by final_score descending,
            truncated to *limit*.
        """
        if not query or not bullets:
            return []

        # Determine current operating mode
        is_full = self._vector_searcher.available

        if not is_full and not self._degraded_logged:
            logger.warning(
                "GeneratorEngine: embedding unavailable, operating in degraded mode "
                "(L4 VectorSearcher skipped)"
            )
            self._degraded_logged = True

        # Reset degraded flag when embeddings recover
        if is_full and self._degraded_logged:
            logger.info(
                "GeneratorEngine: embedding recovered, switching back to full mode"
            )
            self._degraded_logged = False

        # Extract content list for L1/L2 batch matching
        contents = [b.content for b in bullets]

        # -- L1: ExactMatcher --------------------------------------------------
        l1_results = self._run_l1(query, contents)

        # -- L2: FuzzyMatcher --------------------------------------------------
        l2_results = self._run_l2(query, contents)

        # -- L3: MetadataMatcher -----------------------------------------------
        l3_results = self._run_l3(query, bullets)

        # -- L4: VectorSearcher (skip if degraded) -----------------------------
        l4_results: list[tuple[str, float]] = []
        if is_full:
            l4_results = self._run_l4(query, limit, filters)

        # -- Aggregate keyword scores (L1 + L2 + L3) --------------------------
        keyword_results: dict[str, float] = {}
        for i, bullet in enumerate(bullets):
            kw_score = l1_results[i] + l2_results[i] + l3_results[i]
            keyword_results[bullet.bullet_id] = kw_score

        # -- Build semantic results dict (L4) ----------------------------------
        semantic_results: dict[str, float] | None = None
        if is_full and l4_results:
            semantic_results = {bid: score for bid, score in l4_results}
        elif not is_full:
            semantic_results = None

        # -- Build BulletInfo for ScoreMerger ----------------------------------
        bullet_infos: dict[str, BulletInfo] = {}
        for bullet in bullets:
            bullet_infos[bullet.bullet_id] = BulletInfo(
                bullet_id=bullet.bullet_id,
                content=bullet.content,
                created_at=bullet.created_at,
                decay_weight=bullet.decay_weight,
                metadata=dict(bullet.extra),
            )

        # -- Merge and rank ----------------------------------------------------
        scored = self._score_merger.merge(keyword_results, semantic_results, bullet_infos)

        return scored[:limit]

    @property
    def mode(self) -> str:
        """Current operating mode: 'full' or 'degraded'."""
        return "full" if self._vector_searcher.available else "degraded"

    # ------------------------------------------------------------------
    # Internal matchers with error isolation
    # ------------------------------------------------------------------

    def _run_l1(self, query: str, contents: list[str]) -> list[float]:
        """Run L1 ExactMatcher with error isolation. Returns per-bullet scores."""
        try:
            results = self._exact_matcher.match_batch(query, contents)
            return [r.score for r in results]
        except Exception as e:
            logger.warning("L1 ExactMatcher failed: %s", e)
            return [0.0] * len(contents)

    def _run_l2(self, query: str, contents: list[str]) -> list[float]:
        """Run L2 FuzzyMatcher with error isolation. Returns per-bullet scores."""
        try:
            results = self._fuzzy_matcher.match_batch(query, contents)
            return [r.score for r in results]
        except Exception as e:
            logger.warning("L2 FuzzyMatcher failed: %s", e)
            return [0.0] * len(contents)

    def _run_l3(
        self,
        query: str,
        bullets: list[BulletForSearch],
    ) -> list[float]:
        """Run L3 MetadataMatcher with error isolation. Returns per-bullet scores."""
        try:
            scores: list[float] = []
            for bullet in bullets:
                result = self._metadata_matcher.match(query, bullet.metadata)
                scores.append(result.score)
            return scores
        except Exception as e:
            logger.warning("L3 MetadataMatcher failed: %s", e)
            return [0.0] * len(bullets)

    def _run_l4(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[tuple[str, float]]:
        """Run L4 VectorSearcher with error isolation. Returns (bullet_id, score) pairs."""
        try:
            matches = self._vector_searcher.search(query, limit=limit, filters=filters)
            return [(m.bullet_id, m.score) for m in matches]
        except Exception as e:
            logger.warning("L4 VectorSearcher failed: %s", e)
            return []
