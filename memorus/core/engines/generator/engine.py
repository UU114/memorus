"""GeneratorEngine — top-level orchestrator for Memorus hybrid retrieval.

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

from memorus.core.config import GraphExpansionConfig, RetrievalConfig
from memorus.core.engines.generator import graph as graph_mod
from memorus.core.engines.generator.exact_matcher import ExactMatcher
from memorus.core.engines.generator.fuzzy_matcher import FuzzyMatcher
from memorus.core.engines.generator.graph import EdgeProvider, GraphNeighbor
from memorus.core.engines.generator.metadata_matcher import MetadataInfo, MetadataMatcher
from memorus.core.engines.generator.score_merger import BulletInfo, ScoredBullet, ScoreMerger
from memorus.core.engines.generator.vector_searcher import VectorSearcher

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
        scope:        Hierarchical scope for this bullet (default "global").
        extra:        Additional pass-through metadata for the ScoredBullet.
    """

    bullet_id: str
    content: str = ""
    metadata: MetadataInfo = field(default_factory=MetadataInfo)
    created_at: datetime | None = None
    decay_weight: float = 1.0
    scope: str = "global"
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
        inbox: Any | None = None,
        provisional_score_factor: float = 0.5,
        topic_store: Any | None = None,
        topics_config: Any | None = None,
    ) -> None:
        self._config = config or RetrievalConfig()
        self._exact_matcher = ExactMatcher()
        self._fuzzy_matcher = FuzzyMatcher()
        self._metadata_matcher = MetadataMatcher()
        self._vector_searcher = vector_searcher or VectorSearcher()
        self._score_merger = ScoreMerger(self._config)
        # Track whether we have already logged a degradation warning
        self._degraded_logged = False
        # STORY-R095 — optional inbox used for provisional search hits
        self._inbox = inbox
        self._provisional_factor = max(0.0, min(1.0, float(provisional_score_factor)))
        # STORY-R097 — optional TopicPage store for Stage A topic match
        self._topic_store = topic_store
        self._topics_config = topics_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        bullets: list[BulletForSearch],
        limit: int = 20,
        filters: dict[str, Any] | None = None,
        scope: str | None = None,
        now: datetime | None = None,
        include_pending: bool = True,
    ) -> list[ScoredBullet]:
        """Run the full hybrid retrieval pipeline.

        Orchestrates L1->L2->L3->(L4)->ScoreMerger and returns
        results sorted by final_score descending.

        Args:
            query:   Free-text search query.
            bullets: List of bullets to search against.
            limit:   Maximum number of results to return.
            filters: Optional filters passed to VectorSearcher.
            scope:   Target scope for filtering. When set, only bullets
                     matching this scope or "global" are included.

        Returns:
            List of ScoredBullet sorted by final_score descending,
            truncated to *limit*.
        """
        if not query or not bullets:
            return []

        # -- STORY-R097 Stage A: TopicPage match (opt-in) --------------------
        topic_hit = self._try_topic_match(query, bullets, limit)
        if topic_hit is not None:
            return topic_hit

        # Scope filtering: keep only bullets matching the target scope or "global"
        if scope:
            bullets = [
                b for b in bullets
                if b.scope == scope or b.scope == "global"
            ]
            if not bullets:
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
                scope=bullet.scope,
                metadata=dict(bullet.extra),
            )

        # -- Merge and rank ----------------------------------------------------
        scored = self._score_merger.merge(
            keyword_results, semantic_results, bullet_infos, target_scope=scope, now=now,
        )

        # -- STORY-R095 provisional hits from inbox ----------------------------
        if include_pending and self._inbox is not None:
            provisional = self._search_inbox(query, limit)
            if provisional:
                scored = scored + provisional
                scored.sort(key=lambda sb: sb.final_score, reverse=True)

        return scored[:limit]

    def search_with_graph(
        self,
        query: str,
        bullets: list[BulletForSearch],
        edges: EdgeProvider,
        graph_config: GraphExpansionConfig | None = None,
        limit: int = 20,
        filters: dict[str, Any] | None = None,
        scope: str | None = None,
        now: datetime | None = None,
    ) -> list[ScoredBullet]:
        """Run hybrid retrieval + graph-aware 1-hop expansion (STORY-R096).

        When ``graph_config.enabled`` is ``False`` this method is
        byte-identical to :meth:`search` — no edge reads, no new neighbors,
        no re-sort.

        When enabled, the base top-K is expanded 1 hop over *edges* using
        weighted Adamic-Adar. The final score of each bullet becomes
        ``base_score + graph_score_weight * graph_score``. New neighbors
        not already in the base top-K are pulled in via *bullets*
        (matched by ``bullet_id``) and inserted with ``base_score = 0``.

        Callers are responsible for recording co-recall edges after the
        fact (via :func:`memorus.core.storage.edges.record_co_recall`).
        """
        cfg = graph_config or GraphExpansionConfig()
        base = self.search(
            query, bullets, limit=limit, filters=filters, scope=scope, now=now,
        )

        if not cfg.enabled or not base:
            return base

        # Build an id -> BulletForSearch lookup for pulling in new neighbors.
        by_id: dict[str, BulletForSearch] = {b.bullet_id: b for b in bullets}

        seeds: list[tuple[str, float]] = [(sb.bullet_id, sb.final_score) for sb in base]

        neighbors: list[GraphNeighbor] = graph_mod.expand_neighbors(
            seeds,
            edges,
            cfg.k_expand,
            exclude_ids={sid for sid, _ in seeds},
        )

        if not neighbors:
            return base

        # Boost scores of bullets already in base.
        boosted = graph_mod.apply_graph_scores(base, neighbors, cfg.graph_score_weight)

        # Append genuinely-new neighbors (those not in base).
        existing_ids = {sb.bullet_id for sb in boosted}
        for n in neighbors:
            if n.bullet_id in existing_ids:
                continue
            src = by_id.get(n.bullet_id)
            if src is None:
                continue
            stub = graph_mod.scored_bullet_for_new_neighbor(
                bullet_id=n.bullet_id,
                content=src.content,
                graph_score=n.graph_score,
                graph_score_weight=cfg.graph_score_weight,
                decay_weight=src.decay_weight,
                metadata=dict(src.extra),
            )
            boosted.append(stub)

        boosted.sort(key=lambda sb: sb.final_score, reverse=True)
        return boosted[:limit]

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

    # ------------------------------------------------------------------
    # Topic-match Stage A (STORY-R097)
    # ------------------------------------------------------------------

    def _try_topic_match(
        self,
        query: str,
        bullets: list[BulletForSearch],
        limit: int,
    ) -> list[ScoredBullet] | None:
        """Return topic-matched results, or ``None`` if Stage A declines.

        Stage A is a short-circuit: if any persisted :class:`TopicPage`
        scores above ``topics.topic_match_threshold`` against *query* and
        its backing bullets score well, we skip the bullet-level path and
        return a composite result. If no config/store is attached, or the
        threshold is not met, this returns ``None`` and Stage B (the
        existing bullet path) takes over — guaranteeing byte-identical
        behaviour when ``topics.enabled=False``.
        """
        cfg = self._topics_config
        store = self._topic_store
        if cfg is None or store is None:
            return None
        if not getattr(cfg, "enabled", False):
            return None

        try:
            pages = store.list_all()
        except Exception as e:  # defensive — never let topic lookup break search
            logger.warning("GeneratorEngine: topic store list failed: %s", e)
            return None
        if not pages:
            return None

        # Score pages by exact+fuzzy match on their title + summary.
        try:
            combined = [
                f"{p.title}\n{p.summary}" for p in pages
            ]
            exact = self._exact_matcher.match_batch(query, combined)
            fuzzy = self._fuzzy_matcher.match_batch(query, combined)
        except Exception as e:
            logger.warning("GeneratorEngine: topic scoring failed: %s", e)
            return None

        # Normalise — same rough scale as the bullet pipeline (/70 as a
        # conservative cap on exact+fuzzy sum).
        ranked: list[tuple[Any, float]] = []
        for p, e_r, f_r in zip(pages, exact, fuzzy):
            raw = float(e_r.score) + float(f_r.score)
            normalised = min(1.0, raw / 70.0)
            ranked.append((p, normalised))
        ranked.sort(key=lambda t: t[1], reverse=True)

        top_page, top_score = ranked[0]
        threshold = float(getattr(cfg, "topic_match_threshold", 0.65))
        if top_score < threshold:
            return None

        # Compose backing bullets — find the subset of *bullets* that
        # belong to the matched page and score them via the existing path.
        id_set = set(top_page.bullet_ids)
        backing = [b for b in bullets if b.bullet_id in id_set]
        if not backing:
            return None

        # Score backing bullets through the ordinary hybrid pipeline so the
        # top backing score can be combined with the page score.
        try:
            scored_backing = self._score_bullets(query, backing, limit=3)
        except Exception as e:
            logger.warning("GeneratorEngine: backing scoring failed: %s", e)
            return None

        backing_max = scored_backing[0].final_score if scored_backing else 0.0

        page_weight = float(getattr(cfg, "page_score_weight", 0.7))
        backing_weight = float(getattr(cfg, "backing_score_weight", 0.3))

        final = page_weight * top_score + backing_weight * backing_max

        # Build a synthetic ScoredBullet representing the TopicPage itself.
        md: dict[str, Any] = {
            "topic_page": True,
            "topic_id": top_page.id,
            "topic_slug": top_page.slug,
            "title": top_page.title,
            "bullet_ids": list(top_page.bullet_ids),
            "backing_count": len(scored_backing),
        }
        head = ScoredBullet(
            bullet_id=f"topic:{top_page.id}",
            content=top_page.summary,
            final_score=final,
            keyword_score=top_score,
            semantic_score=0.0,
            decay_weight=1.0,
            recency_boost=1.0,
            metadata=md,
        )

        results: list[ScoredBullet] = [head]
        # Include up to 3 backing bullets beneath the topic head.
        for sb in scored_backing[:3]:
            results.append(sb)
        return results[:limit]

    def _score_bullets(
        self,
        query: str,
        bullets: list[BulletForSearch],
        limit: int,
    ) -> list[ScoredBullet]:
        """Run the hybrid pipeline over an explicit bullet list (no topic match).

        Used internally by :meth:`_try_topic_match` to score the backing
        bullets of a matched TopicPage without recursing back into Stage A.
        """
        if not bullets:
            return []

        contents = [b.content for b in bullets]
        l1 = self._run_l1(query, contents)
        l2 = self._run_l2(query, contents)
        l3 = self._run_l3(query, bullets)

        keyword_results = {
            b.bullet_id: l1[i] + l2[i] + l3[i] for i, b in enumerate(bullets)
        }
        bullet_infos: dict[str, BulletInfo] = {
            b.bullet_id: BulletInfo(
                bullet_id=b.bullet_id,
                content=b.content,
                created_at=b.created_at,
                decay_weight=b.decay_weight,
                scope=b.scope,
                metadata=dict(b.extra),
            )
            for b in bullets
        }
        scored = self._score_merger.merge(
            keyword_results, None, bullet_infos,
        )
        return scored[:limit]

    # ------------------------------------------------------------------
    # Provisional search (STORY-R095)
    # ------------------------------------------------------------------

    def _search_inbox(self, query: str, limit: int) -> list[ScoredBullet]:
        """Return keyword-hit ScoredBullets drawn from pending inbox entries.

        Each hit is scored at ``provisional_score_factor`` (default 0.5) of the
        raw L1 ExactMatcher score and flagged ``provisional=True`` inside the
        ScoredBullet metadata so callers can render them differently.
        """
        try:
            pending = self._inbox.list_pending()
        except Exception as e:  # defensive — an inbox failure must not break search
            logger.warning("GeneratorEngine: inbox read failed: %s", e)
            return []
        if not pending:
            return []

        contents = [
            ((e.content or "") + "\n" + (e.assistant_message or "")).strip()
            for e in pending
        ]
        try:
            matches = self._exact_matcher.match_batch(query, contents)
        except Exception as e:
            logger.warning("GeneratorEngine: provisional L1 failed: %s", e)
            return []

        results: list[ScoredBullet] = []
        for entry, m in zip(pending, matches):
            if m.score <= 0.0:
                continue
            base_score = m.score / 35.0  # normalize roughly against keyword max
            final = base_score * self._provisional_factor
            md: dict[str, Any] = {
                "provisional": True,
                "inbox_id": entry.id,
                "conversation_id": entry.conversation_id,
                "turn_offset": entry.turn_offset,
                "pending_reason": "inbox",
            }
            results.append(
                ScoredBullet(
                    bullet_id=f"inbox:{entry.id}",
                    content=(entry.content or "")[:500],
                    final_score=final,
                    keyword_score=m.score,
                    semantic_score=0.0,
                    decay_weight=1.0,
                    recency_boost=1.0,
                    metadata=md,
                )
            )
        results.sort(key=lambda sb: sb.final_score, reverse=True)
        return results[:limit]
