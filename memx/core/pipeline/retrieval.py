"""RetrievalPipeline -- orchestrates the search() processing flow.

Flow: Load bullets -> GeneratorEngine.search() -> TokenBudgetTrimmer.trim()
      -> [async] DecayEngine.reinforce() -> SearchResult

Each stage has independent error handling; failure in any non-critical
stage triggers graceful degradation rather than crashing the pipeline.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from memx.core.engines.decay.engine import DecayEngine
from memx.core.engines.generator.engine import GeneratorEngine
from memx.core.engines.generator.score_merger import ScoredBullet
from memx.core.utils.token_counter import TokenBudgetTrimmer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a retrieval pipeline search.

    Attributes:
        results:          Trimmed list of scored bullets.
        mode:             Operating mode: "full" | "degraded" | "fallback".
        total_candidates: Total number of candidates before trimming.
    """

    results: list[ScoredBullet]
    mode: str  # "full" | "degraded" | "fallback"
    total_candidates: int = 0


class RecallReinforcer:
    """Asynchronously reinforces recall_count for recalled memory bullets.

    Fires a background thread to increment recall_count via DecayEngine.reinforce()
    so that the search response is not blocked by persistence latency.
    """

    def __init__(
        self,
        decay_engine: DecayEngine,
        update_fn: Optional[Callable[[str, dict[str, object]], None]] = None,
    ) -> None:
        self._decay_engine = decay_engine
        self._update_fn = update_fn

    def reinforce_async(self, bullet_ids: list[str]) -> None:
        """Fire-and-forget reinforcement of recalled bullet IDs.

        Spawns a daemon thread so the calling search() is not blocked.
        Any exceptions inside the thread are logged as WARNING and swallowed.
        """
        if not bullet_ids or self._update_fn is None:
            return

        thread = threading.Thread(
            target=self._reinforce_safe,
            args=(bullet_ids,),
            daemon=True,
        )
        thread.start()

    def reinforce_sync(self, bullet_ids: list[str]) -> int:
        """Synchronous reinforcement (for testing). Returns reinforced count."""
        if not bullet_ids or self._update_fn is None:
            return 0
        return self._decay_engine.reinforce(bullet_ids, self._update_fn)

    def _reinforce_safe(self, bullet_ids: list[str]) -> None:
        """Thread-safe reinforcement wrapper. Never raises."""
        try:
            assert self._update_fn is not None
            count = self._decay_engine.reinforce(bullet_ids, self._update_fn)
            logger.debug("Reinforced %d/%d bullets", count, len(bullet_ids))
        except Exception as exc:
            logger.warning("RecallReinforcer error: %s", exc)


class RetrievalPipeline:
    """Pipeline for processing search() operations through Generator + Trimmer + Reinforce.

    Flow: GeneratorEngine.search() -> TokenBudgetTrimmer.trim()
          -> RecallReinforcer.reinforce_async() -> SearchResult

    When Generator fails, the pipeline falls back to mem0 native search.
    When Trimmer fails, untrimmed results are returned.
    When Reinforcer fails, only a WARNING is logged (search results unaffected).
    """

    def __init__(
        self,
        generator: GeneratorEngine,
        trimmer: Optional[TokenBudgetTrimmer] = None,
        decay_engine: Optional[DecayEngine] = None,
        mem0_search_fn: Optional[Callable[..., Any]] = None,
        update_fn: Optional[Callable[[str, dict[str, object]], None]] = None,
    ) -> None:
        self._generator = generator
        self._trimmer = trimmer
        self._mem0_search_fn = mem0_search_fn

        # Build reinforcer only when decay engine is available
        self._reinforcer: Optional[RecallReinforcer] = None
        if decay_engine is not None:
            self._reinforcer = RecallReinforcer(
                decay_engine=decay_engine,
                update_fn=update_fn,
            )

    def search(
        self,
        query: str,
        bullets: Any = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[dict[str, Any]] = None,
        scope: Optional[str] = None,
    ) -> SearchResult:
        """Run the full retrieval pipeline.

        Args:
            query:    Free-text search query.
            bullets:  List of BulletForSearch to search against.
            user_id:  Optional user ID for scoping.
            agent_id: Optional agent ID for scoping.
            limit:    Maximum number of results to return.
            filters:  Optional filters for vector search.
            scope:    Target scope for filtering and boosting.

        Returns:
            SearchResult with trimmed results, operating mode, and candidate count.
        """
        logger.debug(
            "RetrievalPipeline.search query=%r bullets=%d limit=%d scope=%r",
            query[:60] if query else "", len(bullets or []), limit, scope,
        )
        if not query:
            logger.debug("RetrievalPipeline.search -> empty query, returning empty")
            return SearchResult(results=[], mode="full", total_candidates=0)

        # Step 1: Run GeneratorEngine
        logger.debug("RetrievalPipeline step 1: GeneratorEngine.search()")
        try:
            scored = self._generator.search(
                query=query,
                bullets=bullets or [],
                limit=limit * 4,  # Over-fetch for trimming headroom
                filters=filters,
                scope=scope,
            )
            generator_mode = self._generator.mode
            logger.debug(
                "RetrievalPipeline step 1: %d results, mode=%s",
                len(scored), generator_mode,
            )
        except Exception as exc:
            logger.warning(
                "GeneratorEngine failed, falling back to mem0 search: %s", exc
            )
            return self._fallback_search(
                query, user_id=user_id, agent_id=agent_id, limit=limit, filters=filters
            )

        if not scored:
            return SearchResult(results=[], mode=generator_mode, total_candidates=0)

        total_candidates = len(scored)

        # Step 2: TokenBudgetTrimmer
        logger.debug("RetrievalPipeline step 2: trimmer (budget=%s)",
                      self._trimmer.token_budget if self._trimmer else "N/A")
        trimmed = self._run_trimmer(scored)
        logger.debug("RetrievalPipeline step 2: trimmed %d -> %d", len(scored), len(trimmed))

        # Step 3: Async reinforcement (fire-and-forget)
        hit_ids = [b.bullet_id for b in trimmed]
        logger.debug("RetrievalPipeline step 3: reinforce %d bullet(s)", len(hit_ids))
        self._run_reinforcer(hit_ids)

        # Determine final mode
        mode = generator_mode  # "full" or "degraded"

        return SearchResult(
            results=trimmed,
            mode=mode,
            total_candidates=total_candidates,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_trimmer(self, scored: list[ScoredBullet]) -> list[ScoredBullet]:
        """Run TokenBudgetTrimmer with error isolation."""
        if self._trimmer is None:
            return scored

        try:
            return self._trimmer.trim(scored)
        except Exception as exc:
            logger.warning(
                "TokenBudgetTrimmer failed, returning untrimmed results: %s", exc
            )
            return scored

    def _run_reinforcer(self, bullet_ids: list[str]) -> None:
        """Run RecallReinforcer with error isolation. Never raises."""
        if self._reinforcer is None or not bullet_ids:
            return

        try:
            self._reinforcer.reinforce_async(bullet_ids)
        except Exception as exc:
            logger.warning("RecallReinforcer scheduling failed: %s", exc)

    def _fallback_search(
        self,
        query: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> SearchResult:
        """Fallback to mem0 native search when Generator fails."""
        if self._mem0_search_fn is None:
            return SearchResult(results=[], mode="fallback", total_candidates=0)

        try:
            raw = self._mem0_search_fn(
                query,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
                filters=filters,
            )
            # Convert mem0 results to ScoredBullet for uniform interface
            results = self._convert_mem0_results(raw)
            return SearchResult(
                results=results,
                mode="fallback",
                total_candidates=len(results),
            )
        except Exception as exc:
            logger.warning("Fallback mem0 search also failed: %s", exc)
            return SearchResult(results=[], mode="fallback", total_candidates=0)

    @staticmethod
    def _convert_mem0_results(raw: Any) -> list[ScoredBullet]:
        """Convert mem0 search results to ScoredBullet list.

        mem0 returns {"results": [{"id": ..., "memory": ..., "score": ...}, ...]}
        """
        if not isinstance(raw, dict):
            return []

        results_list = raw.get("results", [])
        scored: list[ScoredBullet] = []
        for item in results_list:
            if not isinstance(item, dict):
                continue
            scored.append(
                ScoredBullet(
                    bullet_id=item.get("id", ""),
                    content=item.get("memory", ""),
                    final_score=item.get("score", 0.0),
                    keyword_score=0.0,
                    semantic_score=item.get("score", 0.0),
                    decay_weight=1.0,
                    recency_boost=1.0,
                    metadata=item.get("metadata", {}),
                )
            )
        return scored
