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

from memorus.core.config import VerificationConfig
from memorus.core.engines.decay.engine import DecayEngine
from memorus.core.engines.generator.engine import GeneratorEngine
from memorus.core.engines.generator.score_merger import ScoredBullet
from memorus.core.engines.verifier.engine import (
    VerificationEngine,
    VerificationOutcome,
)
from memorus.core.types import BulletMetadata
from memorus.core.utils.token_counter import TokenBudgetTrimmer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a retrieval pipeline search.

    Attributes:
        results:             Trimmed list of scored bullets.
        mode:                Operating mode: "full" | "degraded" | "fallback".
        total_candidates:    Total number of candidates before trimming.
        dropped_stale_count: STORY-R103 — Number of bullets filtered out when
                             the active verification policy is ``"drop"``.
                             Always 0 for ``"flag"`` / ``"demote"``. Exposed
                             for telemetry / debug via ``ace_search``.
    """

    results: list[ScoredBullet]
    mode: str  # "full" | "degraded" | "fallback"
    total_candidates: int = 0
    dropped_stale_count: int = 0


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
        verification_engine: Optional[VerificationEngine] = None,
        bullet_loader: Optional[Callable[[list[str]], list[BulletMetadata]]] = None,
        verification_config: Optional[VerificationConfig] = None,
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

        # STORY-R102 — Verifier stage. When ``verification_engine`` is None the
        # stage is skipped entirely (zero-cost). ``bullet_loader`` fetches full
        # ``BulletMetadata`` (with anchors) for the trimmed bullet_ids so we
        # can re-run the verifier without keeping anchor payloads on every
        # ScoredBullet.
        self._verification_engine: Optional[VerificationEngine] = verification_engine
        self._bullet_loader = bullet_loader
        # Back-write channel for ``verified_at`` / ``verified_status`` so the
        # next call hits the TTL cache. Shares ``update_fn`` with the
        # reinforcer by design (both are metadata-only writes).
        self._update_fn = update_fn
        # STORY-R103 — verification policy drives the post-verifier dispatch
        # (flag / demote / drop). Defaulting to a fresh ``VerificationConfig``
        # preserves R102 behaviour (``policy="flag"``) for callers that don't
        # thread the config through.
        self._verification_config: VerificationConfig = (
            verification_config if verification_config is not None
            else VerificationConfig()
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

        # Step 3: STORY-R102 — Verifier. Runs only when a VerificationEngine
        # was injected. Mutates ``trimmed`` in place by populating
        # ``verified_status`` + ``trust_score`` and persists ``verified_at``
        # back to the store so the next call hits the TTL cache.
        logger.debug("RetrievalPipeline step 3: verifier")
        self._run_verifier(trimmed)

        # Step 3b: STORY-R103 — apply policy dispatch (flag / demote / drop).
        # For ``"flag"`` this is a no-op, preserving R102 behaviour byte-for-
        # byte. When the verifier never ran (disabled path) every row has
        # ``verified_status is None`` and ``_apply_stale_policy`` is a no-op
        # regardless of policy value.
        trimmed, dropped_stale = self._apply_stale_policy(trimmed)

        # Step 4: Async reinforcement (fire-and-forget)
        hit_ids = [b.bullet_id for b in trimmed]
        logger.debug("RetrievalPipeline step 4: reinforce %d bullet(s)", len(hit_ids))
        self._run_reinforcer(hit_ids)

        # Determine final mode
        mode = generator_mode  # "full" or "degraded"

        return SearchResult(
            results=trimmed,
            mode=mode,
            total_candidates=total_candidates,
            dropped_stale_count=dropped_stale,
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

    def _run_verifier(self, trimmed: list[ScoredBullet]) -> None:
        """STORY-R102 — annotate trimmed results with verification outcomes.

        Runs the injected :class:`VerificationEngine` against the full
        ``BulletMetadata`` resolved via ``bullet_loader``. Populates each
        ``ScoredBullet.verified_status`` / ``trust_score`` in place and
        best-effort writes ``verified_at`` + ``verified_status`` back to the
        vector store via ``update_fn`` so the next call hits the TTL cache.

        Silently degrades (leaves fields as ``None``) when:
        * no ``VerificationEngine`` was injected (verifier disabled);
        * no ``bullet_loader`` can reconstitute full metadata;
        * the loader or engine raises — search must never fail on verifier.

        Non-critical stage: errors are logged at WARNING and swallowed.
        """
        if self._verification_engine is None or not trimmed:
            return
        if self._bullet_loader is None:
            # Without a loader we have no anchors to verify; skip rather than
            # fabricate a NOT_APPLICABLE status (would lie about the bullet).
            logger.debug(
                "Verifier skipped: no bullet_loader wired"
            )
            return

        bullet_ids = [b.bullet_id for b in trimmed]
        try:
            # Contract: loader returns a list aligned 1-1 by position with
            # ``bullet_ids``. A ``None`` entry (or missing trailing slot)
            # means "no metadata available; skip verifier for this row".
            bullets = self._bullet_loader(bullet_ids)
        except Exception as exc:
            logger.warning("Verifier bullet_loader failed: %s", exc)
            return

        if len(bullets) != len(trimmed):
            logger.debug(
                "Verifier loader returned %d for %d ids — falling back to skip",
                len(bullets), len(trimmed),
            )
            return

        # Split into (index_in_trimmed, BulletMetadata) for rows the loader
        # could resolve; skip ``None`` placeholders.
        work: list[tuple[int, BulletMetadata]] = []
        for idx, mb in enumerate(bullets):
            if mb is None:
                continue
            work.append((idx, mb))

        if not work:
            logger.debug("Verifier skipped: loader produced no usable metadata")
            return

        try:
            outcomes = self._verification_engine.verify_many(
                [mb for _, mb in work]
            )
        except Exception as exc:
            logger.warning("VerificationEngine.verify_many failed: %s", exc)
            return

        for (idx, _mb), outcome in zip(work, outcomes):
            sb = trimmed[idx]
            status_value = outcome.verified_status.value
            sb.verified_status = status_value
            sb.trust_score = outcome.trust_score
            # Best-effort write-back; never block search on persistence failure.
            self._persist_verification(sb.bullet_id, outcome)

    def _apply_stale_policy(
        self, bullets: list[ScoredBullet]
    ) -> tuple[list[ScoredBullet], int]:
        """STORY-R103 — dispatch on ``VerificationConfig.policy``.

        Read the policy ONCE per call (captured locally) and branch:

        * ``"flag"`` (default): no-op; preserves STORY-R102 output exactly.
        * ``"demote"``: multiply ``final_score`` by the configured trust
          multiplier for ``stale`` (``stale_trust_score``) and
          ``unverifiable`` (``unverifiable_trust_score``). ``verified`` and
          ``not_applicable`` are untouched. Re-sorts by ``final_score`` so
          the downstream consumer sees the reshuffled order.
        * ``"drop"``: filter out rows whose ``verified_status == "stale"``.
          ``unverifiable`` and ``not_applicable`` are explicitly kept — drop
          is the strictest mode only for definitively-stale rows.

        Returns a tuple of (possibly-mutated list, dropped_stale_count).
        ``dropped_stale_count`` is 0 for ``"flag"`` / ``"demote"`` and only
        populated by the drop branch.

        A row with ``verified_status is None`` (verifier disabled or the
        row-level verifier lookup failed) is always kept and never scaled —
        treat-as-untouched is the only safe default when the signal is
        missing.
        """
        if not bullets:
            return bullets, 0

        policy = self._verification_config.policy

        if policy == "flag":
            # Hot path: zero work, preserves R102 byte-for-byte.
            return bullets, 0

        if policy == "demote":
            stale_mult = self._verification_config.stale_trust_score
            unverifiable_mult = self._verification_config.unverifiable_trust_score
            for b in bullets:
                if b.verified_status == "stale":
                    b.final_score = b.final_score * stale_mult
                elif b.verified_status == "unverifiable":
                    b.final_score = b.final_score * unverifiable_mult
            # Re-sort so the demoted rows sink to the bottom of the final
            # ranking — callers expect the list to be ordered by final_score.
            bullets.sort(key=lambda b: b.final_score, reverse=True)
            return bullets, 0

        if policy == "drop":
            kept: list[ScoredBullet] = []
            dropped = 0
            for b in bullets:
                if b.verified_status == "stale":
                    dropped += 1
                    continue
                kept.append(b)
            if dropped:
                logger.debug(
                    "stale-policy=drop filtered %d bullet(s)", dropped
                )
            return kept, dropped

        # Unknown policy — defensive fallthrough. Pydantic's Literal already
        # guards this at config-load time; we don't re-raise at retrieval.
        logger.warning(
            "Unknown verification.policy=%r; falling back to flag", policy
        )
        return bullets, 0

    def _persist_verification(
        self, bullet_id: str, outcome: VerificationOutcome
    ) -> None:
        """Write ``verified_at`` + ``verified_status`` back to the store.

        MVP path (per STORY-R102 tech notes): emits a minimal metadata patch
        through ``update_fn``. The concrete ``update_fn`` wired by
        :class:`Memory` merges the patch into the existing mem0 payload.
        Silent on failure — search must not degrade because a write failed.
        """
        if self._update_fn is None or not bullet_id:
            return
        try:
            patch: dict[str, object] = {
                "memorus_verified_at": outcome.verified_at.isoformat(),
                "memorus_verified_status": outcome.verified_status.value,
            }
            if outcome.trust_score is not None:
                patch["memorus_trust_score"] = outcome.trust_score
            self._update_fn(bullet_id, patch)
        except Exception as exc:
            logger.debug(
                "Verifier write-back failed for %s: %s", bullet_id, exc
            )

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
