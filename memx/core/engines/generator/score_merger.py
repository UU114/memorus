"""ScoreMerger — unified scoring formula for the Generator search engine.

Blends keyword scores (L1 ExactMatcher + L2 FuzzyMatcher + L3 MetadataMatcher)
with semantic scores (L4 VectorSearcher), then applies DecayWeight and
RecencyBoost to produce final ranked results.

Formula:
    NormKeyword  = KeywordScore / MAX_KEYWORD_SCORE  (0-1)
    NormSemantic = SemanticScore                      (0-1)
    BlendedScore = NormKeyword × kw_weight + NormSemantic × sem_weight
    FinalScore   = BlendedScore × DecayWeight × RecencyBoost × ScopeBoost

Degraded mode (no semantic scores): keyword weight is automatically set to 1.0.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from memx.core.config import RetrievalConfig

logger = logging.getLogger(__name__)

# Maximum raw keyword score: ExactMatcher(15) + FuzzyMatcher(10) + MetadataMatcher(10)
MAX_KEYWORD_SCORE: float = 35.0


@dataclass
class BulletInfo:
    """Metadata about a bullet needed for decay and recency computation.

    Attributes:
        bullet_id:    Unique identifier for the bullet.
        content:      Text content of the bullet.
        created_at:   When the bullet was created (UTC).
        decay_weight: Pre-computed decay weight from DecayEngine [0.0, 1.0].
        scope:        Hierarchical scope of the bullet (default "global").
        metadata:     Additional metadata forwarded to the scored result.
    """

    bullet_id: str
    content: str = ""
    created_at: datetime | None = None
    decay_weight: float = 1.0
    scope: str = "global"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredBullet:
    """Final scored result combining all scoring layers.

    Attributes:
        bullet_id:      Unique identifier for the bullet.
        content:        Text content of the bullet.
        final_score:    Composite score after all multipliers.
        keyword_score:  Raw combined L1+L2+L3 keyword score.
        semantic_score: L4 semantic similarity score [0.0, 1.0].
        decay_weight:   Temporal decay factor [0.0, 1.0].
        recency_boost:  Time-based boost multiplier (>= 1.0).
        metadata:       Pass-through metadata from BulletInfo.
    """

    bullet_id: str
    content: str
    final_score: float
    keyword_score: float
    semantic_score: float
    decay_weight: float
    recency_boost: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ScoreMerger:
    """Merges keyword and semantic scores into a final ranked list.

    Supports full mode (keyword + semantic) and degraded mode
    (keyword only, when semantic scores are unavailable).

    Usage::

        merger = ScoreMerger()
        results = merger.merge(
            keyword_results={"b1": 25.0, "b2": 10.0},
            semantic_results={"b1": 0.8},
            bullet_infos={"b1": BulletInfo(...), "b2": BulletInfo(...)},
        )
    """

    def __init__(self, config: RetrievalConfig | None = None) -> None:
        self._config = config or RetrievalConfig()
        # Pre-compute normalized weights
        self._kw_weight, self._sem_weight = self._normalize_weights(
            self._config.keyword_weight,
            self._config.semantic_weight,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def merge(
        self,
        keyword_results: dict[str, float],
        semantic_results: dict[str, float] | None,
        bullet_infos: dict[str, BulletInfo],
        target_scope: str | None = None,
        now: datetime | None = None,
    ) -> list[ScoredBullet]:
        """Merge keyword and semantic scores into a final ranked list.

        Args:
            keyword_results:  Mapping of bullet_id to raw keyword score (L1+L2+L3).
            semantic_results: Mapping of bullet_id to semantic score [0, 1], or
                              None / empty dict for degraded mode.
            bullet_infos:     Mapping of bullet_id to BulletInfo for each bullet
                              that appears in keyword_results or semantic_results.
            target_scope:     When set, bullets whose scope matches the target
                              receive a configurable scope boost multiplier.

        Returns:
            List of ScoredBullet sorted by final_score in descending order.
        """
        # Determine if we are in degraded mode (no semantic scores)
        degraded = not semantic_results

        # Collect all bullet IDs from both result sets
        all_ids: set[str] = set(keyword_results.keys())
        if not degraded:
            assert semantic_results is not None  # type narrowing
            all_ids |= set(semantic_results.keys())

        # Determine weights for this merge call
        if degraded:
            kw_w, sem_w = 1.0, 0.0
        else:
            kw_w, sem_w = self._kw_weight, self._sem_weight

        if now is None:
            now = datetime.now(timezone.utc)
        scored: list[ScoredBullet] = []

        for bid in all_ids:
            info = bullet_infos.get(bid)
            if info is None:
                logger.warning(
                    "ScoreMerger: bullet_id %r not found in bullet_infos, skipping",
                    bid,
                )
                continue

            raw_kw = keyword_results.get(bid, 0.0)
            raw_sem = (
                semantic_results.get(bid, 0.0)
                if semantic_results is not None and not degraded
                else 0.0
            )

            # Normalize keyword score to [0, 1]
            norm_kw = min(raw_kw / MAX_KEYWORD_SCORE, 1.0) if MAX_KEYWORD_SCORE > 0 else 0.0

            # Semantic score is already [0, 1]
            norm_sem = max(0.0, min(1.0, raw_sem))

            # Weighted blend
            blended = norm_kw * kw_w + norm_sem * sem_w

            # Decay weight from BulletInfo
            decay_w = max(0.0, min(1.0, info.decay_weight))

            # Recency boost
            recency = self.compute_recency_boost(info.created_at, now)

            # Scope boost: bullets matching the target scope get a boost
            scope_b = self._compute_scope_boost(info.scope, target_scope)

            # Final composite score
            final = blended * decay_w * recency * scope_b

            scored.append(ScoredBullet(
                bullet_id=bid,
                content=info.content,
                final_score=final,
                keyword_score=raw_kw,
                semantic_score=raw_sem,
                decay_weight=decay_w,
                recency_boost=recency,
                metadata=dict(info.metadata),
            ))

        # Sort by final_score descending
        scored.sort(key=lambda s: s.final_score, reverse=True)
        return scored

    def compute_recency_boost(
        self,
        created_at: datetime | None,
        now: datetime | None = None,
    ) -> float:
        """Compute recency boost multiplier based on bullet age.

        Bullets created within ``recency_boost_days`` receive a multiplier
        of ``recency_boost_factor`` (default 1.2).  Older bullets receive 1.0.

        Args:
            created_at: When the bullet was created (UTC).
            now:        Reference time for testing; defaults to UTC now.

        Returns:
            Boost multiplier >= 1.0.
        """
        if created_at is None:
            return 1.0

        if now is None:
            now = datetime.now(timezone.utc)

        age_days = (now - created_at).total_seconds() / 86400.0
        if age_days < 0:
            # Future timestamp (clock skew) — treat as recent
            age_days = 0.0

        if age_days <= self._config.recency_boost_days:
            return self._config.recency_boost_factor

        return 1.0

    def _compute_scope_boost(
        self,
        bullet_scope: str,
        target_scope: str | None,
    ) -> float:
        """Compute scope boost multiplier.

        When a target_scope is set and the bullet's scope matches it,
        the bullet receives the configured scope_boost multiplier.
        Bullets with scope "global" or when no target_scope is set
        receive a neutral 1.0 multiplier.

        Args:
            bullet_scope:  The scope of the bullet.
            target_scope:  The requested search scope (may be None).

        Returns:
            Scope boost multiplier >= 1.0.
        """
        if not target_scope:
            return 1.0
        if bullet_scope == target_scope:
            return self._config.scope_boost
        return 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> RetrievalConfig:
        """Current retrieval configuration (read-only)."""
        return self._config

    @property
    def keyword_weight(self) -> float:
        """Normalized keyword weight."""
        return self._kw_weight

    @property
    def semantic_weight(self) -> float:
        """Normalized semantic weight."""
        return self._sem_weight

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_weights(kw: float, sem: float) -> tuple[float, float]:
        """Normalize keyword and semantic weights so they sum to 1.0.

        If both weights are zero, defaults to keyword-only (1.0, 0.0).

        Returns:
            Tuple of (normalized_kw_weight, normalized_sem_weight).
        """
        total = kw + sem
        if total <= 0.0:
            return 1.0, 0.0
        return kw / total, sem / total
