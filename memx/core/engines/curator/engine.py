"""CuratorEngine -- core deduplication logic for the Curator subsystem.

Compares CandidateBullets against existing memories using cosine similarity
(primary) or text-based token overlap (fallback). Marks each candidate as
Insert, Merge, or Skip based on a configurable similarity threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from memx.core.config import CuratorConfig
from memx.core.types import CandidateBullet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExistingBullet:
    """Representation of an existing memory for comparison."""

    bullet_id: str
    content: str
    embedding: list[float] | None = None
    scope: str = "global"
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class MergeCandidate:
    """A candidate that should be merged with an existing bullet."""

    candidate: CandidateBullet
    existing: ExistingBullet
    similarity: float


@dataclass
class CurateResult:
    """Result of curation: partition of candidates into add / merge / skip."""

    to_add: list[CandidateBullet] = field(default_factory=list)
    to_merge: list[MergeCandidate] = field(default_factory=list)
    to_skip: list[CandidateBullet] = field(default_factory=list)
    conflicts: list[object] = field(default_factory=list)  # list[Conflict]


# ---------------------------------------------------------------------------
# CuratorEngine
# ---------------------------------------------------------------------------


class CuratorEngine:
    """Core deduplication engine that decides Insert vs Merge for candidates.

    Uses cosine similarity on embedding vectors when available, falling back
    to a simple token-overlap ratio for text-only comparisons.
    """

    def __init__(self, config: CuratorConfig | None = None) -> None:
        self._config = config or CuratorConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def curate(
        self,
        candidates: list[CandidateBullet],
        existing: list[ExistingBullet],
    ) -> CurateResult:
        """Compare *candidates* against *existing* memories.

        For each candidate:
        - Empty content -> skip
        - No existing memories -> insert
        - Best similarity >= threshold -> merge with the most similar existing
        - Otherwise -> insert

        Each candidate is judged independently, so multiple candidates may
        match the same existing bullet.
        """
        result = CurateResult()
        logger.debug(
            "CuratorEngine.curate: candidates=%d existing=%d threshold=%.2f",
            len(candidates), len(existing), self._config.similarity_threshold,
        )

        # Conflict detection (non-blocking): run if enabled
        if self._config.conflict_detection:
            try:
                from memx.core.engines.curator.conflict import ConflictDetector

                detector = ConflictDetector(self._config)
                conflict_result = detector.detect(existing)
                if conflict_result.conflicts:
                    logger.warning(
                        "Detected %d potential conflicts",
                        len(conflict_result.conflicts),
                    )
                result.conflicts = list(conflict_result.conflicts)
            except Exception as e:
                logger.warning("Conflict detection failed: %s", e)

        for candidate in candidates:
            # Edge case: empty content is not useful
            if not candidate.content or not candidate.content.strip():
                logger.debug("CuratorEngine: SKIP empty content")
                result.to_skip.append(candidate)
                continue

            # Filter existing bullets to only those in the same scope
            same_scope_existing = [
                ex for ex in existing if ex.scope == candidate.scope
            ]
            logger.debug(
                "CuratorEngine: candidate scope=%r -> %d same-scope existing",
                candidate.scope, len(same_scope_existing),
            )

            # No existing memories in same scope -> insert
            if not same_scope_existing:
                logger.debug("CuratorEngine: INSERT (no same-scope existing)")
                result.to_add.append(candidate)
                continue

            # Find the most similar existing bullet within same scope
            best_sim = -1.0
            best_match: ExistingBullet | None = None

            for ex in same_scope_existing:
                sim = self._compare(candidate, ex)
                logger.debug(
                    "CuratorEngine: compare vs %r -> sim=%.3f",
                    ex.bullet_id, sim,
                )
                if sim > best_sim:
                    best_sim = sim
                    best_match = ex

            # Decide: merge or insert
            if best_match is not None and best_sim >= self._config.similarity_threshold:
                logger.debug(
                    "CuratorEngine: MERGE with %r (sim=%.3f >= %.2f)",
                    best_match.bullet_id, best_sim, self._config.similarity_threshold,
                )
                result.to_merge.append(
                    MergeCandidate(
                        candidate=candidate,
                        existing=best_match,
                        similarity=best_sim,
                    )
                )
            else:
                logger.debug(
                    "CuratorEngine: INSERT (best_sim=%.3f < threshold=%.2f)",
                    best_sim, self._config.similarity_threshold,
                )
                result.to_add.append(candidate)

        logger.debug(
            "CuratorEngine.curate result: add=%d merge=%d skip=%d",
            len(result.to_add), len(result.to_merge), len(result.to_skip),
        )
        return result

    # ------------------------------------------------------------------
    # Similarity helpers
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 when either vector is zero-length or when dimensions
        differ.  The result is clamped to [-1.0, 1.0] to guard against
        floating-point rounding.
        """
        if len(a) != len(b) or len(a) == 0:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        raw = dot / (norm_a * norm_b)
        # Clamp to handle floating-point imprecision
        return max(-1.0, min(1.0, raw))

    @staticmethod
    def text_similarity(a: str, b: str) -> float:
        """Fallback: token-overlap ratio when embeddings are unavailable.

        Tokenises both strings into lowercase words, then computes the Jaccard
        coefficient (|intersection| / |union|).  Returns 0.0 for empty inputs.
        """
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        sim = len(intersection) / len(union)
        logger.debug(
            "CuratorEngine.text_similarity: |a|=%d |b|=%d |inter|=%d |union|=%d -> %.3f",
            len(tokens_a), len(tokens_b), len(intersection), len(union), sim,
        )
        return sim

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compare(self, candidate: CandidateBullet, existing: ExistingBullet) -> float:
        """Compare a candidate with an existing bullet.

        Uses cosine similarity on embeddings when both have them, otherwise
        falls back to text similarity.
        """
        # Check if the candidate has an embedding (stored in metadata or as attr)
        candidate_embedding = self._get_candidate_embedding(candidate)
        existing_embedding = existing.embedding

        if candidate_embedding is not None and existing_embedding is not None:
            return self.cosine_similarity(candidate_embedding, existing_embedding)

        # Fallback: text-based similarity
        return self.text_similarity(candidate.content, existing.content)

    @staticmethod
    def _get_candidate_embedding(candidate: CandidateBullet) -> list[float] | None:
        """Extract embedding from a CandidateBullet if available.

        CandidateBullet does not have a native embedding field, so we look
        for it in a conventional location (not yet defined upstream).
        Returns None when no embedding is available.
        """
        # CandidateBullet has no embedding field in current schema.
        # Embeddings will be attached by upstream pipeline (e.g., EmbedService).
        # For now, always return None to trigger text fallback.
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current similarity threshold."""
        return self._config.similarity_threshold
