"""CuratorEngine -- core deduplication logic for the Curator subsystem.

Compares CandidateBullets against existing memories using cosine similarity
(primary) or text-based token overlap (fallback). Marks each candidate as
Insert, Merge, or Skip based on a configurable similarity threshold.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from memorus.core.config import CuratorConfig
from memorus.core.types import CandidateBullet

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
# Corpus consolidation (STORY-R094) — aligned byte-for-byte with the Rust
# ``memorus_ace::curator::consolidate_corpus``.
# ---------------------------------------------------------------------------


@dataclass
class MergeGroup:
    """A group of near-duplicate bullets that should be merged together."""

    bullet_ids: list[str]
    max_similarity: float


@dataclass
class ConsolidateConflictInfo:
    """Conflict between two existing bullets discovered during consolidate.

    Mirrors Rust's ``curator::conflict::ConflictInfo`` but aimed at corpus
    scan (both ids known). Naming kept distinct from the ``Conflict`` type
    produced by the live-curate conflict detector.
    """

    a_id: str
    b_id: str
    a_content: str
    b_content: str
    similarity: float
    conflict_type: str  # "negation" | "contradiction" | "value_difference"
    reason: str


@dataclass
class ConsolidateResult:
    """Result of a corpus-wide consolidation pass."""

    merge_groups: list[MergeGroup] = field(default_factory=list)
    redundant_ids: list[str] = field(default_factory=list)
    conflicts: list[ConsolidateConflictInfo] = field(default_factory=list)
    pairs_compared: int = 0


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
                from memorus.core.engines.curator.conflict import ConflictDetector

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
    # Corpus consolidation (STORY-R094)
    # ------------------------------------------------------------------

    def consolidate_corpus(
        self,
        bullets: list[ExistingBullet],
        max_per_pass: int = 5000,
    ) -> ConsolidateResult:
        """Run corpus-wide deduplication.

        Semantics MUST match ``memorus_ace::curator::Curator::consolidate_corpus``:

        * Pairs (i, j) with ``j > i`` are scanned in order.
        * If similarity >= ``skip_threshold`` → ``bullet[j]`` is marked redundant.
        * Else if similarity >= ``dedup_threshold`` → merge into group seeded
          at ``bullet[i]``.
        * If ``conflict_detection`` is on and ``conflict_min <= sim < conflict_max``,
          run pairwise conflict detection and append results.
        * ``max_per_pass`` caps ``n = min(len, max_per_pass)`` to bound O(n^2).

        Thresholds (mirrors Rust defaults, with Python ``CuratorConfig``
        fallback values for missing fields):

        * ``dedup_threshold``: ``CuratorConfig.similarity_threshold`` (0.8)
        * ``skip_threshold``: 0.95 (Rust default; not exposed in Python config)
        * ``conflict_min_similarity``: ``CuratorConfig.conflict_min_similarity``
        * ``conflict_max_similarity``: ``CuratorConfig.conflict_max_similarity``
        """
        result = ConsolidateResult()
        n = min(len(bullets), max_per_pass)
        if n < 2:
            return result

        dedup_threshold = self._config.similarity_threshold
        skip_threshold = max(0.95, dedup_threshold)
        conflict_min = self._config.conflict_min_similarity
        conflict_max = self._config.conflict_max_similarity
        conflict_on = self._config.conflict_detection

        slice_ = bullets[:n]
        grouped: set[str] = set()

        for i in range(n):
            if slice_[i].bullet_id in grouped:
                continue
            group_ids: list[str] = [slice_[i].bullet_id]
            max_sim = 0.0

            for j in range(i + 1, n):
                if slice_[j].bullet_id in grouped:
                    continue
                result.pairs_compared += 1

                sim = self._compute_bullet_similarity(slice_[i], slice_[j])

                if sim >= skip_threshold:
                    result.redundant_ids.append(slice_[j].bullet_id)
                    grouped.add(slice_[j].bullet_id)
                elif sim >= dedup_threshold:
                    group_ids.append(slice_[j].bullet_id)
                    grouped.add(slice_[j].bullet_id)
                    if sim > max_sim:
                        max_sim = sim

                if conflict_on and conflict_min <= sim < conflict_max:
                    conflicts = _detect_bullet_conflicts(
                        slice_[i], slice_[j], sim,
                    )
                    result.conflicts.extend(conflicts)

            if len(group_ids) > 1:
                grouped.add(slice_[i].bullet_id)
                result.merge_groups.append(
                    MergeGroup(bullet_ids=group_ids, max_similarity=max_sim)
                )

        return result

    def _compute_bullet_similarity(
        self, a: ExistingBullet, b: ExistingBullet
    ) -> float:
        """Compute similarity between two existing bullets.

        Prefers cosine on embeddings; falls back to Jaccard token overlap.
        """
        if a.embedding is not None and b.embedding is not None:
            return self.cosine_similarity(a.embedding, b.embedding)
        return self.text_similarity(a.content, b.content)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current similarity threshold."""
        return self._config.similarity_threshold


# ---------------------------------------------------------------------------
# Conflict detection between two existing bullets (STORY-R094).
# Mirrors Rust's ``detect_bullet_conflicts``.
# ---------------------------------------------------------------------------

_NEGATION_WORDS: frozenset[str] = frozenset({
    # English
    "not", "don't", "doesn't", "never", "avoid", "shouldn't", "won't",
    "cannot", "can't", "no", "without",
    # Chinese (whole-token)
    "不", "不要", "不能", "避免", "禁止", "勿", "别",
})
_NEGATION_ZH: tuple[str, ...] = (
    "不", "不要", "不能", "避免", "禁止", "勿", "别",
)
_ANTONYM_PAIRS: tuple[tuple[str, str], ...] = (
    ("always", "never"),
    ("enable", "disable"),
    ("use", "avoid"),
    ("recommended", "deprecated"),
    ("allow", "deny"),
    ("include", "exclude"),
    ("true", "false"),
    ("yes", "no"),
    ("sync", "async"),
    ("mutable", "immutable"),
)


def _count_negations(text_lower: str) -> int:
    """Count negation-word occurrences. Mirrors Rust helper byte-for-byte.

    * Tokenises on whitespace and strips trailing ASCII punctuation (except
      apostrophes) so ``"don't,"`` still matches ``"don't"``.
    * For each token, if it is a whole-word negation, count once.
    * Otherwise, if the token *contains* any Chinese negation word, count
      once (Chinese text may lack whitespace).
    """
    count = 0
    for token in text_lower.split():
        clean = token.rstrip(
            "!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~"  # ASCII punct minus apostrophe
        )
        if clean in _NEGATION_WORDS:
            count += 1
            continue
        for zh in _NEGATION_ZH:
            if zh in token:
                count += 1
                break
    return count


def _find_antonym_pair(
    text_a_lower: str, text_b_lower: str
) -> tuple[str, str] | None:
    """Find the first antonym pair where text_a has one word and text_b has the other."""
    def _tokenise(s: str) -> list[str]:
        out: list[str] = []
        for tok in s.split():
            out.append(tok.rstrip("!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~"))
        return out

    tokens_a = _tokenise(text_a_lower)
    tokens_b = _tokenise(text_b_lower)
    for word_a, word_b in _ANTONYM_PAIRS:
        if word_a in tokens_a and word_b in tokens_b:
            return (word_a, word_b)
        if word_b in tokens_a and word_a in tokens_b:
            return (word_b, word_a)
    return None


def _detect_bullet_conflicts(
    a: ExistingBullet,
    b: ExistingBullet,
    similarity: float,
) -> list[ConsolidateConflictInfo]:
    """Detect conflicts between two existing bullets (byte-for-byte vs Rust)."""
    a_lower = a.content.lower()
    b_lower = b.content.lower()
    conflicts: list[ConsolidateConflictInfo] = []

    # Negation asymmetry (one has >0 negations, the other has 0)
    neg_a = _count_negations(a_lower)
    neg_b = _count_negations(b_lower)
    if (neg_a > 0) != (neg_b > 0):
        conflicts.append(
            ConsolidateConflictInfo(
                a_id=a.bullet_id,
                b_id=b.bullet_id,
                a_content=a.content,
                b_content=b.content,
                similarity=similarity,
                conflict_type="negation",
                reason=f"Negation mismatch ({neg_a} vs {neg_b} negation words)",
            )
        )

    # Antonym check
    pair = _find_antonym_pair(a_lower, b_lower)
    if pair is not None:
        word_a, word_b = pair
        conflicts.append(
            ConsolidateConflictInfo(
                a_id=a.bullet_id,
                b_id=b.bullet_id,
                a_content=a.content,
                b_content=b.content,
                similarity=similarity,
                conflict_type="contradiction",
                reason=f'Antonym pair: "{word_a}" vs "{word_b}"',
            )
        )
    return conflicts
