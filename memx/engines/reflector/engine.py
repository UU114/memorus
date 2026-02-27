"""ReflectorEngine — orchestrates the 4-stage knowledge distillation pipeline."""

from __future__ import annotations

import logging
from typing import Optional

from memx.config import ReflectorConfig
from memx.engines.reflector.detector import PatternDetector
from memx.engines.reflector.distiller import BulletDistiller
from memx.engines.reflector.scorer import KnowledgeScorer
from memx.privacy.sanitizer import PrivacySanitizer
from memx.types import (
    BulletSection,
    CandidateBullet,
    DetectedPattern,
    InteractionEvent,
    KnowledgeType,
    ScoredCandidate,
    SourceType,
)

logger = logging.getLogger(__name__)


class ReflectorEngine:
    """Orchestrates the 4-stage knowledge distillation pipeline.

    Stage 1: PatternDetector  -- detect learnable patterns from interaction
    Stage 2: KnowledgeScorer  -- classify and score detected patterns
    Stage 3: PrivacySanitizer -- redact sensitive data from scored candidates
    Stage 4: BulletDistiller  -- distill into compact CandidateBullets

    Each stage has independent error handling -- failure in one stage does not
    crash the pipeline.  Fallback logic is applied where possible.
    """

    def __init__(
        self,
        config: Optional[ReflectorConfig] = None,
        sanitizer: Optional[PrivacySanitizer] = None,
    ) -> None:
        self._config = config or ReflectorConfig()
        self._detector = PatternDetector()
        self._scorer = KnowledgeScorer(self._config)
        self._sanitizer = sanitizer or PrivacySanitizer()
        self._distiller = BulletDistiller(self._config)
        self._mode = self._config.mode

        # Only "rules" mode is implemented; warn and fall back for others
        if self._mode in ("llm", "hybrid"):
            logger.warning(
                "Mode '%s' not yet implemented, falling back to 'rules'",
                self._mode,
            )
            self._mode = "rules"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(self, event: InteractionEvent) -> list[CandidateBullet]:
        """Run 4-stage distillation pipeline.

        Each stage has an independent failure boundary -- if one stage fails,
        fallback logic is used rather than crashing the entire pipeline.
        Returns an empty list when there is nothing to learn.
        """
        if event is None:
            return []

        # Stage 1: Pattern Detection
        patterns = self._run_stage1(event)
        if not patterns:
            return []

        # Stage 2: Knowledge Scoring
        scored = self._run_stage2(patterns)
        if not scored:
            return []

        # Stage 3: Privacy Sanitization
        sanitized = self._run_stage3(scored)

        # Stage 4: Bullet Distillation
        bullets = self._run_stage4(sanitized)

        return bullets

    # ------------------------------------------------------------------
    # Stage runners (each catches exceptions independently)
    # ------------------------------------------------------------------

    def _run_stage1(self, event: InteractionEvent) -> list[DetectedPattern]:
        """Stage 1: Detect patterns.  Failure -> empty list."""
        try:
            return self._detector.detect(event)
        except Exception as e:
            logger.warning("Stage 1 (PatternDetector) failed: %s", e)
            return []

    def _run_stage2(
        self, patterns: list[DetectedPattern]
    ) -> list[ScoredCandidate]:
        """Stage 2: Score patterns.  Failure -> fallback scoring."""
        try:
            scored: list[ScoredCandidate] = []
            for p in patterns:
                if s := self._scorer.score(p):
                    scored.append(s)
            return scored
        except Exception as e:
            logger.warning("Stage 2 (KnowledgeScorer) failed: %s", e)
            return self._fallback_scoring(patterns)

    def _run_stage3(
        self, candidates: list[ScoredCandidate]
    ) -> list[ScoredCandidate]:
        """Stage 3: Sanitize content.  Failure -> use original (unsanitized)."""
        try:
            for c in candidates:
                result = self._sanitizer.sanitize(c.pattern.content)
                # Replace pattern content with sanitized version via model_copy
                c.pattern = c.pattern.model_copy(
                    update={"content": result.clean_content}
                )
            return candidates
        except Exception as e:
            logger.warning("Stage 3 (PrivacySanitizer) failed: %s", e)
            return candidates  # graceful degradation: use unsanitized

    def _run_stage4(
        self, candidates: list[ScoredCandidate]
    ) -> list[CandidateBullet]:
        """Stage 4: Distill into Bullets.  Failure -> fallback distill."""
        try:
            return [self._distiller.distill(c) for c in candidates]
        except Exception as e:
            logger.warning("Stage 4 (BulletDistiller) failed: %s", e)
            return self._fallback_distill(candidates)

    # ------------------------------------------------------------------
    # Fallback helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_scoring(
        patterns: list[DetectedPattern],
    ) -> list[ScoredCandidate]:
        """Emergency fallback: create ScoredCandidates with default scores."""
        return [
            ScoredCandidate(
                pattern=p,
                section=BulletSection.GENERAL,
                knowledge_type=KnowledgeType.KNOWLEDGE,
                instructivity_score=50.0,
            )
            for p in patterns
        ]

    @staticmethod
    def _fallback_distill(
        candidates: list[ScoredCandidate],
    ) -> list[CandidateBullet]:
        """Emergency fallback: create minimal CandidateBullets."""
        return [
            CandidateBullet(
                content=c.pattern.content[:500],
                section=c.section,
                knowledge_type=c.knowledge_type,
                instructivity_score=c.instructivity_score,
                source_type=SourceType.INTERACTION,
            )
            for c in candidates
        ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current operating mode (always 'rules' until llm/hybrid are implemented)."""
        return self._mode
