"""ReflectorEngine — orchestrates the 4-stage knowledge distillation pipeline.

Supports three operating modes:
- "rules"  : 0 LLM calls, pure heuristic detection + scoring (default)
- "llm"    : LLM-based evaluation + distillation for semantic understanding
- "hybrid" : rules pre-screen + LLM refinement (best quality/cost balance)
"""

from __future__ import annotations

import logging
from typing import Optional

from memorus.core.config import ReflectorConfig
from memorus.core.engines.reflector.detector import PatternDetector
from memorus.core.engines.reflector.distiller import BulletDistiller
from memorus.core.engines.reflector.scorer import KnowledgeScorer
from memorus.core.privacy.sanitizer import PrivacySanitizer
from memorus.core.types import (
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

    In "llm" mode, Stage 1+2 are replaced by LLMEvaluator and Stage 4
    by LLMDistiller. In "hybrid" mode, rules pre-screen (Stage 1) and
    LLM refines scoring (Stage 2) and distillation (Stage 4).

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

        # Lazy-init LLM components (only when needed)
        self._llm_evaluator: Optional[object] = None
        self._llm_distiller: Optional[object] = None
        if self._mode in ("llm", "hybrid"):
            self._init_llm_components()

    def _init_llm_components(self) -> None:
        """Initialize LLM evaluator and distiller. Falls back to rules on failure."""
        try:
            from memorus.core.engines.reflector.llm_evaluator import LLMEvaluator
            from memorus.core.engines.reflector.llm_distiller import LLMDistiller

            self._llm_evaluator = LLMEvaluator(self._config)
            self._llm_distiller = LLMDistiller(self._config)
            logger.info(
                "ReflectorEngine: LLM components initialized (mode=%s, model=%s)",
                self._mode, self._config.llm_model,
            )
        except Exception as e:
            logger.warning(
                "Failed to initialize LLM components, falling back to 'rules': %s", e,
            )
            self._mode = "rules"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reflect(self, event: InteractionEvent) -> list[CandidateBullet]:
        """Run distillation pipeline based on current mode.

        Each stage has an independent failure boundary -- if one stage fails,
        fallback logic is used rather than crashing the entire pipeline.
        Returns an empty list when there is nothing to learn.
        """
        if event is None:
            logger.debug("ReflectorEngine.reflect: event is None, returning []")
            return []

        logger.debug(
            "ReflectorEngine.reflect: user_msg_len=%d asst_msg_len=%d mode=%s",
            len(event.user_message), len(event.assistant_message), self._mode,
        )

        if self._mode == "llm":
            return self._reflect_llm(event)
        elif self._mode == "hybrid":
            return self._reflect_hybrid(event)
        else:
            return self._reflect_rules(event)

    # ------------------------------------------------------------------
    # Mode: rules (original pipeline)
    # ------------------------------------------------------------------

    def _reflect_rules(self, event: InteractionEvent) -> list[CandidateBullet]:
        """Pure rules pipeline: PatternDetector -> KnowledgeScorer -> Sanitizer -> BulletDistiller."""
        # Stage 1: Pattern Detection
        patterns = self._run_stage1(event)
        if not patterns:
            logger.debug("ReflectorEngine.reflect: stage1 -> 0 patterns, nothing to learn")
            return []

        # Stage 2: Knowledge Scoring
        scored = self._run_stage2(patterns)
        if not scored:
            logger.debug("ReflectorEngine.reflect: stage2 -> 0 scored (all below min_score=%.1f)", self._config.min_score)
            return []

        # Stage 3: Privacy Sanitization
        sanitized = self._run_stage3(scored)

        # Stage 4: Bullet Distillation
        bullets = self._run_stage4(sanitized)
        logger.debug("ReflectorEngine._reflect_rules: pipeline complete -> %d bullet(s)", len(bullets))

        return bullets

    # ------------------------------------------------------------------
    # Mode: llm (full LLM evaluation + distillation)
    # ------------------------------------------------------------------

    def _reflect_llm(self, event: InteractionEvent) -> list[CandidateBullet]:
        """LLM pipeline: LLMEvaluator -> Sanitizer -> LLMDistiller.

        Falls back to rules pipeline on LLM failure.
        """
        from memorus.core.engines.reflector.llm_evaluator import LLMEvaluator
        from memorus.core.engines.reflector.llm_distiller import LLMDistiller

        evaluator = self._llm_evaluator
        distiller = self._llm_distiller
        if not isinstance(evaluator, LLMEvaluator) or not isinstance(distiller, LLMDistiller):
            logger.warning("LLM components not available, falling back to rules")
            return self._reflect_rules(event)

        # Stage 1+2: LLM Evaluation
        try:
            scored = evaluator.evaluate(event)
        except Exception as e:
            logger.warning("LLMEvaluator failed, falling back to rules: %s", e)
            return self._reflect_rules(event)

        if scored is None:
            logger.debug("ReflectorEngine._reflect_llm: LLM says nothing to learn")
            return []

        # Check min_score
        if scored.instructivity_score < self._config.min_score:
            logger.debug(
                "ReflectorEngine._reflect_llm: LLM score %.1f < min %.1f, skipping",
                scored.instructivity_score, self._config.min_score,
            )
            return []

        # Stage 3: Privacy Sanitization
        sanitized = self._run_stage3([scored])

        # Stage 4: LLM Distillation
        try:
            bullet = distiller.distill(sanitized[0])
            logger.debug(
                "ReflectorEngine._reflect_llm: complete -> section=%s rule=%r",
                bullet.section.value,
                (bullet.distilled_rule or "")[:60],
            )
            return [bullet]
        except Exception as e:
            logger.warning("LLMDistiller failed, using fallback: %s", e)
            return self._run_stage4(sanitized)

    # ------------------------------------------------------------------
    # Mode: hybrid (rules pre-screen + LLM refinement)
    # ------------------------------------------------------------------

    def _reflect_hybrid(self, event: InteractionEvent) -> list[CandidateBullet]:
        """Hybrid pipeline: rules detect -> LLM refine scoring -> Sanitizer -> LLM distill.

        If rules detect nothing, LLMEvaluator gets a chance to evaluate directly.
        Falls back to rules pipeline on LLM failure.
        """
        from memorus.core.engines.reflector.llm_evaluator import LLMEvaluator
        from memorus.core.engines.reflector.llm_distiller import LLMDistiller

        evaluator = self._llm_evaluator
        distiller = self._llm_distiller
        if not isinstance(evaluator, LLMEvaluator) or not isinstance(distiller, LLMDistiller):
            logger.warning("LLM components not available, falling back to rules")
            return self._reflect_rules(event)

        # Stage 1: Rules-based pre-screening
        patterns = self._run_stage1(event)
        scored_candidates: list[ScoredCandidate] = []

        if patterns:
            # Stage 2a: Rule-based scoring
            rule_scored = self._run_stage2(patterns)
            # Stage 2b: LLM refinement of rule-detected candidates
            for candidate in rule_scored:
                try:
                    refined = evaluator.refine(candidate)
                    scored_candidates.append(refined)
                except Exception as e:
                    logger.debug("LLM refine failed for candidate, keeping rule score: %s", e)
                    scored_candidates.append(candidate)
        else:
            # Rules found nothing — give LLM a direct shot
            logger.debug("ReflectorEngine._reflect_hybrid: rules found nothing, trying LLM evaluation")
            try:
                llm_scored = evaluator.evaluate(event)
                if llm_scored is not None and llm_scored.instructivity_score >= self._config.min_score:
                    scored_candidates.append(llm_scored)
            except Exception as e:
                logger.debug("LLM direct evaluation failed: %s", e)

        if not scored_candidates:
            logger.debug("ReflectorEngine._reflect_hybrid: nothing to learn")
            return []

        # Stage 3: Privacy Sanitization
        sanitized = self._run_stage3(scored_candidates)

        # Stage 4: LLM Distillation (with fallback)
        bullets: list[CandidateBullet] = []
        for candidate in sanitized:
            try:
                bullet = distiller.distill(candidate)
                bullets.append(bullet)
            except Exception as e:
                logger.debug("LLM distill failed for candidate, using fallback: %s", e)
                bullets.append(self._fallback_distill_single(candidate))

        logger.debug("ReflectorEngine._reflect_hybrid: complete -> %d bullet(s)", len(bullets))
        return bullets

    # ------------------------------------------------------------------
    # Stage runners (each catches exceptions independently)
    # ------------------------------------------------------------------

    def _run_stage1(self, event: InteractionEvent) -> list[DetectedPattern]:
        """Stage 1: Detect patterns.  Failure -> empty list."""
        try:
            result = self._detector.detect(event)
            logger.debug("ReflectorEngine stage1: detected %d pattern(s): %s",
                         len(result), [p.pattern_type for p in result])
            return result
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
                    logger.debug(
                        "ReflectorEngine stage2: scored %s -> section=%s type=%s score=%.1f",
                        p.pattern_type, s.section.value, s.knowledge_type.value,
                        s.instructivity_score,
                    )
                else:
                    logger.debug("ReflectorEngine stage2: %s rejected (below min_score)", p.pattern_type)
            return scored
        except Exception as e:
            logger.warning("Stage 2 (KnowledgeScorer) failed: %s", e)
            return self._fallback_scoring(patterns)

    def _run_stage3(
        self, candidates: list[ScoredCandidate]
    ) -> list[ScoredCandidate]:
        """Stage 3: Sanitize content.  Failure -> use original (unsanitized)."""
        try:
            modified_count = 0
            for c in candidates:
                result = self._sanitizer.sanitize(c.pattern.content)
                if result.was_modified:
                    modified_count += 1
                    logger.debug(
                        "ReflectorEngine stage3: sanitized content (filtered %d items)",
                        len(result.filtered_items),
                    )
                # Replace pattern content with sanitized version via model_copy
                c.pattern = c.pattern.model_copy(
                    update={"content": result.clean_content}
                )
            logger.debug("ReflectorEngine stage3: sanitized %d/%d candidates", modified_count, len(candidates))
            return candidates
        except Exception as e:
            logger.warning("Stage 3 (PrivacySanitizer) failed: %s", e)
            return candidates  # graceful degradation: use unsanitized

    def _run_stage4(
        self, candidates: list[ScoredCandidate]
    ) -> list[CandidateBullet]:
        """Stage 4: Distill into Bullets.  Failure -> fallback distill."""
        try:
            bullets = [self._distiller.distill(c) for c in candidates]
            for b in bullets:
                logger.debug(
                    "ReflectorEngine stage4: bullet section=%s tools=%s entities=%s content=%r",
                    b.section.value, b.related_tools, b.key_entities[:3],
                    b.content[:60],
                )
            return bullets
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
                sources=BulletDistiller._build_sources(c),
            )
            for c in candidates
        ]

    @staticmethod
    def _fallback_distill_single(candidate: ScoredCandidate) -> CandidateBullet:
        """Fallback for a single candidate when LLM distillation fails."""
        return CandidateBullet(
            content=candidate.pattern.content[:500],
            section=candidate.section,
            knowledge_type=candidate.knowledge_type,
            instructivity_score=candidate.instructivity_score,
            source_type=SourceType.INTERACTION,
            sources=BulletDistiller._build_sources(candidate),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        """Current operating mode: 'rules', 'llm', or 'hybrid'."""
        return self._mode
