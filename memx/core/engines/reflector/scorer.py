"""KnowledgeScorer -- Stage 2 of the Reflector pipeline. Classifies and scores patterns."""

from __future__ import annotations

import logging
from typing import Optional

from memx.core.config import ReflectorConfig
from memx.core.types import (
    BulletSection,
    DetectedPattern,
    KnowledgeType,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

# Pattern type -> knowledge type mapping
_TYPE_MAP: dict[str, KnowledgeType] = {
    "error_fix": KnowledgeType.PITFALL,
    "retry_success": KnowledgeType.METHOD,
    "config_change": KnowledgeType.PREFERENCE,
    "new_tool": KnowledgeType.TRICK,
    "repetitive_op": KnowledgeType.METHOD,
}

# Section keywords (ordered by priority -- first match wins)
_SECTION_KEYWORDS: list[tuple[BulletSection, frozenset[str]]] = [
    (BulletSection.DEBUGGING, frozenset({
        "debug", "error", "fix", "crash", "traceback", "exception",
        "bug", "stacktrace", "segfault", "panic",
    })),
    (BulletSection.WORKFLOW, frozenset({
        "install", "config", "env", "setup", "deploy", "build",
        "ci", "cd", "pipeline", "release", "migration",
    })),
    (BulletSection.TOOLS, frozenset({
        "git", "docker", "npm", "pip", "cargo", "brew",
        "kubectl", "terraform", "ansible", "make", "cmake",
    })),
    (BulletSection.ARCHITECTURE, frozenset({
        "architect", "design", "pattern", "refactor", "structure",
        "module", "layer", "interface", "abstract",
    })),
    (BulletSection.COMMANDS, frozenset({
        "run", "exec", "command", "cli", "terminal", "shell",
        "bash", "script", "alias",
    })),
    (BulletSection.PREFERENCES, frozenset({
        "prefer", "always", "never", "default", "convention",
        "style", "habit", "like", "dislike",
    })),
    (BulletSection.PATTERNS, frozenset({
        "pattern", "practice", "idiom", "recipe", "technique",
        "approach", "strategy",
    })),
]

# Actionable keywords that boost score
_ACTIONABLE_KEYWORDS = frozenset({
    "use", "run", "try", "avoid", "instead", "should", "must",
    "tip", "remember", "always", "never", "better", "best",
})


class KnowledgeScorer:
    """Score and classify DetectedPatterns into ScoredCandidates."""

    def __init__(self, config: Optional[ReflectorConfig] = None):
        self._config = config or ReflectorConfig()
        self._min_score = self._config.min_score

    def score(self, pattern: DetectedPattern) -> Optional[ScoredCandidate]:
        """Score a pattern. Returns None if below min_score threshold."""
        knowledge_type = self._classify_type(pattern)
        section = self._assign_section(pattern.content)
        score_val = self._compute_score(pattern)

        logger.debug(
            "KnowledgeScorer: pattern=%s type=%s section=%s raw_score=%.1f min=%.1f",
            pattern.pattern_type, knowledge_type.value, section.value,
            score_val, self._min_score,
        )

        if score_val < self._min_score:
            logger.debug("KnowledgeScorer: REJECTED (%.1f < %.1f)", score_val, self._min_score)
            return None

        # Clamp to valid range
        score_val = min(100.0, max(0.0, score_val))

        return ScoredCandidate(
            pattern=pattern,
            section=section,
            knowledge_type=knowledge_type,
            instructivity_score=score_val,
        )

    def _classify_type(self, pattern: DetectedPattern) -> KnowledgeType:
        """Map pattern_type to KnowledgeType."""
        return _TYPE_MAP.get(pattern.pattern_type, KnowledgeType.KNOWLEDGE)

    @staticmethod
    def _assign_section(content: str) -> BulletSection:
        """Assign section based on keyword matching (first match wins)."""
        content_lower = content.lower()
        words = set(content_lower.split())
        for section, keywords in _SECTION_KEYWORDS:
            if words & keywords:
                return section
        return BulletSection.GENERAL

    @staticmethod
    def _compute_score(pattern: DetectedPattern) -> float:
        """Compute instructivity score.

        Formula: base_score * density_penalty + distill_bonus
        """
        base = pattern.confidence * 100

        # Density penalty: penalize very short content
        word_count = len(pattern.content.split())
        density_penalty = min(1.0, word_count / 20)

        # Distill bonus: reward actionable content
        distill_bonus = 0.0
        content_lower = pattern.content.lower()
        if any(kw in content_lower for kw in _ACTIONABLE_KEYWORDS):
            distill_bonus += 10.0
        # Bonus for specific tool/file references
        if any(c in content_lower for c in (".", "/", "\\")):
            distill_bonus += 5.0

        total = base * density_penalty + distill_bonus
        logger.debug(
            "KnowledgeScorer._compute_score: base=%.1f density=%.2f(words=%d) "
            "bonus=%.1f -> total=%.1f",
            base, density_penalty, word_count, distill_bonus, total,
        )
        return total
