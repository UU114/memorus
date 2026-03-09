"""Unit tests for memx.engines.reflector.scorer -- KnowledgeScorer."""

from __future__ import annotations

from memx.core.config import ReflectorConfig
from memx.core.engines.reflector.scorer import KnowledgeScorer
from memx.core.types import (
    BulletSection,
    DetectedPattern,
    KnowledgeType,
    ScoredCandidate,
)


# -- Helper factories --------------------------------------------------------


def _pattern(
    pattern_type: str = "",
    content: str = (
        "this is a sufficiently long piece of generic content "
        "with enough words to avoid any density penalty being applied here"
    ),
    confidence: float = 0.5,
) -> DetectedPattern:
    return DetectedPattern(
        pattern_type=pattern_type,
        content=content,
        confidence=confidence,
    )


# -- Type classification tests -----------------------------------------------


class TestClassifyType:
    """Tests for _classify_type: pattern_type -> KnowledgeType mapping."""

    def test_score_error_fix_type(self) -> None:
        """error_fix pattern_type maps to KnowledgeType.PITFALL."""
        scorer = KnowledgeScorer()
        p = _pattern(pattern_type="error_fix", confidence=0.8)
        result = scorer.score(p)
        assert result is not None
        assert result.knowledge_type == KnowledgeType.PITFALL

    def test_score_retry_success_type(self) -> None:
        """retry_success pattern_type maps to KnowledgeType.METHOD."""
        scorer = KnowledgeScorer()
        p = _pattern(pattern_type="retry_success", confidence=0.8)
        result = scorer.score(p)
        assert result is not None
        assert result.knowledge_type == KnowledgeType.METHOD

    def test_score_config_change_type(self) -> None:
        """config_change pattern_type maps to KnowledgeType.PREFERENCE."""
        scorer = KnowledgeScorer()
        p = _pattern(pattern_type="config_change", confidence=0.8)
        result = scorer.score(p)
        assert result is not None
        assert result.knowledge_type == KnowledgeType.PREFERENCE

    def test_score_new_tool_type(self) -> None:
        """new_tool pattern_type maps to KnowledgeType.TRICK."""
        scorer = KnowledgeScorer()
        p = _pattern(pattern_type="new_tool", confidence=0.8)
        result = scorer.score(p)
        assert result is not None
        assert result.knowledge_type == KnowledgeType.TRICK

    def test_score_unknown_type(self) -> None:
        """Unknown pattern_type falls back to KnowledgeType.KNOWLEDGE."""
        scorer = KnowledgeScorer()
        p = _pattern(pattern_type="something_entirely_unknown", confidence=0.8)
        result = scorer.score(p)
        assert result is not None
        assert result.knowledge_type == KnowledgeType.KNOWLEDGE


# -- Section assignment tests -------------------------------------------------


class TestAssignSection:
    """Tests for _assign_section: keyword-based section classification."""

    def test_section_debugging(self) -> None:
        """Content containing 'error' and 'fix' should map to DEBUGGING."""
        scorer = KnowledgeScorer()
        p = _pattern(
            content="Found an error fix for the database connection timeout issue we encountered",
            confidence=0.8,
        )
        result = scorer.score(p)
        assert result is not None
        assert result.section == BulletSection.DEBUGGING

    def test_section_tools(self) -> None:
        """Content containing 'docker' should map to TOOLS."""
        scorer = KnowledgeScorer()
        p = _pattern(
            content="Use docker compose up to start the full development stack containers",
            confidence=0.8,
        )
        result = scorer.score(p)
        assert result is not None
        assert result.section == BulletSection.TOOLS

    def test_section_workflow(self) -> None:
        """Content containing 'install' and 'setup' should map to WORKFLOW."""
        scorer = KnowledgeScorer()
        p = _pattern(
            content="To install the dependencies you need to setup the virtual environment first",
            confidence=0.8,
        )
        result = scorer.score(p)
        assert result is not None
        assert result.section == BulletSection.WORKFLOW

    def test_section_preferences(self) -> None:
        """Content containing 'always' and 'prefer' should map to PREFERENCES."""
        scorer = KnowledgeScorer()
        p = _pattern(
            content="I always prefer using tabs over spaces for indentation in all projects",
            confidence=0.8,
        )
        result = scorer.score(p)
        assert result is not None
        assert result.section == BulletSection.PREFERENCES

    def test_section_general(self) -> None:
        """Content with no matching keywords should fall back to GENERAL."""
        scorer = KnowledgeScorer()
        p = _pattern(
            content="The sky is blue and the sun is bright on a warm summer afternoon today",
            confidence=0.8,
        )
        result = scorer.score(p)
        assert result is not None
        assert result.section == BulletSection.GENERAL


# -- Threshold tests ----------------------------------------------------------


class TestScoreThreshold:
    """Tests for min_score threshold filtering."""

    def test_score_above_threshold(self) -> None:
        """High confidence + good content -> ScoredCandidate returned (not None)."""
        scorer = KnowledgeScorer()
        # confidence=0.8, 20 words -> base=80, penalty=1.0 -> score=80
        p = _pattern(
            content=(
                "When the database connection times out you need to increase the "
                "pool size and add a retry mechanism for the connection logic handling"
            ),
            confidence=0.8,
        )
        result = scorer.score(p)
        assert result is not None
        assert isinstance(result, ScoredCandidate)
        assert result.instructivity_score > 30.0

    def test_score_below_threshold(self) -> None:
        """Low confidence + short content -> None returned."""
        scorer = KnowledgeScorer()
        # confidence=0.1, 2 words -> base=10, penalty=2/20=0.1 -> score=1.0
        p = _pattern(
            content="hi there",
            confidence=0.1,
        )
        result = scorer.score(p)
        assert result is None

    def test_score_at_threshold_boundary(self) -> None:
        """Score exactly at min_score (30.0) -> ScoredCandidate returned."""
        scorer = KnowledgeScorer()
        # confidence=0.6, 10 words -> base=60, penalty=10/20=0.5 -> score=30.0
        # No actionable keywords, no file refs -> no bonus
        p = _pattern(
            content="the quick brown fox jumped over the lazy sleeping dog",
            confidence=0.6,
        )
        result = scorer.score(p)
        assert result is not None
        assert result.instructivity_score == 30.0

    def test_score_just_below_threshold(self) -> None:
        """Score just under min_score -> None returned."""
        scorer = KnowledgeScorer()
        # confidence=0.6, 9 words -> base=60, penalty=9/20=0.45 -> score=27.0
        # No actionable keywords, no file refs -> no bonus
        p = _pattern(
            content="the quick brown fox jumped over the lazy dog",
            confidence=0.6,
        )
        result = scorer.score(p)
        assert result is None


# -- Density penalty tests ----------------------------------------------------


class TestDensityPenalty:
    """Tests for the word-count-based density penalty."""

    def test_density_penalty_short(self) -> None:
        """Very short content gets penalized heavily."""
        scorer = KnowledgeScorer()
        # confidence=0.5, 2 words -> base=50, penalty=2/20=0.1 -> score=5.0
        p = _pattern(content="hi there", confidence=0.5)
        result = scorer.score(p)
        # score = 5.0 < 30.0 (default min_score) -> None
        assert result is None

        # Verify the raw score value via static method
        raw = KnowledgeScorer._compute_score(p)
        assert raw < 30.0
        assert raw == 5.0

    def test_density_penalty_long(self) -> None:
        """Long content (>= 20 words) gets no penalty (density_penalty = 1.0)."""
        scorer = KnowledgeScorer()
        # 25 words, confidence=0.5 -> base=50, penalty=min(1.0, 25/20)=1.0 -> score=50
        long_content = " ".join(["word"] * 25)
        p = _pattern(content=long_content, confidence=0.5)
        raw = KnowledgeScorer._compute_score(p)
        assert raw == 50.0

        result = scorer.score(p)
        assert result is not None
        assert result.instructivity_score == 50.0


# -- Distill bonus tests ------------------------------------------------------


class TestDistillBonus:
    """Tests for actionable keyword and file-reference bonuses."""

    def test_distill_bonus_actionable(self) -> None:
        """Content with actionable keywords ('use', 'avoid', etc.) gets +10 bonus."""
        scorer = KnowledgeScorer()
        # 20 words with "use" keyword, confidence=0.5
        # base=50, penalty=1.0, actionable_bonus=+10, no file ref -> score=60
        content = (
            "you can use this particular technique to make your code "
            "much more readable and maintainable for the whole team members"
        )
        p = _pattern(content=content, confidence=0.5)
        raw = KnowledgeScorer._compute_score(p)
        # base=50, penalty=1.0 (20 words), actionable=+10 -> 60
        assert raw == 60.0

        # Compare to same content without actionable keyword (also 20 words)
        neutral_content = (
            "the particular technique makes the code "
            "much more readable and maintainable for the whole team members around here today"
        )
        p_neutral = _pattern(content=neutral_content, confidence=0.5)
        raw_neutral = KnowledgeScorer._compute_score(p_neutral)
        # 19 words -> penalty=19/20=0.95, base=50*0.95=47.5
        assert raw_neutral == 47.5

        assert raw - raw_neutral == 12.5

    def test_distill_bonus_file_ref(self) -> None:
        """Content with file path characters ('.', '/') gets +5 bonus."""
        scorer = KnowledgeScorer()
        # 20 words with a file path, confidence=0.5
        # base=50, penalty=1.0, file_ref=+5 -> score=55
        content = (
            "check the configuration at /etc/nginx/nginx conf "
            "to see the upstream settings for the proxy server backend nodes here"
        )
        p = _pattern(content=content, confidence=0.5)
        raw = KnowledgeScorer._compute_score(p)
        # 18 words -> penalty=18/20=0.9, base=50*0.9=45, file_ref=+5 -> 50
        assert raw == 50.0


# -- Custom config tests ------------------------------------------------------


class TestCustomConfig:
    """Tests for custom ReflectorConfig overrides."""

    def test_custom_min_score(self) -> None:
        """ReflectorConfig(min_score=50) raises the threshold."""
        config = ReflectorConfig(min_score=50.0)
        scorer = KnowledgeScorer(config=config)

        # This pattern scores ~40 (base=80, penalty=10/20=0.5 -> 40)
        # No actionable keywords, no file ref
        content = "the quick brown fox jumped over the lazy sleeping dog"
        p = _pattern(content=content, confidence=0.8)

        raw = KnowledgeScorer._compute_score(p)
        assert raw == 40.0

        # With default min_score=30, this would pass; with 50, it should not
        result = scorer.score(p)
        assert result is None

        # Same pattern with default scorer passes
        default_scorer = KnowledgeScorer()
        result_default = default_scorer.score(p)
        assert result_default is not None
        assert result_default.instructivity_score == 40.0
