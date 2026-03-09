"""Unit tests for memx.engines.reflector.distiller — BulletDistiller."""

from __future__ import annotations

from memx.core.config import ReflectorConfig
from memx.core.engines.reflector.distiller import BulletDistiller
from memx.core.types import (
    BulletSection,
    CandidateBullet,
    DetectedPattern,
    KnowledgeType,
    ScoredCandidate,
    SourceType,
)


# -- Helper factories --------------------------------------------------------


def _pattern(
    content: str = "Use git rebase instead of merge for linear history.",
    pattern_type: str = "error_fix",
    confidence: float = 0.8,
    metadata: dict | None = None,
) -> DetectedPattern:
    return DetectedPattern(
        pattern_type=pattern_type,
        content=content,
        confidence=confidence,
        metadata=metadata or {},
    )


def _candidate(
    content: str = "Use git rebase instead of merge for linear history.",
    section: BulletSection = BulletSection.GENERAL,
    knowledge_type: KnowledgeType = KnowledgeType.KNOWLEDGE,
    score: float = 60.0,
    metadata: dict | None = None,
) -> ScoredCandidate:
    return ScoredCandidate(
        pattern=_pattern(content=content, metadata=metadata),
        section=section,
        knowledge_type=knowledge_type,
        instructivity_score=score,
    )


# -- distill basic -----------------------------------------------------------


class TestDistillBasic:
    def test_distill_basic(self) -> None:
        """Normal ScoredCandidate produces a CandidateBullet with correct fields."""
        distiller = BulletDistiller()
        candidate = _candidate(
            content="Use git rebase instead of merge for linear history.",
            section=BulletSection.TOOLS,
            knowledge_type=KnowledgeType.METHOD,
            score=75.0,
        )
        bullet = distiller.distill(candidate)

        assert isinstance(bullet, CandidateBullet)
        assert bullet.content == "Use git rebase instead of merge for linear history."
        assert bullet.section == BulletSection.TOOLS
        assert bullet.knowledge_type == KnowledgeType.METHOD
        assert bullet.source_type == SourceType.INTERACTION
        assert bullet.instructivity_score == 75.0
        assert "git" in bullet.related_tools


# -- _truncate_content -------------------------------------------------------


class TestTruncateContent:
    def test_truncate_short_content(self) -> None:
        """Content shorter than max_content_length is returned unchanged."""
        distiller = BulletDistiller()
        short = "This is short content."
        assert distiller._truncate_content(short) == short

    def test_truncate_long_content(self) -> None:
        """Content longer than 500 chars is truncated."""
        distiller = BulletDistiller()
        long_content = "a " * 300  # 600 chars
        result = distiller._truncate_content(long_content)
        assert len(result) <= 505  # max 500 + small separator/ellipsis

    def test_truncate_at_period(self) -> None:
        """Truncation finds '. ' sentence boundary."""
        distiller = BulletDistiller(ReflectorConfig(max_content_length=50))
        # Build content: ~30 chars sentence + ". " + more chars to exceed 50
        content = "First sentence of the text. Second sentence that pushes past the limit easily."
        result = distiller._truncate_content(content)
        assert result == "First sentence of the text."
        assert len(result) <= 50

    def test_truncate_at_newline(self) -> None:
        r"""Truncation finds '\n' boundary when no period is available."""
        distiller = BulletDistiller(ReflectorConfig(max_content_length=50))
        content = "A line without period ending\nSecond line that extends past the fifty character boundary"
        result = distiller._truncate_content(content)
        assert result == "A line without period ending"
        assert "\n" not in result

    def test_truncate_no_boundary(self) -> None:
        """No sentence boundary found -> hard truncate with '...'."""
        distiller = BulletDistiller(ReflectorConfig(max_content_length=30))
        # Single long word with no boundary separators
        content = "abcdefghij" * 5  # 50 chars, no separators
        result = distiller._truncate_content(content)
        assert result.endswith("...")
        assert len(result) <= 33  # 30 chars + "..."


# -- _truncate_code ----------------------------------------------------------


class TestTruncateCode:
    def test_truncate_code_short(self) -> None:
        """Code with fewer lines than max_code_lines is returned unchanged."""
        distiller = BulletDistiller()
        code = "import os\nimport sys"
        assert distiller._truncate_code(code) == code

    def test_truncate_code_long(self) -> None:
        """Code with more lines than max_code_lines is truncated + '...'."""
        distiller = BulletDistiller()  # default max_code_lines=3
        code = "line1\nline2\nline3\nline4\nline5"
        result = distiller._truncate_code(code)
        assert result == "line1\nline2\nline3\n..."
        assert result.count("\n") == 3  # 3 newlines: between lines + before "..."


# -- _extract_tools ----------------------------------------------------------


class TestExtractTools:
    def test_extract_tools_from_content(self) -> None:
        """Known tool names in content text are extracted."""
        tools = BulletDistiller._extract_tools(
            "Use git and docker to deploy the service.", {}
        )
        assert tools == ["docker", "git"]

    def test_extract_tools_from_context(self) -> None:
        """Tool name from context metadata is included."""
        tools = BulletDistiller._extract_tools(
            "Run the linter on your code.",
            {"tool": "Ruff"},
        )
        assert "ruff" in tools

    def test_extract_tools_from_context_list(self) -> None:
        """Multiple tools from context['tools'] list are included."""
        tools = BulletDistiller._extract_tools(
            "Check the code.",
            {"tools": ["mypy", "black"]},
        )
        assert "mypy" in tools
        assert "black" in tools

    def test_extract_tools_no_tools(self) -> None:
        """Content with no tool references returns empty list."""
        tools = BulletDistiller._extract_tools(
            "The weather is nice today.", {}
        )
        assert tools == []


# -- _extract_entities -------------------------------------------------------


class TestExtractEntities:
    def test_extract_entities_quoted(self) -> None:
        """Backtick-quoted terms are extracted."""
        entities = BulletDistiller._extract_entities(
            "Run `cargo build` and check `main.rs` output."
        )
        assert "cargo build" in entities
        assert "main.rs" in entities

    def test_extract_entities_camelcase(self) -> None:
        """PascalCase / CamelCase identifiers are extracted."""
        entities = BulletDistiller._extract_entities(
            "The BulletDistiller class handles CandidateBullet creation."
        )
        assert "BulletDistiller" in entities
        assert "CandidateBullet" in entities

    def test_extract_entities_dotted(self) -> None:
        """File-like dotted names (config.yaml, main.py) are extracted."""
        entities = BulletDistiller._extract_entities(
            "Edit config.yaml and update main.py to fix the issue."
        )
        assert "config.yaml" in entities
        assert "main.py" in entities

    def test_extract_entities_limit_10(self) -> None:
        """More than 10 entities -> only first 10 (sorted) are returned."""
        # Build content with 15+ backtick-quoted unique entities
        parts = [f"`entity_{chr(65 + i)}`" for i in range(15)]
        content = " ".join(parts)
        entities = BulletDistiller._extract_entities(content)
        assert len(entities) == 10
        # Sorted alphabetically, so first 10 of the sorted set
        assert entities == sorted(entities)

    def test_extract_entities_no_version(self) -> None:
        """Version-like numbers starting with digit (e.g., '2.0') are not extracted."""
        entities = BulletDistiller._extract_entities(
            "Upgrade from version 2.0 to 3.1 of the library."
        )
        # 2.0 and 3.1 should not appear because they start with a digit
        for e in entities:
            assert not e[0].isdigit(), f"Version-like entity extracted: {e}"


# -- distill preserves fields ------------------------------------------------


class TestDistillPreservesFields:
    def test_distill_preserves_section(self) -> None:
        """Section from candidate flows through to bullet."""
        distiller = BulletDistiller()
        for section in [BulletSection.DEBUGGING, BulletSection.WORKFLOW, BulletSection.COMMANDS]:
            candidate = _candidate(section=section)
            bullet = distiller.distill(candidate)
            assert bullet.section == section

    def test_distill_preserves_knowledge_type(self) -> None:
        """Knowledge type from candidate flows through to bullet."""
        distiller = BulletDistiller()
        for kt in [KnowledgeType.PITFALL, KnowledgeType.TRICK, KnowledgeType.PREFERENCE]:
            candidate = _candidate(knowledge_type=kt)
            bullet = distiller.distill(candidate)
            assert bullet.knowledge_type == kt


# -- custom config -----------------------------------------------------------


class TestCustomConfig:
    def test_custom_config(self) -> None:
        """ReflectorConfig(max_content_length=100) limits truncation to 100."""
        cfg = ReflectorConfig(max_content_length=100)
        distiller = BulletDistiller(config=cfg)
        assert distiller._max_content == 100

        long_content = "word " * 50  # 250 chars
        result = distiller._truncate_content(long_content)
        assert len(result) <= 103  # 100 + "..."
