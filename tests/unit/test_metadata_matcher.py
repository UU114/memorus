"""Unit tests for memx.engines.generator.metadata_matcher -- MetadataMatcher (L3)."""

from __future__ import annotations

from memx.engines.generator.metadata_matcher import (
    MatchResult,
    MetadataInfo,
    MetadataMatcher,
)

# -- Helper factories --------------------------------------------------------


def _meta(
    tools: list[str] | None = None,
    entities: list[str] | None = None,
    tags: list[str] | None = None,
) -> MetadataInfo:
    return MetadataInfo(
        related_tools=tools or [],
        key_entities=entities or [],
        tags=tags or [],
    )


# -- MetadataInfo defaults ---------------------------------------------------


class TestMetadataInfoDefaults:
    """MetadataInfo fields default to empty lists."""

    def test_zero_arg_construction(self) -> None:
        m = MetadataInfo()
        assert m.related_tools == []
        assert m.key_entities == []
        assert m.tags == []

    def test_list_isolation(self) -> None:
        """Different instances should not share list state."""
        a = MetadataInfo()
        b = MetadataInfo()
        # frozen=True prevents mutation, but ensure distinct defaults
        assert a.related_tools is not b.related_tools


# -- MatchResult defaults ---------------------------------------------------


class TestMatchResultDefaults:
    """MatchResult stores per-field scores and matched items."""

    def test_zero_score_result(self) -> None:
        r = MatchResult(score=0.0, tools_score=0.0, entities_score=0.0, tags_score=0.0)
        assert r.score == 0.0
        assert r.matched_tools == []
        assert r.matched_entities == []
        assert r.matched_tags == []

    def test_full_score_result(self) -> None:
        r = MatchResult(
            score=10.0,
            tools_score=4.0,
            entities_score=3.0,
            tags_score=3.0,
            matched_tools=["git"],
            matched_entities=["React"],
            matched_tags=["python"],
        )
        assert r.score == 10.0
        assert r.matched_tools == ["git"]


# -- related_tools prefix matching ------------------------------------------


class TestToolsPrefixMatching:
    """related_tools: prefix matching ('git' matches 'git-rebase', 'git-stash')."""

    def test_exact_tool_name(self) -> None:
        """Query 'git' matches tool 'git' exactly."""
        matcher = MetadataMatcher()
        result = matcher.match("git", _meta(tools=["git"]))
        assert result.tools_score == 4.0
        assert "git" in result.matched_tools

    def test_prefix_match_hyphenated(self) -> None:
        """Query 'git' matches tools 'git-rebase' and 'git-stash' by prefix."""
        matcher = MetadataMatcher()
        result = matcher.match("git", _meta(tools=["git-rebase", "git-stash", "docker"]))
        assert result.tools_score == 4.0
        assert "git-rebase" in result.matched_tools
        assert "git-stash" in result.matched_tools
        assert "docker" not in result.matched_tools

    def test_no_match(self) -> None:
        """Query 'npm' does not match tool 'cargo'."""
        matcher = MetadataMatcher()
        result = matcher.match("npm", _meta(tools=["cargo"]))
        assert result.tools_score == 0.0
        assert result.matched_tools == []

    def test_case_insensitive(self) -> None:
        """Prefix matching is case-insensitive."""
        matcher = MetadataMatcher()
        result = matcher.match("GIT", _meta(tools=["git-rebase"]))
        assert result.tools_score == 4.0
        assert "git-rebase" in result.matched_tools

    def test_case_insensitive_candidate(self) -> None:
        """Candidate in uppercase also matches lowercase query."""
        matcher = MetadataMatcher()
        result = matcher.match("docker", _meta(tools=["Docker-Compose"]))
        assert result.tools_score == 4.0
        assert "Docker-Compose" in result.matched_tools

    def test_empty_tools(self) -> None:
        """Empty related_tools -> 0 score."""
        matcher = MetadataMatcher()
        result = matcher.match("git", _meta(tools=[]))
        assert result.tools_score == 0.0

    def test_partial_prefix_no_false_positive(self) -> None:
        """'car' should NOT match 'cargo' as a prefix (it should match as prefix)."""
        matcher = MetadataMatcher()
        result = matcher.match("car", _meta(tools=["cargo"]))
        # "car" IS a prefix of "cargo", so this SHOULD match
        assert result.tools_score == 4.0
        assert "cargo" in result.matched_tools

    def test_substring_not_prefix(self) -> None:
        """'argo' should NOT match 'cargo' (substring but not prefix)."""
        matcher = MetadataMatcher()
        result = matcher.match("argo", _meta(tools=["cargo"]))
        assert result.tools_score == 0.0
        assert result.matched_tools == []


# -- key_entities prefix matching -------------------------------------------


class TestEntitiesPrefixMatching:
    """key_entities: prefix matching ('React' matches 'ReactDOM')."""

    def test_exact_entity_name(self) -> None:
        """Query 'react' matches entity 'React' exactly."""
        matcher = MetadataMatcher()
        result = matcher.match("react", _meta(entities=["React"]))
        assert result.entities_score == 3.0
        assert "React" in result.matched_entities

    def test_prefix_match(self) -> None:
        """Query 'React' matches 'ReactDOM' and 'React' by prefix."""
        matcher = MetadataMatcher()
        result = matcher.match("React", _meta(entities=["React", "ReactDOM", "Vue"]))
        assert result.entities_score == 3.0
        assert "React" in result.matched_entities
        assert "ReactDOM" in result.matched_entities
        assert "Vue" not in result.matched_entities

    def test_no_match(self) -> None:
        """Query 'Angular' does not match entity 'React'."""
        matcher = MetadataMatcher()
        result = matcher.match("Angular", _meta(entities=["React"]))
        assert result.entities_score == 0.0

    def test_case_insensitive(self) -> None:
        """Entity matching is case-insensitive."""
        matcher = MetadataMatcher()
        result = matcher.match("react", _meta(entities=["ReactDOM"]))
        assert result.entities_score == 3.0
        assert "ReactDOM" in result.matched_entities


# -- tags exact matching ----------------------------------------------------


class TestTagsExactMatching:
    """tags: exact matching ('python' matches 'python' but not 'python3')."""

    def test_exact_match(self) -> None:
        """Query 'python' matches tag 'python'."""
        matcher = MetadataMatcher()
        result = matcher.match("python", _meta(tags=["python"]))
        assert result.tags_score == 3.0
        assert "python" in result.matched_tags

    def test_no_prefix_match(self) -> None:
        """Tags use exact match: 'python' does NOT match 'python3'."""
        matcher = MetadataMatcher()
        result = matcher.match("python", _meta(tags=["python3"]))
        assert result.tags_score == 0.0
        assert result.matched_tags == []

    def test_case_insensitive(self) -> None:
        """Tag matching is case-insensitive."""
        matcher = MetadataMatcher()
        result = matcher.match("Python", _meta(tags=["python"]))
        assert result.tags_score == 3.0
        assert "python" in result.matched_tags

    def test_multiple_tags(self) -> None:
        """Multiple matching tags still contribute up to tags_score."""
        matcher = MetadataMatcher()
        result = matcher.match(
            "python rust", _meta(tags=["python", "rust", "java"])
        )
        assert result.tags_score == 3.0
        assert "python" in result.matched_tags
        assert "rust" in result.matched_tags
        assert "java" not in result.matched_tags

    def test_empty_tags(self) -> None:
        """Empty tags list -> 0 score."""
        matcher = MetadataMatcher()
        result = matcher.match("python", _meta(tags=[]))
        assert result.tags_score == 0.0

    def test_no_match(self) -> None:
        """Query 'go' does not match tag 'rust'."""
        matcher = MetadataMatcher()
        result = matcher.match("go", _meta(tags=["rust"]))
        assert result.tags_score == 0.0


# -- Combined scoring -------------------------------------------------------


class TestCombinedScoring:
    """Total score is the sum of per-field scores, clamped to [0, 10]."""

    def test_all_fields_match(self) -> None:
        """All three field types match -> full 10.0 score."""
        matcher = MetadataMatcher()
        meta = _meta(tools=["git"], entities=["React"], tags=["python"])
        result = matcher.match("git react python", meta)
        assert result.score == 10.0
        assert result.tools_score == 4.0
        assert result.entities_score == 3.0
        assert result.tags_score == 3.0

    def test_only_tools_match(self) -> None:
        """Only tools match -> 4.0 total."""
        matcher = MetadataMatcher()
        result = matcher.match("git", _meta(tools=["git"], entities=["Vue"], tags=["java"]))
        assert result.score == 4.0
        assert result.tools_score == 4.0
        assert result.entities_score == 0.0
        assert result.tags_score == 0.0

    def test_only_entities_match(self) -> None:
        """Only entities match -> 3.0 total."""
        matcher = MetadataMatcher()
        result = matcher.match("react", _meta(tools=["cargo"], entities=["React"], tags=["java"]))
        assert result.score == 3.0

    def test_only_tags_match(self) -> None:
        """Only tags match -> 3.0 total."""
        matcher = MetadataMatcher()
        result = matcher.match("python", _meta(tools=["cargo"], entities=["Vue"], tags=["python"]))
        assert result.score == 3.0

    def test_tools_and_entities_match(self) -> None:
        """Tools + entities match -> 7.0 total."""
        matcher = MetadataMatcher()
        result = matcher.match(
            "git react", _meta(tools=["git"], entities=["React"], tags=["java"])
        )
        assert result.score == 7.0

    def test_no_matches_at_all(self) -> None:
        """No fields match -> 0.0 total."""
        matcher = MetadataMatcher()
        result = matcher.match("xyz", _meta(tools=["git"], entities=["React"], tags=["python"]))
        assert result.score == 0.0


# -- Empty metadata ----------------------------------------------------------


class TestEmptyMetadata:
    """Empty metadata fields always produce 0 score."""

    def test_all_empty(self) -> None:
        matcher = MetadataMatcher()
        result = matcher.match("git", MetadataInfo())
        assert result.score == 0.0

    def test_empty_query(self) -> None:
        matcher = MetadataMatcher()
        result = matcher.match("", _meta(tools=["git"], entities=["React"], tags=["python"]))
        assert result.score == 0.0

    def test_whitespace_query(self) -> None:
        matcher = MetadataMatcher()
        result = matcher.match("   ", _meta(tools=["git"]))
        assert result.score == 0.0


# -- Custom score weights ---------------------------------------------------


class TestCustomWeights:
    """Custom score weights change per-field contributions."""

    def test_custom_tools_score(self) -> None:
        matcher = MetadataMatcher(tools_score=6.0, entities_score=2.0, tags_score=2.0)
        result = matcher.match("git", _meta(tools=["git"]))
        assert result.tools_score == 6.0
        assert result.score == 6.0

    def test_custom_all_scores(self) -> None:
        matcher = MetadataMatcher(tools_score=5.0, entities_score=3.0, tags_score=2.0)
        meta = _meta(tools=["git"], entities=["React"], tags=["python"])
        result = matcher.match("git react python", meta)
        assert result.score == 10.0  # Clamped to 10.0

    def test_score_clamped_to_10(self) -> None:
        """Even with large custom weights, total is clamped to 10.0."""
        matcher = MetadataMatcher(tools_score=10.0, entities_score=10.0, tags_score=10.0)
        meta = _meta(tools=["git"], entities=["React"], tags=["python"])
        result = matcher.match("git react python", meta)
        assert result.score == 10.0

    def test_max_score_property(self) -> None:
        matcher = MetadataMatcher(tools_score=5.0, entities_score=3.0, tags_score=2.0)
        assert matcher.max_score == 10.0


# -- Tokenizer edge cases ---------------------------------------------------


class TestTokenizer:
    """Tokenizer splits on whitespace and common punctuation."""

    def test_comma_separated(self) -> None:
        matcher = MetadataMatcher()
        result = matcher.match("git, react", _meta(tools=["git"], entities=["React"]))
        assert result.tools_score == 4.0
        assert result.entities_score == 3.0

    def test_semicolon_separated(self) -> None:
        matcher = MetadataMatcher()
        result = matcher.match("git; python", _meta(tools=["git"], tags=["python"]))
        assert result.tools_score == 4.0
        assert result.tags_score == 3.0

    def test_multi_word_query(self) -> None:
        """Multi-word query with only some tokens matching."""
        matcher = MetadataMatcher()
        result = matcher.match(
            "how to use git rebase", _meta(tools=["git-rebase", "docker"])
        )
        assert result.tools_score == 4.0
        assert "git-rebase" in result.matched_tools


# -- Case-insensitivity comprehensive tests ---------------------------------


class TestCaseInsensitivity:
    """All matching is case-insensitive."""

    def test_uppercase_query_lowercase_metadata(self) -> None:
        matcher = MetadataMatcher()
        result = matcher.match("GIT", _meta(tools=["git"]))
        assert result.tools_score == 4.0

    def test_lowercase_query_uppercase_metadata(self) -> None:
        matcher = MetadataMatcher()
        result = matcher.match("react", _meta(entities=["REACTDOM"]))
        assert result.entities_score == 3.0

    def test_mixed_case_tags(self) -> None:
        matcher = MetadataMatcher()
        result = matcher.match("Python", _meta(tags=["PYTHON"]))
        assert result.tags_score == 3.0
