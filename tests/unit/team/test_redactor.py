"""Unit tests for memx.team.redactor — three-layer sanitization engine."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memx.core.privacy.sanitizer import FilteredItem, PrivacySanitizer
from memx.team.config import RedactorConfig
from memx.team.redactor import (
    LLMGeneralizer,
    RedactedResult,
    Redactor,
    ReviewPayload,
    TEAM_PATTERNS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> RedactorConfig:
    """RedactorConfig with defaults (no LLM, no custom patterns)."""
    return RedactorConfig()


@pytest.fixture
def config_with_custom() -> RedactorConfig:
    """RedactorConfig with custom regex patterns."""
    return RedactorConfig(
        custom_patterns=[r"PROJ-\d{4,}", r"secret_value_\w+"],
    )


@pytest.fixture
def config_with_llm() -> RedactorConfig:
    """RedactorConfig with LLM generalization enabled."""
    return RedactorConfig(llm_generalize=True)


@pytest.fixture
def redactor(default_config: RedactorConfig) -> Redactor:
    """Redactor with default config."""
    return Redactor(config=default_config)


@pytest.fixture
def redactor_custom(config_with_custom: RedactorConfig) -> Redactor:
    """Redactor with custom patterns."""
    return Redactor(config=config_with_custom)


# ---------------------------------------------------------------------------
# 1. L1 sanitization with builtin patterns
# ---------------------------------------------------------------------------


class TestL1BuiltinPatterns:
    """L1 sanitization catches core builtin patterns (API keys, etc.)."""

    def test_openai_key_redacted(self, redactor: Redactor) -> None:
        content = "Use key sk-proj-abcdefghij1234567890xyz"
        result = redactor.redact_l1(content)
        assert "<OPENAI_KEY>" in result.clean_content
        assert "sk-proj-" not in result.clean_content
        assert result.was_modified is True
        assert len(result.filtered_items) > 0

    def test_github_token_redacted(self, redactor: Redactor) -> None:
        content = "Token: ghp_aBcDeFgHiJkLmNoPqRsT1234567890"
        result = redactor.redact_l1(content)
        assert "<GITHUB_TOKEN>" in result.clean_content
        assert result.was_modified is True

    def test_aws_key_redacted(self, redactor: Redactor) -> None:
        content = "AWS key: AKIAIOSFODNN7EXAMPLE"
        result = redactor.redact_l1(content)
        assert "<AWS_KEY>" in result.clean_content
        assert result.was_modified is True

    def test_clean_content_unchanged(self, redactor: Redactor) -> None:
        content = "Use pytest to run tests with -v flag"
        result = redactor.redact_l1(content)
        assert result.clean_content == content
        assert result.was_modified is False
        assert result.filtered_items == []

    def test_original_content_preserved(self, redactor: Redactor) -> None:
        content = "Key: sk-proj-abcdefghij1234567890xyz"
        result = redactor.redact_l1(content)
        assert result.original_content == content


# ---------------------------------------------------------------------------
# 2. L1 with custom_patterns (regex)
# ---------------------------------------------------------------------------


class TestL1CustomPatterns:
    """L1 sanitization with user-defined regex custom_patterns."""

    def test_custom_pattern_match(self, redactor_custom: Redactor) -> None:
        content = "Reference PROJ-12345 for details"
        result = redactor_custom.redact_l1(content)
        assert "PROJ-12345" not in result.clean_content
        assert "<REDACTED>" in result.clean_content
        assert result.was_modified is True

    def test_custom_pattern_secret_value(self, redactor_custom: Redactor) -> None:
        content = "Set secret_value_database to 42"
        result = redactor_custom.redact_l1(content)
        assert "secret_value_database" not in result.clean_content
        assert result.was_modified is True

    def test_custom_pattern_no_match(self, redactor_custom: Redactor) -> None:
        content = "PROJ-12 is too short to match"
        result = redactor_custom.redact_l1(content)
        assert result.clean_content == content
        assert result.was_modified is False

    def test_invalid_custom_pattern_ignored(self) -> None:
        """Invalid regex in custom_patterns should not crash."""
        config = RedactorConfig(custom_patterns=[r"[invalid"])
        # Should not raise
        redactor = Redactor(config=config)
        result = redactor.redact_l1("some content")
        assert result.clean_content == "some content"


# ---------------------------------------------------------------------------
# 3. Team-specific patterns (internal IPs, project paths, etc.)
# ---------------------------------------------------------------------------


class TestTeamPatterns:
    """Team-specific patterns for internal infrastructure."""

    def test_internal_ip_10(self, redactor: Redactor) -> None:
        content = "Connect to 10.0.1.42 for the database"
        result = redactor.redact_l1(content)
        assert "[INTERNAL_IP]" in result.clean_content
        assert "10.0.1.42" not in result.clean_content

    def test_internal_ip_192(self, redactor: Redactor) -> None:
        content = "Server at 192.168.1.100"
        result = redactor.redact_l1(content)
        assert "[INTERNAL_IP]" in result.clean_content
        assert "192.168.1.100" not in result.clean_content

    def test_internal_ip_172(self, redactor: Redactor) -> None:
        content = "Gateway 172.16.0.1"
        result = redactor.redact_l1(content)
        assert "[INTERNAL_IP]" in result.clean_content
        assert "172.16.0.1" not in result.clean_content

    def test_public_ip_not_redacted(self, redactor: Redactor) -> None:
        content = "Public IP 8.8.8.8 is Google DNS"
        result = redactor.redact_l1(content)
        assert "8.8.8.8" in result.clean_content

    def test_internal_url(self, redactor: Redactor) -> None:
        content = "See https://internal.company.com/wiki/page for docs"
        result = redactor.redact_l1(content)
        assert "[INTERNAL_URL]" in result.clean_content
        assert "internal.company.com" not in result.clean_content

    def test_project_path(self, redactor: Redactor) -> None:
        # Core sanitizer catches /home/alice -> <USER_PATH> first,
        # so project_path pattern only fires on unsanitized core paths.
        # Use a path form that bypasses core's unix_user_path pattern.
        content = "Code lives in /home/alice/projects/secret-api/src"
        result = redactor.redact_l1(content)
        # Core catches /home/alice, team catches the full project path
        assert "secret-api" not in result.clean_content
        assert result.was_modified is True

    def test_project_path_direct(self) -> None:
        """Test project_path pattern directly with a PrivacySanitizer that has no core patterns."""
        import re
        from memx.team.redactor import TEAM_PATTERNS

        project_patterns = [(n, r, rep) for n, r, rep in TEAM_PATTERNS if "project_path" in n]
        assert len(project_patterns) > 0
        pattern = re.compile(project_patterns[0][1])
        assert pattern.search("/home/user/projects/myapp/src") is not None

    def test_db_connection_string(self, redactor: Redactor) -> None:
        content = "Set DB_URL=postgresql://admin:p4ss@db.internal:5432/mydb"
        result = redactor.redact_l1(content)
        assert "[DB_CONNECTION]" in result.clean_content
        assert "admin" not in result.clean_content
        assert "p4ss" not in result.clean_content

    def test_cloud_arn(self, redactor: Redactor) -> None:
        content = "Use arn:aws:s3:us-east-1:123456789012:mybucket/key"
        result = redactor.redact_l1(content)
        assert "[CLOUD_RESOURCE]" in result.clean_content
        assert "123456789012" not in result.clean_content

    def test_multiple_team_patterns(self, redactor: Redactor) -> None:
        content = "Server 192.168.1.5 at https://staging.example.com/api"
        result = redactor.redact_l1(content)
        assert "[INTERNAL_IP]" in result.clean_content
        assert "[INTERNAL_URL]" in result.clean_content
        assert result.was_modified is True
        assert len(result.filtered_items) >= 2


# ---------------------------------------------------------------------------
# 4. L2 review payload generation with diff
# ---------------------------------------------------------------------------


class TestL2ReviewPayload:
    """L2 prepare_for_review produces correct diff and warnings."""

    def test_diff_generated(self, redactor: Redactor) -> None:
        result = redactor.redact_l1("Key: sk-proj-abcdefghij1234567890xyz")
        review = redactor.prepare_for_review(result)
        assert isinstance(review, ReviewPayload)
        assert len(review.diff_lines) > 0
        assert review.original_content == result.original_content
        assert review.redacted_content == result.clean_content

    def test_no_diff_for_clean_content(self, redactor: Redactor) -> None:
        result = redactor.redact_l1("Nothing sensitive here")
        review = redactor.prepare_for_review(result)
        assert review.diff_lines == []
        assert review.warning is None

    def test_filtered_items_in_payload(self, redactor: Redactor) -> None:
        result = redactor.redact_l1("Token ghp_aBcDeFgHiJkLmNoPqRsT1234567890")
        review = redactor.prepare_for_review(result)
        assert len(review.filtered_items) > 0
        assert review.filtered_items[0].pattern_name == "github_token"

    def test_is_fully_redacted_flag(self, redactor: Redactor) -> None:
        result = RedactedResult(
            original_content="secret",
            clean_content="[REDACTED]",
            was_modified=True,
        )
        review = redactor.prepare_for_review(result)
        assert review.is_fully_redacted is True


# ---------------------------------------------------------------------------
# 5. User edit re-scanning
# ---------------------------------------------------------------------------


class TestUserEditRescanning:
    """apply_user_edits re-runs L1 to catch new sensitive data."""

    def test_clean_edit_accepted(self, redactor: Redactor) -> None:
        original_result = redactor.redact_l1("Key: sk-proj-abcdefghij1234567890xyz")
        edited = "Use a configuration file for API keys"
        new_result = redactor.apply_user_edits(original_result, edited)
        assert new_result.clean_content == edited
        assert new_result.original_content == original_result.original_content

    def test_new_sensitive_data_caught(self, redactor: Redactor) -> None:
        original_result = redactor.redact_l1("Some safe text")
        # User accidentally introduces a new secret in their edit
        edited = "Connect to 192.168.0.1 for the service"
        new_result = redactor.apply_user_edits(original_result, edited)
        assert "[INTERNAL_IP]" in new_result.clean_content
        assert "192.168.0.1" not in new_result.clean_content

    def test_context_summary_preserved(self, redactor: Redactor) -> None:
        result = RedactedResult(
            original_content="original",
            clean_content="clean",
            was_modified=False,
            context_summary="Debugging tip for API auth",
        )
        new_result = redactor.apply_user_edits(result, "safe edit")
        assert new_result.context_summary == "Debugging tip for API auth"


# ---------------------------------------------------------------------------
# 6. Fully redacted content warning
# ---------------------------------------------------------------------------


class TestFullyRedacted:
    """Warning when content is entirely redacted."""

    def test_empty_content_is_fully_redacted(self) -> None:
        result = RedactedResult(
            original_content="secret", clean_content="", was_modified=True
        )
        assert result.is_fully_redacted is True

    def test_only_placeholders_is_fully_redacted(self) -> None:
        result = RedactedResult(
            original_content="secret",
            clean_content="[INTERNAL_IP] [DB_CONNECTION]",
            was_modified=True,
        )
        assert result.is_fully_redacted is True

    def test_mixed_content_not_fully_redacted(self) -> None:
        result = RedactedResult(
            original_content="Use [REDACTED] for auth",
            clean_content="Use [REDACTED] for auth",
            was_modified=True,
        )
        assert result.is_fully_redacted is False

    def test_warning_in_review_payload(self, redactor: Redactor) -> None:
        result = RedactedResult(
            original_content="192.168.1.1",
            clean_content="[INTERNAL_IP]",
            was_modified=True,
        )
        review = redactor.prepare_for_review(result)
        assert review.warning is not None
        assert "fully redacted" in review.warning.lower()

    def test_angle_bracket_placeholders_fully_redacted(self) -> None:
        result = RedactedResult(
            original_content="key",
            clean_content="<OPENAI_KEY>",
            was_modified=True,
        )
        assert result.is_fully_redacted is True


# ---------------------------------------------------------------------------
# 7. L3 LLM generalization (mock LLM)
# ---------------------------------------------------------------------------


class TestL3LLMGeneralization:
    """L3 optional LLM generalization with mock."""

    @pytest.fixture
    def mock_generalizer(self) -> LLMGeneralizer:
        mock = MagicMock(spec=LLMGeneralizer)
        mock.generalize = AsyncMock(
            return_value="Generalized: use env vars for credentials"
        )
        return mock

    @pytest.mark.asyncio
    async def test_l3_generalizes_content(
        self, config_with_llm: RedactorConfig, mock_generalizer: LLMGeneralizer
    ) -> None:
        redactor = Redactor(
            config=config_with_llm, llm_generalizer=mock_generalizer
        )
        result = redactor.redact_l1("Use <REDACTED> for API auth")
        l3_result = await redactor.redact_l3(result)
        assert l3_result.clean_content == "Generalized: use env vars for credentials"
        assert l3_result.was_modified is True
        assert l3_result.original_content == result.original_content
        mock_generalizer.generalize.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_l3_noop_when_disabled(self, default_config: RedactorConfig) -> None:
        redactor = Redactor(config=default_config)
        result = RedactedResult(
            original_content="test",
            clean_content="test",
            was_modified=False,
        )
        l3_result = await redactor.redact_l3(result)
        assert l3_result is result  # same object, no-op

    @pytest.mark.asyncio
    async def test_l3_noop_when_no_generalizer(
        self, config_with_llm: RedactorConfig
    ) -> None:
        redactor = Redactor(config=config_with_llm, llm_generalizer=None)
        result = RedactedResult(
            original_content="test",
            clean_content="test",
            was_modified=False,
        )
        l3_result = await redactor.redact_l3(result)
        assert l3_result is result


# ---------------------------------------------------------------------------
# 8. L3 fallback on error
# ---------------------------------------------------------------------------


class TestL3Fallback:
    """L3 falls back to L1 result when LLM fails."""

    @pytest.mark.asyncio
    async def test_fallback_on_llm_exception(
        self, config_with_llm: RedactorConfig
    ) -> None:
        mock_gen = MagicMock(spec=LLMGeneralizer)
        mock_gen.generalize = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        redactor = Redactor(config=config_with_llm, llm_generalizer=mock_gen)

        result = RedactedResult(
            original_content="test",
            clean_content="sanitized test",
            was_modified=True,
        )
        l3_result = await redactor.redact_l3(result)
        # Should fall back to original result
        assert l3_result is result
        assert l3_result.clean_content == "sanitized test"


# ---------------------------------------------------------------------------
# 9. Finalize output format
# ---------------------------------------------------------------------------


class TestFinalize:
    """finalize() produces correct dict for TeamBullet construction."""

    def test_basic_finalize(self, redactor: Redactor) -> None:
        result = RedactedResult(
            original_content="key: sk-proj-abcdefghij1234567890xyz",
            clean_content="key: <OPENAI_KEY>",
            filtered_items=[
                FilteredItem(pattern_name="openai_key", snippet="sk-proj-...", position=5)
            ],
            was_modified=True,
        )
        output = redactor.finalize(result)
        assert output["content"] == "key: <OPENAI_KEY>"
        assert output["was_redacted"] is True
        assert output["redacted_items_count"] == 1
        assert output["context_summary"] is None

    def test_finalize_with_context_summary(self, redactor: Redactor) -> None:
        result = RedactedResult(
            original_content="original",
            clean_content="clean",
            was_modified=False,
        )
        output = redactor.finalize(result, context_summary="Auth setup tip")
        assert output["context_summary"] == "Auth setup tip"

    def test_finalize_preserves_result_context_summary(
        self, redactor: Redactor
    ) -> None:
        result = RedactedResult(
            original_content="original",
            clean_content="clean",
            was_modified=False,
            context_summary="From result",
        )
        output = redactor.finalize(result)
        assert output["context_summary"] == "From result"

    def test_finalize_kwarg_overrides_result_summary(
        self, redactor: Redactor
    ) -> None:
        result = RedactedResult(
            original_content="original",
            clean_content="clean",
            was_modified=False,
            context_summary="From result",
        )
        output = redactor.finalize(result, context_summary="Override")
        assert output["context_summary"] == "Override"


# ---------------------------------------------------------------------------
# 10. Empty content handling
# ---------------------------------------------------------------------------


class TestEmptyContent:
    """Edge case: empty or whitespace-only content."""

    def test_empty_string(self, redactor: Redactor) -> None:
        result = redactor.redact_l1("")
        assert result.clean_content == ""
        assert result.was_modified is False
        assert result.filtered_items == []

    def test_whitespace_only(self, redactor: Redactor) -> None:
        result = redactor.redact_l1("   ")
        # Whitespace should pass through (no patterns match pure whitespace)
        assert result.clean_content == "   "
        assert result.was_modified is False


# ---------------------------------------------------------------------------
# 11. context_summary attachment
# ---------------------------------------------------------------------------


class TestContextSummary:
    """context_summary flows through the pipeline correctly."""

    def test_context_summary_on_redacted_result(self) -> None:
        result = RedactedResult(
            original_content="test",
            clean_content="test",
            was_modified=False,
            context_summary="Debugging tip",
        )
        assert result.context_summary == "Debugging tip"

    def test_context_summary_default_none(self) -> None:
        result = RedactedResult(
            original_content="test",
            clean_content="test",
            was_modified=False,
        )
        assert result.context_summary is None

    def test_context_summary_through_finalize(self, redactor: Redactor) -> None:
        result = redactor.redact_l1("Safe content about debugging")
        output = redactor.finalize(result, context_summary="Debug workflow tip")
        assert output["context_summary"] == "Debug workflow tip"
        assert output["content"] == "Safe content about debugging"


# ---------------------------------------------------------------------------
# Injected sanitizer test
# ---------------------------------------------------------------------------


class TestInjectedSanitizer:
    """Verify that a pre-configured PrivacySanitizer can be injected."""

    def test_custom_sanitizer_used(self) -> None:
        sanitizer = PrivacySanitizer(custom_patterns=[r"INJECT-\d+"])
        config = RedactorConfig()
        redactor = Redactor(config=config, sanitizer=sanitizer)
        result = redactor.redact_l1("See INJECT-9999 for details")
        assert "INJECT-9999" not in result.clean_content
        assert result.was_modified is True
