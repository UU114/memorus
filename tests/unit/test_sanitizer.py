"""Unit tests for memx.privacy.sanitizer -- PrivacySanitizer and privacy patterns."""

from __future__ import annotations

import re

import pytest

from memx.core.privacy.patterns import BUILTIN_PATTERNS
from memx.core.privacy.sanitizer import FilteredItem, PrivacySanitizer, SanitizeResult


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def sanitizer() -> PrivacySanitizer:
    """Default PrivacySanitizer with no custom patterns."""
    return PrivacySanitizer()


# ── API Key Tests ─────────────────────────────────────────────────────


class TestOpenAIKey:
    def test_openai_key(self, sanitizer: PrivacySanitizer) -> None:
        """OpenAI sk-proj-... key is redacted."""
        text = "my key is sk-proj-abc123xyz456uvw789def012ghi"
        result = sanitizer.sanitize(text)
        assert "<OPENAI_KEY>" in result.clean_content
        assert "sk-proj-" not in result.clean_content
        assert result.was_modified is True

    def test_openai_key_plain_sk(self, sanitizer: PrivacySanitizer) -> None:
        """OpenAI sk-... key without proj prefix is redacted."""
        text = "key=sk-abcdefghijklmnopqrstuvwxyz"
        result = sanitizer.sanitize(text)
        assert "<OPENAI_KEY>" in result.clean_content

    def test_openai_key_normal_text(self, sanitizer: PrivacySanitizer) -> None:
        """Normal words like 'skill-set' should NOT be matched."""
        text = "She has a great skill-set for the job."
        result = sanitizer.sanitize(text)
        assert result.clean_content == text
        assert result.was_modified is False


class TestAnthropicKey:
    def test_anthropic_key(self, sanitizer: PrivacySanitizer) -> None:
        """Anthropic sk-ant-api03-... key is redacted."""
        text = "export KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234"
        result = sanitizer.sanitize(text)
        assert "<ANTHROPIC_KEY>" in result.clean_content
        assert "sk-ant-" not in result.clean_content
        assert result.was_modified is True

    def test_anthropic_key_normal_text(self, sanitizer: PrivacySanitizer) -> None:
        """Short text like 'skat' should NOT be matched."""
        text = "I like playing skat with friends."
        result = sanitizer.sanitize(text)
        assert result.clean_content == text
        assert result.was_modified is False


class TestGitHubToken:
    def test_github_token_ghp(self, sanitizer: PrivacySanitizer) -> None:
        """GitHub personal access token ghp_ is redacted."""
        text = "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTuvwx in env"
        result = sanitizer.sanitize(text)
        assert "<GITHUB_TOKEN>" in result.clean_content
        assert "ghp_ABCDEF" not in result.clean_content

    def test_github_token_gho(self, sanitizer: PrivacySanitizer) -> None:
        """GitHub OAuth token gho_ is redacted."""
        text = "auth: gho_ABCDEFGHIJKLMNOPQRSTuvwx"
        result = sanitizer.sanitize(text)
        assert "<GITHUB_TOKEN>" in result.clean_content

    def test_github_pat(self, sanitizer: PrivacySanitizer) -> None:
        """GitHub fine-grained PAT github_pat_ is redacted."""
        text = "GITHUB_TOKEN=github_pat_ABCDEFGHIJKLMNOPQRSTuvwx1234"
        result = sanitizer.sanitize(text)
        assert "<GITHUB_TOKEN>" in result.clean_content


class TestAWSKey:
    def test_aws_key(self, sanitizer: PrivacySanitizer) -> None:
        """AWS access key AKIA... is redacted."""
        text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
        result = sanitizer.sanitize(text)
        assert "<AWS_KEY>" in result.clean_content
        assert "AKIAIOSFODNN7EXAMPLE" not in result.clean_content

    def test_aws_key_normal_text(self, sanitizer: PrivacySanitizer) -> None:
        """Short AKIA prefix without full 20 chars should NOT match."""
        text = "The word AKIATEST is not a valid key."
        result = sanitizer.sanitize(text)
        # AKIATEST is only 8 chars total (AKIA + 4), need AKIA + 16 = 20
        assert "AKIATEST" in result.clean_content


class TestBearerToken:
    def test_bearer_token(self, sanitizer: PrivacySanitizer) -> None:
        """Bearer JWT token is redacted."""
        # Realistic JWT: header.payload.signature
        jwt = (
            "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9"
            ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
            ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        )
        text = f"Authorization: Bearer {jwt}"
        result = sanitizer.sanitize(text)
        assert "<BEARER_TOKEN>" in result.clean_content
        assert "eyJ" not in result.clean_content
        assert result.was_modified is True


# ── Password / Secret Field Tests ─────────────────────────────────────


class TestPasswordSecretFields:
    def test_password_field(self, sanitizer: PrivacySanitizer) -> None:
        """password=value is redacted."""
        text = "config: password=mySuper$ecretP4ss!"
        result = sanitizer.sanitize(text)
        assert "password=<REDACTED>" in result.clean_content
        assert "mySuper$ecretP4ss!" not in result.clean_content

    def test_secret_field(self, sanitizer: PrivacySanitizer) -> None:
        """secret=value is redacted."""
        text = "secret=abc123def456"
        result = sanitizer.sanitize(text)
        assert "secret=<REDACTED>" in result.clean_content
        assert "abc123def456" not in result.clean_content

    def test_api_key_param(self, sanitizer: PrivacySanitizer) -> None:
        """api_key=value is redacted."""
        text = "curl -H 'api_key=xyz789_secret_value'"
        result = sanitizer.sanitize(text)
        assert "api_key=<REDACTED>" in result.clean_content
        assert "xyz789_secret_value" not in result.clean_content

    def test_token_field(self, sanitizer: PrivacySanitizer) -> None:
        """token=value is redacted via password_field pattern."""
        text = "token=some_token_value_here"
        result = sanitizer.sanitize(text)
        assert "token=<REDACTED>" in result.clean_content


# ── Path Tests ────────────────────────────────────────────────────────


class TestPathSanitization:
    def test_windows_path(self, sanitizer: PrivacySanitizer) -> None:
        r"""C:\Users\JohnDoe is redacted to <USER_PATH>."""
        text = r"Loading config from C:\Users\JohnDoe\AppData\settings.json"
        result = sanitizer.sanitize(text)
        assert "<USER_PATH>" in result.clean_content
        assert "JohnDoe" not in result.clean_content
        assert result.was_modified is True

    def test_unix_home_path(self, sanitizer: PrivacySanitizer) -> None:
        """/home/johndoe is redacted to <USER_PATH>."""
        text = "Reading /home/johndoe/.config/app.yaml"
        result = sanitizer.sanitize(text)
        assert "<USER_PATH>" in result.clean_content
        assert "johndoe" not in result.clean_content

    def test_unix_users_path(self, sanitizer: PrivacySanitizer) -> None:
        """/Users/johndoe is redacted to <USER_PATH>."""
        text = "Path: /Users/johndoe/Documents/project"
        result = sanitizer.sanitize(text)
        assert "<USER_PATH>" in result.clean_content
        assert "johndoe" not in result.clean_content

    def test_system_path_not_matched(self, sanitizer: PrivacySanitizer) -> None:
        """/usr/bin/python should NOT be redacted."""
        text = "Using /usr/bin/python3 as interpreter."
        result = sanitizer.sanitize(text)
        assert result.clean_content == text
        assert result.was_modified is False


# ── Private Key / DB URL Tests ────────────────────────────────────────


class TestPrivateKey:
    def test_private_key_block(self, sanitizer: PrivacySanitizer) -> None:
        """PEM private key block is fully redacted."""
        key_block = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF5PiGLaDZkiz\n"
            "BGhHKJ7mPkBR4kDgGM4Hxn2F5+HShT9dG3nEpf1Z5DUxTMi\n"
            "-----END RSA PRIVATE KEY-----"
        )
        text = f"Here is the key:\n{key_block}\nEnd of key."
        result = sanitizer.sanitize(text)
        assert "<PRIVATE_KEY>" in result.clean_content
        assert "BEGIN RSA PRIVATE KEY" not in result.clean_content
        assert "MIIEpAIBAAK" not in result.clean_content
        assert result.was_modified is True

    def test_ec_private_key(self, sanitizer: PrivacySanitizer) -> None:
        """EC private key variant is also detected."""
        text = (
            "-----BEGIN EC PRIVATE KEY-----\n"
            "MHQCAQEEIODaqlNgTnxhC/TTkWOF2LFgMCINXwS98jnyp8\n"
            "-----END EC PRIVATE KEY-----"
        )
        result = sanitizer.sanitize(text)
        assert "<PRIVATE_KEY>" in result.clean_content


class TestDatabaseURL:
    def test_db_url(self, sanitizer: PrivacySanitizer) -> None:
        """Postgres URL credentials are redacted."""
        text = "DATABASE_URL=postgres://admin:s3cretP@ss@db.example.com:5432/mydb"
        result = sanitizer.sanitize(text)
        assert "<REDACTED>:<REDACTED>@" in result.clean_content
        assert "admin" not in result.clean_content
        assert "s3cretP@ss" not in result.clean_content
        assert "db.example.com" in result.clean_content

    def test_mysql_url(self, sanitizer: PrivacySanitizer) -> None:
        """MySQL URL credentials are redacted."""
        text = "mysql://root:password123@localhost/testdb"
        result = sanitizer.sanitize(text)
        assert "mysql://<REDACTED>:<REDACTED>@" in result.clean_content

    def test_mongodb_url(self, sanitizer: PrivacySanitizer) -> None:
        """MongoDB URL credentials are redacted."""
        text = "mongodb://dbuser:dbpass@mongo.host.com:27017/admin"
        result = sanitizer.sanitize(text)
        assert "mongodb://<REDACTED>:<REDACTED>@" in result.clean_content


# ── Composite / Edge Case Tests ───────────────────────────────────────


class TestCompositeAndEdgeCases:
    def test_multiple_secrets(self, sanitizer: PrivacySanitizer) -> None:
        """Multiple different secrets in one text are all redacted."""
        text = (
            "Config:\n"
            "  OPENAI_KEY=sk-proj-abcdefghijklmnopqrstuvwxyz\n"
            "  password=hunter2\n"
            "  DB=postgres://user:pass@host/db\n"
        )
        result = sanitizer.sanitize(text)
        assert "<OPENAI_KEY>" in result.clean_content
        assert "password=<REDACTED>" in result.clean_content
        assert "<REDACTED>:<REDACTED>@" in result.clean_content
        assert result.was_modified is True
        assert len(result.filtered_items) >= 3

    def test_clean_content_unchanged(self, sanitizer: PrivacySanitizer) -> None:
        """Normal text without secrets is returned unchanged."""
        text = "The quick brown fox jumps over the lazy dog."
        result = sanitizer.sanitize(text)
        assert result.clean_content == text
        assert result.was_modified is False
        assert result.filtered_items == []

    def test_empty_content(self, sanitizer: PrivacySanitizer) -> None:
        """Empty string returns empty result."""
        result = sanitizer.sanitize("")
        assert result.clean_content == ""
        assert result.was_modified is False
        assert result.filtered_items == []

    def test_none_like_empty(self, sanitizer: PrivacySanitizer) -> None:
        """Empty-ish content does not crash."""
        result = sanitizer.sanitize("")
        assert result.clean_content == ""


# ── Custom Pattern Tests ──────────────────────────────────────────────


class TestCustomPatterns:
    def test_custom_patterns(self) -> None:
        """Custom regex pattern is applied after builtins."""
        sanitizer = PrivacySanitizer(custom_patterns=[r"INTERNAL-\d{6}"])
        text = "Reference: INTERNAL-123456 is classified."
        result = sanitizer.sanitize(text)
        assert "<REDACTED>" in result.clean_content
        assert "INTERNAL-123456" not in result.clean_content
        assert result.was_modified is True

    def test_custom_pattern_invalid_regex(self) -> None:
        """Invalid custom regex logs a warning but doesn't crash."""
        # Should not raise -- bad pattern is skipped
        sanitizer = PrivacySanitizer(custom_patterns=[r"(unclosed"])
        text = "hello world"
        result = sanitizer.sanitize(text)
        assert result.clean_content == text

    def test_builtin_patterns_not_removable(self) -> None:
        """Builtin patterns are always present even with custom patterns."""
        sanitizer = PrivacySanitizer(custom_patterns=[r"CUSTOM_MATCH"])
        # Builtins should still work
        text = "sk-proj-abcdefghijklmnopqrstuvwxyz"
        result = sanitizer.sanitize(text)
        assert "<OPENAI_KEY>" in result.clean_content

        # Verify the builtin count is correct
        builtin_count = len(BUILTIN_PATTERNS)
        builtin_names_in_sanitizer = [
            name for name, _, _ in sanitizer._patterns if not name.startswith("custom_")
        ]
        assert len(builtin_names_in_sanitizer) == builtin_count

    def test_multiple_custom_patterns(self) -> None:
        """Multiple custom patterns all work."""
        sanitizer = PrivacySanitizer(
            custom_patterns=[r"SECRET-\w+", r"CONFIDENTIAL-\d+"]
        )
        text = "SECRET-abc123 and CONFIDENTIAL-9999"
        result = sanitizer.sanitize(text)
        assert "SECRET-abc123" not in result.clean_content
        assert "CONFIDENTIAL-9999" not in result.clean_content
        assert result.clean_content.count("<REDACTED>") == 2


# ── FilteredItem / SanitizeResult Tests ───────────────────────────────


class TestFilteredItems:
    def test_filtered_items_count(self, sanitizer: PrivacySanitizer) -> None:
        """Each matched secret produces a FilteredItem."""
        text = (
            "ghp_ABCDEFGHIJKLMNOPQRSTuvwx "
            "AKIAIOSFODNN7EXAMPLE "
            "password=hunter2"
        )
        result = sanitizer.sanitize(text)
        # At least 3 items: github_token, aws_key, password_field
        assert len(result.filtered_items) >= 3
        names = [item.pattern_name for item in result.filtered_items]
        assert "github_token" in names
        assert "aws_access_key" in names

    def test_filtered_item_has_position(self, sanitizer: PrivacySanitizer) -> None:
        """FilteredItem records the match position in the (current) string."""
        text = "key: ghp_ABCDEFGHIJKLMNOPQRSTuvwx"
        result = sanitizer.sanitize(text)
        gh_items = [i for i in result.filtered_items if i.pattern_name == "github_token"]
        assert len(gh_items) == 1
        assert gh_items[0].position >= 0

    def test_filtered_item_snippet_truncated(self, sanitizer: PrivacySanitizer) -> None:
        """Long secrets are truncated in the snippet field."""
        text = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789ABCD"
        result = sanitizer.sanitize(text)
        items = [i for i in result.filtered_items if i.pattern_name == "openai_key"]
        assert len(items) >= 1
        snippet = items[0].snippet
        # Should be truncated: first 8 + "..." + last 4
        assert "..." in snippet
        assert len(snippet) <= 15  # 8 + 3 + 4 = 15


# ── Truncation Helper Tests ───────────────────────────────────────────


class TestTruncateMatch:
    def test_truncate_match_short(self) -> None:
        """Short strings (<=12 chars) are truncated to first 4 + '...'."""
        result = PrivacySanitizer._truncate_match("abcdefgh")
        assert result == "abcd..."

    def test_truncate_match_exactly_12(self) -> None:
        """12-char string uses the short path."""
        result = PrivacySanitizer._truncate_match("123456789012")
        assert result == "1234..."

    def test_truncate_match_long(self) -> None:
        """Long strings (>12 chars) are truncated to first 8 + '...' + last 4."""
        result = PrivacySanitizer._truncate_match("abcdefghijklmnop")
        assert result == "abcdefgh...mnop"

    def test_truncate_match_13_chars(self) -> None:
        """13-char string uses the long path."""
        result = PrivacySanitizer._truncate_match("1234567890123")
        assert result == "12345678...0123"

    def test_truncate_match_very_short(self) -> None:
        """Very short string (< 4 chars)."""
        result = PrivacySanitizer._truncate_match("ab")
        assert result == "ab..."


# ── Pattern Ordering Tests ────────────────────────────────────────────


class TestPatternOrdering:
    def test_anthropic_before_openai(self, sanitizer: PrivacySanitizer) -> None:
        """sk-ant-... should be matched as Anthropic, not OpenAI."""
        text = "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234"
        result = sanitizer.sanitize(text)
        assert "<ANTHROPIC_KEY>" in result.clean_content
        # Should NOT also produce an OpenAI match on the cleaned text
        assert "<OPENAI_KEY>" not in result.clean_content

    def test_builtin_patterns_are_compiled_regex(self) -> None:
        """All builtin patterns compile without error."""
        for name, regex_str, _replacement in BUILTIN_PATTERNS:
            compiled = re.compile(regex_str)
            assert compiled is not None, f"Pattern '{name}' failed to compile"


# ── SanitizeResult Dataclass Tests ────────────────────────────────────


class TestSanitizeResultDataclass:
    def test_default_values(self) -> None:
        """SanitizeResult defaults are correct."""
        sr = SanitizeResult(clean_content="test")
        assert sr.clean_content == "test"
        assert sr.filtered_items == []
        assert sr.was_modified is False

    def test_with_items(self) -> None:
        """SanitizeResult stores items and modification flag."""
        item = FilteredItem(pattern_name="test", snippet="snip...", position=0)
        sr = SanitizeResult(
            clean_content="redacted",
            filtered_items=[item],
            was_modified=True,
        )
        assert len(sr.filtered_items) == 1
        assert sr.was_modified is True


class TestFilteredItemDataclass:
    def test_construction(self) -> None:
        """FilteredItem stores all fields."""
        fi = FilteredItem(pattern_name="openai_key", snippet="sk-proj-..._xyz", position=42)
        assert fi.pattern_name == "openai_key"
        assert fi.snippet == "sk-proj-..._xyz"
        assert fi.position == 42
