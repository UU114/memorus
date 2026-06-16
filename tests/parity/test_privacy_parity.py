"""Cross-language privacy (PII) parity suite.

Asserts that the Python ``PrivacySanitizer`` redacts every secret type that the
Rust ``PrivacyEngine`` redacts (and vice-versa) -- the UNION mandated by the
2026-06-01 divergence audit (``doc/optimization-roadmap.md`` PRIVACY track).
The README advertises a multi-layer PII filter, so any gap is a security defect.

Design notes
------------
* The Rust ``PrivacyEngine`` is NOT exposed through the ``memorus_r`` Python
  binding (only ``Memory`` / ``AsyncMemory`` are). We therefore cannot call the
  Rust sanitizer directly from Python. Instead this suite:
    1. Drives the Python sanitizer over a fixture containing EVERY secret type
       and asserts the secret *body* is gone (not merely that a token appears).
    2. Encodes the canonical per-secret redaction contract (the union spec) as
       ``EXPECTED`` below. The Rust side is independently verified by the
       in-crate tests in ``memorus-ace/src/privacy/mod.rs`` (one test per new
       secret type), so the two languages are pinned to the same contract from
       both ends.
* Replacement TOKEN strings legitimately differ between languages by existing
  convention -- Rust uses ``[REDACTED:X]`` and Python uses ``<X>``. Parity is
  about *which secret types are redacted* and *that the secret body is gone*,
  NOT about byte-identical replacement tokens. We assert the Python token where
  it is load-bearing and record the Rust token alongside for documentation.
"""

from __future__ import annotations

import pytest

from memorus.core.privacy.sanitizer import PrivacySanitizer

# ---------------------------------------------------------------------------
# Per-secret-type parity contract.
#
# Each entry: a unique key -> (input_text, secret_body_that_must_disappear,
#   python_replacement_token, rust_replacement_token).
# ``secret_body`` is a substring that MUST NOT survive sanitization (the actual
# secret). ``python_token`` is asserted to appear in the Python output. The
# Rust token is documentation of the canonical Rust behaviour (verified by the
# Rust in-crate tests), kept here so the two contracts live side by side.
# ---------------------------------------------------------------------------
EXPECTED: dict[str, tuple[str, str, str, str]] = {
    # provider API keys
    "openai_plain": (
        "key=sk-abcdefghijklmnopqrstuvwxyz",
        "sk-abcdefghijklmnopqrstuvwxyz",
        "<OPENAI_KEY>",
        "[REDACTED:API_KEY]",
    ),
    "openai_proj": (
        "key=sk-proj-abc_def-ghi123jkl456mno789pqr",
        "sk-proj-abc_def-ghi123jkl456mno789pqr",
        "<OPENAI_KEY>",
        "[REDACTED:API_KEY]",
    ),
    "anthropic": (
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234",
        "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234",
        "<ANTHROPIC_KEY>",
        "[REDACTED:API_KEY]",
    ),
    "groq": (
        "gsk_abcdefghijklmnopqrstuvwx1234",
        "gsk_abcdefghijklmnopqrstuvwx1234",
        "<GROQ_KEY>",
        "[REDACTED:API_KEY]",
    ),
    # GitHub token family (all five prefixes must redact in BOTH languages)
    "github_ghp": (
        "ghp_ABCDEFGHIJKLMNOPQRSTuvwx1234",
        "ghp_ABCDEFGHIJKLMNOPQRSTuvwx1234",
        "<GITHUB_TOKEN>",
        "[REDACTED:API_KEY]",
    ),
    "github_gho": (
        "gho_ABCDEFGHIJKLMNOPQRSTuvwx",
        "gho_ABCDEFGHIJKLMNOPQRSTuvwx",
        "<GITHUB_TOKEN>",
        "[REDACTED:API_KEY]",
    ),
    "github_ghs": (
        "ghs_ABCDEFGHIJKLMNOPQRSTuvwx",
        "ghs_ABCDEFGHIJKLMNOPQRSTuvwx",
        "<GITHUB_TOKEN>",
        "[REDACTED:API_KEY]",
    ),
    "github_ghr": (
        "ghr_ABCDEFGHIJKLMNOPQRSTuvwx",
        "ghr_ABCDEFGHIJKLMNOPQRSTuvwx",
        "<GITHUB_TOKEN>",
        "[REDACTED:API_KEY]",
    ),
    "github_pat": (
        "github_pat_ABCDEFGHIJKLMNOPQRSTuvwx1234",
        "github_pat_ABCDEFGHIJKLMNOPQRSTuvwx1234",
        "<GITHUB_TOKEN>",
        "[REDACTED:API_KEY]",
    ),
    # JWT (bare, not Bearer-prefixed)
    "jwt_bare": (
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c",
        "<JWT>",
        "[REDACTED:JWT]",
    ),
    # AWS access key id
    "aws_access_key": (
        "aws_access_key_id = AKIAIOSFODNN7EXAMPLE",
        "AKIAIOSFODNN7EXAMPLE",
        "<AWS_KEY>",
        "[REDACTED:AWS_CREDENTIALS]",
    ),
    # email / phone / credit card / SSN (general PII)
    "email": (
        "contact user@example.com please",
        "user@example.com",
        "<EMAIL>",
        "[REDACTED:EMAIL]",
    ),
    "phone": (
        "call +1-5551234567 now",
        "+1-5551234567",
        "<PHONE>",
        "[REDACTED:PHONE]",
    ),
    "credit_card": (
        "card 4111-1111-1111-1111 ok",
        "4111-1111-1111-1111",
        "<CREDIT_CARD>",
        "[REDACTED:CREDIT_CARD]",
    ),
    "ssn": (
        "ssn 123-45-6789 here",
        "123-45-6789",
        "<SSN>",
        "[REDACTED:SSN]",
    ),
    # private key block
    "private_key": (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEAsecretbodymaterialXYZ\n"
        "-----END RSA PRIVATE KEY-----",
        "secretbodymaterialXYZ",
        "<PRIVATE_KEY>",
        "[REDACTED:PRIVATE_KEY]",
    ),
    # short token=<value> (< 20 chars, below the oauth_token length floor):
    # only the keyword-anchored secrets/password_field layer catches it, so this
    # pins the `token` keyword union added on the Rust side.
    "token_short": (
        "token=abc123",
        "abc123",
        "<REDACTED>",
        "[REDACTED:SECRET]",
    ),
    # credential=<value> (any length): the `credential` keyword had no other
    # layer covering it, so this pins the `credential` keyword union.
    "credential_kv": (
        "credential=mysecretvalue",
        "mysecretvalue",
        "<REDACTED>",
        "[REDACTED:SECRET]",
    ),
    # db connection (Python preserves host; both must remove the credentials)
    "db_postgres": (
        "postgres://admin:s3cretPass@db.example.com/mydb",
        "s3cretPass",
        "<REDACTED>",
        "[REDACTED:DB_CONNECTION]",
    ),
    # password-only userinfo form (empty username) — X-PARITY-5 leak fix.
    "db_redis_passonly": (
        "redis://:s3cretPass@cache.example.com:6379/0",
        "s3cretPass",
        "<REDACTED>",
        "[REDACTED:DB_CONNECTION]",
    ),
    # credential carried in the URL query string — must not leak.
    "db_redis_query": (
        "redis://cache.example.com:6379?password=s3cretPass",
        "s3cretPass",
        "<REDACTED>",
        "[REDACTED:DB_CONNECTION]",
    ),
    # windows path (any drive)
    "win_path_d": (
        r"file at D:\Users\JaneDoe\config",
        "JaneDoe",
        "<USER_PATH>",
        "[USER_PATH]",
    ),
}


@pytest.fixture(scope="module")
def sanitizer() -> PrivacySanitizer:
    return PrivacySanitizer()


@pytest.mark.parametrize("key", sorted(EXPECTED))
def test_python_redacts_every_secret_type(key: str, sanitizer: PrivacySanitizer) -> None:
    """Every secret body is GONE and the expected Python token appears."""
    text, secret_body, py_token, _rust_token = EXPECTED[key]
    out = sanitizer.sanitize(text).clean_content
    assert secret_body not in out, f"[{key}] secret body leaked: {out!r}"
    assert py_token in out, f"[{key}] expected token {py_token!r} missing: {out!r}"


def test_all_secret_types_in_one_blob(sanitizer: PrivacySanitizer) -> None:
    """A single blob containing EVERY secret type has all bodies removed."""
    blob = "\n".join(text for text, *_ in EXPECTED.values())
    out = sanitizer.sanitize(blob).clean_content
    leaked = [
        key
        for key, (_text, secret_body, *_rest) in EXPECTED.items()
        if secret_body in out
    ]
    assert not leaked, f"secret bodies leaked from combined blob: {leaked}\n{out!r}"


def test_clean_text_untouched(sanitizer: PrivacySanitizer) -> None:
    """Plain prose with no secrets is returned byte-identical."""
    text = "The quick brown fox jumps over the lazy dog near /usr/bin/python3."
    result = sanitizer.sanitize(text)
    assert result.clean_content == text
    assert result.was_modified is False
