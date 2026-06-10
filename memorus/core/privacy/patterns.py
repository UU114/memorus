"""Hardcoded privacy patterns -- cannot be removed via configuration."""

from __future__ import annotations

# Each pattern: (name, regex_pattern, replacement)
# Order matters: more specific patterns first
BUILTIN_PATTERNS: list[tuple[str, str, str]] = [
    # 1. Private key blocks (must be first -- multiline)
    (
        "private_key",
        r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----",
        "<PRIVATE_KEY>",
    ),
    # 2. Bearer JWT tokens (Bearer-prefixed -- more specific, must run before
    #    the bare jwt layer below so it wins on "Bearer eyJ...").
    (
        "bearer_token",
        r"Bearer\s+eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)?",
        "<BEARER_TOKEN>",
    ),
    # 2b. Bare JWT (NOT Bearer-prefixed). UNION with Rust `jwt` layer: a JWT that
    #     is not preceded by "Bearer " was previously missed by Python entirely.
    (
        "jwt",
        r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}",
        "<JWT>",
    ),
    # 3. Anthropic API Key (before OpenAI -- sk-ant is more specific than sk-)
    (
        "anthropic_key",
        r"sk-ant-(?:api\d+-)?[A-Za-z0-9_-]{20,}",
        "<ANTHROPIC_KEY>",
    ),
    # 4. OpenAI API Key
    (
        "openai_key",
        r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}",
        "<OPENAI_KEY>",
    ),
    # 4b. Groq API Key. UNION with Rust `api_key` alternation (gsk_); previously
    #     missed by Python.
    (
        "groq_key",
        r"gsk_[A-Za-z0-9]{20,}",
        "<GROQ_KEY>",
    ),
    # 5. GitHub Token
    (
        "github_token",
        r"(?:ghp_|gho_|ghs_|ghr_|github_pat_)[A-Za-z0-9_]{20,}",
        "<GITHUB_TOKEN>",
    ),
    # 6. AWS Access Key
    (
        "aws_access_key",
        r"AKIA[A-Z0-9]{16}",
        "<AWS_KEY>",
    ),
    # 7. AWS Secret Key (after = or : delimiter)
    # Uses a capturing group instead of lookbehind to avoid variable-width issues.
    (
        "aws_secret_key",
        r"((?:aws_secret_access_key|secret_key|SecretAccessKey)\s{0,3}[=:]\s{0,3})"
        r"[A-Za-z0-9/+=]{40}",
        r"\1<AWS_SECRET>",
    ),
    # 8. Database URL with credentials
    (
        "db_url_creds",
        r"((?:postgres|postgresql|mysql|mongodb|redis|amqp)(?:ql)?://)([^:]+):([^@]+)@",
        r"\1<REDACTED>:<REDACTED>@",
    ),
    # 9. Generic API key parameters
    (
        "api_key_param",
        r"((?:api_key|apikey|api-key|access_token|auth_token)\s*[=:]\s*)[^\s,;\"']+",
        r"\1<REDACTED>",
    ),
    # 10. Password/secret fields
    (
        "password_field",
        r"((?:password|passwd|secret|token|credential)\s*[=:]\s*)[^\s,;\"']+",
        r"\1<REDACTED>",
    ),
    # 10b-e: General PII (UNION with Rust email/phone/credit_card/ssn layers;
    # previously missing from Python). Placed after the keyword-anchored secret
    # layers (so labelled secrets win) but before the path layers.
    (
        "email",
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "<EMAIL>",
    ),
    (
        "phone",
        r"\+\d{1,3}[\-.\s]?\d{3,14}",
        "<PHONE>",
    ),
    (
        "credit_card",
        r"\d{4}[\-\s]?\d{4}[\-\s]?\d{4}[\-\s]?\d{4}",
        "<CREDIT_CARD>",
    ),
    (
        "ssn",
        r"\d{3}\-\d{2}\-\d{4}",
        "<SSN>",
    ),
    # 11. Windows user path
    (
        "win_user_path",
        r"[A-Z]:\\Users\\[^\\\s]+",
        "<USER_PATH>",
    ),
    # 12. Unix user path
    (
        "unix_user_path",
        r"(?:/home/|/Users/)[^\s/]+",
        "<USER_PATH>",
    ),
]
