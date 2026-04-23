"""Unit tests for STORY-R098 PoolPurpose (loader + scoring + nominator gate)."""

from __future__ import annotations

from pathlib import Path

import pytest

from memorus.core.purpose import (
    DEFAULT_NOMINATE_THRESHOLD,
    PoolPurpose,
    apply_purpose,
    cosine_similarity,
    load_pool_purpose,
    load_purpose_file,
    purpose_prompt_context,
    purpose_template,
    resolve_purpose_paths,
    tokenize,
)
from memorus.team.nominator import should_nominate


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def test_tokenize_basic_word_boundary() -> None:
    # "rust" should match a token, not a substring of "rusty".
    tokens = tokenize("I use Rust for systems programming.")
    assert "rust" in tokens
    assert "systems" in tokens
    assert "programming" in tokens


def test_tokenize_unicode_chinese_and_emoji() -> None:
    tokens = tokenize("后端 backend 开发 🚀 perf")
    assert "后端" in tokens
    assert "开发" in tokens
    assert "backend" in tokens
    assert "perf" in tokens


def test_tokenize_lowercase() -> None:
    tokens = tokenize("CARGO CARGO cargo")
    assert tokens == {"cargo"}


def test_tokenize_empty() -> None:
    assert tokenize("") == set()


# ---------------------------------------------------------------------------
# apply_purpose — core scoring rules
# ---------------------------------------------------------------------------


def test_apply_purpose_none_is_noop() -> None:
    assert apply_purpose(80.0, "any content", None) == 80.0


def test_apply_purpose_empty_is_noop() -> None:
    p = PoolPurpose()
    assert apply_purpose(80.0, "any content", p) == 80.0


def test_apply_purpose_keyword_boost() -> None:
    p = PoolPurpose(keywords=["rust", "async"])
    out = apply_purpose(50.0, "Async Rust tips for production", p)
    assert out == pytest.approx(60.0)


def test_apply_purpose_excluded_penalty() -> None:
    p = PoolPurpose(
        keywords=["rust"], excluded_topics=["frontend", "ui-design"]
    )
    out = apply_purpose(80.0, "Frontend UI tweaks", p)
    assert out == pytest.approx(80.0 * 0.3)


def test_apply_purpose_exclusion_dominates_keyword() -> None:
    # Content matches both a keyword and an excluded topic → exclusion wins.
    p = PoolPurpose(keywords=["rust"], excluded_topics=["frontend"])
    out = apply_purpose(80.0, "Rust frontend tweaks", p)
    assert out == pytest.approx(80.0 * 0.3)


def test_apply_purpose_no_match_unchanged() -> None:
    p = PoolPurpose(keywords=["kubernetes"], excluded_topics=["marketing"])
    assert apply_purpose(42.0, "Writing documentation", p) == 42.0


def test_apply_purpose_word_boundary_not_substring() -> None:
    # Keyword "rust" must NOT match the substring "rusty".
    p = PoolPurpose(keywords=["rust"])
    assert apply_purpose(50.0, "a rusty old pipe", p) == 50.0


def test_apply_purpose_multi_token_keyword() -> None:
    # "ui-design" keyword matches content that contains BOTH "ui" and "design".
    p = PoolPurpose(excluded_topics=["ui-design"])
    out = apply_purpose(50.0, "Some UI design improvements", p)
    assert out == pytest.approx(50.0 * 0.3)


def test_apply_purpose_unicode_keyword() -> None:
    p = PoolPurpose(keywords=["后端"])
    assert apply_purpose(50.0, "后端 服务", p) == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Loader: path priority + malformed YAML + missing file
# ---------------------------------------------------------------------------


def _write_purpose(dir_: Path, body: str) -> Path:
    (dir_ / ".ace").mkdir(parents=True, exist_ok=True)
    target = dir_ / ".ace" / "purpose.md"
    target.write_text(body, encoding="utf-8")
    return target


def test_load_missing_returns_empty(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    p = load_pool_purpose(project_dir=proj, home_dir=home)
    assert p.is_empty()
    assert p.source_path is None


def test_load_project_only(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    _write_purpose(
        proj,
        "---\n"
        "scope: project:backend\n"
        "keywords: [rust, async]\n"
        "excluded_topics: [frontend]\n"
        "---\n\n"
        "# Purpose\nBackend best practices.\n",
    )
    p = load_pool_purpose(project_dir=proj, home_dir=home)
    assert p.scope == "project:backend"
    assert p.keywords == ["rust", "async"]
    assert p.excluded_topics == ["frontend"]
    assert "Backend best practices" in p.intent_body


def test_load_global_only(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    _write_purpose(
        home,
        "---\nscope: global\nkeywords: [rust]\n---\n\nGlobal body.\n",
    )
    p = load_pool_purpose(project_dir=proj, home_dir=home)
    assert p.scope == "global"
    assert p.keywords == ["rust"]


def test_load_project_overrides_global(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    _write_purpose(
        home,
        "---\nscope: global\nkeywords: [python]\nexcluded_topics: [marketing]\n---\nGlobal\n",
    )
    _write_purpose(
        proj,
        "---\nscope: project:backend\nkeywords: [rust]\n---\nProject\n",
    )
    p = load_pool_purpose(project_dir=proj, home_dir=home)
    assert p.scope == "project:backend"  # project wins
    assert p.keywords == ["rust"]  # project wins
    # Project did NOT specify excluded_topics, so global fills the gap.
    assert p.excluded_topics == ["marketing"]


def test_load_malformed_yaml_returns_empty_body(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    # Broken YAML: unterminated string
    _write_purpose(
        proj,
        "---\nkeywords: [unterminated\n---\n\nbody only\n",
    )
    p = load_pool_purpose(project_dir=proj, home_dir=home)
    # Body should still come through even when frontmatter fails.
    assert p.keywords == []
    assert "body only" in p.intent_body


def test_load_no_frontmatter_treated_as_body(tmp_path: Path) -> None:
    home = tmp_path / "home"
    proj = tmp_path / "proj"
    home.mkdir()
    proj.mkdir()
    _write_purpose(proj, "# Purpose\nJust a body, no YAML.\n")
    p = load_pool_purpose(project_dir=proj, home_dir=home)
    assert p.keywords == []
    assert "Just a body" in p.intent_body


def test_resolve_paths(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    home = tmp_path / "home"
    pp, gp = resolve_purpose_paths(project_dir=proj, home_dir=home)
    assert pp == proj / ".ace" / "purpose.md"
    assert gp == home / ".ace" / "purpose.md"


def test_load_direct_file_returns_none_when_missing(tmp_path: Path) -> None:
    assert load_purpose_file(tmp_path / "nope.md") is None


def test_default_threshold_constant() -> None:
    assert DEFAULT_NOMINATE_THRESHOLD == 0.6


# ---------------------------------------------------------------------------
# Prompt context + template
# ---------------------------------------------------------------------------


def test_prompt_context_empty_on_none() -> None:
    assert purpose_prompt_context(None) == ""


def test_prompt_context_empty_body() -> None:
    p = PoolPurpose(keywords=["rust"])
    assert purpose_prompt_context(p) == ""


def test_prompt_context_renders_body_and_keywords() -> None:
    p = PoolPurpose(keywords=["rust", "async"], intent_body="Backend best practices.")
    out = purpose_prompt_context(p)
    assert "Pool purpose: Backend best practices." in out
    assert "Keywords: rust, async" in out


def test_template_parses_roundtrip() -> None:
    from memorus.core.purpose import _parse_purpose_text

    text = purpose_template("team:backend")
    p = _parse_purpose_text(text, source_path=None)
    assert p.scope == "team:backend"
    assert p.keywords == []
    assert p.excluded_topics == []


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical() -> None:
    v = [1.0, 2.0, 3.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_dim_mismatch() -> None:
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_cosine_similarity_zero_norm() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ---------------------------------------------------------------------------
# should_nominate — embedding cosine + fallback
# ---------------------------------------------------------------------------


def test_should_nominate_accept_when_purpose_empty() -> None:
    bullet = {"content": "Random thought", "embedding": [0.0]}
    assert should_nominate(bullet, None) is True
    assert should_nominate(bullet, PoolPurpose()) is True


def test_should_nominate_cosine_passes_threshold() -> None:
    # Embedder that returns the same vector for purpose and bullet.
    bullet = {"content": "Rust perf tricks", "embedding": [1.0, 0.0, 0.0]}
    purpose = PoolPurpose(
        keywords=["rust"],
        intent_body="Rust pool",
        nominate_threshold=0.9,
    )
    sim = should_nominate(
        bullet, purpose, embed_fn=lambda _t: [1.0, 0.0, 0.0]
    )
    assert sim is True


def test_should_nominate_cosine_below_threshold() -> None:
    bullet = {"content": "Rust perf tricks", "embedding": [1.0, 0.0]}
    purpose = PoolPurpose(
        keywords=["rust"],
        intent_body="Rust pool",
        nominate_threshold=0.9,
    )
    # Purpose embedding orthogonal to bullet → cosine = 0.0
    assert should_nominate(bullet, purpose, embed_fn=lambda _t: [0.0, 1.0]) is False


def test_should_nominate_fallback_keyword_hit() -> None:
    bullet = {"content": "Rust async runtime tips"}  # no embedding
    purpose = PoolPurpose(keywords=["rust"], intent_body="backend")
    assert should_nominate(bullet, purpose) is True


def test_should_nominate_fallback_token_overlap() -> None:
    bullet = {"content": "backend service deploy"}
    purpose = PoolPurpose(intent_body="backend service", nominate_threshold=0.5)
    # intent_tokens = {backend, service}; overlap = 2/2 = 1.0 >= 0.5
    assert should_nominate(bullet, purpose) is True


def test_should_nominate_fallback_no_overlap() -> None:
    bullet = {"content": "frontend ui tweaks"}
    purpose = PoolPurpose(keywords=["rust"], intent_body="backend service")
    assert should_nominate(bullet, purpose) is False


# ---------------------------------------------------------------------------
# Scorer integration
# ---------------------------------------------------------------------------


def test_scorer_applies_purpose_penalty() -> None:
    from memorus.core.config import ReflectorConfig
    from memorus.core.engines.reflector.scorer import KnowledgeScorer
    from memorus.core.types import DetectedPattern

    cfg = ReflectorConfig(min_score=0.0)
    purpose = PoolPurpose(excluded_topics=["frontend"])
    scorer = KnowledgeScorer(cfg, purpose=purpose)

    # Confidence drives _compute_score; actionable 'use' adds +10 bonus.
    p = DetectedPattern(
        pattern_type="error_fix",
        content="Use the frontend button to submit forms and handle errors",
        confidence=0.9,
    )
    candidate = scorer.score(p)
    # Content hits excluded "frontend" → adjusted score is 0.3 * raw.
    # Without purpose: base = 90 * (14/20) + 10 = 73. With exclusion → 21.9
    # We care about the penalty, not the exact number.
    assert candidate is not None
    # Without purpose the score would be >= 30; with 0.3 multiplier it must
    # be notably smaller than the un-penalized equivalent.
    un_penalized = KnowledgeScorer(cfg, purpose=None).score(p)
    assert un_penalized is not None
    assert candidate.instructivity_score < un_penalized.instructivity_score


def test_scorer_without_purpose_is_backward_compatible() -> None:
    from memorus.core.config import ReflectorConfig
    from memorus.core.engines.reflector.scorer import KnowledgeScorer
    from memorus.core.types import DetectedPattern

    cfg = ReflectorConfig(min_score=0.0)
    scorer_no = KnowledgeScorer(cfg)
    scorer_empty = KnowledgeScorer(cfg, purpose=PoolPurpose())
    p = DetectedPattern(
        pattern_type="error_fix",
        content="Always use structured logging for debugging errors",
        confidence=0.8,
    )
    c1 = scorer_no.score(p)
    c2 = scorer_empty.score(p)
    assert c1 is not None and c2 is not None
    assert c1.instructivity_score == pytest.approx(c2.instructivity_score)


# ---------------------------------------------------------------------------
# Cross-language parity (STORY-R098)
# ---------------------------------------------------------------------------


_CROSS_LANG_FIXTURE = (
    "---\n"
    "scope: team:backend\n"
    "keywords: [rust, async, database]\n"
    "excluded_topics: [frontend, ui-design]\n"
    "nominate_threshold: 0.7\n"
    "---\n"
    "\n"
    "# Purpose\n"
    "Backend best practices for Rust services.\n"
)


def test_cross_language_purpose_values(tmp_path: Path) -> None:
    """Parity with `memorus-r/memorus-core/tests/r098_cross_lang_purpose.rs`.

    The exact fixture and expected values must match the Rust test byte-for-
    byte so the two runtimes agree on what the file means.
    """
    proj = tmp_path / "proj"
    home = tmp_path / "home"
    (proj / ".ace").mkdir(parents=True)
    home.mkdir()
    (proj / ".ace" / "purpose.md").write_text(_CROSS_LANG_FIXTURE, encoding="utf-8")

    p = load_pool_purpose(project_dir=proj, home_dir=home)

    assert p.scope == "team:backend"
    assert p.keywords == ["rust", "async", "database"]
    assert p.excluded_topics == ["frontend", "ui-design"]
    assert "Backend best practices" in p.intent_body
    assert p.nominate_threshold == pytest.approx(0.7)

    assert apply_purpose(50.0, "Async Rust tips", p) == pytest.approx(60.0)
    assert apply_purpose(80.0, "Frontend UI tweaks", p) == pytest.approx(80.0 * 0.3)
    assert apply_purpose(42.0, "Writing documentation", p) == pytest.approx(42.0)
    # Unicode: "后端" NOT in fixture keywords, so score unchanged.
    assert apply_purpose(50.0, "后端 服务", p) == pytest.approx(50.0)
