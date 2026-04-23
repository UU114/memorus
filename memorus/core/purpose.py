"""PoolPurpose — pool-level intent declaration (STORY-R098).

Loads `.ace/purpose.md` files (YAML frontmatter + Markdown body) that steer
Reflector scoring and Team nomination behavior.

Path priority:
  1. ``<project>/.ace/purpose.md`` (project-local)
  2. ``~/.ace/purpose.md`` (user-global)
  3. None → fallback: empty PoolPurpose (no scoring impact)

Project purpose overrides Global purpose at the **field level** (merge,
not file-level replacement).

Scoring rules (MUST be identical across languages):
  - if any excluded_topic token matches content → score *= 0.3
  - elif any keyword token matches content      → score *= 1.2
  - else                                         → score unchanged

Token matching: lowercase + Unicode word-boundary (NOT substring).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# Default similarity threshold for team nomination (can be overridden).
DEFAULT_NOMINATE_THRESHOLD = 0.6

# Standard purpose file location inside a pool.
PURPOSE_DIR = ".ace"
PURPOSE_FILENAME = "purpose.md"

# Track loaders that have already warned about malformed YAML so we only
# warn once per path per process lifetime.
_WARNED_PATHS: set[str] = set()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PoolPurpose:
    """A pool-level intent declaration.

    Attributes:
        scope: Logical pool identifier, e.g. "team:backend" / "project:xyz" / "global".
        keywords: Tokens that boost matching content.
        excluded_topics: Tokens that penalize matching content.
        intent_body: The free-form Markdown body (minus the YAML frontmatter).
        source_path: The file the purpose was loaded from (None if synthesized).
        nominate_threshold: Similarity threshold for team nomination.
    """

    scope: str = ""
    keywords: list[str] = field(default_factory=list)
    excluded_topics: list[str] = field(default_factory=list)
    intent_body: str = ""
    source_path: Optional[Path] = None
    nominate_threshold: float = DEFAULT_NOMINATE_THRESHOLD

    def is_empty(self) -> bool:
        """Return True if this purpose carries no signals."""
        return (
            not self.scope
            and not self.keywords
            and not self.excluded_topics
            and not self.intent_body.strip()
        )


# ---------------------------------------------------------------------------
# Tokenization (Unicode-aware, word-boundary)
# ---------------------------------------------------------------------------

# Word = one or more alphanumeric or underscore code points across any script
# (handles Chinese, Japanese, Korean, Arabic, Cyrillic, etc.). Hyphens are
# treated as word separators to keep "ui-design" tokenizable as {"ui", "design"}
# on one side while still allowing exact keyword match "ui-design" on the
# other — see `_token_hits` below.
_WORD_RE = re.compile(r"[^\W]+", flags=re.UNICODE)


def tokenize(text: str) -> set[str]:
    """Lowercase Unicode-aware word tokenization.

    Normalizes to NFKC so that full-width digits / compatibility characters
    collapse to their canonical form before matching.
    """
    if not text:
        return set()
    normalized = unicodedata.normalize("NFKC", text).lower()
    return {m.group(0) for m in _WORD_RE.finditer(normalized)}


def _token_hits(needles: Iterable[str], haystack_tokens: set[str]) -> bool:
    """Return True if any *needle* matches as a whole token in *haystack_tokens*.

    Multi-word needles (e.g. "ui-design") match iff **every** sub-token is
    present in the haystack. This gives sensible behavior for compound keywords
    without relying on fragile substring matching.
    """
    for needle in needles:
        needle_tokens = tokenize(needle)
        if not needle_tokens:
            continue
        if needle_tokens.issubset(haystack_tokens):
            return True
    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def apply_purpose(
    base_score: float,
    content: str,
    purpose: Optional[PoolPurpose],
) -> float:
    """Apply purpose-driven score adjustment.

    Rules (order matters — exclusion dominates):
      - excluded match → score * 0.3 (near-kill)
      - keyword match  → score * 1.2 (boost)
      - neither        → unchanged

    A `None` or empty PoolPurpose is a no-op: the base_score is returned
    unchanged.
    """
    if purpose is None or purpose.is_empty():
        return base_score

    tokens = tokenize(content)
    if not tokens:
        return base_score

    if _token_hits(purpose.excluded_topics, tokens):
        logger.debug(
            "apply_purpose: excluded_topic hit, score %.2f -> %.2f",
            base_score,
            base_score * 0.3,
        )
        return base_score * 0.3
    if _token_hits(purpose.keywords, tokens):
        logger.debug(
            "apply_purpose: keyword hit, score %.2f -> %.2f",
            base_score,
            base_score * 1.2,
        )
        return base_score * 1.2
    return base_score


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\r?\n(?P<front>.*?)\r?\n---\s*\r?\n?(?P<body>.*)\Z",
    flags=re.DOTALL,
)


def _parse_purpose_text(text: str, source_path: Optional[Path]) -> PoolPurpose:
    """Parse YAML frontmatter + body into a PoolPurpose.

    On YAML parse error, warn once and return an empty PoolPurpose (still
    carrying the source_path for observability).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        # No frontmatter: treat entire file as body.
        return PoolPurpose(intent_body=text.strip(), source_path=source_path)

    front = match.group("front")
    body = match.group("body").strip()

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("PyYAML not installed; cannot parse %s", source_path)
        return PoolPurpose(intent_body=body, source_path=source_path)

    try:
        data = yaml.safe_load(front) or {}
    except Exception as exc:
        key = str(source_path) if source_path else "<memory>"
        if key not in _WARNED_PATHS:
            logger.warning("Malformed YAML frontmatter in %s: %s", key, exc)
            _WARNED_PATHS.add(key)
        return PoolPurpose(intent_body=body, source_path=source_path)

    if not isinstance(data, dict):
        return PoolPurpose(intent_body=body, source_path=source_path)

    scope = str(data.get("scope", "") or "")
    keywords = _as_str_list(data.get("keywords"))
    excluded = _as_str_list(data.get("excluded_topics"))
    threshold = data.get("nominate_threshold", DEFAULT_NOMINATE_THRESHOLD)
    try:
        threshold_f = float(threshold)
    except (TypeError, ValueError):
        threshold_f = DEFAULT_NOMINATE_THRESHOLD

    return PoolPurpose(
        scope=scope,
        keywords=keywords,
        excluded_topics=excluded,
        intent_body=body,
        source_path=source_path,
        nominate_threshold=threshold_f,
    )


def _as_str_list(value: object) -> list[str]:
    """Coerce a YAML list-of-strings (or CSV) into a normalized list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()] if str(value).strip() else []


def load_purpose_file(path: Path) -> Optional[PoolPurpose]:
    """Load a single purpose.md file. Returns None if the file is missing."""
    try:
        if not path.is_file():
            return None
    except OSError:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None
    return _parse_purpose_text(text, source_path=path)


def _merge_fields(primary: PoolPurpose, fallback: PoolPurpose) -> PoolPurpose:
    """Merge *fallback* into *primary* at the field level.

    Primary (project) wins per field; fallback (global) fills gaps.
    """
    return PoolPurpose(
        scope=primary.scope or fallback.scope,
        keywords=primary.keywords or list(fallback.keywords),
        excluded_topics=primary.excluded_topics or list(fallback.excluded_topics),
        intent_body=primary.intent_body or fallback.intent_body,
        source_path=primary.source_path or fallback.source_path,
        nominate_threshold=(
            primary.nominate_threshold
            if primary.nominate_threshold != DEFAULT_NOMINATE_THRESHOLD
            else fallback.nominate_threshold
        ),
    )


def resolve_purpose_paths(
    project_dir: Optional[Path] = None,
    home_dir: Optional[Path] = None,
) -> tuple[Optional[Path], Optional[Path]]:
    """Return (project_purpose_path, global_purpose_path) candidates.

    Existence is not checked here — callers use `load_purpose_file`.
    """
    pdir = project_dir if project_dir is not None else Path.cwd()
    hdir = home_dir if home_dir is not None else Path.home()
    project_path = pdir / PURPOSE_DIR / PURPOSE_FILENAME
    global_path = hdir / PURPOSE_DIR / PURPOSE_FILENAME
    return project_path, global_path


def load_pool_purpose(
    project_dir: Optional[Path] = None,
    home_dir: Optional[Path] = None,
) -> PoolPurpose:
    """Load the effective PoolPurpose following the path-priority rules.

    Missing files are **not** warnings — this is the common, zero-config case.
    """
    project_path, global_path = resolve_purpose_paths(project_dir, home_dir)
    project = load_purpose_file(project_path)
    global_ = load_purpose_file(global_path)

    if project is not None and global_ is not None:
        return _merge_fields(project, global_)
    if project is not None:
        return project
    if global_ is not None:
        return global_
    return PoolPurpose()


# ---------------------------------------------------------------------------
# Template helpers (for `memorus purpose init`)
# ---------------------------------------------------------------------------


def purpose_template(scope: str = "global") -> str:
    """Return a ready-to-commit template purpose.md body."""
    return (
        "---\n"
        f"scope: {scope}\n"
        "keywords: []\n"
        "excluded_topics: []\n"
        "nominate_threshold: 0.6\n"
        "---\n"
        "\n"
        "# Purpose\n"
        "Describe what this pool cares about. Mention the domain,\n"
        "the kinds of knowledge you want to capture, and anything that\n"
        "should explicitly stay out of this pool.\n"
    )


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------


def purpose_prompt_context(purpose: Optional[PoolPurpose]) -> str:
    """Render purpose for LLM prompt injection.

    Returns an empty string when the purpose is missing or has no body,
    so callers can unconditionally prepend the result to a prompt.
    """
    if purpose is None or not purpose.intent_body.strip():
        return ""
    kw = ", ".join(purpose.keywords) if purpose.keywords else "(none)"
    return (
        "[Context]\n"
        f"Pool purpose: {purpose.intent_body.strip()}\n"
        f"Keywords: {kw}\n"
    )


# ---------------------------------------------------------------------------
# Embedding / similarity helpers (for the Nominator)
# ---------------------------------------------------------------------------


def purpose_text_for_embedding(purpose: PoolPurpose) -> str:
    """Concatenate intent_body + keywords into a single string for embedding.

    Empty fields are skipped so that purely-keyword or purely-body purposes
    still produce a meaningful signal.
    """
    parts: list[str] = []
    if purpose.intent_body.strip():
        parts.append(purpose.intent_body.strip())
    if purpose.keywords:
        parts.append(" ".join(purpose.keywords))
    return "\n".join(parts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two dense vectors.

    Returns 0.0 if either vector is empty or has zero norm.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a**0.5) * (norm_b**0.5))
