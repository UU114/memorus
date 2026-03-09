"""L1 ExactMatcher — exact keyword matching layer for the Generator search engine.

Performs word-boundary-aware matching for English tokens and character-level
substring matching for Chinese tokens.  Each keyword hit adds a configurable
score (default +15).  Multiple hits accumulate additively.

Usage::

    matcher = ExactMatcher(hit_score=15.0)
    result = matcher.match("git rebase", "Use git rebase -i for interactive rebase")
    # result.score == 15.0 (two hits: "git" and "rebase")
    # Wait — both hit, so score == 30.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# English stopwords — common words that should not be matched as keywords
# ---------------------------------------------------------------------------

_ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "as", "be", "was", "are",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "its", "our", "their", "what", "which", "who", "whom", "when",
    "where", "why", "how", "not", "no", "so", "if", "then", "than",
    "too", "very", "just", "about", "up", "out", "all", "each", "every",
    "both", "few", "more", "most", "other", "some", "such", "only",
})

# Regex to split English text on whitespace and punctuation boundaries
_WORD_SPLIT_RE: re.Pattern[str] = re.compile(r"[A-Za-z0-9_]+(?:'[A-Za-z]+)?")

# Regex to detect CJK Unified Ideographs (Chinese characters)
_CJK_RE: re.Pattern[str] = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df"
    r"\U0002a700-\U0002b73f\U0002b740-\U0002b81f"
    r"\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef"
    r"\U00030000-\U0003134f]+"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    """Result from a single matcher layer."""

    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


def _is_cjk_char(ch: str) -> bool:
    """Return True if *ch* is a CJK Unified Ideograph."""
    cp = ord(ch)
    return (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F
        or 0x2B740 <= cp <= 0x2B81F
        or 0x2B820 <= cp <= 0x2CEAF
        or 0x2CEB0 <= cp <= 0x2EBEF
        or 0x30000 <= cp <= 0x3134F
    )


def _extract_cjk_segments(text: str) -> list[str]:
    """Extract contiguous CJK character runs from *text*."""
    return _CJK_RE.findall(text)


def tokenize_query(query: str) -> tuple[list[str], list[str]]:
    """Split *query* into (english_tokens, chinese_segments).

    English tokens are lowercased, de-duplicated, and stripped of stopwords.
    Chinese segments are contiguous runs of CJK characters (length >= 2).
    Single-character Chinese segments are kept as-is for matching.
    """
    english_tokens: list[str] = []
    seen_en: set[str] = set()
    for m in _WORD_SPLIT_RE.finditer(query):
        word = m.group().lower()
        if word not in _ENGLISH_STOPWORDS and word not in seen_en:
            seen_en.add(word)
            english_tokens.append(word)

    # Chinese: extract CJK segments; keep segments of any length
    chinese_segments: list[str] = []
    seen_zh: set[str] = set()
    for seg in _extract_cjk_segments(query):
        if seg not in seen_zh:
            seen_zh.add(seg)
            chinese_segments.append(seg)

    return english_tokens, chinese_segments


# ---------------------------------------------------------------------------
# Pattern cache for compiled word-boundary regexes
# ---------------------------------------------------------------------------


def _build_word_pattern(token: str) -> re.Pattern[str]:
    """Build a compiled case-insensitive word-boundary regex for *token*.

    Uses ASCII-only boundaries (lookbehind/lookahead for ``[A-Za-z0-9_]``)
    instead of ``\\b`` so that English words embedded in CJK text are matched
    correctly.  Python's ``\\b`` treats CJK characters as ``\\w``, which means
    ``\\bgit\\b`` would NOT match "git" in "使用git进行" — this pattern fixes that.
    """
    escaped = re.escape(token)
    return re.compile(
        rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])",
        re.IGNORECASE,
    )


# ---------------------------------------------------------------------------
# ExactMatcher
# ---------------------------------------------------------------------------


class ExactMatcher:
    """L1 exact keyword matcher.

    For each keyword extracted from the query:
    - **English**: word-boundary-aware matching (``\\b`` regex), case-insensitive.
    - **Chinese**: character-level substring matching (case-insensitive is N/A).

    Each hit adds ``hit_score`` (default 15.0) to the result.  Multiple keyword
    hits accumulate additively.
    """

    def __init__(self, hit_score: float = 15.0) -> None:
        self._hit_score = hit_score
        # Cache compiled regex patterns keyed by lowered token
        self._pattern_cache: dict[str, re.Pattern[str]] = {}

    @property
    def hit_score(self) -> float:
        """Score added per keyword hit."""
        return self._hit_score

    # -- internal helpers ---------------------------------------------------

    def _get_word_pattern(self, token: str) -> re.Pattern[str]:
        """Return a cached compiled word-boundary pattern for *token*."""
        if token not in self._pattern_cache:
            self._pattern_cache[token] = _build_word_pattern(token)
        return self._pattern_cache[token]

    # -- public API ---------------------------------------------------------

    def match(self, query: str, content: str) -> MatchResult:
        """Score *content* against *query* using exact keyword matching.

        Returns a :class:`MatchResult` with cumulative score and details about
        which terms matched and at what positions.
        """
        if not query or not content:
            return MatchResult()

        english_tokens, chinese_segments = tokenize_query(query)
        matched_terms: list[str] = []
        positions: dict[str, list[int]] = {}

        # English: word-boundary matching
        for token in english_tokens:
            pattern = self._get_word_pattern(token)
            matches = list(pattern.finditer(content))
            if matches:
                matched_terms.append(token)
                positions[token] = [m.start() for m in matches]

        # Chinese: substring matching
        for seg in chinese_segments:
            # Find all occurrences of the CJK segment in content
            seg_positions: list[int] = []
            start = 0
            while True:
                idx = content.find(seg, start)
                if idx == -1:
                    break
                seg_positions.append(idx)
                start = idx + 1
            if seg_positions:
                matched_terms.append(seg)
                positions[seg] = seg_positions

        score = len(matched_terms) * self._hit_score

        return MatchResult(
            score=score,
            matched_terms=matched_terms,
            details={
                "positions": positions,
                "english_tokens": english_tokens,
                "chinese_segments": chinese_segments,
            },
        )

    def match_batch(
        self,
        query: str,
        contents: list[str],
    ) -> list[MatchResult]:
        """Score each item in *contents* against *query*.

        Pre-tokenizes the query once and reuses compiled patterns for all items.
        Returns a list of :class:`MatchResult` in the same order as *contents*.
        """
        if not query:
            return [MatchResult() for _ in contents]

        # Pre-tokenize once
        english_tokens, chinese_segments = tokenize_query(query)

        # Pre-compile all English patterns
        en_patterns: list[tuple[str, re.Pattern[str]]] = [
            (token, self._get_word_pattern(token)) for token in english_tokens
        ]

        results: list[MatchResult] = []
        for content in contents:
            if not content:
                results.append(MatchResult())
                continue

            matched_terms: list[str] = []
            positions: dict[str, list[int]] = {}

            # English: word-boundary matching
            for token, pattern in en_patterns:
                matches = list(pattern.finditer(content))
                if matches:
                    matched_terms.append(token)
                    positions[token] = [m.start() for m in matches]

            # Chinese: substring matching
            for seg in chinese_segments:
                seg_positions: list[int] = []
                start = 0
                while True:
                    idx = content.find(seg, start)
                    if idx == -1:
                        break
                    seg_positions.append(idx)
                    start = idx + 1
                if seg_positions:
                    matched_terms.append(seg)
                    positions[seg] = seg_positions

            score = len(matched_terms) * self._hit_score
            results.append(MatchResult(
                score=score,
                matched_terms=matched_terms,
                details={
                    "positions": positions,
                    "english_tokens": english_tokens,
                    "chinese_segments": chinese_segments,
                },
            ))

        return results
