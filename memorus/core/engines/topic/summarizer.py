"""TopicPage summarizer — LLM-backed with deterministic fallback.

The :class:`Summarizer` protocol declares the minimal surface consumed by
:class:`~memorus.core.engines.topic.engine.TopicEngine`.

:class:`LLMSummarizer` calls an LLM client to produce a title + 200-word
narrative; if the LLM path fails (client missing, exception, empty
response) the caller falls back to :class:`FallbackSummarizer` which
derives a title from the first bullet's noun-ish prefix and a summary
from truncated concatenation — fully deterministic so tests can run
offline.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

MAX_SUMMARY_WORDS = 200
FALLBACK_TITLE_WORDS = 8


# ---------------------------------------------------------------------------
# Public protocol + data types
# ---------------------------------------------------------------------------


@dataclass
class SummaryResult:
    """Output of :meth:`Summarizer.summarize` — title + narrative + meta."""

    title: str
    summary: str
    model_hash: str  # e.g. "gpt-4o-mini" or "fallback:noun-phrase"


class Summarizer(Protocol):
    """Minimal duck-typed summarizer surface."""

    def summarize(
        self,
        bullet_contents: list[str],
        *,
        hint: Optional[str] = None,
    ) -> SummaryResult:
        """Produce a title + ≤200-word summary for the given bullet contents."""
        ...


# ---------------------------------------------------------------------------
# Fallback — deterministic, LLM-free
# ---------------------------------------------------------------------------


_WORD_RE = re.compile(r"[\w一-鿿][\w一-鿿\-]*", re.UNICODE)


def _truncate_words(text: str, max_words: int) -> str:
    """Truncate *text* to *max_words* tokens, preserving original spacing
    between the retained tokens. Returns the truncated string trimmed."""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


def _first_noun_phrase(text: str, max_words: int = FALLBACK_TITLE_WORDS) -> str:
    """Best-effort title: first sentence up to *max_words* word tokens."""
    head = text.splitlines()[0] if text else ""
    # Cut at the first sentence terminator if any.
    for sep in ("。", "！", "？", ". ", "! ", "? "):
        idx = head.find(sep)
        if idx > 0:
            head = head[:idx]
            break
    tokens = _WORD_RE.findall(head)
    if not tokens:
        return head.strip() or "Untitled Topic"
    return " ".join(tokens[:max_words]).strip()


class FallbackSummarizer:
    """Deterministic summarizer used when the LLM path is unavailable."""

    model_hash = "fallback:noun-phrase"

    def summarize(
        self,
        bullet_contents: list[str],
        *,
        hint: Optional[str] = None,
    ) -> SummaryResult:
        if not bullet_contents:
            return SummaryResult(
                title="Empty Topic",
                summary="",
                model_hash=self.model_hash,
            )
        title_seed = next((c for c in bullet_contents if c.strip()), "")
        title = _first_noun_phrase(title_seed) or "Untitled Topic"

        # Summary = concatenated bullet contents, truncated at 200 words.
        joined = "\n".join(c.strip() for c in bullet_contents if c and c.strip())
        summary = _truncate_words(joined, MAX_SUMMARY_WORDS)
        return SummaryResult(title=title, summary=summary, model_hash=self.model_hash)


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------


_PROMPT_TEMPLATE = (
    "You are summarising a cluster of related engineering notes into a "
    "single TopicPage.\n\n"
    "Return *exactly* this structure:\n"
    "TITLE: <kebab-case title, up to 8 words>\n"
    "SUMMARY: <flowing narrative, ≤200 words, English>\n\n"
    "Cluster bullets:\n{bullets}\n"
)


class LLMSummarizer:
    """LLM-backed summarizer. On any failure it falls back to
    :class:`FallbackSummarizer` — callers get a valid result either way."""

    def __init__(
        self,
        *,
        client: Any = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 800,
    ) -> None:
        self._client = client  # duck-typed: must expose .complete(prompt, **kw) -> str
        self._model = model
        self._max_tokens = max_tokens
        self._fallback = FallbackSummarizer()

    @property
    def model_hash(self) -> str:
        return self._model

    def summarize(
        self,
        bullet_contents: list[str],
        *,
        hint: Optional[str] = None,
    ) -> SummaryResult:
        if self._client is None:
            return self._fallback.summarize(bullet_contents, hint=hint)

        try:
            raw = self._call_llm(bullet_contents, hint=hint)
        except Exception as e:
            logger.warning("LLMSummarizer failed (%s) — using fallback", e)
            return self._fallback.summarize(bullet_contents, hint=hint)

        title, summary = _parse_llm_response(raw)
        if not title or not summary:
            logger.info("LLMSummarizer returned unparseable body — using fallback")
            return self._fallback.summarize(bullet_contents, hint=hint)

        summary = _truncate_words(summary, MAX_SUMMARY_WORDS)
        return SummaryResult(title=title, summary=summary, model_hash=self._model)

    def _call_llm(
        self,
        bullet_contents: list[str],
        *,
        hint: Optional[str] = None,
    ) -> str:
        enumerated = "\n".join(f"- {c.strip()}" for c in bullet_contents if c.strip())
        prompt = _PROMPT_TEMPLATE.format(bullets=enumerated)
        if hint:
            prompt = f"{prompt}\n\nContext hint: {hint.strip()}\n"
        # Accept both sync and async-ish client interfaces that return str.
        complete = getattr(self._client, "complete", None)
        if callable(complete):
            return str(complete(prompt, max_tokens=self._max_tokens, model=self._model))
        raise RuntimeError(
            "LLM client does not expose a .complete(prompt, **kw) method"
        )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


_TITLE_RE = re.compile(r"^\s*TITLE\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_SUMMARY_RE = re.compile(
    r"^\s*SUMMARY\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def _parse_llm_response(body: str) -> tuple[str, str]:
    """Extract ``TITLE:`` and ``SUMMARY:`` fields — robust to extra text."""
    if not body:
        return "", ""
    t_match = _TITLE_RE.search(body)
    s_match = _SUMMARY_RE.search(body)
    title = t_match.group(1).strip() if t_match else ""
    summary = s_match.group(1).strip() if s_match else ""
    return title, summary


__all__ = [
    "FallbackSummarizer",
    "LLMSummarizer",
    "MAX_SUMMARY_WORDS",
    "Summarizer",
    "SummaryResult",
]
