"""Unit tests for the TopicPage summarizer (STORY-R097)."""

from __future__ import annotations

from memorus.core.engines.topic.summarizer import (
    FallbackSummarizer,
    LLMSummarizer,
    MAX_SUMMARY_WORDS,
    _parse_llm_response,
)


def test_fallback_produces_title_and_truncated_summary():
    bullets = [
        "always use pnpm install --frozen-lockfile in CI",
        "avoid npm install in CI — lockfile drift breaks reproducibility",
        "for node projects enable corepack to pin pnpm version",
    ]
    sr = FallbackSummarizer().summarize(bullets)
    assert sr.title.strip() != ""
    assert sr.summary.strip() != ""
    assert sr.model_hash.startswith("fallback:")
    # Summary word count within bounds.
    assert len(sr.summary.split()) <= MAX_SUMMARY_WORDS


def test_fallback_on_empty_input():
    sr = FallbackSummarizer().summarize([])
    assert sr.title == "Empty Topic"
    assert sr.summary == ""


def test_llm_falls_back_when_client_missing():
    sr = LLMSummarizer(client=None).summarize(["one", "two"])
    assert sr.model_hash.startswith("fallback:")


class _ErrClient:
    def complete(self, _prompt, **_kw):
        raise RuntimeError("llm down")


def test_llm_falls_back_when_client_raises():
    sr = LLMSummarizer(client=_ErrClient()).summarize(["a", "b"])
    assert sr.model_hash.startswith("fallback:")


class _EchoClient:
    def __init__(self, body: str):
        self.body = body
        self.last_prompt: str | None = None

    def complete(self, prompt: str, **_kw) -> str:
        self.last_prompt = prompt
        return self.body


def test_llm_parses_title_and_summary():
    body = (
        "TITLE: Package Manager Selection\n"
        "SUMMARY: Pnpm is preferred for CI because its lockfile is strict.\n"
    )
    client = _EchoClient(body)
    sr = LLMSummarizer(client=client, model="gpt-4o-mini").summarize([
        "always use pnpm install --frozen-lockfile",
    ])
    assert sr.title == "Package Manager Selection"
    assert "Pnpm is preferred" in sr.summary
    assert sr.model_hash == "gpt-4o-mini"
    # The prompt must mention the bullet content.
    assert "pnpm install --frozen-lockfile" in (client.last_prompt or "")


def test_llm_parses_falls_back_on_unparseable():
    sr = LLMSummarizer(client=_EchoClient("gibberish response")).summarize(["x"])
    assert sr.model_hash.startswith("fallback:")


def test_parse_llm_response_direct():
    t, s = _parse_llm_response("TITLE: Foo\nSUMMARY: Bar baz")
    assert t == "Foo"
    assert s.startswith("Bar baz")
    t2, s2 = _parse_llm_response("nope")
    assert t2 == ""
    assert s2 == ""


def test_llm_truncates_long_summary():
    long_body = "TITLE: T\nSUMMARY: " + " ".join(["word"] * 500)
    sr = LLMSummarizer(client=_EchoClient(long_body)).summarize(["x"])
    assert len(sr.summary.split()) <= MAX_SUMMARY_WORDS
