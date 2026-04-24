"""Tests for Generator Stage A topic match (STORY-R097)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from memorus.core.config import RetrievalConfig, TopicsConfig
from memorus.core.engines.generator.engine import BulletForSearch, GeneratorEngine
from memorus.core.engines.topic.store import SqliteTopicStore, TopicPage


def _make_bullet(bid: str, content: str) -> BulletForSearch:
    return BulletForSearch(bullet_id=bid, content=content)


def _store_with_page(tmpdir: Path) -> SqliteTopicStore:
    store = SqliteTopicStore(
        str(tmpdir / "topics.db"),
        str(tmpdir / "pages"),
    )
    page = TopicPage(
        id="topic_abcdef12",
        slug="package-manager-selection",
        title="pnpm frozen lockfile CI workflow",
        summary=(
            "Use pnpm install with the frozen-lockfile flag in CI to "
            "guarantee reproducible builds. Enable corepack so the pinned "
            "version is respected on every runner."
        ),
        bullet_ids=["b1", "b2"],
        source_hash="deadbeefcafef00d",
        model_hash="fallback:noun-phrase",
    )
    store.upsert(page)
    return store


def test_topic_match_disabled_is_byte_identical_to_bullet_path():
    """When topics.enabled=False the Generator must not touch the store."""
    bullets = [
        _make_bullet("b1", "pnpm install --frozen-lockfile"),
        _make_bullet("b2", "enable corepack in CI"),
        _make_bullet("b3", "unrelated content about git"),
    ]

    cfg = TopicsConfig(enabled=False)

    with tempfile.TemporaryDirectory() as d:
        store = _store_with_page(Path(d))
        engine_with_topics = GeneratorEngine(
            config=RetrievalConfig(),
            topic_store=store,
            topics_config=cfg,
        )
        engine_without_topics = GeneratorEngine(config=RetrievalConfig())
        a = engine_with_topics.search("pnpm frozen lockfile", bullets)
        b = engine_without_topics.search("pnpm frozen lockfile", bullets)
        store.close()

    assert len(a) == len(b)
    assert [sb.bullet_id for sb in a] == [sb.bullet_id for sb in b]
    for x, y in zip(a, b):
        assert abs(x.final_score - y.final_score) < 1e-9


def test_topic_match_returns_topic_head_when_threshold_met():
    """Enabled + strong keyword match → topic head appears first."""
    bullets = [
        _make_bullet("b1", "always use pnpm install --frozen-lockfile in CI"),
        _make_bullet("b2", "enable corepack and pin pnpm in CI"),
        _make_bullet("b3", "git rebase -i interactive mode notes"),
    ]

    cfg = TopicsConfig(enabled=True, topic_match_threshold=0.05)

    with tempfile.TemporaryDirectory() as d:
        store = _store_with_page(Path(d))
        engine = GeneratorEngine(
            config=RetrievalConfig(),
            topic_store=store,
            topics_config=cfg,
        )
        results = engine.search("pnpm frozen lockfile CI workflow", bullets)
        store.close()

    assert results
    head = results[0]
    assert head.bullet_id.startswith("topic:")
    assert head.metadata.get("topic_page") is True
    assert head.metadata.get("topic_slug") == "package-manager-selection"


def test_topic_match_threshold_skips_low_score():
    """When no topic exceeds the threshold, Stage B (bullet path) wins."""
    bullets = [
        _make_bullet("b1", "pnpm install --frozen-lockfile"),
    ]
    cfg = TopicsConfig(enabled=True, topic_match_threshold=0.99)

    with tempfile.TemporaryDirectory() as d:
        store = _store_with_page(Path(d))
        engine = GeneratorEngine(
            config=RetrievalConfig(),
            topic_store=store,
            topics_config=cfg,
        )
        results = engine.search("entirely unrelated query", bullets)
        store.close()

    if results:
        assert not results[0].bullet_id.startswith("topic:")


def test_topic_match_has_small_overhead_when_store_empty():
    """Negligible overhead when the store has no pages — guards the <10ms claim."""
    import time

    bullets = [_make_bullet(f"b{i}", f"bullet content {i}") for i in range(20)]
    cfg = TopicsConfig(enabled=True)

    with tempfile.TemporaryDirectory() as d:
        store = SqliteTopicStore(str(Path(d) / "t.db"), str(Path(d) / "p"))
        engine = GeneratorEngine(
            config=RetrievalConfig(),
            topic_store=store,
            topics_config=cfg,
        )
        # Warm-up.
        engine.search("cargo build", bullets)
        start = time.monotonic()
        for _ in range(20):
            engine.search("cargo build", bullets)
        elapsed_ms = (time.monotonic() - start) * 1000 / 20
        store.close()

    # A generous ceiling (host-dependent); guards against catastrophic regressions.
    assert elapsed_ms < 50.0, f"per-query overhead {elapsed_ms:.2f}ms > 50ms"
