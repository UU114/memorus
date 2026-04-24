"""Unit tests for the SqliteTopicStore + MD rendering (STORY-R097)."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path

from memorus.core.engines.topic.store import (
    SqliteTopicStore,
    TopicPage,
    compute_drift_fraction,
    compute_source_hash,
    generate_topic_id,
    render_markdown,
    slugify,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_source_hash_is_deterministic_and_order_independent():
    h1 = compute_source_hash(["b", "a", "c"])
    h2 = compute_source_hash(["c", "b", "a"])
    assert h1 == h2
    assert len(h1) == 16


def test_source_hash_cross_language_parity():
    """Lock the hash used by Rust+Python so the two share the same SQLite file."""
    # These values are asserted in memorus-ace/src/topic/store.rs — they
    # MUST match sha256(canonical(sorted(bullet_ids)))[:16].
    assert compute_source_hash(["a", "b"]) == "0473ef2dc0d324ab"
    assert compute_source_hash(["a", "b", "c"]) == "fa1844c2988ad15a"


def test_source_hash_dedup():
    h1 = compute_source_hash(["a", "a", "b"])
    h2 = compute_source_hash(["a", "b"])
    assert h1 == h2


def test_drift_fraction_bounds():
    # Jaccard distance: 1 - |inter| / |union|
    assert compute_drift_fraction([], []) == 0.0
    assert compute_drift_fraction(["a"], ["a"]) == 0.0
    assert compute_drift_fraction(["a"], ["b"]) == 1.0
    # {a,b} vs {a,c}: inter={a}=1, union={a,b,c}=3 → 1 - 1/3
    assert abs(compute_drift_fraction(["a", "b"], ["a", "c"]) - (1.0 - 1.0 / 3.0)) < 1e-9
    # {a,b,c,d} vs {a,b,c,e}: inter=3, union=5 → 1 - 3/5 = 0.4
    df = compute_drift_fraction(["a", "b", "c", "d"], ["a", "b", "c", "e"])
    assert abs(df - 0.4) < 1e-9


def test_generate_topic_id_format():
    h = compute_source_hash(["a", "b"])
    tid = generate_topic_id(h)
    assert tid.startswith("topic_")
    assert len(tid) == len("topic_") + 8


def test_slugify_english():
    assert slugify("Package Manager Selection") == "package-manager-selection"
    assert slugify("   multiple   spaces  ") == "multiple-spaces"
    assert slugify("") == "untitled"
    # Keep CJK characters.
    s = slugify("包管理器 选型")
    assert s and s != "untitled"


# ---------------------------------------------------------------------------
# SqliteTopicStore
# ---------------------------------------------------------------------------


def _fresh_store() -> tuple[SqliteTopicStore, Path]:
    tmpdir = Path(tempfile.mkdtemp())
    store = SqliteTopicStore(str(tmpdir / "topics.db"), str(tmpdir / "pages"))
    return store, tmpdir


def _make_page(ids: list[str], title: str = "Topic") -> TopicPage:
    sh = compute_source_hash(ids)
    return TopicPage(
        id=generate_topic_id(sh),
        slug=slugify(title),
        title=title,
        summary="A test summary.",
        bullet_ids=list(ids),
        source_hash=sh,
        model_hash="fallback:noun-phrase",
    )


def test_store_upsert_and_get_by_id():
    store, _ = _fresh_store()
    page = _make_page(["b1", "b2", "b3"], title="First topic")
    saved = store.upsert(page)
    got = store.get_by_id(saved.id)
    assert got is not None
    assert got.title == "First topic"
    assert sorted(got.bullet_ids) == ["b1", "b2", "b3"]
    store.close()


def test_store_get_by_source_hash():
    store, _ = _fresh_store()
    page = _make_page(["b1", "b2"], title="T")
    store.upsert(page)
    hit = store.get_by_source_hash(page.source_hash)
    assert hit is not None
    assert hit.id == page.id
    assert store.get_by_source_hash("nonexistent") is None
    store.close()


def test_store_slug_disambiguation():
    store, _ = _fresh_store()
    p1 = TopicPage(
        id="topic_deadbeef",
        slug="shared",
        title="A",
        summary="x",
        bullet_ids=["x"],
        source_hash="h1",
        model_hash="m",
    )
    p2 = TopicPage(
        id="topic_feedf00d",
        slug="shared",
        title="B",
        summary="y",
        bullet_ids=["y"],
        source_hash="h2",
        model_hash="m",
    )
    store.upsert(p1)
    store.upsert(p2)
    # Both must persist under distinct slugs.
    assert store.get_by_id("topic_deadbeef") is not None
    assert store.get_by_id("topic_feedf00d") is not None
    slugs = {p.slug for p in store.list_all()}
    assert len(slugs) == 2
    store.close()


def test_store_list_all_ordered_by_update_time():
    store, _ = _fresh_store()
    p1 = _make_page(["a1"], title="Older")
    store.upsert(p1)
    p2 = _make_page(["a2"], title="Newer")
    store.upsert(p2)
    pages = store.list_all()
    assert len(pages) == 2
    # Newer should come first (list_all orders by updated_at DESC).
    assert pages[0].title == "Newer"
    store.close()


def test_store_delete_removes_md_file(tmp_path):
    store = SqliteTopicStore(
        str(tmp_path / "topics.db"),
        str(tmp_path / "pages"),
    )
    page = _make_page(["a", "b", "c"], title="Zip topic")
    saved = store.upsert(page)
    store.write_md(saved, bullet_lookup={"a": "Content A"})
    md = store.md_path(saved.slug)
    assert md.exists()
    # Delete and verify MD is gone.
    ok = store.delete(saved.id)
    assert ok
    assert store.get_by_id(saved.id) is None
    assert not md.exists()
    store.close()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_markdown_obsidian_compatible():
    page = TopicPage(
        id="topic_12345678",
        slug="sample",
        title="Sample",
        summary="This is a summary.",
        bullet_ids=["b1", "b2"],
        source_hash="aaaaaaaaaaaaaaaa",
        model_hash="gpt-4o-mini",
        created_at=datetime(2026, 4, 23, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 23, tzinfo=timezone.utc),
    )
    md = render_markdown(
        page,
        bullet_lookup={"b1": "First backing bullet", "b2": "Second"},
        related_titles=["Related One", "Related Two"],
    )
    # YAML front matter present.
    assert md.startswith("---\n")
    assert "id: topic_12345678" in md
    assert "slug: sample" in md
    assert "bullet_ids:" in md
    assert "  - b1" in md
    assert "  - b2" in md
    assert "model_hash: gpt-4o-mini" in md
    # Heading + summary body.
    assert "# Sample" in md
    assert "This is a summary." in md
    # Backing Bullets section with snippet preview.
    assert "## Backing Bullets" in md
    assert "- b1: First backing bullet" in md
    # Wikilinks for related topics (Obsidian-compatible).
    assert "[[Related One]]" in md
    assert "[[Related Two]]" in md
