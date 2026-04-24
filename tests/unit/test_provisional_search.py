"""Unit tests for Generator provisional search (STORY-R095)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from memorus.core.config import RetrievalConfig
from memorus.core.engines.generator.engine import BulletForSearch, GeneratorEngine
from memorus.core.engines.generator.metadata_matcher import MetadataInfo
from memorus.core.engines.reflector.inbox import Inbox, make_entry


def _bullet(bid: str, content: str) -> BulletForSearch:
    return BulletForSearch(
        bullet_id=bid,
        content=content,
        metadata=MetadataInfo(),
        created_at=datetime.now(timezone.utc),
        decay_weight=1.0,
    )


def test_search_without_inbox_is_unchanged(tmp_path: Path) -> None:
    engine = GeneratorEngine(config=RetrievalConfig())
    results = engine.search(
        "rebase",
        [_bullet("b1", "git rebase is useful"), _bullet("b2", "unrelated content")],
        limit=10,
    )
    assert results
    assert all(r.metadata.get("provisional") is not True for r in results)


def test_include_pending_adds_provisional_hits(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox.jsonl")
    inbox.append(
        make_entry(
            user_message="how do I run git rebase interactively?",
            assistant_message="use git rebase -i HEAD~N",
            conversation_id="c1",
            turn_offset=1,
        )
    )

    engine = GeneratorEngine(config=RetrievalConfig(), inbox=inbox)
    results = engine.search(
        "git rebase",
        [_bullet("b1", "unrelated text")],
        limit=10,
    )
    provisional = [r for r in results if r.metadata.get("provisional")]
    assert len(provisional) == 1
    p = provisional[0]
    assert p.bullet_id.startswith("inbox:")
    assert p.metadata["conversation_id"] == "c1"
    # Score should be halved relative to raw keyword score ( * 0.5 default )
    assert 0.0 < p.final_score < 1.0


def test_provisional_score_factor_applied(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox.jsonl")
    inbox.append(
        make_entry(
            user_message="git rebase is hard",
            assistant_message="not really",
        )
    )
    full = GeneratorEngine(
        config=RetrievalConfig(), inbox=inbox, provisional_score_factor=1.0,
    )
    halved = GeneratorEngine(
        config=RetrievalConfig(), inbox=inbox, provisional_score_factor=0.5,
    )

    full_results = full.search("git rebase", [_bullet("x", "irrelevant")], limit=10)
    halved_results = halved.search("git rebase", [_bullet("x", "irrelevant")], limit=10)

    full_scores = [r.final_score for r in full_results if r.metadata.get("provisional")]
    half_scores = [r.final_score for r in halved_results if r.metadata.get("provisional")]
    assert full_scores and half_scores
    assert abs(half_scores[0] * 2 - full_scores[0]) < 1e-6


def test_include_pending_false_skips_inbox(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox.jsonl")
    inbox.append(make_entry(user_message="git rebase", assistant_message="."))
    engine = GeneratorEngine(config=RetrievalConfig(), inbox=inbox)
    results = engine.search(
        "git rebase", [_bullet("x", "irrelevant")], limit=10, include_pending=False,
    )
    assert all(not r.metadata.get("provisional") for r in results)


def test_consumed_entries_do_not_appear(tmp_path: Path) -> None:
    inbox = Inbox(tmp_path / "inbox.jsonl")
    inbox.append(make_entry(user_message="git rebase", assistant_message="..."))
    claimed = inbox.claim(1)
    inbox.mark_consumed([claimed[0].id])

    engine = GeneratorEngine(config=RetrievalConfig(), inbox=inbox)
    results = engine.search("git rebase", [_bullet("x", "irrelevant")], limit=10)
    assert all(not r.metadata.get("provisional") for r in results)
