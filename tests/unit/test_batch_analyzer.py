"""Unit tests for BatchAnalyzer (STORY-R095)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memorus.core.config import ReflectorConfig
from memorus.core.engines.reflector.batch_analyzer import (
    BatchAnalyzer,
    BatchReport,
    estimate_tokens,
)
from memorus.core.engines.reflector.inbox import Inbox, make_entry
from memorus.core.types import CandidateBullet


def _seed_inbox(path: Path, n: int) -> Inbox:
    box = Inbox(path / "inbox.jsonl")
    for i in range(n):
        box.append(
            make_entry(
                user_message=f"turn {i} about git rebase",
                assistant_message=f"answer {i} explaining the rebase flow",
                conversation_id="c1",
                turn_offset=i,
            )
        )
    return box


def _cfg(**overrides: Any) -> ReflectorConfig:
    cfg = ReflectorConfig(mode="hybrid")
    cfg.batch.batch_enabled = True
    for k, v in overrides.items():
        setattr(cfg.batch, k, v)
    return cfg


def test_should_run_respects_min_turns(tmp_path: Path) -> None:
    cfg = _cfg(batch_min_turns=5, batch_max_age_seconds=10**9)
    inbox = _seed_inbox(tmp_path, 3)
    analyzer = BatchAnalyzer(cfg, inbox, llm_caller=lambda p, **kw: "{}")
    assert analyzer.should_run() is False
    inbox.append(make_entry(user_message="x", assistant_message="y"))
    inbox.append(make_entry(user_message="x", assistant_message="y"))
    assert analyzer.should_run() is True


def test_two_step_flow_produces_bullets(tmp_path: Path) -> None:
    cfg = _cfg(batch_min_turns=1)
    inbox = _seed_inbox(tmp_path, 3)
    pending_ids = [e.id for e in inbox.list_pending()]

    calls: list[str] = []

    def llm(prompt: str, *, max_tokens: int, stage: str) -> str:
        calls.append(stage)
        if stage == "analyze":
            return json.dumps(
                {
                    "themes": [
                        {
                            "id": "t1",
                            "title": "git rebase",
                            "turn_ids": pending_ids,
                            "entities": [],
                            "tools": ["git"],
                        }
                    ]
                }
            )
        return json.dumps(
            {
                "bullets": [
                    {
                        "theme_id": "t1",
                        "distilled_rule": "When rebasing, prefer interactive mode.",
                        "content": "Git rebase is flexible; interactive mode lets you edit commits.",
                        "section": "commands",
                        "knowledge_type": "method",
                        "related_tools": ["git"],
                        "key_entities": ["git rebase"],
                        "tags": ["git"],
                        "turn_ids": pending_ids,
                    }
                ]
            }
        )

    emitted: list[list[CandidateBullet]] = []
    analyzer = BatchAnalyzer(cfg, inbox, llm_caller=llm)
    analyzer.set_emit_callback(lambda bs: emitted.append(list(bs)))

    report = analyzer.run_once(force=True)
    assert isinstance(report, BatchReport)
    assert report.turns_processed == 3
    assert report.bullets_produced == 1
    assert report.bullets_fallback == 0
    assert calls == ["analyze", "generate"]
    assert len(emitted) == 1
    bullet = emitted[0][0]
    assert bullet.content.startswith("Git rebase")
    # Source refs should track all three turns.
    assert len(bullet.sources) == 3

    # Inbox should mark all entries consumed.
    assert inbox.list_pending() == []
    statuses = [e.status for e in inbox.list_all()]
    assert set(statuses) == {"consumed"}


def test_retry_then_fallback_on_persistent_failure(tmp_path: Path) -> None:
    cfg = _cfg(batch_min_turns=1, max_batch_retries=2)
    inbox = _seed_inbox(tmp_path, 2)

    fb_called: list[str] = []

    def fallback(entry: Any) -> list[CandidateBullet]:
        fb_called.append(entry.id)
        return []

    def failing_llm(prompt: str, *, max_tokens: int, stage: str) -> str:
        raise RuntimeError("boom")

    analyzer = BatchAnalyzer(
        cfg, inbox, llm_caller=failing_llm, fallback_distill=fallback,
    )
    report = analyzer.run_once(force=True)
    assert report.batches_retried == 2
    assert report.batches_failed == 1
    assert len(fb_called) == 2
    # All entries still marked consumed so they don't loop forever.
    assert inbox.list_pending() == []


def test_batch_splits_by_max_size(tmp_path: Path) -> None:
    cfg = _cfg(batch_min_turns=1, max_batch_size=3)
    inbox = _seed_inbox(tmp_path, 7)

    calls = {"analyze": 0, "generate": 0}

    def llm(prompt: str, *, max_tokens: int, stage: str) -> str:
        calls[stage] += 1
        if stage == "analyze":
            return json.dumps({"themes": []})
        return json.dumps({"bullets": []})

    analyzer = BatchAnalyzer(cfg, inbox, llm_caller=llm)
    report = analyzer.run_once(force=True)
    # 7 turns, max 3 per batch -> ceil(7/3) = 3 batches
    assert report.batches == 3
    assert report.turns_processed == 7
    # analyze called once per batch
    assert calls["analyze"] == 3
    # generate not called when themes are empty
    assert calls["generate"] == 0


def test_batch_splits_by_token_budget(tmp_path: Path) -> None:
    # Shrink budget so each turn takes its own batch
    cfg = _cfg(batch_min_turns=1, max_batch_size=50, prompt_token_budget=80)
    inbox = _seed_inbox(tmp_path, 4)

    calls = {"analyze": 0}

    def llm(prompt: str, *, max_tokens: int, stage: str) -> str:
        calls[stage] = calls.get(stage, 0) + 1
        if stage == "analyze":
            return json.dumps({"themes": []})
        return json.dumps({"bullets": []})

    analyzer = BatchAnalyzer(cfg, inbox, llm_caller=llm)
    report = analyzer.run_once(force=True)
    # Each turn renders to well over 80 tokens worth of prompt after headers,
    # so each turn should go into its own batch.
    assert report.batches >= 2
    assert calls["analyze"] >= 2


def test_estimate_tokens_monotonic() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") >= 1
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)
