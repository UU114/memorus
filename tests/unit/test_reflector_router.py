"""Unit tests for ReflectorRouter (STORY-R095)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memorus.core.config import ReflectorConfig
from memorus.core.engines.reflector.inbox import Inbox
from memorus.core.engines.reflector.router import ReflectorRouter, RouteKind
from memorus.core.types import InteractionEvent


def _cfg(mode: str = "hybrid", batch_enabled: bool = True) -> ReflectorConfig:
    cfg = ReflectorConfig(mode=mode)
    cfg.batch.batch_enabled = batch_enabled
    return cfg


# Note: batch_enabled defaults to False — tests explicitly opt-in.


def _router(tmp_path: Path, cfg: ReflectorConfig) -> ReflectorRouter:
    inbox = Inbox(tmp_path / "inbox.jsonl")
    return ReflectorRouter(cfg, inbox=inbox)


def _event(user: str = "hi", asst: str = "hello", **meta: object) -> InteractionEvent:
    return InteractionEvent(
        user_message=user,
        assistant_message=asst,
        metadata=dict(meta),
    )


def test_correction_flag_routes_realtime(tmp_path: Path) -> None:
    r = _router(tmp_path, _cfg())
    decision = r.route(_event(is_correction=True))
    assert decision.kind == RouteKind.REALTIME
    assert decision.reason == "correction"
    assert decision.entry is None
    assert r.inbox.list_pending() == []


def test_rules_mode_routes_realtime(tmp_path: Path) -> None:
    r = _router(tmp_path, _cfg(mode="rules"))
    decision = r.route(_event())
    assert decision.kind == RouteKind.REALTIME
    assert decision.reason == "rules_mode"
    assert r.inbox.list_pending() == []


def test_batch_disabled_routes_realtime(tmp_path: Path) -> None:
    r = _router(tmp_path, _cfg(batch_enabled=False))
    decision = r.route(_event())
    assert decision.kind == RouteKind.REALTIME
    assert decision.reason == "batch_disabled"


def test_normal_turn_routes_to_inbox(tmp_path: Path) -> None:
    r = _router(tmp_path, _cfg())
    decision = r.route(_event(user="please add a CI workflow", asst="sure, here is a yaml"))
    assert decision.kind == RouteKind.INBOX
    assert decision.reason == "enqueued"
    assert decision.entry is not None
    pending = r.inbox.list_pending()
    assert len(pending) == 1
    assert pending[0].id == decision.entry.id
    assert pending[0].status == "pending"


def test_correction_regex_fallback(tmp_path: Path) -> None:
    r = _router(tmp_path, _cfg())
    # no explicit flag, but content looks like a correction
    decision = r.route(
        InteractionEvent(
            user_message="Actually, that command is wrong — do X instead",
            assistant_message="got it",
        )
    )
    assert decision.kind == RouteKind.REALTIME
    assert decision.reason == "correction"


def test_multiple_turns_accumulate(tmp_path: Path) -> None:
    r = _router(tmp_path, _cfg())
    for i in range(5):
        r.route(_event(user=f"question {i}", asst=f"answer {i}",
                       conversation_id="c1", turn_offset=i))
    pending = r.inbox.list_pending()
    assert len(pending) == 5
    # Ensure file is valid JSONL
    lines = (tmp_path / "inbox.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5
    for ln in lines:
        raw = json.loads(ln)
        assert raw["status"] == "pending"
        assert raw["conversation_id"] == "c1"
