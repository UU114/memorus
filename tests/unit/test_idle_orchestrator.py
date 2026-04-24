"""Unit tests for IdleOrchestrator (STORY-R094)."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from memorus.core.config import ConsolidateConfig, CuratorConfig
from memorus.core.daemon.orchestrator import IdleOrchestrator
from memorus.core.engines.curator.engine import ExistingBullet
from memorus.core.engines.curator.executor import BulletRecord, MetadataPatch


class NoopAdapter:
    """MemoryAdapter that reports everything as missing."""

    def get_bullet(self, bullet_id: str) -> BulletRecord | None:
        return None

    def update_metadata(self, bullet_id: str, patch: MetadataPatch) -> None:
        pass

    def mark_deleted(self, bullet_id: str, when: datetime) -> None:
        pass


def _build(tmp_path: Path) -> IdleOrchestrator:
    cfg = ConsolidateConfig(
        check_interval_secs=1,
        min_idle_secs=0,
        min_gap_secs=0,
        log_path=str(tmp_path / "log.md"),
        review_queue_path=str(tmp_path / "q.jsonl"),
    )
    return IdleOrchestrator(
        config=cfg,
        curator_config=CuratorConfig(),
        adapter=NoopAdapter(),
        load_bullets=lambda: [],
    )


@pytest.mark.asyncio
async def test_force_run_executes_and_writes_log(tmp_path: Path) -> None:
    orch = _build(tmp_path)
    report = await orch.run_once(force=True)
    assert report is not None
    assert orch.last_run is not None
    log_file = Path(orch._config.log_path)
    assert log_file.exists()
    assert "consolidate" in log_file.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_should_run_rejects_active_sessions(tmp_path: Path) -> None:
    orch = _build(tmp_path)

    async def _active() -> bool:
        return False

    assert await orch.should_run(_active) is False


@pytest.mark.asyncio
async def test_should_run_accepts_idle(tmp_path: Path) -> None:
    orch = _build(tmp_path)

    async def _empty() -> bool:
        return True

    assert await orch.should_run(_empty) is True


@pytest.mark.asyncio
async def test_should_run_respects_min_idle(tmp_path: Path) -> None:
    cfg = ConsolidateConfig(
        check_interval_secs=1,
        min_idle_secs=999,
        min_gap_secs=0,
        log_path=str(tmp_path / "log.md"),
        review_queue_path=str(tmp_path / "q.jsonl"),
    )
    orch = IdleOrchestrator(
        cfg, CuratorConfig(), NoopAdapter(), load_bullets=lambda: []
    )
    orch.touch()

    async def _empty() -> bool:
        return True

    assert await orch.should_run(_empty) is False


@pytest.mark.asyncio
async def test_should_run_respects_min_gap(tmp_path: Path) -> None:
    cfg = ConsolidateConfig(
        check_interval_secs=1,
        min_idle_secs=0,
        min_gap_secs=999,
        log_path=str(tmp_path / "log.md"),
        review_queue_path=str(tmp_path / "q.jsonl"),
    )
    orch = IdleOrchestrator(
        cfg, CuratorConfig(), NoopAdapter(), load_bullets=lambda: []
    )
    # Simulate a recent last_run
    await orch.run_once(force=True)

    async def _empty() -> bool:
        return True

    # last_run is now "now" → gap < 999s so gate is closed
    assert await orch.should_run(_empty) is False


@pytest.mark.asyncio
async def test_exclusive_lock_skips_second_caller(tmp_path: Path) -> None:
    """A non-forced run while another is ongoing returns None."""
    orch = _build(tmp_path)

    # Acquire the lock manually as a proxy for an ongoing run
    await orch._run_lock.acquire()
    try:
        report = await orch.run_once(force=False)
        assert report is None
    finally:
        orch._run_lock.release()


@pytest.mark.asyncio
async def test_stop_terminates_loop(tmp_path: Path) -> None:
    orch = _build(tmp_path)

    async def _empty() -> bool:
        return True

    task = asyncio.create_task(orch.run_forever(_empty))
    # Yield once so the loop can enter the wait
    await asyncio.sleep(0)
    orch.stop()
    await asyncio.wait_for(task, timeout=2.0)
    assert task.done()


@pytest.mark.asyncio
async def test_exception_in_load_does_not_crash(tmp_path: Path) -> None:
    cfg = ConsolidateConfig(
        check_interval_secs=1,
        min_idle_secs=0,
        min_gap_secs=0,
        log_path=str(tmp_path / "log.md"),
        review_queue_path=str(tmp_path / "q.jsonl"),
    )

    def _bad_load() -> list[ExistingBullet]:
        raise RuntimeError("backend down")

    orch = IdleOrchestrator(cfg, CuratorConfig(), NoopAdapter(), load_bullets=_bad_load)
    report = await orch.run_once(force=True)
    assert report is None  # handled internally, no exception raised


@pytest.mark.asyncio
async def test_log_file_created_with_correct_format(tmp_path: Path) -> None:
    orch = _build(tmp_path)
    await orch.run_once(force=True)
    content = Path(orch._config.log_path).read_text(encoding="utf-8")
    # All five count lines present
    assert "merged:" in content
    assert "soft-deleted:" in content
    assert "auto-supersede:" in content
    assert "queued-for-review:" in content
    assert "marked-conflict:" in content
    assert "duration:" in content
