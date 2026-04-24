"""Unit tests for the three-tier conflict triage in ConsolidateExecutor."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memorus.core.config import ConsolidateConfig
from memorus.core.engines.curator.engine import (
    ConsolidateConflictInfo,
    ConsolidateResult,
    MergeGroup,
)
from memorus.core.engines.curator.executor import (
    BulletRecord,
    ConsolidateExecutor,
    ExecutionReport,
    MetadataPatch,
    append_consolidate_log,
    conflict_confidence,
)


class FakeAdapter:
    """In-memory MemoryAdapter implementation."""

    def __init__(self) -> None:
        self.records: dict[str, BulletRecord] = {}
        self.updates: list[tuple[str, MetadataPatch]] = []
        self.deletes: list[tuple[str, datetime]] = []

    def add(self, rec: BulletRecord) -> None:
        self.records[rec.bullet_id] = rec

    def get_bullet(self, bullet_id: str) -> BulletRecord | None:
        return self.records.get(bullet_id)

    def update_metadata(self, bullet_id: str, patch: MetadataPatch) -> None:
        self.updates.append((bullet_id, patch))

    def mark_deleted(self, bullet_id: str, when: datetime) -> None:
        self.deletes.append((bullet_id, when))


def _mk(content: str, recall: int = 0, offset_secs: int = 0) -> BulletRecord:
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return BulletRecord(
        bullet_id=f"id-{content[:3]}-{offset_secs}",
        content=content,
        recall_count=recall,
        created_at=base + timedelta(seconds=offset_secs),
    )


def _conflict(
    a: BulletRecord,
    b: BulletRecord,
    sim: float = 0.9,
    ctype: str = "negation",
) -> ConsolidateConflictInfo:
    return ConsolidateConflictInfo(
        a_id=a.bullet_id,
        b_id=b.bullet_id,
        a_content=a.content,
        b_content=b.content,
        similarity=sim,
        conflict_type=ctype,
        reason="test",
    )


class TestHighConfidenceAutoSupersede:
    def test_auto_supersede_kills_loser(self) -> None:
        store = FakeAdapter()
        a = _mk("foo", recall=1, offset_secs=0)
        b = _mk("bar", recall=5, offset_secs=10)  # higher recall = winner
        store.add(a)
        store.add(b)

        result = ConsolidateResult(conflicts=[_conflict(a, b, sim=0.9, ctype="negation")])
        exec_ = ConsolidateExecutor(store, ConsolidateConfig())
        report = exec_.execute(result)

        assert report.auto_supersede == 1
        assert report.queued_for_review == 0
        assert report.marked_conflict == 0
        deleted = [d[0] for d in store.deletes]
        assert a.bullet_id in deleted
        assert b.bullet_id not in deleted

    def test_winner_gets_audit_metadata(self) -> None:
        store = FakeAdapter()
        a = _mk("aaa", recall=0, offset_secs=0)
        b = _mk("bbb", recall=10, offset_secs=10)
        store.add(a)
        store.add(b)

        exec_ = ConsolidateExecutor(store, ConsolidateConfig())
        exec_.execute(
            ConsolidateResult(conflicts=[_conflict(a, b, sim=0.9, ctype="negation")])
        )

        # Find the winner (b) patch with conflict_resolution
        winner_patches = [p for (bid, p) in store.updates if bid == b.bullet_id]
        assert any(p.conflict_resolution is not None for p in winner_patches)


class TestMediumConfidenceReviewQueue:
    def test_review_queue_written(self, tmp_path: Path) -> None:
        store = FakeAdapter()
        a = _mk("aaa", recall=0)
        b = _mk("bbb", recall=0, offset_secs=10)
        store.add(a)
        store.add(b)

        q_path = tmp_path / "q.jsonl"
        cfg = ConsolidateConfig(review_queue_path=str(q_path))
        # similarity 0.6 * weight(negation)=1.0 = 0.6 -> in review band
        result = ConsolidateResult(
            conflicts=[_conflict(a, b, sim=0.6, ctype="negation")]
        )
        exec_ = ConsolidateExecutor(store, cfg)
        report = exec_.execute(result)

        assert report.queued_for_review == 1
        assert report.auto_supersede == 0
        assert q_path.exists()
        lines = q_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        row = json.loads(lines[0])
        assert row["a_id"] == a.bullet_id
        assert row["b_id"] == b.bullet_id
        assert row["hint"] == f"memorus review {b.bullet_id}"
        assert 0.55 <= row["confidence"] <= 0.65


class TestLowConfidenceMark:
    def test_low_confidence_marks_both_sides(self) -> None:
        store = FakeAdapter()
        a = _mk("aaa")
        b = _mk("bbb", offset_secs=10)
        store.add(a)
        store.add(b)

        cfg = ConsolidateConfig()
        # sim 0.2 → confidence ~0.2 → below review band (0.50)
        result = ConsolidateResult(
            conflicts=[_conflict(a, b, sim=0.2, ctype="negation")]
        )
        exec_ = ConsolidateExecutor(store, cfg)
        report = exec_.execute(result)

        assert report.marked_conflict == 1
        assert report.auto_supersede == 0
        assert report.queued_for_review == 0
        # Both sides received a conflict_with patch
        marked = {bid for (bid, p) in store.updates if p.conflict_with}
        assert a.bullet_id in marked
        assert b.bullet_id in marked


class TestConfidenceWeights:
    def test_negation_highest_weight(self) -> None:
        b = _mk("b")
        base = _conflict(b, b, sim=0.8, ctype="negation")
        w_neg = conflict_confidence(base)

        base_contra = _conflict(b, b, sim=0.8, ctype="contradiction")
        w_con = conflict_confidence(base_contra)

        base_val = _conflict(b, b, sim=0.8, ctype="value_difference")
        w_val = conflict_confidence(base_val)

        assert w_neg > w_con > w_val


class TestMergeGroupSupersede:
    def test_picks_highest_recall(self) -> None:
        store = FakeAdapter()
        a = _mk("aaa", recall=1, offset_secs=0)
        b = _mk("bbb", recall=5, offset_secs=10)
        store.add(a)
        store.add(b)

        cfg = ConsolidateConfig()
        result = ConsolidateResult(
            merge_groups=[
                MergeGroup(bullet_ids=[a.bullet_id, b.bullet_id], max_similarity=0.9)
            ]
        )
        exec_ = ConsolidateExecutor(store, cfg)
        report = exec_.execute(result)

        assert report.merged_groups == 1
        deleted_ids = [d[0] for d in store.deletes]
        assert a.bullet_id in deleted_ids
        assert b.bullet_id not in deleted_ids

    def test_missing_bullets_noop(self) -> None:
        store = FakeAdapter()
        a = _mk("aaa")
        store.add(a)

        cfg = ConsolidateConfig()
        result = ConsolidateResult(
            merge_groups=[
                MergeGroup(bullet_ids=[a.bullet_id, "ghost-id"], max_similarity=0.9)
            ]
        )
        exec_ = ConsolidateExecutor(store, cfg)
        report = exec_.execute(result)

        assert report.merged_groups == 0
        assert not report.errors


class TestSoftDelete:
    def test_only_live_counted(self) -> None:
        store = FakeAdapter()
        a = _mk("aaa")
        store.add(a)

        cfg = ConsolidateConfig()
        result = ConsolidateResult(redundant_ids=[a.bullet_id, "ghost-id"])
        exec_ = ConsolidateExecutor(store, cfg)
        report = exec_.execute(result)

        assert report.soft_deleted == 1
        deleted_ids = [d[0] for d in store.deletes]
        assert deleted_ids == [a.bullet_id]


class TestLogFormat:
    def test_block_matches_spec(self, tmp_path: Path) -> None:
        path = tmp_path / "log.md"
        report = ExecutionReport(
            merged_groups=3,
            merged_bullets_total=8,
            soft_deleted=1,
            auto_supersede=1,
            queued_for_review=1,
            marked_conflict=1,
            duration_seconds=1.2,
        )
        ts = datetime(2026, 4, 23, 3, 15, 12, tzinfo=timezone.utc)
        append_consolidate_log(path, report, ts)
        content = path.read_text(encoding="utf-8")
        assert "## 2026-04-23T03:15:12Z consolidate" in content
        assert "merged: 3 groups (8 bullets -> 3)" in content
        assert "soft-deleted: 1" in content
        assert "auto-supersede: 1" in content
        assert "queued-for-review: 1" in content
        assert "marked-conflict: 1" in content
        assert "duration: 1.2s" in content


class TestReviewQueueFallback:
    def test_unwritable_falls_back_to_mark(self, tmp_path: Path, monkeypatch) -> None:
        store = FakeAdapter()
        a = _mk("aaa")
        b = _mk("bbb", offset_secs=10)
        store.add(a)
        store.add(b)

        # Force Path.mkdir to raise OSError
        import memorus.core.engines.curator.executor as exec_mod

        def _boom(self, *args, **kwargs):
            raise OSError("permission denied")

        monkeypatch.setattr("pathlib.Path.mkdir", _boom)
        cfg = ConsolidateConfig(
            review_queue_path=str(tmp_path / "no_perm" / "q.jsonl"),
        )
        result = ConsolidateResult(
            conflicts=[_conflict(a, b, sim=0.6, ctype="negation")]
        )
        exec_ = ConsolidateExecutor(store, cfg)
        report = exec_.execute(result)
        # The failed enqueue still counts as queued_for_review but marks as fallback
        assert report.queued_for_review == 1
        # After fallback, there should be a conflict_with marking instead
        marked = {bid for (bid, p) in store.updates if p.conflict_with}
        assert marked
