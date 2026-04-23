"""Unit tests for memorus.core.types.SourceRef and merge_sources helper.

Covers:
- SourceRef construction, serialization, frozen semantics
- merge_sources: dedup by (conv_id, turn_offset), sort by timestamp, cap at 50
- BulletMetadata / CandidateBullet carry the sources field
- JSONL backward compatibility (legacy rows without sources load cleanly)
- Reflector emits at least one SourceRef per CandidateBullet
- Curator merger unions candidate sources with existing metadata sources
- Cross-language JSON shape is stable (keys + ISO-8601 timestamp)
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from memorus.core.engines.curator.engine import ExistingBullet
from memorus.core.engines.curator.merger import (
    KeepBestStrategy,
    MergeContentStrategy,
)
from memorus.core.engines.reflector.distiller import BulletDistiller
from memorus.core.types import (
    SOURCES_CAP,
    BulletMetadata,
    BulletSection,
    CandidateBullet,
    DetectedPattern,
    InteractionEvent,
    KnowledgeType,
    ScoredCandidate,
    SourceRef,
    merge_sources,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _src(
    conversation_id: str = "conv-1",
    turn_offset: int = 0,
    seconds: int = 0,
    role: str = "assistant",
    turn_hash: str | None = None,
) -> SourceRef:
    """Build a deterministic SourceRef for tests."""
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    return SourceRef(
        conversation_id=conversation_id,
        turn_hash=turn_hash or SourceRef.compute_turn_hash(f"{conversation_id}:{turn_offset}"),
        turn_offset=turn_offset,
        timestamp=ts,
        role=role,
    )


# ---------------------------------------------------------------------------
# SourceRef basics
# ---------------------------------------------------------------------------


class TestSourceRef:
    def test_compute_turn_hash_is_16_hex(self) -> None:
        h = SourceRef.compute_turn_hash("hello world")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_turn_hash_is_sha256_prefix(self) -> None:
        import hashlib

        expected = hashlib.sha256(b"hello").hexdigest()[:16]
        assert SourceRef.compute_turn_hash("hello") == expected

    def test_frozen_semantics(self) -> None:
        s = _src()
        with pytest.raises(Exception):
            s.conversation_id = "other"  # type: ignore[misc]

    def test_serialization_roundtrip(self) -> None:
        s = _src(conversation_id="abc", turn_offset=3, seconds=10, role="user")
        data = s.model_dump(mode="json")
        assert data["conversation_id"] == "abc"
        assert data["turn_offset"] == 3
        assert data["role"] == "user"
        assert isinstance(data["timestamp"], str)
        restored = SourceRef.model_validate(data)
        assert restored == s

    def test_negative_turn_offset_rejected(self) -> None:
        with pytest.raises(Exception):
            SourceRef(
                conversation_id="x",
                turn_hash="a" * 16,
                turn_offset=-1,
                timestamp=datetime.now(timezone.utc),
                role="user",
            )


# ---------------------------------------------------------------------------
# merge_sources semantics
# ---------------------------------------------------------------------------


class TestMergeSources:
    def test_empty(self) -> None:
        assert merge_sources([], []) == []

    def test_dedup_by_conversation_and_offset(self) -> None:
        a = [_src("c1", 0, seconds=0), _src("c1", 1, seconds=1)]
        b = [_src("c1", 0, seconds=5), _src("c2", 0, seconds=2)]
        merged = merge_sources(a, b)
        # 3 unique (conv_id, offset) combinations
        assert len(merged) == 3
        keys = {(s.conversation_id, s.turn_offset) for s in merged}
        assert keys == {("c1", 0), ("c1", 1), ("c2", 0)}

    def test_sorted_ascending_by_timestamp(self) -> None:
        a = [_src("c1", 2, seconds=30), _src("c1", 0, seconds=10)]
        b = [_src("c2", 0, seconds=20)]
        merged = merge_sources(a, b)
        timestamps = [s.timestamp for s in merged]
        assert timestamps == sorted(timestamps)

    def test_cap_50_keeps_earliest_25_plus_latest_25(self) -> None:
        # Build 60 unique sources
        items = [_src(f"c-{i:03d}", 0, seconds=i) for i in range(60)]
        merged = merge_sources(items, [])
        assert len(merged) == SOURCES_CAP
        # First 25 should match items 0..24
        assert [s.conversation_id for s in merged[:25]] == [f"c-{i:03d}" for i in range(25)]
        # Last 25 should match items 35..59
        assert [s.conversation_id for s in merged[25:]] == [f"c-{i:03d}" for i in range(35, 60)]

    def test_cap_under_limit_preserves_all(self) -> None:
        items = [_src(f"c-{i}", 0, seconds=i) for i in range(10)]
        merged = merge_sources(items, [])
        assert len(merged) == 10

    def test_first_occurrence_wins_on_dedup(self) -> None:
        a = [_src("c1", 0, seconds=0, role="user")]
        b = [_src("c1", 0, seconds=99, role="assistant")]
        merged = merge_sources(a, b)
        assert len(merged) == 1
        assert merged[0].role == "user"
        assert merged[0].timestamp == a[0].timestamp


# ---------------------------------------------------------------------------
# BulletMetadata / CandidateBullet field + JSONL compat
# ---------------------------------------------------------------------------


class TestBulletMetadataSources:
    def test_default_empty_sources(self) -> None:
        meta = BulletMetadata()
        assert meta.sources == []

    def test_sources_round_trip_json(self) -> None:
        meta = BulletMetadata(sources=[_src("c1", 0), _src("c1", 1)])
        dumped = meta.model_dump_json()
        reloaded = BulletMetadata.model_validate_json(dumped)
        assert len(reloaded.sources) == 2
        assert reloaded.sources[0].conversation_id == "c1"

    def test_backward_compat_legacy_jsonl_without_sources(self) -> None:
        # A JSONL line written before STORY-R092 has no `sources` key.
        legacy_line = json.dumps(
            {
                "section": "general",
                "knowledge_type": "knowledge",
                "instructivity_score": 80.0,
                "recall_count": 2,
                "decay_weight": 0.9,
                "related_tools": ["git"],
                "related_files": [],
                "key_entities": [],
                "tags": [],
                "distilled_rule": None,
                "source_type": "interaction",
                "scope": "global",
                "schema_version": 1,
                "incompatible_tags": [],
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        )
        meta = BulletMetadata.model_validate_json(legacy_line)
        assert meta.sources == []
        assert meta.related_tools == ["git"]

    def test_candidate_bullet_has_sources(self) -> None:
        cb = CandidateBullet(content="x", sources=[_src()])
        assert len(cb.sources) == 1


# ---------------------------------------------------------------------------
# Reflector emits SourceRef
# ---------------------------------------------------------------------------


class TestReflectorEmitsSources:
    def test_distiller_emits_source_from_pattern(self) -> None:
        event = InteractionEvent(
            user_message="how do I rebase?",
            assistant_message="Run 'git rebase -i main' to interactively rebase.",
            session_id="session-abc",
            metadata={"turn_offset": 5},
        )
        pattern = DetectedPattern(
            pattern_type="new_tool",
            content="Run 'git rebase -i main' to interactively rebase.",
            confidence=0.8,
            source_event=event,
        )
        scored = ScoredCandidate(
            pattern=pattern,
            section=BulletSection.TOOLS,
            knowledge_type=KnowledgeType.METHOD,
            instructivity_score=80.0,
        )
        bullet = BulletDistiller().distill(scored)
        assert len(bullet.sources) == 1
        s = bullet.sources[0]
        assert s.conversation_id == "session-abc"
        assert s.turn_offset == 5
        assert s.role == "assistant"
        assert s.turn_hash == SourceRef.compute_turn_hash(event.assistant_message)

    def test_distiller_no_source_event_yields_empty_sources(self) -> None:
        pattern = DetectedPattern(
            pattern_type="new_tool",
            content="some knowledge",
            confidence=0.8,
            source_event=None,
        )
        scored = ScoredCandidate(
            pattern=pattern,
            section=BulletSection.GENERAL,
            knowledge_type=KnowledgeType.KNOWLEDGE,
            instructivity_score=60.0,
        )
        bullet = BulletDistiller().distill(scored)
        assert bullet.sources == []

    def test_distiller_no_conversation_id_yields_empty(self) -> None:
        event = InteractionEvent(
            user_message="q",
            assistant_message="a",
            session_id="",
            user_id="",
        )
        pattern = DetectedPattern(
            pattern_type="new_tool",
            content="a",
            confidence=0.8,
            source_event=event,
        )
        scored = ScoredCandidate(
            pattern=pattern,
            section=BulletSection.GENERAL,
            knowledge_type=KnowledgeType.KNOWLEDGE,
            instructivity_score=60.0,
        )
        bullet = BulletDistiller().distill(scored)
        assert bullet.sources == []


# ---------------------------------------------------------------------------
# Curator preserves and merges sources
# ---------------------------------------------------------------------------


class TestCuratorMergesSources:
    def _candidate_with_sources(self, *refs: SourceRef) -> CandidateBullet:
        return CandidateBullet(content="candidate content", sources=list(refs))

    def _existing_with_sources(self, *refs: SourceRef) -> ExistingBullet:
        return ExistingBullet(
            bullet_id="b1",
            content="existing content",
            metadata={
                "instructivity_score": 50.0,
                "recall_count": 3,
                "related_tools": [],
                "key_entities": [],
                "tags": [],
                "sources": [r.model_dump(mode="json") for r in refs],
            },
        )

    def test_keep_best_unions_sources(self) -> None:
        candidate = self._candidate_with_sources(
            _src("c1", 0, seconds=10),
            _src("c1", 1, seconds=20),
        )
        existing = self._existing_with_sources(
            _src("c1", 0, seconds=10),  # duplicate
            _src("c2", 0, seconds=5),
        )
        result = KeepBestStrategy().merge(candidate, existing)
        sources = result.merged_metadata["sources"]
        assert len(sources) == 3
        # sorted ascending by timestamp
        ts = [s["timestamp"] for s in sources]
        assert ts == sorted(ts)

    def test_merge_content_unions_sources(self) -> None:
        candidate = self._candidate_with_sources(_src("c1", 0, seconds=1))
        existing = self._existing_with_sources(_src("c1", 1, seconds=2))
        result = MergeContentStrategy().merge(candidate, existing)
        assert len(result.merged_metadata["sources"]) == 2

    def test_merge_preserves_when_candidate_has_none(self) -> None:
        candidate = self._candidate_with_sources()
        existing = self._existing_with_sources(_src("c1", 0, seconds=0))
        result = KeepBestStrategy().merge(candidate, existing)
        assert len(result.merged_metadata["sources"]) == 1

    def test_merge_caps_at_50(self) -> None:
        candidate = self._candidate_with_sources(
            *(_src(f"c-{i:03d}", 0, seconds=i) for i in range(40))
        )
        existing = self._existing_with_sources(
            *(_src(f"d-{i:03d}", 0, seconds=1000 + i) for i in range(30))
        )
        result = KeepBestStrategy().merge(candidate, existing)
        assert len(result.merged_metadata["sources"]) == SOURCES_CAP


# ---------------------------------------------------------------------------
# Cross-language JSON shape (must match Rust's serde output)
# ---------------------------------------------------------------------------


class TestCrossLanguageShape:
    def test_source_ref_json_has_expected_keys(self) -> None:
        s = _src("conv-1", 2, seconds=7, role="user")
        data = json.loads(s.model_dump_json())
        assert set(data.keys()) == {
            "conversation_id",
            "turn_hash",
            "turn_offset",
            "timestamp",
            "role",
        }
        # timestamp must be ISO-8601 parseable
        assert datetime.fromisoformat(data["timestamp"])
