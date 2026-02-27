"""Unit tests for memx.types — BulletMetadata, enums, and auxiliary models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from memx.types import (
    BulletMetadata,
    BulletSection,
    CandidateBullet,
    DetectedPattern,
    InteractionEvent,
    KnowledgeType,
    ScoredCandidate,
    SourceType,
)


# ── Enum tests ──────────────────────────────────────────────────────────


class TestBulletSection:
    def test_has_at_least_8_members(self) -> None:
        assert len(BulletSection) >= 8

    def test_expected_members(self) -> None:
        expected = {
            "commands",
            "debugging",
            "architecture",
            "workflow",
            "tools",
            "patterns",
            "preferences",
            "general",
        }
        assert {s.value for s in BulletSection} == expected

    def test_str_mixin(self) -> None:
        assert BulletSection.GENERAL == "general"
        assert BulletSection.COMMANDS.value == "commands"


class TestKnowledgeType:
    def test_has_5_members(self) -> None:
        assert len(KnowledgeType) == 5

    def test_expected_members(self) -> None:
        expected = {"method", "trick", "pitfall", "preference", "knowledge"}
        assert {k.value for k in KnowledgeType} == expected


class TestSourceType:
    def test_has_3_members(self) -> None:
        assert len(SourceType) == 3

    def test_expected_members(self) -> None:
        expected = {"interaction", "manual", "import"}
        assert {s.value for s in SourceType} == expected


# ── BulletMetadata tests ────────────────────────────────────────────────


class TestBulletMetadataDefaults:
    """All fields have defaults — zero-arg construction must succeed."""

    def test_zero_arg_construction(self) -> None:
        b = BulletMetadata()
        assert b.section == BulletSection.GENERAL
        assert b.knowledge_type == KnowledgeType.KNOWLEDGE
        assert b.instructivity_score == 50.0
        assert b.recall_count == 0
        assert b.last_recall is None
        assert b.decay_weight == 1.0
        assert b.related_tools == []
        assert b.related_files == []
        assert b.key_entities == []
        assert b.tags == []
        assert b.distilled_rule is None
        assert b.source_type == SourceType.INTERACTION
        assert b.scope == "global"
        assert isinstance(b.created_at, datetime)
        assert isinstance(b.updated_at, datetime)

    def test_created_at_is_utc(self) -> None:
        b = BulletMetadata()
        assert b.created_at.tzinfo is not None


class TestBulletMetadataValidation:
    """Field-level validation rules."""

    def test_instructivity_score_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            BulletMetadata(instructivity_score=-1)

    def test_instructivity_score_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            BulletMetadata(instructivity_score=101)

    def test_instructivity_score_boundary_values(self) -> None:
        assert BulletMetadata(instructivity_score=0).instructivity_score == 0
        assert BulletMetadata(instructivity_score=100).instructivity_score == 100

    def test_decay_weight_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            BulletMetadata(decay_weight=-0.01)

    def test_decay_weight_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            BulletMetadata(decay_weight=1.01)

    def test_decay_weight_boundary_values(self) -> None:
        assert BulletMetadata(decay_weight=0).decay_weight == 0
        assert BulletMetadata(decay_weight=1).decay_weight == 1

    def test_recall_count_negative(self) -> None:
        with pytest.raises(ValidationError):
            BulletMetadata(recall_count=-1)

    def test_recall_count_zero(self) -> None:
        assert BulletMetadata(recall_count=0).recall_count == 0

    def test_empty_scope_allowed(self) -> None:
        b = BulletMetadata(scope="")
        assert b.scope == ""


class TestBulletMetadataSerialization:
    """model_dump / model_validate round-trip."""

    def test_model_dump(self) -> None:
        b = BulletMetadata()
        d = b.model_dump()
        assert isinstance(d, dict)
        assert "section" in d
        assert "knowledge_type" in d
        assert "created_at" in d

    def test_model_validate_round_trip(self) -> None:
        original = BulletMetadata(
            section=BulletSection.DEBUGGING,
            knowledge_type=KnowledgeType.PITFALL,
            instructivity_score=75.0,
            recall_count=3,
            decay_weight=0.87,
            related_tools=["cargo", "rustc"],
            key_entities=["async", "await"],
            tags=["rust"],
            scope="project:my-app",
        )
        d = original.model_dump()
        restored = BulletMetadata.model_validate(d)
        assert restored == original

    def test_json_round_trip(self) -> None:
        original = BulletMetadata(
            last_recall=datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc),
        )
        json_str = original.model_dump_json()
        restored = BulletMetadata.model_validate_json(json_str)
        assert restored.last_recall == original.last_recall

    def test_datetime_iso_format_in_json(self) -> None:
        b = BulletMetadata(
            created_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        )
        json_str = b.model_dump_json()
        assert "2026-01-01" in json_str

    def test_enum_serialized_as_value(self) -> None:
        b = BulletMetadata(section=BulletSection.COMMANDS)
        d = b.model_dump()
        assert d["section"] == "commands"

    def test_model_validate_from_raw_dict(self) -> None:
        raw = {
            "section": "debugging",
            "knowledge_type": "pitfall",
            "instructivity_score": 80,
            "recall_count": 5,
            "decay_weight": 0.5,
            "source_type": "manual",
        }
        b = BulletMetadata.model_validate(raw)
        assert b.section == BulletSection.DEBUGGING
        assert b.knowledge_type == KnowledgeType.PITFALL
        assert b.source_type == SourceType.MANUAL

    def test_invalid_enum_value_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BulletMetadata.model_validate({"section": "nonexistent"})

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BulletMetadata.model_validate({"instructivity_score": "not_a_number"})


class TestBulletMetadataListIsolation:
    """Mutable default fields should not share state between instances."""

    def test_list_fields_are_independent(self) -> None:
        a = BulletMetadata()
        b = BulletMetadata()
        a.related_tools.append("git")
        assert b.related_tools == []


# ── Auxiliary type tests ────────────────────────────────────────────────


class TestInteractionEvent:
    def test_defaults(self) -> None:
        e = InteractionEvent(user_message="hello", assistant_message="hi")
        assert e.user_id == ""
        assert e.session_id == ""
        assert isinstance(e.timestamp, datetime)
        assert e.metadata == {}

    def test_full_construction(self) -> None:
        e = InteractionEvent(
            user_message="how?",
            assistant_message="like this",
            user_id="alice",
            session_id="sess-1",
        )
        assert e.user_id == "alice"


class TestDetectedPattern:
    def test_defaults(self) -> None:
        p = DetectedPattern()
        assert p.confidence == 0.0
        assert p.source_event is None

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DetectedPattern(confidence=1.5)
        with pytest.raises(ValidationError):
            DetectedPattern(confidence=-0.1)


class TestScoredCandidate:
    def test_defaults(self) -> None:
        sc = ScoredCandidate()
        assert sc.instructivity_score == 50.0
        assert sc.section == BulletSection.GENERAL

    def test_score_validation(self) -> None:
        with pytest.raises(ValidationError):
            ScoredCandidate(instructivity_score=101)


class TestCandidateBullet:
    def test_defaults(self) -> None:
        cb = CandidateBullet()
        assert cb.content == ""
        assert cb.scope == "global"

    def test_full_construction(self) -> None:
        cb = CandidateBullet(
            content="Always pin futures in Rust",
            distilled_rule="Pin futures before polling",
            section=BulletSection.PATTERNS,
            knowledge_type=KnowledgeType.PITFALL,
            instructivity_score=75.0,
            key_entities=["async", "Pin"],
            related_tools=["cargo"],
            tags=["rust"],
        )
        assert cb.content == "Always pin futures in Rust"
        assert cb.section == BulletSection.PATTERNS
