"""Unit tests for memorus.types — BulletMetadata, enums, and auxiliary models."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from memorus.core.types import (
    Anchor,
    BulletMetadata,
    BulletSection,
    CandidateBullet,
    DetectedPattern,
    InteractionEvent,
    KnowledgeType,
    ScoredCandidate,
    SourceType,
    VerifiedStatus,
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


class TestBulletMetadataNewFields:
    """Tests for schema_version and incompatible_tags fields."""

    def test_schema_version_default(self) -> None:
        b = BulletMetadata()
        assert b.schema_version == 1

    def test_schema_version_custom(self) -> None:
        b = BulletMetadata(schema_version=2)
        assert b.schema_version == 2

    def test_incompatible_tags_default(self) -> None:
        b = BulletMetadata()
        assert b.incompatible_tags == []

    def test_incompatible_tags_custom(self) -> None:
        b = BulletMetadata(incompatible_tags=["v2-only", "team-ext"])
        assert b.incompatible_tags == ["v2-only", "team-ext"]

    def test_incompatible_tags_list_isolation(self) -> None:
        a = BulletMetadata()
        b = BulletMetadata()
        a.incompatible_tags.append("x")
        assert b.incompatible_tags == []

    def test_schema_version_in_model_dump(self) -> None:
        b = BulletMetadata(schema_version=2, incompatible_tags=["a"])
        d = b.model_dump()
        assert d["schema_version"] == 2
        assert d["incompatible_tags"] == ["a"]

    def test_round_trip_with_new_fields(self) -> None:
        original = BulletMetadata(
            schema_version=3,
            incompatible_tags=["team-v2", "conflict-marker"],
        )
        d = original.model_dump()
        restored = BulletMetadata.model_validate(d)
        assert restored.schema_version == original.schema_version
        assert restored.incompatible_tags == original.incompatible_tags

    def test_backward_compat_missing_new_fields(self) -> None:
        """Old data without schema_version/incompatible_tags uses defaults."""
        raw = {"section": "general", "instructivity_score": 50}
        b = BulletMetadata.model_validate(raw)
        assert b.schema_version == 1
        assert b.incompatible_tags == []


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


# ── Verification layer tests (STORY-R099) ──────────────────────────────


_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "anchor_sample.json"


class TestVerifiedStatus:
    def test_string_values_match_wire_form(self) -> None:
        # The exact lowercase snake_case strings are part of the wire contract
        # with the Rust side; if these change, the cross-language fixture
        # tests will break.
        assert VerifiedStatus.VERIFIED.value == "verified"
        assert VerifiedStatus.STALE.value == "stale"
        assert VerifiedStatus.UNVERIFIABLE.value == "unverifiable"
        assert VerifiedStatus.NOT_APPLICABLE.value == "not_applicable"


class TestAnchor:
    def test_construction_and_compute_hash(self) -> None:
        text = "fn handle_request(req: Request) -> Response"
        a = Anchor(
            file_path="src/handlers/router.rs",
            anchor_text=text,
            anchor_hash=Anchor.compute_hash(text),
            created_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        )
        assert a.file_path == "src/handlers/router.rs"
        assert a.anchor_hash == Anchor.compute_hash(text)
        assert len(a.anchor_hash) == 64  # sha256 hex

    def test_forward_slash_normalization_from_backslash(self) -> None:
        a = Anchor(
            file_path="src\\handlers\\router.rs",
            anchor_text="fn x() {}",
            anchor_hash=Anchor.compute_hash("fn x() {}"),
            created_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        )
        assert a.file_path == "src/handlers/router.rs"

    def test_anchor_text_max_length_enforced(self) -> None:
        too_long = "x" * 201
        with pytest.raises(ValidationError):
            Anchor(
                file_path="src/x.rs",
                anchor_text=too_long,
                anchor_hash=Anchor.compute_hash(too_long),
                created_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
            )

    def test_json_round_trip(self) -> None:
        text = "fn verify() -> bool"
        original = Anchor(
            file_path="src/verifier.rs",
            anchor_text=text,
            anchor_hash=Anchor.compute_hash(text),
            created_at=datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc),
        )
        restored = Anchor.model_validate_json(original.model_dump_json())
        assert restored == original


class TestBulletMetadataVerificationFields:
    def test_default_verification_fields(self) -> None:
        b = BulletMetadata()
        assert b.anchors == []
        assert b.verified_at is None
        assert b.verified_status is None
        assert b.trust_score is None

    def test_legacy_json_without_anchors_loads_with_defaults(self) -> None:
        # Pre-R099 wire shape: no anchors / verified_status / trust_score
        # / verified_at keys. Must load with defaults (empty list, None).
        legacy = {
            "section": "general",
            "knowledge_type": "method",
            "instructivity_score": 50,
            "recall_count": 0,
            "decay_weight": 1.0,
            "scope": "global",
            "schema_version": 1,
        }
        b = BulletMetadata.model_validate(legacy)
        assert b.anchors == []
        assert b.verified_status is None
        assert b.verified_at is None
        assert b.trust_score is None

    def test_invalid_verified_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            BulletMetadata(verified_status="not_a_real_status")

    def test_verified_status_accepts_enum_value(self) -> None:
        b = BulletMetadata(verified_status=VerifiedStatus.VERIFIED)
        assert b.verified_status == "verified"

    def test_with_anchor_round_trip(self) -> None:
        text = "fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>)"
        anchor = Anchor(
            file_path="src/future.rs",
            anchor_text=text,
            anchor_hash=Anchor.compute_hash(text),
            created_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        )
        original = BulletMetadata(
            anchors=[anchor],
            verified_at=datetime(2026, 4, 24, 13, 0, tzinfo=timezone.utc),
            verified_status="verified",
            trust_score=1.0,
        )
        restored = BulletMetadata.model_validate_json(original.model_dump_json())
        assert restored.anchors == original.anchors
        assert restored.verified_at == original.verified_at
        assert restored.verified_status == "verified"
        assert restored.trust_score == 1.0


class TestAnchorCrossLanguageFixture:
    """Cross-language wire compatibility — same JSON loads in both runtimes.

    The Rust side has an analogous test that loads
    `tests/fixtures/anchor_sample.json` and asserts the same semantic values.
    """

    def test_loads_committed_fixture(self) -> None:
        raw = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
        a = Anchor.model_validate(raw)
        assert a.file_path == "src/handlers/router.rs"
        assert a.anchor_text == "fn handle_request(req: Request) -> Response"
        # The committed hash must match what compute_hash produces for the
        # text — protects against accidental drift.
        assert a.anchor_hash == Anchor.compute_hash(a.anchor_text)
        assert a.created_at == datetime(2026, 4, 24, 12, 34, 56, tzinfo=timezone.utc)

    def test_round_trip_emits_compatible_shape(self) -> None:
        raw = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
        a = Anchor.model_validate(raw)
        rebuilt = json.loads(a.model_dump_json())
        # All four keys present, file_path still forward-slash, hash stable.
        assert set(rebuilt.keys()) == {
            "file_path",
            "anchor_text",
            "anchor_hash",
            "created_at",
        }
        assert "\\" not in rebuilt["file_path"]
        assert rebuilt["anchor_hash"] == raw["anchor_hash"]
