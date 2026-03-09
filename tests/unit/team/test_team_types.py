"""Tests for memx.team.types — TeamBullet model."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memx.team.types import TeamBullet


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------


class TestTeamBulletDefaults:
    """Verify that TeamBullet can be constructed with sensible defaults."""

    def test_minimal_construction(self) -> None:
        b = TeamBullet()
        assert b.schema_version == 2
        assert b.author_id == ""
        assert b.enforcement == "suggestion"
        assert b.upvotes == 0
        assert b.downvotes == 0
        assert b.status == "approved"
        assert b.deleted_at is None
        assert b.origin_id is None
        assert b.context_summary is None

    def test_schema_version_auto_set(self) -> None:
        """schema_version defaults to 2 even if not provided."""
        b = TeamBullet()
        assert b.schema_version == 2

    def test_schema_version_override(self) -> None:
        """Explicit schema_version is respected."""
        b = TeamBullet(schema_version=3)
        assert b.schema_version == 3

    def test_inherited_fields_accessible(self) -> None:
        """BulletMetadata fields are still accessible."""
        b = TeamBullet(instructivity_score=75.0, tags=["python"])
        assert b.instructivity_score == 75.0
        assert b.tags == ["python"]
        assert b.recall_count == 0


# ---------------------------------------------------------------------------
# Effective score
# ---------------------------------------------------------------------------


class TestEffectiveScore:
    """Verify effective_score computation and bounds."""

    def test_basic_score(self) -> None:
        b = TeamBullet(instructivity_score=50.0, upvotes=10, downvotes=3)
        assert b.effective_score == 57.0

    def test_score_lower_bound(self) -> None:
        b = TeamBullet(instructivity_score=5.0, upvotes=0, downvotes=100)
        assert b.effective_score == 0.0

    def test_score_upper_bound(self) -> None:
        b = TeamBullet(instructivity_score=90.0, upvotes=50, downvotes=0)
        assert b.effective_score == 100.0

    def test_score_exact_zero(self) -> None:
        b = TeamBullet(instructivity_score=0.0, upvotes=0, downvotes=0)
        assert b.effective_score == 0.0

    def test_score_exact_hundred(self) -> None:
        b = TeamBullet(instructivity_score=100.0, upvotes=0, downvotes=0)
        assert b.effective_score == 100.0


# ---------------------------------------------------------------------------
# is_active property
# ---------------------------------------------------------------------------


class TestIsActive:
    def test_approved_is_active(self) -> None:
        assert TeamBullet(status="approved").is_active is True

    def test_staging_is_active(self) -> None:
        assert TeamBullet(status="staging").is_active is True

    def test_deprecated_is_not_active(self) -> None:
        assert TeamBullet(status="deprecated").is_active is False

    def test_tombstone_is_not_active(self) -> None:
        assert TeamBullet(status="tombstone").is_active is False


# ---------------------------------------------------------------------------
# v1 -> v2 backward compatibility
# ---------------------------------------------------------------------------


class TestV1ToV2Compat:
    """Simulate loading v1 data (missing team fields) into TeamBullet."""

    def test_v1_data_gets_defaults(self) -> None:
        v1_data = {
            "section": "general",
            "knowledge_type": "knowledge",
            "instructivity_score": 60.0,
            "schema_version": 1,
        }
        b = TeamBullet(**v1_data)
        # schema_version is preserved when explicitly passed
        assert b.schema_version == 1
        # team fields filled with defaults
        assert b.enforcement == "suggestion"
        assert b.upvotes == 0
        assert b.downvotes == 0
        assert b.status == "approved"
        assert b.author_id == ""

    def test_v1_data_without_schema_version(self) -> None:
        """When v1 data omits schema_version, it defaults to 2."""
        v1_data = {
            "section": "general",
            "instructivity_score": 40.0,
        }
        b = TeamBullet(**v1_data)
        assert b.schema_version == 2
        assert b.enforcement == "suggestion"


# ---------------------------------------------------------------------------
# v2 -> v1 forward compatibility (extra="allow")
# ---------------------------------------------------------------------------


class TestV2ToV1Compat:
    """Verify extra fields are preserved during serialization."""

    def test_extra_fields_preserved(self) -> None:
        b = TeamBullet(
            author_id="user-1",
            future_field_xyz="some-value",  # type: ignore[call-arg]
        )
        dumped = b.model_dump()
        assert dumped["future_field_xyz"] == "some-value"
        assert dumped["author_id"] == "user-1"

    def test_round_trip_json(self) -> None:
        b = TeamBullet(
            author_id="user-2",
            enforcement="mandatory",
            unknown_v3_field=42,  # type: ignore[call-arg]
        )
        json_str = b.model_dump_json()
        restored = TeamBullet.model_validate_json(json_str)
        assert restored.author_id == "user-2"
        assert restored.enforcement == "mandatory"
        assert restored.model_extra is not None
        assert restored.model_extra.get("unknown_v3_field") == 42


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_model_dump_contains_team_fields(self) -> None:
        b = TeamBullet(author_id="alice", upvotes=5)
        d = b.model_dump()
        assert d["author_id"] == "alice"
        assert d["upvotes"] == 5
        assert d["schema_version"] == 2

    def test_deleted_at_serialization(self) -> None:
        now = datetime.now(timezone.utc)
        b = TeamBullet(deleted_at=now)
        d = b.model_dump()
        assert d["deleted_at"] == now

    def test_origin_id_and_context_summary(self) -> None:
        b = TeamBullet(
            origin_id="bullet-old-123",
            context_summary="Redacted context about deployment",
        )
        assert b.origin_id == "bullet-old-123"
        assert b.context_summary == "Redacted context about deployment"
