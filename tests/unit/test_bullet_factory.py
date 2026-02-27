"""Unit tests for memx.utils.bullet_factory — BulletFactory."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from memx.types import BulletMetadata, BulletSection, KnowledgeType, SourceType
from memx.utils.bullet_factory import MEMX_PREFIX, BulletFactory

# ── BulletFactory.create ──────────────────────────────────────────────


class TestCreate:
    def test_returns_dict_with_content_and_metadata(self) -> None:
        result = BulletFactory.create("use cargo check")
        assert result["content"] == "use cargo check"
        assert isinstance(result["metadata"], BulletMetadata)

    def test_default_metadata(self) -> None:
        result = BulletFactory.create("hello")
        meta: BulletMetadata = result["metadata"]
        assert meta.section == BulletSection.GENERAL
        assert meta.knowledge_type == KnowledgeType.KNOWLEDGE
        assert meta.instructivity_score == 50.0

    def test_custom_kwargs(self) -> None:
        result = BulletFactory.create(
            "prefer dark mode",
            section=BulletSection.PREFERENCES,
            knowledge_type=KnowledgeType.PREFERENCE,
            instructivity_score=80.0,
            tags=["ui"],
        )
        meta: BulletMetadata = result["metadata"]
        assert meta.section == BulletSection.PREFERENCES
        assert meta.knowledge_type == KnowledgeType.PREFERENCE
        assert meta.instructivity_score == 80.0
        assert meta.tags == ["ui"]


# ── BulletFactory.to_mem0_metadata ────────────────────────────────────


class TestToMem0Metadata:
    def test_all_keys_have_prefix(self) -> None:
        meta = BulletMetadata()
        result = BulletFactory.to_mem0_metadata(meta)
        for key in result:
            assert key.startswith(MEMX_PREFIX), f"key {key!r} missing prefix"

    def test_enum_serialised_as_string(self) -> None:
        meta = BulletMetadata(
            section=BulletSection.DEBUGGING,
            knowledge_type=KnowledgeType.PITFALL,
        )
        result = BulletFactory.to_mem0_metadata(meta)
        assert result["memx_section"] == "debugging"
        assert result["memx_knowledge_type"] == "pitfall"

    def test_datetime_serialised_as_iso_string(self) -> None:
        ts = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
        meta = BulletMetadata(created_at=ts)
        result = BulletFactory.to_mem0_metadata(meta)
        assert "2026-03-15" in result["memx_created_at"]

    def test_list_fields_serialised_as_json_string(self) -> None:
        meta = BulletMetadata(
            related_tools=["cargo", "rustc"],
            tags=["rust", "async"],
        )
        result = BulletFactory.to_mem0_metadata(meta)
        assert isinstance(result["memx_related_tools"], str)
        assert json.loads(result["memx_related_tools"]) == ["cargo", "rustc"]
        assert isinstance(result["memx_tags"], str)
        assert json.loads(result["memx_tags"]) == ["rust", "async"]

    def test_none_last_recall(self) -> None:
        meta = BulletMetadata(last_recall=None)
        result = BulletFactory.to_mem0_metadata(meta)
        assert result["memx_last_recall"] is None

    def test_numeric_fields(self) -> None:
        meta = BulletMetadata(
            instructivity_score=75.0,
            recall_count=3,
            decay_weight=0.85,
        )
        result = BulletFactory.to_mem0_metadata(meta)
        assert result["memx_instructivity_score"] == 75.0
        assert result["memx_recall_count"] == 3
        assert result["memx_decay_weight"] == 0.85


# ── BulletFactory.from_mem0_payload ───────────────────────────────────


class TestFromMem0Payload:
    def test_empty_payload_returns_defaults(self) -> None:
        meta = BulletFactory.from_mem0_payload({})
        assert meta.section == BulletSection.GENERAL
        assert meta.instructivity_score == 50.0
        assert meta.tags == []

    def test_payload_without_metadata_key(self) -> None:
        meta = BulletFactory.from_mem0_payload({"id": "abc", "memory": "hello"})
        assert meta.section == BulletSection.GENERAL
        assert meta.instructivity_score == 50.0
        assert meta.tags == []

    def test_full_round_trip(self) -> None:
        original = BulletMetadata(
            section=BulletSection.DEBUGGING,
            knowledge_type=KnowledgeType.PITFALL,
            instructivity_score=75.0,
            recall_count=3,
            decay_weight=0.87,
            related_tools=["cargo", "rustc"],
            key_entities=["async", "await"],
            tags=["rust"],
            source_type=SourceType.MANUAL,
            scope="project:my-app",
        )
        mem0_meta = BulletFactory.to_mem0_metadata(original)
        payload = {"metadata": mem0_meta}
        restored = BulletFactory.from_mem0_payload(payload)

        assert restored.section == original.section
        assert restored.knowledge_type == original.knowledge_type
        assert restored.instructivity_score == original.instructivity_score
        assert restored.recall_count == original.recall_count
        assert restored.decay_weight == original.decay_weight
        assert restored.related_tools == original.related_tools
        assert restored.key_entities == original.key_entities
        assert restored.tags == original.tags
        assert restored.source_type == original.source_type
        assert restored.scope == original.scope

    def test_non_memx_keys_are_ignored(self) -> None:
        payload = {
            "metadata": {
                "user_id": "alice",
                "custom_key": 42,
                "memx_section": "tools",
            }
        }
        meta = BulletFactory.from_mem0_payload(payload)
        assert meta.section == BulletSection.TOOLS
        assert not hasattr(meta, "user_id")

    def test_partial_fields_use_defaults(self) -> None:
        payload = {
            "metadata": {
                "memx_section": "commands",
                "memx_instructivity_score": 90.0,
            }
        }
        meta = BulletFactory.from_mem0_payload(payload)
        assert meta.section == BulletSection.COMMANDS
        assert meta.instructivity_score == 90.0
        # Remaining fields should be default
        assert meta.knowledge_type == KnowledgeType.KNOWLEDGE
        assert meta.recall_count == 0
        assert meta.tags == []

    def test_list_field_as_json_string(self) -> None:
        payload = {
            "metadata": {
                "memx_related_tools": '["git", "npm"]',
                "memx_tags": '["js"]',
            }
        }
        meta = BulletFactory.from_mem0_payload(payload)
        assert meta.related_tools == ["git", "npm"]
        assert meta.tags == ["js"]

    def test_list_field_as_native_list(self) -> None:
        """If mem0 returns lists natively (not JSON string), still works."""
        payload = {
            "metadata": {
                "memx_related_tools": ["git", "npm"],
            }
        }
        meta = BulletFactory.from_mem0_payload(payload)
        assert meta.related_tools == ["git", "npm"]

    def test_list_field_invalid_json_falls_back_to_empty(self) -> None:
        payload = {
            "metadata": {
                "memx_tags": "not-valid-json{",
            }
        }
        meta = BulletFactory.from_mem0_payload(payload)
        assert meta.tags == []

    def test_string_number_coerced(self) -> None:
        """Pydantic should coerce '75' to 75.0 for float fields."""
        payload = {
            "metadata": {
                "memx_instructivity_score": "75",
            }
        }
        meta = BulletFactory.from_mem0_payload(payload)
        assert meta.instructivity_score == 75.0

    def test_datetime_iso_string_restored(self) -> None:
        ts = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc)
        original = BulletMetadata(created_at=ts, last_recall=ts)
        mem0_meta = BulletFactory.to_mem0_metadata(original)
        restored = BulletFactory.from_mem0_payload({"metadata": mem0_meta})
        assert restored.created_at == ts
        assert restored.last_recall == ts

    def test_none_last_recall_round_trip(self) -> None:
        original = BulletMetadata(last_recall=None)
        mem0_meta = BulletFactory.to_mem0_metadata(original)
        restored = BulletFactory.from_mem0_payload({"metadata": mem0_meta})
        assert restored.last_recall is None


# ── BulletFactory.merge_metadata ──────────────────────────────────────


class TestMergeMetadata:
    def test_merge_partial_update(self) -> None:
        existing = BulletMetadata(
            section=BulletSection.GENERAL,
            instructivity_score=50.0,
        )
        updated = BulletFactory.merge_metadata(
            existing, {"instructivity_score": 80.0, "recall_count": 5}
        )
        assert updated.instructivity_score == 80.0
        assert updated.recall_count == 5
        # Unchanged fields preserved
        assert updated.section == BulletSection.GENERAL

    def test_merge_returns_new_instance(self) -> None:
        existing = BulletMetadata()
        updated = BulletFactory.merge_metadata(existing, {"recall_count": 1})
        assert updated is not existing

    def test_merge_empty_update(self) -> None:
        existing = BulletMetadata(instructivity_score=75.0)
        updated = BulletFactory.merge_metadata(existing, {})
        assert updated.instructivity_score == 75.0
