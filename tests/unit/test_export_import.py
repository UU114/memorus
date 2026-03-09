"""Unit tests for STORY-044: export/import functionality.

Covers Memory.export(), Memory.import_data(), and the CLI export/import commands.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memx.core.cli.main import cli

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_memories() -> list[dict[str, Any]]:
    """Sample memory records as returned by mem0 get_all()."""
    return [
        {
            "id": "abc123def456",
            "memory": "Use pytest -v for verbose output",
            "metadata": {
                "memx_section": "commands",
                "memx_knowledge_type": "method",
                "memx_instructivity_score": 92.0,
                "memx_decay_weight": 0.95,
                "memx_scope": "global",
                "memx_tags": '["pytest", "testing"]',
                "memx_related_tools": '["pytest"]',
                "memx_key_entities": '[]',
                "memx_related_files": '[]',
                "memx_source_type": "interaction",
                "memx_distilled_rule": "Always use -v flag",
                "memx_recall_count": 3,
                "memx_created_at": "2026-01-15T10:00:00+00:00",
                "memx_updated_at": "2026-01-20T10:00:00+00:00",
            },
        },
        {
            "id": "xyz789abc012",
            "memory": "Avoid mutable default arguments in Python",
            "metadata": {
                "memx_section": "patterns",
                "memx_knowledge_type": "pitfall",
                "memx_instructivity_score": 85.0,
                "memx_decay_weight": 0.88,
                "memx_scope": "project:myapp",
                "memx_tags": '["python", "best-practices"]',
                "memx_related_tools": '[]',
                "memx_key_entities": '[]',
                "memx_related_files": '[]',
                "memx_source_type": "manual",
                "memx_distilled_rule": None,
                "memx_recall_count": 1,
                "memx_created_at": "2026-02-10T08:00:00+00:00",
                "memx_updated_at": "2026-02-10T08:00:00+00:00",
            },
        },
        {
            "id": "empty_content_id",
            "memory": "",
            "metadata": {
                "memx_section": "general",
                "memx_knowledge_type": "knowledge",
                "memx_scope": "global",
            },
        },
    ]


@pytest.fixture
def mock_memory() -> MagicMock:
    """Create a mocked Memory instance with sensible defaults."""
    m = MagicMock()
    m.get_all.return_value = {"results": []}
    m.export.return_value = {
        "version": "1.0",
        "exported_at": "2026-02-27T00:00:00+00:00",
        "total": 0,
        "memories": [],
    }
    m.import_data.return_value = {"imported": 0, "skipped": 0, "merged": 0}
    return m


def _patch_create_memory(mock_memory: MagicMock) -> Any:
    """Patch _create_memory to return the mock."""
    return patch("memx.core.cli.main._create_memory", return_value=mock_memory)


# ---------------------------------------------------------------------------
# Memory.export() tests
# ---------------------------------------------------------------------------


class TestExportJSON:
    """Tests for Memory.export(format='json')."""

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_export_empty_database(self, mock_init: MagicMock) -> None:
        """Empty database exports with version header and zero memories."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": []}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        result = mem.export(format="json")
        assert isinstance(result, dict)
        assert result["version"] == "1.0"
        assert result["total"] == 0
        assert result["memories"] == []
        assert "exported_at" in result

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_export_json_with_memories(
        self, mock_init: MagicMock, sample_memories: list[dict[str, Any]]
    ) -> None:
        """Export includes all memories in the envelope."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": sample_memories}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        result = mem.export(format="json")
        assert isinstance(result, dict)
        assert result["version"] == "1.0"
        assert result["total"] == 3
        assert len(result["memories"]) == 3

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_export_json_scope_filter(
        self, mock_init: MagicMock, sample_memories: list[dict[str, Any]]
    ) -> None:
        """Export with scope filter only returns matching memories."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": sample_memories}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        result = mem.export(format="json", scope="project:myapp")
        assert isinstance(result, dict)
        assert result["total"] == 1
        assert result["memories"][0]["id"] == "xyz789abc012"

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_export_json_scope_filter_no_match(
        self, mock_init: MagicMock, sample_memories: list[dict[str, Any]]
    ) -> None:
        """Export with non-matching scope returns empty."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": sample_memories}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        result = mem.export(format="json", scope="project:nonexistent")
        assert isinstance(result, dict)
        assert result["total"] == 0

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_export_unsupported_format(self, mock_init: MagicMock) -> None:
        """Unsupported format raises ValueError."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": []}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        with pytest.raises(ValueError, match="Unsupported format"):
            mem.export(format="csv")


class TestExportMarkdown:
    """Tests for Memory.export(format='markdown')."""

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_export_markdown_header(self, mock_init: MagicMock) -> None:
        """Markdown export starts with the standard header."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": []}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        result = mem.export(format="markdown")
        assert isinstance(result, str)
        assert result.startswith("# MemX Knowledge Export")
        assert "Total: 0 memories" in result

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_export_markdown_groups_by_section(
        self, mock_init: MagicMock, sample_memories: list[dict[str, Any]]
    ) -> None:
        """Markdown export groups memories by section."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": sample_memories}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        result = mem.export(format="markdown")
        assert isinstance(result, str)
        assert "## Commands" in result
        assert "## Patterns" in result
        assert "## General" in result
        assert "pytest -v" in result
        assert "`abc123`" in result

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_export_markdown_with_scope(
        self, mock_init: MagicMock, sample_memories: list[dict[str, Any]]
    ) -> None:
        """Markdown export respects scope filter."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": sample_memories}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        result = mem.export(format="markdown", scope="project:myapp")
        assert isinstance(result, str)
        assert "Total: 1 memories" in result
        assert "mutable default" in result


# ---------------------------------------------------------------------------
# Memory.import_data() tests
# ---------------------------------------------------------------------------


class TestImportData:
    """Tests for Memory.import_data()."""

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_empty_payload(self, mock_init: MagicMock) -> None:
        """Import with no memories returns zero counts."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        result = mem.import_data({"version": "1.0", "memories": []})
        assert result == {"imported": 0, "skipped": 0, "merged": 0}

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_json_string(self, mock_init: MagicMock) -> None:
        """Import accepts a JSON string."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._mem0.add.return_value = {"id": "new1"}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        payload = json.dumps({
            "version": "1.0",
            "memories": [
                {"memory": "test content", "metadata": {}},
            ],
        })
        result = mem.import_data(payload, format="json")
        assert result["imported"] == 1
        assert result["skipped"] == 0

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_invalid_json_string(self, mock_init: MagicMock) -> None:
        """Import with invalid JSON string raises ValueError."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        with pytest.raises(ValueError, match="Invalid JSON"):
            mem.import_data("{broken json!!", format="json")

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_unsupported_format(self, mock_init: MagicMock) -> None:
        """Import with unsupported format raises ValueError."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        with pytest.raises(ValueError, match="Unsupported import format"):
            mem.import_data({}, format="csv")

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_skips_empty_content(self, mock_init: MagicMock) -> None:
        """Import skips memories with empty content."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        data = {
            "version": "1.0",
            "memories": [
                {"memory": "", "metadata": {}},
                {"memory": "   ", "metadata": {}},
                {"memory": "valid content", "metadata": {}},
            ],
        }
        mem._mem0.add.return_value = {"id": "new1"}
        result = mem.import_data(data)
        assert result["skipped"] == 2
        assert result["imported"] == 1

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_skips_non_dict_entries(self, mock_init: MagicMock) -> None:
        """Import skips entries that are not dicts."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        data: dict[str, Any] = {
            "version": "1.0",
            "memories": ["not a dict", 42, None],
        }
        result = mem.import_data(data)
        assert result["skipped"] == 3
        assert result["imported"] == 0

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_missing_version_treated_as_v1(
        self, mock_init: MagicMock
    ) -> None:
        """Import with missing version field still processes memories."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._mem0.add.return_value = {"id": "new1"}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        data = {
            "memories": [{"memory": "some knowledge", "metadata": {}}],
        }
        result = mem.import_data(data)
        assert result["imported"] == 1

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_handles_add_failure(self, mock_init: MagicMock) -> None:
        """Import counts add failures as skipped."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._mem0.add.side_effect = RuntimeError("storage full")
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        data = {
            "version": "1.0",
            "memories": [{"memory": "will fail", "metadata": {}}],
        }
        result = mem.import_data(data)
        assert result["skipped"] == 1
        assert result["imported"] == 0

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_not_a_dict_raises(self, mock_init: MagicMock) -> None:
        """Import with non-dict/non-string data raises ValueError."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._mem0 = MagicMock()
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        with pytest.raises(ValueError, match="must be a JSON object"):
            mem.import_data(["not", "a", "dict"])  # type: ignore[arg-type]

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_legacy_no_memx_prefix(self, mock_init: MagicMock) -> None:
        """Import handles legacy payloads with no memx_ prefix in metadata."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._mem0.add.return_value = {"id": "new1"}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        data = {
            "version": "1.0",
            "memories": [
                {
                    "memory": "old format knowledge",
                    "metadata": {"custom_field": "value"},
                },
            ],
        }
        result = mem.import_data(data)
        assert result["imported"] == 1


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify export -> import round-trip preserves data."""

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_json_round_trip(
        self, mock_init: MagicMock, sample_memories: list[dict[str, Any]]
    ) -> None:
        """Exporting then importing JSON preserves memory content."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._mem0.get_all.return_value = {"results": sample_memories}
        mem._mem0.add.return_value = {"id": "reimported"}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        # Export
        exported = mem.export(format="json")
        assert isinstance(exported, dict)

        # Serialize to string (simulates writing to file)
        json_str = json.dumps(exported)

        # Import from string
        result = mem.import_data(json_str, format="json")

        # 3 memories total: 2 with content, 1 empty (skipped)
        assert result["imported"] == 2
        assert result["skipped"] == 1

        # Verify add was called with the right content
        add_calls = mem._mem0.add.call_args_list
        contents = [call[0][0] for call in add_calls]
        assert "Use pytest -v for verbose output" in contents
        assert "Avoid mutable default arguments in Python" in contents


# ---------------------------------------------------------------------------
# CLI export command tests
# ---------------------------------------------------------------------------


class TestCLIExport:
    """Tests for the CLI export command."""

    def test_export_json_stdout(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Export JSON prints to stdout."""
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["export"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["version"] == "1.0"

    def test_export_json_to_file(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Export JSON writes to a file with --output."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            tmp_path = f.name

        try:
            with _patch_create_memory(mock_memory):
                result = runner.invoke(cli, ["export", "-o", tmp_path])
            assert result.exit_code == 0
            assert "Exported to" in result.output

            with open(tmp_path, encoding="utf-8") as f:
                data = json.load(f)
            assert data["version"] == "1.0"
        finally:
            os.unlink(tmp_path)

    def test_export_markdown_stdout(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Export markdown prints to stdout."""
        mock_memory.export.return_value = "# MemX Knowledge Export\n> Exported: ..."
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["export", "-f", "markdown"])
        assert result.exit_code == 0
        assert "MemX Knowledge Export" in result.output

    def test_export_with_scope(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Export respects --scope option."""
        with _patch_create_memory(mock_memory):
            result = runner.invoke(
                cli, ["export", "--scope", "project:myapp"]
            )
        assert result.exit_code == 0
        mock_memory.export.assert_called_once_with(
            format="json", scope="project:myapp"
        )

    def test_export_error_handling(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Export reports errors to stderr."""
        mock_memory.export.side_effect = RuntimeError("backend down")
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["export"])
        assert result.exit_code != 0
        assert "Error" in result.output


# ---------------------------------------------------------------------------
# CLI import command tests
# ---------------------------------------------------------------------------


class TestCLIImport:
    """Tests for the CLI import command."""

    def test_import_json_file(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Import reads a JSON file and calls import_data."""
        mock_memory.import_data.return_value = {
            "imported": 5,
            "skipped": 1,
            "merged": 2,
        }
        payload = json.dumps({
            "version": "1.0",
            "memories": [{"memory": "test", "metadata": {}}],
        })
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(payload)
            tmp_path = f.name

        try:
            with _patch_create_memory(mock_memory):
                result = runner.invoke(cli, ["import", tmp_path])
            assert result.exit_code == 0
            assert "Imported: 5" in result.output
            assert "Skipped:  1" in result.output
            assert "Merged:   2" in result.output
        finally:
            os.unlink(tmp_path)

    def test_import_invalid_json(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Import with invalid JSON shows error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write("{broken json!!")
            tmp_path = f.name

        try:
            with _patch_create_memory(mock_memory):
                result = runner.invoke(cli, ["import", tmp_path])
            assert result.exit_code != 0
            assert "Invalid JSON" in result.output
        finally:
            os.unlink(tmp_path)

    def test_import_nonexistent_file(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Import with nonexistent file shows error."""
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["import", "/nonexistent/path.json"])
        assert result.exit_code != 0

    def test_import_error_from_memory(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Import reports ValueError from import_data."""
        mock_memory.import_data.side_effect = ValueError("bad data")
        payload = json.dumps({"version": "1.0", "memories": []})
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            f.write(payload)
            tmp_path = f.name

        try:
            with _patch_create_memory(mock_memory):
                result = runner.invoke(cli, ["import", tmp_path])
            assert result.exit_code != 0
            assert "bad data" in result.output
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# BulletFactory.from_export_payload tests
# ---------------------------------------------------------------------------


class TestBulletFactoryExport:
    """Tests for BulletFactory.from_export_payload()."""

    def test_from_export_payload_full(
        self, sample_memories: list[dict[str, Any]]
    ) -> None:
        """Reconstruct a full memory with all metadata."""
        from memx.core.utils.bullet_factory import BulletFactory

        result = BulletFactory.from_export_payload(sample_memories[0])
        assert result["content"] == "Use pytest -v for verbose output"
        meta = result["metadata"]
        assert meta.section.value == "commands"
        assert meta.knowledge_type.value == "method"
        assert meta.scope == "global"

    def test_from_export_payload_empty(self) -> None:
        """Reconstruct from empty payload uses defaults."""
        from memx.core.utils.bullet_factory import BulletFactory

        result = BulletFactory.from_export_payload({})
        assert result["content"] == ""
        meta = result["metadata"]
        assert meta.section.value == "general"
        assert meta.knowledge_type.value == "knowledge"

    def test_from_export_payload_legacy_no_prefix(self) -> None:
        """Legacy payload without memx_ prefix gets default metadata."""
        from memx.core.utils.bullet_factory import BulletFactory

        payload = {
            "memory": "legacy content",
            "metadata": {"custom_field": "value"},
        }
        result = BulletFactory.from_export_payload(payload)
        assert result["content"] == "legacy content"
        meta = result["metadata"]
        assert meta.section.value == "general"


# ---------------------------------------------------------------------------
# Large import batch test
# ---------------------------------------------------------------------------


class TestLargeImport:
    """Test batch processing for large imports."""

    @patch("memx.core.memory.Memory.__init__", return_value=None)
    def test_import_large_dataset(self, mock_init: MagicMock) -> None:
        """Import > 500 memories processes in batches without error."""
        from memx.core.memory import Memory

        mem = Memory.__new__(Memory)
        mem._config = MagicMock()
        mem._config.ace_enabled = False
        mem._mem0 = MagicMock()
        mem._mem0.add.return_value = {"id": "new"}
        mem._ingest_pipeline = None
        mem._sanitizer = None
        mem._daemon_fallback = None

        # Create 600 memories (spans 2 batches)
        memories = [
            {"memory": f"knowledge item {i}", "metadata": {}}
            for i in range(600)
        ]
        data = {"version": "1.0", "memories": memories}

        result = mem.import_data(data)
        assert result["imported"] == 600
        assert result["skipped"] == 0
        assert mem._mem0.add.call_count == 600
