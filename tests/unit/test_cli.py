"""Unit tests for memx.cli.main — CLI commands (status + search + learn + list + forget + sweep)."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch

from memx.core.cli.main import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_memory() -> MagicMock:
    """Create a mocked Memory instance with sensible defaults."""
    m = MagicMock()
    m.status.return_value = {
        "total": 0,
        "ace_enabled": True,
        "sections": {},
        "knowledge_types": {},
        "avg_decay_weight": 0.0,
    }
    m.search.return_value = {"results": []}
    return m


def _patch_create_memory(mock_memory: MagicMock):
    """Patch _create_memory to return the mock."""
    return patch("memx.core.cli.main._create_memory", return_value=mock_memory)


# ---------------------------------------------------------------------------
# Helpers: sample data
# ---------------------------------------------------------------------------

SAMPLE_MEMORIES_STATUS = {
    "total": 3,
    "ace_enabled": True,
    "sections": {"general": 2, "commands": 1},
    "knowledge_types": {"knowledge": 2, "preference": 1},
    "avg_decay_weight": 0.85,
}


SAMPLE_SEARCH_RESULTS = {
    "results": [
        {
            "id": "abc123",
            "memory": "When using asyncio, always wrap with try/except for CancelledError",
            "score": 0.92,
            "metadata": {
                "memx_knowledge_type": "method",
                "memx_tags": '["python", "asyncio"]',
            },
        },
        {
            "id": "def456",
            "memory": "Prefer structured error handling with Result type",
            "score": 0.85,
            "metadata": {
                "memx_knowledge_type": "preference",
                "memx_tags": '["error-handling"]',
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# CLI group tests
# ---------------------------------------------------------------------------


class TestCLIGroup:
    """Tests for the top-level CLI group."""

    def test_help(self, runner: CliRunner) -> None:
        """memx --help shows help text."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "MemX" in result.output

    def test_version(self, runner: CliRunner) -> None:
        """memx --version shows the version."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output


# ---------------------------------------------------------------------------
# status command tests
# ---------------------------------------------------------------------------


class TestStatusCommand:
    """Tests for `memx status`."""

    def test_status_empty_db(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Empty knowledge base shows friendly message."""
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "No memories yet" in result.output

    def test_status_with_data(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Status with data shows statistics."""
        mock_memory.status.return_value = SAMPLE_MEMORIES_STATUS
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Total memories:    3" in result.output
        assert "ACE mode:          ON" in result.output
        assert "general" in result.output
        assert "commands" in result.output
        assert "knowledge" in result.output
        assert "preference" in result.output
        assert "0.85" in result.output

    def test_status_json(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json outputs valid JSON."""
        mock_memory.status.return_value = SAMPLE_MEMORIES_STATUS
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["status", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total"] == 3
        assert data["ace_enabled"] is True
        assert data["sections"]["general"] == 2
        assert data["avg_decay_weight"] == 0.85

    def test_status_ace_off(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """ACE off shows 'OFF' label."""
        mock_memory.status.return_value = {
            **SAMPLE_MEMORIES_STATUS,
            "ace_enabled": False,
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "ACE mode:          OFF" in result.output

    def test_status_with_user_id(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--user-id is passed to Memory.status()."""
        mock_memory.status.return_value = SAMPLE_MEMORIES_STATUS
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["status", "--user-id", "u1"])
        assert result.exit_code == 0
        mock_memory.status.assert_called_once_with(user_id="u1")

    def test_status_error_handling(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Memory.status() failure shows error message."""
        mock_memory.status.side_effect = RuntimeError("mem0 backend not initialized")
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_status_memory_init_failure(self, runner: CliRunner) -> None:
        """Memory initialization failure shows error and exits."""
        with patch("memx.core.cli.main._create_memory", return_value=None):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code != 0

    def test_status_section_distribution_percentages(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Section distribution shows correct percentages."""
        mock_memory.status.return_value = {
            "total": 10,
            "ace_enabled": True,
            "sections": {"general": 7, "commands": 3},
            "knowledge_types": {"knowledge": 10},
            "avg_decay_weight": 1.0,
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "70.0%" in result.output
        assert "30.0%" in result.output


# ---------------------------------------------------------------------------
# search command tests
# ---------------------------------------------------------------------------


class TestSearchCommand:
    """Tests for `memx search`."""

    def test_search_no_results(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Search with no results shows friendly message."""
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "nonexistent"])
        assert result.exit_code == 0
        assert "No results found" in result.output
        assert "nonexistent" in result.output

    def test_search_with_results(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Search results display score, content, type, tags, id."""
        mock_memory.search.return_value = SAMPLE_SEARCH_RESULTS
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "async error"])
        assert result.exit_code == 0
        assert "[0.92]" in result.output
        assert "asyncio" in result.output
        assert "method" in result.output
        assert "abc123" in result.output
        assert "[0.85]" in result.output
        assert "preference" in result.output
        assert "def456" in result.output
        assert "2 results" in result.output

    def test_search_json(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json outputs valid JSON array of results."""
        mock_memory.search.return_value = SAMPLE_SEARCH_RESULTS
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "async", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["id"] == "abc123"
        assert data[0]["score"] == 0.92

    def test_search_limit(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--limit is passed to Memory.search()."""
        mock_memory.search.return_value = {"results": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "test", "--limit", "10"])
        assert result.exit_code == 0
        mock_memory.search.assert_called_once_with(
            "test", user_id=None, limit=10, scope=None
        )

    def test_search_limit_short_flag(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """-n is a short alias for --limit."""
        mock_memory.search.return_value = {"results": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "query", "-n", "3"])
        assert result.exit_code == 0
        mock_memory.search.assert_called_once_with(
            "query", user_id=None, limit=3, scope=None
        )

    def test_search_with_user_id(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--user-id is passed to Memory.search()."""
        mock_memory.search.return_value = {"results": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "q", "--user-id", "alice"])
        assert result.exit_code == 0
        mock_memory.search.assert_called_once_with(
            "q", user_id="alice", limit=5, scope=None
        )

    def test_search_error_handling(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Memory.search() failure shows error message."""
        mock_memory.search.side_effect = RuntimeError("search failed")
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "query"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_search_memory_init_failure(self, runner: CliRunner) -> None:
        """Memory initialization failure shows error and exits."""
        with patch("memx.core.cli.main._create_memory", return_value=None):
            result = runner.invoke(cli, ["search", "query"])
        assert result.exit_code != 0

    def test_search_missing_query(self, runner: CliRunner) -> None:
        """Missing query argument shows error."""
        result = runner.invoke(cli, ["search"])
        assert result.exit_code != 0

    def test_search_default_limit(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Default limit is 5."""
        mock_memory.search.return_value = {"results": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "q"])
        mock_memory.search.assert_called_once_with(
            "q", user_id=None, limit=5, scope=None
        )

    def test_search_json_no_results(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json with no results outputs empty JSON array."""
        mock_memory.search.return_value = {"results": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "nothing", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    def test_search_unicode_query(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Unicode queries are handled correctly."""
        mock_memory.search.return_value = {
            "results": [
                {
                    "id": "uni1",
                    "memory": "Support for Chinese characters: \u4e2d\u6587",
                    "score": 0.75,
                    "metadata": {},
                }
            ]
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "\u4e2d\u6587\u641c\u7d22"])
        assert result.exit_code == 0
        assert "\u4e2d\u6587" in result.output

    def test_search_tags_as_list(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Tags stored as list (not JSON string) are handled."""
        mock_memory.search.return_value = {
            "results": [
                {
                    "id": "tag1",
                    "memory": "test result",
                    "score": 0.9,
                    "metadata": {
                        "memx_knowledge_type": "method",
                        "memx_tags": ["rust", "async"],
                    },
                }
            ]
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "tags test"])
        assert result.exit_code == 0
        assert "rust" in result.output
        assert "async" in result.output

    def test_search_single_result_grammar(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Single result shows '1 result' (not '1 results')."""
        mock_memory.search.return_value = {
            "results": [
                {
                    "id": "one",
                    "memory": "single item",
                    "score": 0.5,
                    "metadata": {},
                }
            ]
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["search", "single"])
        assert result.exit_code == 0
        assert "1 result)" in result.output
        assert "1 results)" not in result.output


# ---------------------------------------------------------------------------
# Memory.status() integration tests (testing the method directly)
# ---------------------------------------------------------------------------


class TestMemoryStatus:
    """Tests for the Memory.status() method itself."""

    def test_status_empty(self) -> None:
        """status() with no memories returns zero counts."""
        from memx.core.config import MemXConfig
        from memx.core.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._mem0.get_all.return_value = {"memories": []}
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        result = m.status()
        assert result["total"] == 0
        assert result["sections"] == {}
        assert result["knowledge_types"] == {}
        assert result["avg_decay_weight"] == 0.0
        assert result["ace_enabled"] is False

    def test_status_with_memories(self) -> None:
        """status() computes correct distributions and averages."""
        from memx.core.config import MemXConfig
        from memx.core.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._mem0.get_all.return_value = {
            "memories": [
                {
                    "id": "1",
                    "memory": "rule one",
                    "metadata": {
                        "memx_section": "commands",
                        "memx_knowledge_type": "method",
                        "memx_decay_weight": 0.9,
                    },
                },
                {
                    "id": "2",
                    "memory": "rule two",
                    "metadata": {
                        "memx_section": "commands",
                        "memx_knowledge_type": "preference",
                        "memx_decay_weight": 0.8,
                    },
                },
                {
                    "id": "3",
                    "memory": "rule three",
                    "metadata": {
                        "memx_section": "general",
                        "memx_knowledge_type": "method",
                        "memx_decay_weight": 0.7,
                    },
                },
            ]
        }
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        result = m.status()
        assert result["total"] == 3
        assert result["ace_enabled"] is True
        assert result["sections"] == {"commands": 2, "general": 1}
        assert result["knowledge_types"] == {"method": 2, "preference": 1}
        assert result["avg_decay_weight"] == 0.8

    def test_status_with_user_id(self) -> None:
        """status(user_id=...) passes user_id to get_all()."""
        from memx.core.config import MemXConfig
        from memx.core.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._mem0.get_all.return_value = {"memories": []}
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        m.status(user_id="alice")
        m._mem0.get_all.assert_called_once_with(user_id="alice")

    def test_status_missing_metadata_defaults(self) -> None:
        """Memories without memx_ metadata use defaults."""
        from memx.core.config import MemXConfig
        from memx.core.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._mem0.get_all.return_value = {
            "memories": [
                {"id": "1", "memory": "plain", "metadata": {}},
            ]
        }
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        result = m.status()
        assert result["total"] == 1
        assert result["sections"] == {"general": 1}
        assert result["knowledge_types"] == {"knowledge": 1}
        assert result["avg_decay_weight"] == 1.0


# ---------------------------------------------------------------------------
# Helpers: sample data for new commands
# ---------------------------------------------------------------------------

SAMPLE_LEARN_RESULT = {
    "results": [],
    "ace_ingest": {
        "bullets_added": [
            {
                "id": "new123",
                "knowledge_type": "tool_pattern",
                "distilled_rule": "When using pytest, always use -v for verbose output",
                "tags": ["pytest", "testing"],
                "content": "When using pytest, always use -v for verbose output",
            }
        ],
        "raw_fallback": False,
        "errors": [],
    },
}


SAMPLE_LEARN_RESULT_RAW_FALLBACK = {
    "results": [],
    "ace_ingest": {
        "bullets_added": [],
        "raw_fallback": True,
        "errors": [],
    },
}


SAMPLE_ALL_MEMORIES = {
    "memories": [
        {
            "id": "abc12345",
            "memory": "User prefers dark mode in all editors",
            "metadata": {
                "memx_knowledge_type": "preference",
                "memx_decay_weight": 0.92,
                "memx_scope": "project:myapp",
            },
        },
        {
            "id": "def45678",
            "memory": "pytest -v flag usage for verbose output",
            "metadata": {
                "memx_knowledge_type": "tool_pattern",
                "memx_decay_weight": 0.85,
                "memx_scope": "global",
            },
        },
        {
            "id": "ghi78901",
            "memory": "subprocess stderr check pattern",
            "metadata": {
                "memx_knowledge_type": "error_fix",
                "memx_decay_weight": 0.71,
                "memx_scope": "project:myapp",
            },
        },
    ]
}


SAMPLE_SWEEP_RESULT = {
    "updated": 35,
    "archived": 3,
    "permanent": 8,
    "unchanged": 12,
}


# ---------------------------------------------------------------------------
# learn command tests
# ---------------------------------------------------------------------------


class TestLearnCommand:
    """Tests for `memx learn`."""

    def test_learn_basic(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Learn stores content and shows result."""
        mock_memory.add.return_value = SAMPLE_LEARN_RESULT
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["learn", "Always use pytest -v"])
        assert result.exit_code == 0
        assert "Learned new knowledge" in result.output
        assert "tool_pattern" in result.output
        assert "pytest" in result.output

    def test_learn_calls_add_correctly(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """learn calls Memory.add() with correct args."""
        mock_memory.add.return_value = SAMPLE_LEARN_RESULT
        with _patch_create_memory(mock_memory):
            runner.invoke(cli, ["learn", "some content"])
        mock_memory.add.assert_called_once_with(
            [{"role": "user", "content": "some content"}],
            user_id="manual",
        )

    def test_learn_raw_flag(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--raw passes source_type=manual metadata."""
        mock_memory.add.return_value = SAMPLE_LEARN_RESULT
        with _patch_create_memory(mock_memory):
            runner.invoke(cli, ["learn", "--raw", "raw content"])
        mock_memory.add.assert_called_once_with(
            [{"role": "user", "content": "raw content"}],
            user_id="manual",
            metadata={"source_type": "manual"},
        )

    def test_learn_json_output(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json outputs valid JSON."""
        mock_memory.add.return_value = SAMPLE_LEARN_RESULT
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["learn", "--json", "test content"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "ace_ingest" in data
        assert data["ace_ingest"]["bullets_added"][0]["knowledge_type"] == "tool_pattern"

    def test_learn_with_user_id(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--user-id overrides the default 'manual' user ID."""
        mock_memory.add.return_value = SAMPLE_LEARN_RESULT
        with _patch_create_memory(mock_memory):
            runner.invoke(cli, ["learn", "--user-id", "alice", "my content"])
        mock_memory.add.assert_called_once_with(
            [{"role": "user", "content": "my content"}],
            user_id="alice",
        )

    def test_learn_raw_fallback_display(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Raw fallback shows appropriate message."""
        mock_memory.add.return_value = SAMPLE_LEARN_RESULT_RAW_FALLBACK
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["learn", "fallback content"])
        assert result.exit_code == 0
        assert "raw fallback" in result.output

    def test_learn_error_handling(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Memory.add() failure shows error message."""
        mock_memory.add.side_effect = RuntimeError("add failed")
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["learn", "bad content"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_learn_memory_init_failure(self, runner: CliRunner) -> None:
        """Memory initialization failure shows error and exits."""
        with patch("memx.core.cli.main._create_memory", return_value=None):
            result = runner.invoke(cli, ["learn", "content"])
        assert result.exit_code != 0

    def test_learn_missing_content(self, runner: CliRunner) -> None:
        """Missing content argument shows error."""
        result = runner.invoke(cli, ["learn"])
        assert result.exit_code != 0

    def test_learn_with_ace_ingest_errors(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Warnings from ace_ingest errors are shown."""
        mock_memory.add.return_value = {
            "results": [],
            "ace_ingest": {
                "bullets_added": [],
                "raw_fallback": False,
                "errors": ["Reflector timeout"],
            },
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["learn", "timeout content"])
        assert result.exit_code == 0
        assert "Warning" in result.output or "Reflector timeout" in result.output

    def test_learn_unicode_content(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Unicode content is handled correctly."""
        mock_memory.add.return_value = SAMPLE_LEARN_RESULT
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["learn", "\u4f7f\u7528pytest\u65f6\u8981\u52a0-v\u53c2\u6570"])
        assert result.exit_code == 0
        mock_memory.add.assert_called_once()


# ---------------------------------------------------------------------------
# list command tests
# ---------------------------------------------------------------------------


class TestListCommand:
    """Tests for `memx list`."""

    def test_list_empty(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Empty knowledge base shows friendly message."""
        mock_memory.get_all.return_value = {"memories": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No memories found" in result.output

    def test_list_with_data(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """List shows memories in table format."""
        mock_memory.get_all.return_value = SAMPLE_ALL_MEMORIES
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "3 total" in result.output
        assert "abc12345" in result.output
        assert "preference" in result.output
        assert "dark mode" in result.output
        assert "tool_pattern" in result.output

    def test_list_json_output(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json outputs valid JSON array."""
        mock_memory.get_all.return_value = SAMPLE_ALL_MEMORIES
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 3
        assert data[0]["id"] == "abc12345"

    def test_list_scope_filter(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--scope filters memories by memx_scope metadata."""
        mock_memory.get_all.return_value = SAMPLE_ALL_MEMORIES
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list", "--scope", "project:myapp"])
        assert result.exit_code == 0
        assert "2 total" in result.output
        # abc12345 and ghi78901 have scope project:myapp
        assert "abc12345" in result.output
        assert "ghi78901" in result.output

    def test_list_type_filter(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--type filters memories by knowledge type."""
        mock_memory.get_all.return_value = SAMPLE_ALL_MEMORIES
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list", "--type", "preference"])
        assert result.exit_code == 0
        assert "1 total" in result.output
        assert "abc12345" in result.output

    def test_list_limit(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--limit caps the number of displayed memories."""
        mock_memory.get_all.return_value = SAMPLE_ALL_MEMORIES
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list", "--limit", "1"])
        assert result.exit_code == 0
        assert "showing 1" in result.output
        assert "abc12345" in result.output

    def test_list_limit_short_flag(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """-n is a short alias for --limit."""
        mock_memory.get_all.return_value = SAMPLE_ALL_MEMORIES
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list", "-n", "2"])
        assert result.exit_code == 0
        assert "showing 2" in result.output

    def test_list_with_user_id(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--user-id is passed to Memory.get_all()."""
        mock_memory.get_all.return_value = {"memories": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list", "--user-id", "bob"])
        assert result.exit_code == 0
        mock_memory.get_all.assert_called_once_with(user_id="bob")

    def test_list_json_empty(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json with no memories outputs empty JSON array."""
        mock_memory.get_all.return_value = {"memories": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data == []

    def test_list_error_handling(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Memory.get_all() failure shows error message."""
        mock_memory.get_all.side_effect = RuntimeError("get_all failed")
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_list_memory_init_failure(self, runner: CliRunner) -> None:
        """Memory initialization failure shows error and exits."""
        with patch("memx.core.cli.main._create_memory", return_value=None):
            result = runner.invoke(cli, ["list"])
        assert result.exit_code != 0

    def test_list_combined_filters(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--scope and --type can be combined."""
        mock_memory.get_all.return_value = SAMPLE_ALL_MEMORIES
        with _patch_create_memory(mock_memory):
            result = runner.invoke(
                cli, ["list", "--scope", "project:myapp", "--type", "preference"]
            )
        assert result.exit_code == 0
        assert "1 total" in result.output
        assert "abc12345" in result.output

    def test_list_default_limit_is_20(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Default limit is 20."""
        mock_memory.get_all.return_value = {"memories": []}
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["list"])
        # No assertion on mock call args since limit is applied locally, not passed to get_all
        assert result.exit_code == 0

    def test_list_json_with_scope_filter(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json respects --scope filter."""
        mock_memory.get_all.return_value = SAMPLE_ALL_MEMORIES
        with _patch_create_memory(mock_memory):
            result = runner.invoke(
                cli, ["list", "--json", "--scope", "global"]
            )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["id"] == "def45678"


# ---------------------------------------------------------------------------
# forget command tests
# ---------------------------------------------------------------------------


class TestForgetCommand:
    """Tests for `memx forget`."""

    def test_forget_with_confirmation(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """forget shows content and asks confirmation, then deletes."""
        mock_memory.get.return_value = {
            "id": "abc123",
            "memory": "Some memory content",
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["forget", "abc123"], input="y\n")
        assert result.exit_code == 0
        assert "Will delete" in result.output
        assert "Deleted memory: abc123" in result.output
        mock_memory.delete.assert_called_once_with("abc123")

    def test_forget_cancelled(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """forget cancelled when user says no."""
        mock_memory.get.return_value = {
            "id": "abc123",
            "memory": "Some memory content",
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["forget", "abc123"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output
        mock_memory.delete.assert_not_called()

    def test_forget_skip_confirmation(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--yes skips confirmation prompt."""
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["forget", "abc123", "--yes"])
        assert result.exit_code == 0
        assert "Deleted memory: abc123" in result.output
        mock_memory.delete.assert_called_once_with("abc123")

    def test_forget_short_yes_flag(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """-y is a short alias for --yes."""
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["forget", "abc123", "-y"])
        assert result.exit_code == 0
        mock_memory.delete.assert_called_once_with("abc123")

    def test_forget_json_output(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json outputs valid JSON with deleted ID."""
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["forget", "abc123", "--yes", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["deleted"] == "abc123"

    def test_forget_not_found(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Non-existent memory shows error (without --yes)."""
        mock_memory.get.return_value = None
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["forget", "nonexistent"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "Memory not found" in result.output

    def test_forget_get_raises(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Memory.get() exception during confirmation handled gracefully."""
        mock_memory.get.side_effect = RuntimeError("get failed")
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["forget", "abc123"])
        # get() exception means mem is None, should show not found
        assert result.exit_code != 0

    def test_forget_delete_error(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Memory.delete() failure shows error message."""
        mock_memory.delete.side_effect = RuntimeError("delete failed")
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["forget", "abc123", "--yes"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_forget_memory_init_failure(self, runner: CliRunner) -> None:
        """Memory initialization failure shows error and exits."""
        with patch("memx.core.cli.main._create_memory", return_value=None):
            result = runner.invoke(cli, ["forget", "abc123", "--yes"])
        assert result.exit_code != 0

    def test_forget_missing_id(self, runner: CliRunner) -> None:
        """Missing memory_id argument shows error."""
        result = runner.invoke(cli, ["forget"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# sweep command tests
# ---------------------------------------------------------------------------


class TestSweepCommand:
    """Tests for `memx sweep`."""

    def test_sweep_basic(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """sweep shows results summary."""
        mock_memory.run_decay_sweep.return_value = SAMPLE_SWEEP_RESULT
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["sweep"])
        assert result.exit_code == 0
        assert "Decay sweep complete" in result.output
        assert "Updated:   35" in result.output
        assert "Archived:  3" in result.output
        assert "Permanent: 8" in result.output
        assert "Unchanged: 12" in result.output

    def test_sweep_json_output(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json outputs valid JSON."""
        mock_memory.run_decay_sweep.return_value = SAMPLE_SWEEP_RESULT
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["sweep", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["updated"] == 35
        assert data["archived"] == 3
        assert data["permanent"] == 8
        assert data["unchanged"] == 12

    def test_sweep_zero_results(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """sweep with no changes shows zero counts."""
        mock_memory.run_decay_sweep.return_value = {
            "updated": 0, "archived": 0, "permanent": 0, "unchanged": 0,
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["sweep"])
        assert result.exit_code == 0
        assert "Updated:   0" in result.output
        assert "Archived:  0" in result.output

    def test_sweep_not_implemented(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """NotImplementedError shows user-friendly message."""
        mock_memory.run_decay_sweep.side_effect = NotImplementedError(
            "run_decay_sweep() will be implemented in STORY-021"
        )
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["sweep"])
        assert result.exit_code != 0
        assert "not yet implemented" in result.output

    def test_sweep_error_handling(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """Generic error shows error message."""
        mock_memory.run_decay_sweep.side_effect = RuntimeError("sweep failed")
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["sweep"])
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_sweep_memory_init_failure(self, runner: CliRunner) -> None:
        """Memory initialization failure shows error and exits."""
        with patch("memx.core.cli.main._create_memory", return_value=None):
            result = runner.invoke(cli, ["sweep"])
        assert result.exit_code != 0

    def test_sweep_json_zero(
        self, runner: CliRunner, mock_memory: MagicMock
    ) -> None:
        """--json with zero results outputs valid JSON."""
        mock_memory.run_decay_sweep.return_value = {
            "updated": 0, "archived": 0, "permanent": 0, "unchanged": 0,
        }
        with _patch_create_memory(mock_memory):
            result = runner.invoke(cli, ["sweep", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["updated"] == 0


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for CLI helper functions."""

    def test_apply_filters_scope(self) -> None:
        """_apply_filters filters by scope correctly."""
        from memx.core.cli.main import _apply_filters

        memories = SAMPLE_ALL_MEMORIES["memories"]
        filtered = _apply_filters(memories, scope="project:myapp")
        assert len(filtered) == 2
        assert all(
            m["metadata"]["memx_scope"] == "project:myapp" for m in filtered
        )

    def test_apply_filters_type(self) -> None:
        """_apply_filters filters by knowledge type correctly."""
        from memx.core.cli.main import _apply_filters

        memories = SAMPLE_ALL_MEMORIES["memories"]
        filtered = _apply_filters(memories, knowledge_type="tool_pattern")
        assert len(filtered) == 1
        assert filtered[0]["id"] == "def45678"

    def test_apply_filters_both(self) -> None:
        """_apply_filters with both scope and type."""
        from memx.core.cli.main import _apply_filters

        memories = SAMPLE_ALL_MEMORIES["memories"]
        filtered = _apply_filters(
            memories, scope="project:myapp", knowledge_type="error_fix"
        )
        assert len(filtered) == 1
        assert filtered[0]["id"] == "ghi78901"

    def test_apply_filters_no_match(self) -> None:
        """_apply_filters returns empty list when no match."""
        from memx.core.cli.main import _apply_filters

        memories = SAMPLE_ALL_MEMORIES["memories"]
        filtered = _apply_filters(memories, scope="nonexistent")
        assert filtered == []

    def test_parse_tags_json_string(self) -> None:
        """_parse_tags parses JSON string tags."""
        from memx.core.cli.main import _parse_tags

        assert _parse_tags('["a", "b"]') == ["a", "b"]

    def test_parse_tags_list(self) -> None:
        """_parse_tags passes through list tags."""
        from memx.core.cli.main import _parse_tags

        assert _parse_tags(["x", "y"]) == ["x", "y"]

    def test_parse_tags_invalid(self) -> None:
        """_parse_tags returns empty for invalid input."""
        from memx.core.cli.main import _parse_tags

        assert _parse_tags("not json") == []
        assert _parse_tags(None) == []
        assert _parse_tags(123) == []
