"""Tests for memorus.ext.mcp_server."""

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, mock_open, patch

import pytest


class TestCreateMcpServer:
    """Test MCP server factory and tool registration."""

    def test_import_error_without_mcp(self):
        """create_mcp_server raises ImportError when mcp is not installed."""
        with patch("memorus.ext.mcp_server.FastMCP", None):
            from memorus.ext.mcp_server import create_mcp_server

            with pytest.raises(ImportError, match="mcp"):
                create_mcp_server()

    def test_create_server_returns_fastmcp(self):
        """Factory returns a FastMCP instance when mcp is available."""
        mock_fastmcp_cls = MagicMock()
        mock_server = MagicMock()
        mock_fastmcp_cls.return_value = mock_server
        mock_server.tool.return_value = lambda fn: fn

        with patch("memorus.ext.mcp_server.FastMCP", mock_fastmcp_cls):
            from memorus.ext.mcp_server import create_mcp_server

            server = create_mcp_server()

        mock_fastmcp_cls.assert_called_once_with("memorus", description="Memorus Memory Server")
        assert server is mock_server

    def test_nine_tools_registered(self):
        """All 9 tools are registered via @mcp.tool()."""
        mock_fastmcp_cls = MagicMock()
        mock_server = MagicMock()
        mock_fastmcp_cls.return_value = mock_server
        mock_server.tool.return_value = lambda fn: fn

        with patch("memorus.ext.mcp_server.FastMCP", mock_fastmcp_cls):
            from memorus.ext.mcp_server import create_mcp_server

            create_mcp_server()

        assert mock_server.tool.call_count == 9


def _make_tools_and_mock():
    """Create a server, capture tool functions, and return (tools_dict, mock_memory)."""
    mock_memory = MagicMock()
    mock_memory.search.return_value = {"results": []}
    mock_memory.add.return_value = {"results": [{"id": "abc"}]}
    mock_memory.get_all.return_value = {"results": []}
    mock_memory.delete.return_value = None
    mock_memory.status.return_value = {"total": 5}
    mock_memory.detect_conflicts.return_value = []
    mock_memory.run_decay_sweep.return_value = {
        "updated": 0, "archived": 0, "permanent": 0, "unchanged": 0, "errors": []
    }
    mock_memory.export.return_value = {
        "version": "1.0", "exported_at": "2026-01-01T00:00:00", "total": 0, "memories": []
    }
    mock_memory.import_data.return_value = {"imported": 0, "skipped": 0, "merged": 0}

    tool_funcs = {}

    def capture_tool():
        def decorator(fn):
            tool_funcs[fn.__name__] = fn
            return fn
        return decorator

    mock_fastmcp_cls = MagicMock()
    mock_server = MagicMock()
    mock_fastmcp_cls.return_value = mock_server
    mock_server.tool.side_effect = capture_tool

    return tool_funcs, mock_memory, mock_fastmcp_cls


class TestMcpToolFunctions:
    """Test individual MCP tool functions by calling them directly."""

    async def test_search_memory(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["search_memory"]("test query", user_id="u1", limit=10)
        mock_memory.search.assert_called_once_with("test query", user_id="u1", limit=10, scope=None)
        assert result == {"results": []}

    async def test_search_memory_with_scope(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["search_memory"](
                "test query", user_id="u1", limit=10, scope="project:myapp"
            )
        mock_memory.search.assert_called_once_with(
            "test query", user_id="u1", limit=10, scope="project:myapp"
        )
        assert result == {"results": []}

    async def test_add_memory(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["add_memory"]("hello world", user_id="u1")
        mock_memory.add.assert_called_once_with("hello world", user_id="u1", scope=None)
        assert "results" in result

    async def test_add_memory_with_scope(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["add_memory"](
                "hello world", user_id="u1", scope="project:myapp"
            )
        mock_memory.add.assert_called_once_with(
            "hello world", user_id="u1", scope="project:myapp"
        )
        assert "results" in result

    async def test_list_memories(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["list_memories"](user_id="u1")
        mock_memory.get_all.assert_called_once_with(user_id="u1")
        assert result == {"results": []}

    async def test_forget_memory(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["forget_memory"]("mem-123")
        mock_memory.delete.assert_called_once_with("mem-123")
        assert result["status"] == "deleted"

    async def test_memory_status(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["memory_status"](user_id="u1")
        mock_memory.status.assert_called_once_with(user_id="u1")
        assert result == {"total": 5}

    async def test_detect_conflicts(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["detect_conflicts"](user_id="u1")
        mock_memory.detect_conflicts.assert_called_once_with(user_id="u1")
        assert result == []

    async def test_detect_conflicts_with_pydantic_objects(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        mock_conflict = MagicMock()
        mock_conflict.model_dump.return_value = {"id": "c1", "memory_a": "a", "memory_b": "b"}
        mock_memory.detect_conflicts.return_value = [mock_conflict]
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["detect_conflicts"]()
        assert result == [{"id": "c1", "memory_a": "a", "memory_b": "b"}]

    async def test_run_decay_sweep(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["run_decay_sweep"](user_id="u1", archive=False)
        mock_memory.run_decay_sweep.assert_called_once_with(user_id="u1", archive=False)
        assert "updated" in result

    async def test_export_memories(self, tmp_path):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        export_path = str(tmp_path / "export.json")
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["export_memories"](filepath=export_path)
        mock_memory.export.assert_called_once_with(format="json", scope=None)
        assert result["status"] == "exported"
        assert result["filepath"] == export_path
        assert os.path.exists(export_path)
        with open(export_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == "1.0"

    async def test_export_memories_markdown(self, tmp_path):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        mock_memory.export.return_value = "# Memories\n\n- Hello world"
        export_path = str(tmp_path / "export.md")
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["export_memories"](
                filepath=export_path, format="markdown"
            )
        assert result["total"] == "unknown"
        assert os.path.exists(export_path)

    async def test_import_memories(self, tmp_path):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        import_path = str(tmp_path / "import.json")
        import_data = {"version": "1.0", "total": 1, "memories": [{"content": "hello"}]}
        with open(import_path, "w", encoding="utf-8") as f:
            json.dump(import_data, f)
        with (
            patch("memorus.ext.mcp_server.FastMCP", mock_cls),
            patch("memorus.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memorus.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["import_memories"](filepath=import_path)
        mock_memory.import_data.assert_called_once_with(import_data, format="json")
        assert result == {"imported": 0, "skipped": 0, "merged": 0}


class TestGetMemory:
    """Test the lazy Memory singleton initialization."""

    def test_lazy_init(self):
        """_get_memory creates Memory only once."""
        import memorus.ext.mcp_server as mod

        mod._memory_singleton = None  # reset

        mock_memory = MagicMock()
        with patch("memorus.core.memory.Memory", return_value=mock_memory) as mock_cls:
            result1 = mod._get_memory()
            result2 = mod._get_memory()

        mock_cls.assert_called_once_with(config=None)
        assert result1 is result2 is mock_memory

        mod._memory_singleton = None  # cleanup


class TestMainConfigParsing:
    """Test the main() entry point --config argument handling."""

    def test_main_no_config(self):
        """main() works without --config."""
        with (
            patch("memorus.ext.mcp_server.create_mcp_server") as mock_create,
            patch("sys.argv", ["memorus-mcp"]),
        ):
            mock_server = MagicMock()
            mock_create.return_value = mock_server
            from memorus.ext.mcp_server import main
            main()
        mock_create.assert_called_once_with(None)
        mock_server.run.assert_called_once()

    def test_main_with_config(self, tmp_path):
        """main() reads --config and passes parsed dict."""
        config_file = tmp_path / "config.json"
        config_data = {"ace_enabled": True}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")
        with (
            patch("memorus.ext.mcp_server.create_mcp_server") as mock_create,
            patch("sys.argv", ["memorus-mcp", "--config", str(config_file)]),
        ):
            mock_server = MagicMock()
            mock_create.return_value = mock_server
            from memorus.ext.mcp_server import main
            main()
        mock_create.assert_called_once_with(config_data)

    def test_main_config_not_found(self):
        """main() exits with error for missing config file."""
        with (
            patch("sys.argv", ["memorus-mcp", "--config", "/nonexistent/config.json"]),
            pytest.raises(SystemExit, match="1"),
        ):
            from memorus.ext.mcp_server import main
            main()

    def test_main_config_invalid_json(self, tmp_path):
        """main() exits with error for invalid JSON config."""
        config_file = tmp_path / "bad.json"
        config_file.write_text("{invalid json", encoding="utf-8")
        with (
            patch("sys.argv", ["memorus-mcp", "--config", str(config_file)]),
            pytest.raises(SystemExit, match="1"),
        ):
            from memorus.ext.mcp_server import main
            main()
