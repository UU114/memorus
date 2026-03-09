"""Tests for memx.ext.mcp_server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestCreateMcpServer:
    """Test MCP server factory and tool registration."""

    def test_import_error_without_mcp(self):
        """create_mcp_server raises ImportError when mcp is not installed."""
        with patch("memx.ext.mcp_server.FastMCP", None):
            from memx.ext.mcp_server import create_mcp_server

            with pytest.raises(ImportError, match="mcp"):
                create_mcp_server()

    def test_create_server_returns_fastmcp(self):
        """Factory returns a FastMCP instance when mcp is available."""
        mock_fastmcp_cls = MagicMock()
        mock_server = MagicMock()
        mock_fastmcp_cls.return_value = mock_server
        mock_server.tool.return_value = lambda fn: fn

        with patch("memx.ext.mcp_server.FastMCP", mock_fastmcp_cls):
            from memx.ext.mcp_server import create_mcp_server

            server = create_mcp_server()

        mock_fastmcp_cls.assert_called_once_with("memx", description="MemX Memory Server")
        assert server is mock_server

    def test_five_tools_registered(self):
        """All 5 tools are registered via @mcp.tool()."""
        mock_fastmcp_cls = MagicMock()
        mock_server = MagicMock()
        mock_fastmcp_cls.return_value = mock_server
        mock_server.tool.return_value = lambda fn: fn

        with patch("memx.ext.mcp_server.FastMCP", mock_fastmcp_cls):
            from memx.ext.mcp_server import create_mcp_server

            create_mcp_server()

        assert mock_server.tool.call_count == 5


def _make_tools_and_mock():
    """Create a server, capture tool functions, and return (tools_dict, mock_memory)."""
    mock_memory = MagicMock()
    mock_memory.search.return_value = {"results": []}
    mock_memory.add.return_value = {"results": [{"id": "abc"}]}
    mock_memory.get_all.return_value = {"results": []}
    mock_memory.delete.return_value = None
    mock_memory.status.return_value = {"total": 5}

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
            patch("memx.ext.mcp_server.FastMCP", mock_cls),
            patch("memx.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memx.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["search_memory"]("test query", user_id="u1", limit=10)
        mock_memory.search.assert_called_once_with("test query", user_id="u1", limit=10)
        assert result == {"results": []}

    async def test_add_memory(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memx.ext.mcp_server.FastMCP", mock_cls),
            patch("memx.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memx.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["add_memory"]("hello world", user_id="u1")
        mock_memory.add.assert_called_once_with("hello world", user_id="u1")
        assert "results" in result

    async def test_list_memories(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memx.ext.mcp_server.FastMCP", mock_cls),
            patch("memx.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memx.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["list_memories"](user_id="u1")
        mock_memory.get_all.assert_called_once_with(user_id="u1")
        assert result == {"results": []}

    async def test_forget_memory(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memx.ext.mcp_server.FastMCP", mock_cls),
            patch("memx.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memx.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["forget_memory"]("mem-123")
        mock_memory.delete.assert_called_once_with("mem-123")
        assert result["status"] == "deleted"

    async def test_memory_status(self):
        tool_funcs, mock_memory, mock_cls = _make_tools_and_mock()
        with (
            patch("memx.ext.mcp_server.FastMCP", mock_cls),
            patch("memx.ext.mcp_server._get_memory", return_value=mock_memory),
        ):
            from memx.ext.mcp_server import create_mcp_server
            create_mcp_server()
            result = await tool_funcs["memory_status"](user_id="u1")
        mock_memory.status.assert_called_once_with(user_id="u1")
        assert result == {"total": 5}


class TestGetMemory:
    """Test the lazy Memory singleton initialization."""

    def test_lazy_init(self):
        """_get_memory creates Memory only once."""
        import memx.ext.mcp_server as mod

        mod._memory_singleton = None  # reset

        mock_memory = MagicMock()
        with patch("memx.core.memory.Memory", return_value=mock_memory) as mock_cls:
            result1 = mod._get_memory()
            result2 = mod._get_memory()

        mock_cls.assert_called_once_with(config=None)
        assert result1 is result2 is mock_memory

        mod._memory_singleton = None  # cleanup
