"""Tests for memx.ext.agent_tools."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from memx.ext.agent_tools import get_langchain_tools, get_openai_tools


class TestGetOpenaiTools:
    """Test OpenAI Agents SDK tool factory."""

    def test_memory_required(self):
        """Passing None raises TypeError."""
        with pytest.raises(TypeError, match="memory argument is required"):
            get_openai_tools(None)

    def test_import_error_without_agents(self):
        """Raises ImportError when openai-agents is missing."""
        mock_memory = MagicMock()
        with patch.dict("sys.modules", {"agents": None}):
            with pytest.raises(ImportError, match="openai-agents"):
                get_openai_tools(mock_memory)

    def test_returns_two_tools(self):
        """Returns a list of 2 function tools."""
        mock_memory = MagicMock()

        # Mock the agents module with a passthrough function_tool decorator
        mock_agents = MagicMock()
        mock_agents.function_tool = lambda fn: fn

        with patch.dict("sys.modules", {"agents": mock_agents}):
            tools = get_openai_tools(mock_memory)

        assert len(tools) == 2

    def test_search_tool_calls_memory(self):
        """search_memory tool delegates to Memory.search()."""
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}

        mock_agents = MagicMock()
        mock_agents.function_tool = lambda fn: fn

        with patch.dict("sys.modules", {"agents": mock_agents}):
            tools = get_openai_tools(mock_memory)

        search_fn = tools[0]
        result = search_fn("test query", user_id="u1")
        mock_memory.search.assert_called_once_with("test query", user_id="u1")
        assert result == {"results": []}

    def test_add_tool_calls_memory(self):
        """add_memory tool delegates to Memory.add()."""
        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": []}

        mock_agents = MagicMock()
        mock_agents.function_tool = lambda fn: fn

        with patch.dict("sys.modules", {"agents": mock_agents}):
            tools = get_openai_tools(mock_memory)

        add_fn = tools[1]
        result = add_fn("hello", user_id="u1")
        mock_memory.add.assert_called_once_with("hello", user_id="u1")


class TestGetLangchainTools:
    """Test LangChain tool factory."""

    def test_memory_required(self):
        """Passing None raises TypeError."""
        with pytest.raises(TypeError, match="memory argument is required"):
            get_langchain_tools(None)

    def test_import_error_without_langchain(self):
        """Raises ImportError when langchain-core is missing."""
        mock_memory = MagicMock()
        with patch.dict("sys.modules", {"langchain_core": None, "langchain_core.tools": None}):
            with pytest.raises(ImportError, match="langchain-core"):
                get_langchain_tools(mock_memory)

    def test_returns_two_tools_with_correct_names(self):
        """Returns 2 BaseTool instances with expected names (requires langchain-core)."""
        langchain_core = pytest.importorskip("langchain_core")

        mock_memory = MagicMock()
        tools = get_langchain_tools(mock_memory)

        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"search_memory", "add_memory"}

    def test_search_tool_delegates(self):
        """Search tool _run delegates to Memory.search() (requires langchain-core)."""
        pytest.importorskip("langchain_core")

        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}

        tools = get_langchain_tools(mock_memory)
        search_tool = next(t for t in tools if t.name == "search_memory")
        result = search_tool._run(query="hello", user_id="u1")
        mock_memory.search.assert_called_once_with("hello", user_id="u1")

    def test_add_tool_delegates(self):
        """Add tool _run delegates to Memory.add() (requires langchain-core)."""
        pytest.importorskip("langchain_core")

        mock_memory = MagicMock()
        mock_memory.add.return_value = {"results": []}

        tools = get_langchain_tools(mock_memory)
        add_tool = next(t for t in tools if t.name == "add_memory")
        result = add_tool._run(content="hello", user_id="u1")
        mock_memory.add.assert_called_once_with("hello", user_id="u1")
