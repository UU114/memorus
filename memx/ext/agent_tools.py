"""MemX Agent Tool Wrappers — OpenAI Agents SDK & LangChain integration.

Provides factory functions that return framework-native tool objects
backed by a caller-supplied Memory instance.

Usage:
    from memx import Memory
    from memx.ext.agent_tools import get_openai_tools

    memory = Memory()
    tools = get_openai_tools(memory)
"""

from __future__ import annotations

from typing import Any, Optional


def get_openai_tools(memory: Any) -> list[Any]:
    """Return OpenAI Agents SDK function tools backed by *memory*.

    Args:
        memory: A ``memx.core.memory.Memory`` instance (required).

    Returns:
        List of ``FunctionTool`` objects for the OpenAI Agents SDK.

    Raises:
        ImportError: If ``openai-agents`` is not installed.
        TypeError: If *memory* is None.
    """
    if memory is None:
        raise TypeError("memory argument is required and cannot be None")

    try:
        from agents import function_tool
    except ImportError:
        raise ImportError(
            "OpenAI Agents SDK tools require the 'openai-agents' package. "
            "Install with: pip install memx[agents]"
        )

    @function_tool
    def search_memory(query: str, user_id: Optional[str] = None) -> dict[str, Any]:
        """Search memories by semantic similarity."""
        return memory.search(query, user_id=user_id)

    @function_tool
    def add_memory(content: str, user_id: Optional[str] = None) -> dict[str, Any]:
        """Add a new memory entry."""
        return memory.add(content, user_id=user_id)

    return [search_memory, add_memory]


def get_langchain_tools(memory: Any) -> list[Any]:
    """Return LangChain tool objects backed by *memory*.

    Args:
        memory: A ``memx.core.memory.Memory`` instance (required).

    Returns:
        List of ``BaseTool`` subclass instances for LangChain.

    Raises:
        ImportError: If ``langchain-core`` is not installed.
        TypeError: If *memory* is None.
    """
    if memory is None:
        raise TypeError("memory argument is required and cannot be None")

    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        raise ImportError(
            "LangChain tools require the 'langchain-core' package. "
            "Install with: pip install memx[langchain]"
        )

    from pydantic import BaseModel, Field

    # -- Search tool --------------------------------------------------------

    class SearchInput(BaseModel):
        query: str = Field(description="Search query string")
        user_id: Optional[str] = Field(default=None, description="Optional user ID filter")

    class MemXSearchTool(BaseTool):
        name: str = "search_memory"
        description: str = "Search memories by semantic similarity"
        args_schema: type[BaseModel] = SearchInput

        def _run(self, query: str, user_id: Optional[str] = None) -> dict[str, Any]:
            return memory.search(query, user_id=user_id)

    # -- Add tool -----------------------------------------------------------

    class AddInput(BaseModel):
        content: str = Field(description="Memory content to add")
        user_id: Optional[str] = Field(default=None, description="Optional user ID")

    class MemXAddTool(BaseTool):
        name: str = "add_memory"
        description: str = "Add a new memory entry"
        args_schema: type[BaseModel] = AddInput

        def _run(self, content: str, user_id: Optional[str] = None) -> dict[str, Any]:
            return memory.add(content, user_id=user_id)

    return [MemXSearchTool(), MemXAddTool()]
