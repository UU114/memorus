"""mem0 API compatibility tests — verify memx.Memory is a drop-in replacement."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from memx.core.config import MemXConfig
from memx.core.memory import Memory


@pytest.fixture()
def mem0_mock() -> MagicMock:
    """Create a realistic mem0 mock with standard return values."""
    mock = MagicMock()
    mock.add.return_value = {
        "results": [
            {"id": "mem-001", "memory": "User prefers dark mode", "event": "ADD"}
        ]
    }
    mock.search.return_value = {
        "results": [
            {
                "id": "mem-001",
                "memory": "User prefers dark mode",
                "score": 0.95,
            }
        ]
    }
    mock.get_all.return_value = {
        "results": [
            {"id": "mem-001", "memory": "User prefers dark mode"},
            {"id": "mem-002", "memory": "Uses pytest for testing"},
        ]
    }
    mock.get.return_value = {
        "id": "mem-001",
        "memory": "User prefers dark mode",
        "metadata": {},
    }
    mock.update.return_value = {
        "id": "mem-001",
        "memory": "User prefers dark mode in VS Code",
        "event": "UPDATE",
    }
    mock.delete.return_value = {"message": "Memory deleted"}
    mock.delete_all.return_value = {"message": "All memories deleted"}
    mock.history.return_value = {
        "changes": [
            {
                "id": "change-1",
                "event": "ADD",
                "old_memory": None,
                "new_memory": "User prefers dark mode",
            }
        ]
    }
    mock.reset.return_value = {"message": "Reset successful"}
    return mock


@pytest.fixture()
def memory(mem0_mock: MagicMock) -> Memory:
    """Create a Memory in proxy mode (ace_enabled=False) with mocked mem0."""
    m = Memory.__new__(Memory)
    m._config = MemXConfig()  # ace_enabled=False by default
    m._mem0 = mem0_mock
    m._mem0_init_error = None
    m._ingest_pipeline = None
    m._retrieval_pipeline = None
    m._sanitizer = None
    return m


class TestAddCompat:
    """Test add() API compatibility."""

    def test_add_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        result = memory.add("Python is great", user_id="u1")
        mem0_mock.add.assert_called_once_with(
            "Python is great",
            user_id="u1",
            agent_id=None,
            run_id=None,
            metadata=None,
            filters=None,
            prompt=None,
        )
        assert "results" in result

    def test_add_with_user_id(self, memory: Memory, mem0_mock: MagicMock) -> None:
        memory.add("test", user_id="user_123")
        assert mem0_mock.add.call_args.kwargs["user_id"] == "user_123"

    def test_add_with_agent_id(self, memory: Memory, mem0_mock: MagicMock) -> None:
        memory.add("test", agent_id="agent_456")
        assert mem0_mock.add.call_args.kwargs["agent_id"] == "agent_456"

    def test_add_with_metadata(self, memory: Memory, mem0_mock: MagicMock) -> None:
        meta = {"category": "tools", "priority": "high"}
        memory.add("Use pytest", user_id="u1", metadata=meta)
        assert mem0_mock.add.call_args.kwargs["metadata"] == meta

    def test_add_with_filters(self, memory: Memory, mem0_mock: MagicMock) -> None:
        filters = {"user_id": "u1"}
        memory.add("test", filters=filters)
        assert mem0_mock.add.call_args.kwargs["filters"] == filters

    def test_add_with_prompt(self, memory: Memory, mem0_mock: MagicMock) -> None:
        memory.add("test", prompt="Extract preferences")
        assert mem0_mock.add.call_args.kwargs["prompt"] == "Extract preferences"

    def test_add_return_value(self, memory: Memory) -> None:
        result = memory.add("test")
        assert isinstance(result, dict)
        assert "results" in result
        assert result["results"][0]["id"] == "mem-001"

    def test_add_list_messages(self, memory: Memory, mem0_mock: MagicMock) -> None:
        msgs = [
            {"role": "user", "content": "I prefer dark mode"},
            {"role": "assistant", "content": "Noted!"},
        ]
        memory.add(msgs, user_id="u1")
        mem0_mock.add.assert_called_once()
        assert mem0_mock.add.call_args[0][0] == msgs


class TestSearchCompat:
    """Test search() API compatibility."""

    def test_search_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        result = memory.search("dark mode", user_id="u1")
        mem0_mock.search.assert_called_once_with(
            "dark mode",
            user_id="u1",
            agent_id=None,
            run_id=None,
            limit=100,
            filters=None,
        )
        assert "results" in result

    def test_search_with_limit(self, memory: Memory, mem0_mock: MagicMock) -> None:
        memory.search("test", limit=5)
        assert mem0_mock.search.call_args.kwargs["limit"] == 5

    def test_search_with_filters(self, memory: Memory, mem0_mock: MagicMock) -> None:
        filters = {"user_id": "u1", "agent_id": "a1"}
        memory.search("test", filters=filters)
        assert mem0_mock.search.call_args.kwargs["filters"] == filters

    def test_search_return_value(self, memory: Memory) -> None:
        result = memory.search("dark mode")
        assert isinstance(result, dict)
        assert "results" in result
        assert result["results"][0]["score"] == 0.95


class TestGetAllCompat:
    """Test get_all() API compatibility."""

    def test_get_all_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        result = memory.get_all()
        mem0_mock.get_all.assert_called_once()
        assert "results" in result

    def test_get_all_with_user_id(self, memory: Memory, mem0_mock: MagicMock) -> None:
        memory.get_all(user_id="u1")
        mem0_mock.get_all.assert_called_once_with(user_id="u1")

    def test_get_all_return_value(self, memory: Memory) -> None:
        result = memory.get_all()
        assert len(result["results"]) == 2


class TestGetCompat:
    """Test get() API compatibility."""

    def test_get_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        result = memory.get("mem-001")
        mem0_mock.get.assert_called_once_with("mem-001")
        assert result["id"] == "mem-001"


class TestUpdateCompat:
    """Test update() API compatibility."""

    def test_update_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        result = memory.update("mem-001", "Updated preference")
        mem0_mock.update.assert_called_once_with("mem-001", "Updated preference")
        assert result["event"] == "UPDATE"


class TestDeleteCompat:
    """Test delete() and delete_all() API compatibility."""

    def test_delete_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        memory.delete("mem-001")
        mem0_mock.delete.assert_called_once_with("mem-001")

    def test_delete_all_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        memory.delete_all()
        mem0_mock.delete_all.assert_called_once()

    def test_delete_all_with_user_id(
        self, memory: Memory, mem0_mock: MagicMock
    ) -> None:
        memory.delete_all(user_id="u1")
        mem0_mock.delete_all.assert_called_once_with(user_id="u1")


class TestHistoryCompat:
    """Test history() API compatibility."""

    def test_history_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        result = memory.history("mem-001")
        mem0_mock.history.assert_called_once_with("mem-001")
        assert "changes" in result


class TestResetCompat:
    """Test reset() API compatibility."""

    def test_reset_basic(self, memory: Memory, mem0_mock: MagicMock) -> None:
        memory.reset()
        mem0_mock.reset.assert_called_once()


class TestConfigCompat:
    """Test config dict compatibility."""

    def test_mem0_config_passthrough(self) -> None:
        """mem0 config fields should be preserved in mem0_config."""
        config = {
            "vector_store": {"provider": "qdrant", "config": {"host": "localhost"}},
            "llm": {"provider": "openai", "config": {"model": "gpt-4"}},
        }
        m = Memory.__new__(Memory)
        m._config = MemXConfig.from_dict(config)
        assert m._config.mem0_config["vector_store"]["provider"] == "qdrant"
        assert m._config.mem0_config["llm"]["provider"] == "openai"

    def test_mixed_config(self) -> None:
        """Both ACE and mem0 fields in same config dict."""
        config = {
            "ace_enabled": True,
            "vector_store": {"provider": "qdrant"},
        }
        m = Memory.__new__(Memory)
        m._config = MemXConfig.from_dict(config)
        assert m._config.ace_enabled is True
        assert m._config.mem0_config["vector_store"]["provider"] == "qdrant"

    def test_default_config(self) -> None:
        """Empty config works (ace_enabled=False by default)."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig.from_dict({})
        assert m._config.ace_enabled is False


class TestProxyModeGuarantee:
    """Verify proxy mode doesn't alter behavior."""

    def test_proxy_mode_no_transformation(
        self, memory: Memory, mem0_mock: MagicMock
    ) -> None:
        """In proxy mode, add() should pass through without any transformation."""
        original_msg = "Keep this exactly as-is"
        memory.add(original_msg, user_id="u1")
        assert mem0_mock.add.call_args[0][0] == original_msg

    def test_proxy_mode_preserves_return(
        self, memory: Memory, mem0_mock: MagicMock
    ) -> None:
        """Return value from mem0 should be returned unmodified."""
        custom_return: dict[str, Any] = {"custom": "value", "results": []}
        mem0_mock.add.return_value = custom_return
        result = memory.add("test")
        assert result is custom_return
