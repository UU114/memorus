"""Unit tests for memx.async_memory — AsyncMemory async wrapper class."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from memx.async_memory import AsyncMemory
from memx.config import MemXConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def async_memory() -> AsyncMemory:
    """Create AsyncMemory with mocked mem0 async backend.

    We bypass __init__ entirely and wire up the internal state manually so
    that tests do not require mem0 to be installed or an API key to be set.
    """
    m = AsyncMemory.__new__(AsyncMemory)
    m._config = MemXConfig()
    m._mem0 = AsyncMock()
    m._mem0_init_error = None

    # Configure mock return values for all proxy methods
    m._mem0.add.return_value = {"results": [{"id": "1", "memory": "test"}]}
    m._mem0.search.return_value = {"results": []}
    m._mem0.get_all.return_value = {"memories": []}
    m._mem0.get.return_value = {"id": "1", "memory": "test"}
    m._mem0.update.return_value = {"id": "1", "memory": "updated"}
    m._mem0.delete.return_value = None
    m._mem0.delete_all.return_value = None
    m._mem0.history.return_value = {"changes": []}
    m._mem0.reset.return_value = None

    m._ingest_pipeline = None
    m._retrieval_pipeline = None
    m._sanitizer = None
    return m


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    """AsyncMemory initialization tests."""

    def test_init_default_config(self, async_memory: AsyncMemory) -> None:
        """AsyncMemory() uses default MemXConfig (ace_enabled=False)."""
        assert async_memory._config.ace_enabled is False
        assert isinstance(async_memory._config, MemXConfig)

    def test_from_config(self) -> None:
        """AsyncMemory.from_config({}) works as an alternate constructor."""
        with patch.object(AsyncMemory, "__init__", return_value=None) as mock_init:
            result = AsyncMemory.from_config({"ace_enabled": True})
            mock_init.assert_called_once_with(config={"ace_enabled": True})
            assert isinstance(result, AsyncMemory)


# ---------------------------------------------------------------------------
# Proxy mode tests (ACE disabled)
# ---------------------------------------------------------------------------


class TestProxyMode:
    """All core async API calls proxy through to _mem0 backend when ACE is off."""

    async def test_add_proxy(self, async_memory: AsyncMemory) -> None:
        """ace_enabled=False: add() awaits _mem0.add."""
        result = await async_memory.add("test message")
        async_memory._mem0.add.assert_called_once_with(
            "test message",
            user_id=None,
            agent_id=None,
            run_id=None,
            metadata=None,
            filters=None,
            prompt=None,
        )
        assert result == {"results": [{"id": "1", "memory": "test"}]}

    async def test_search_proxy(self, async_memory: AsyncMemory) -> None:
        """ace_enabled=False: search() awaits _mem0.search."""
        result = await async_memory.search("query")
        async_memory._mem0.search.assert_called_once_with(
            "query",
            user_id=None,
            agent_id=None,
            run_id=None,
            limit=100,
            filters=None,
        )
        assert result == {"results": []}

    async def test_get_all_proxy(self, async_memory: AsyncMemory) -> None:
        """get_all() proxies to _mem0.get_all."""
        result = await async_memory.get_all(user_id="u1")
        async_memory._mem0.get_all.assert_called_once_with(user_id="u1")
        assert result == {"memories": []}

    async def test_get_proxy(self, async_memory: AsyncMemory) -> None:
        """get() proxies to _mem0.get."""
        result = await async_memory.get("mem-123")
        async_memory._mem0.get.assert_called_once_with("mem-123")
        assert result == {"id": "1", "memory": "test"}

    async def test_update_proxy(self, async_memory: AsyncMemory) -> None:
        """update() proxies to _mem0.update."""
        result = await async_memory.update("mem-123", "new data")
        async_memory._mem0.update.assert_called_once_with("mem-123", "new data")
        assert result == {"id": "1", "memory": "updated"}

    async def test_delete_proxy(self, async_memory: AsyncMemory) -> None:
        """delete() proxies to _mem0.delete."""
        await async_memory.delete("mem-123")
        async_memory._mem0.delete.assert_called_once_with("mem-123")

    async def test_delete_all_proxy(self, async_memory: AsyncMemory) -> None:
        """delete_all() proxies to _mem0.delete_all."""
        await async_memory.delete_all(user_id="u1")
        async_memory._mem0.delete_all.assert_called_once_with(user_id="u1")

    async def test_history_proxy(self, async_memory: AsyncMemory) -> None:
        """history() proxies to _mem0.history."""
        result = await async_memory.history("mem-123")
        async_memory._mem0.history.assert_called_once_with("mem-123")
        assert result == {"changes": []}

    async def test_reset_proxy(self, async_memory: AsyncMemory) -> None:
        """reset() proxies to _mem0.reset."""
        await async_memory.reset()
        async_memory._mem0.reset.assert_called_once()


# ---------------------------------------------------------------------------
# Argument forwarding tests
# ---------------------------------------------------------------------------


class TestArgForwarding:
    """Tests that arguments are correctly forwarded to the mem0 backend."""

    async def test_add_with_args(self, async_memory: AsyncMemory) -> None:
        """add() forwards user_id, metadata, and other kwargs."""
        result = await async_memory.add(
            "test message",
            user_id="u1",
            agent_id="a1",
            run_id="r1",
            metadata={"key": "val"},
            filters={"user_id": "u1"},
            prompt="custom prompt",
        )
        async_memory._mem0.add.assert_called_once_with(
            "test message",
            user_id="u1",
            agent_id="a1",
            run_id="r1",
            metadata={"key": "val"},
            filters={"user_id": "u1"},
            prompt="custom prompt",
        )
        assert result == {"results": [{"id": "1", "memory": "test"}]}

    async def test_search_with_args(self, async_memory: AsyncMemory) -> None:
        """search() forwards user_id, limit, and other kwargs."""
        result = await async_memory.search(
            "query",
            user_id="u1",
            limit=50,
        )
        async_memory._mem0.search.assert_called_once_with(
            "query",
            user_id="u1",
            agent_id=None,
            run_id=None,
            limit=50,
            filters=None,
        )
        assert result == {"results": []}


# ---------------------------------------------------------------------------
# NotImplementedError tests
# ---------------------------------------------------------------------------


class TestNotImplemented:
    """ACE-specific async methods raise NotImplementedError until their stories land."""

    async def test_status_not_implemented(self, async_memory: AsyncMemory) -> None:
        """status() raises NotImplementedError with STORY reference."""
        with pytest.raises(NotImplementedError, match="STORY-041"):
            await async_memory.status()


# ---------------------------------------------------------------------------
# Config property tests
# ---------------------------------------------------------------------------


class TestConfigProperty:
    """Tests for the config property."""

    def test_config_property(self, async_memory: AsyncMemory) -> None:
        """config property returns the MemXConfig instance."""
        assert isinstance(async_memory.config, MemXConfig)
        assert async_memory.config.ace_enabled is False


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for _ensure_mem0 error handling."""

    def test_ensure_mem0_raises(self) -> None:
        """_ensure_mem0 raises RuntimeError when _mem0 is None."""
        m = AsyncMemory.__new__(AsyncMemory)
        m._config = MemXConfig()
        m._mem0 = None
        m._mem0_init_error = ImportError("No module named 'mem0'")
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        with pytest.raises(RuntimeError, match="mem0 async backend not initialized"):
            m._ensure_mem0()
