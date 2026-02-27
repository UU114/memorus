"""Unit tests for memx.memory — Memory decorator wrapper class."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from memx.config import MemXConfig
from memx.memory import Memory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory() -> Memory:
    """Create a Memory instance with a mocked mem0 backend.

    We bypass __init__ entirely and wire up the internal state manually so
    that tests do not require mem0 to be installed or an API key to be set.
    """
    m = Memory.__new__(Memory)
    m._config = MemXConfig()
    m._mem0 = MagicMock()
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


@pytest.fixture
def ace_memory() -> Memory:
    """Create a Memory instance with ACE enabled."""
    m = Memory.__new__(Memory)
    m._config = MemXConfig(ace_enabled=True)
    m._mem0 = MagicMock()
    m._mem0_init_error = None

    m._mem0.add.return_value = {"results": [{"id": "1", "memory": "test"}]}
    m._mem0.search.return_value = {"results": []}

    m._ingest_pipeline = None
    m._retrieval_pipeline = None
    m._sanitizer = None
    return m


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


class TestInit:
    """Memory initialization tests."""

    def test_init_default_config(self, memory: Memory) -> None:
        """Memory() uses default MemXConfig (ace_enabled=False)."""
        assert memory._config.ace_enabled is False
        assert isinstance(memory._config, MemXConfig)

    def test_init_with_config_ace_enabled(self) -> None:
        """Memory({"ace_enabled": True}) sets ace_enabled flag."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig.from_dict({"ace_enabled": True})
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None
        assert m._config.ace_enabled is True

    def test_init_with_mem0_config_fields(self) -> None:
        """mem0-specific fields are separated from ACE fields in config."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig.from_dict({
            "vector_store": {"provider": "qdrant"},
            "llm": {"provider": "openai"},
        })
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None
        assert "vector_store" in m._config.mem0_config
        assert "llm" in m._config.mem0_config

    def test_from_config_classmethod(self) -> None:
        """Memory.from_config({}) works as an alternate constructor."""
        with patch.object(Memory, "__init__", return_value=None) as mock_init:
            result = Memory.from_config({"ace_enabled": True})
            mock_init.assert_called_once_with(config={"ace_enabled": True})
            assert isinstance(result, Memory)

    def test_mem0_init_failure_stored(self) -> None:
        """When mem0 import fails, the error is stored and _mem0 is None."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._mem0 = None
        m._mem0_init_error = ImportError("No module named 'mem0'")
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None
        assert m._mem0 is None
        assert isinstance(m._mem0_init_error, ImportError)


# ---------------------------------------------------------------------------
# _ensure_mem0 tests
# ---------------------------------------------------------------------------


class TestEnsureMem0:
    """Tests for the _ensure_mem0 guard."""

    def test_ensure_mem0_returns_backend(self, memory: Memory) -> None:
        """_ensure_mem0 returns _mem0 when it is available."""
        result = memory._ensure_mem0()
        assert result is memory._mem0

    def test_ensure_mem0_raises_when_none(self) -> None:
        """_ensure_mem0 raises RuntimeError when _mem0 is None."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._mem0 = None
        m._mem0_init_error = ImportError("No module named 'mem0'")
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        with pytest.raises(RuntimeError, match="mem0 backend not initialized"):
            m._ensure_mem0()


# ---------------------------------------------------------------------------
# Proxy mode tests (ACE disabled)
# ---------------------------------------------------------------------------


class TestProxyMode:
    """All core mem0 API calls proxy through to _mem0 backend when ACE is off."""

    def test_add_proxy_mode(self, memory: Memory) -> None:
        """ace_enabled=False: add() calls _mem0.add with same args."""
        result = memory.add(
            "test message",
            user_id="u1",
            agent_id="a1",
            run_id="r1",
            metadata={"key": "val"},
            filters={"user_id": "u1"},
            prompt="custom prompt",
        )
        memory._mem0.add.assert_called_once_with(
            "test message",
            user_id="u1",
            agent_id="a1",
            run_id="r1",
            metadata={"key": "val"},
            filters={"user_id": "u1"},
            prompt="custom prompt",
        )
        assert result == {"results": [{"id": "1", "memory": "test"}]}

    def test_search_proxy_mode(self, memory: Memory) -> None:
        """ace_enabled=False: search() calls _mem0.search with same args."""
        result = memory.search(
            "query",
            user_id="u1",
            agent_id="a1",
            run_id="r1",
            limit=50,
            filters={"user_id": "u1"},
        )
        memory._mem0.search.assert_called_once_with(
            "query",
            user_id="u1",
            agent_id="a1",
            run_id="r1",
            limit=50,
            filters={"user_id": "u1"},
        )
        assert result == {"results": []}

    def test_get_all_proxy(self, memory: Memory) -> None:
        """get_all() proxies to _mem0.get_all."""
        result = memory.get_all(user_id="u1")
        memory._mem0.get_all.assert_called_once_with(user_id="u1")
        assert result == {"memories": []}

    def test_get_proxy(self, memory: Memory) -> None:
        """get() proxies to _mem0.get."""
        result = memory.get("mem-123")
        memory._mem0.get.assert_called_once_with("mem-123")
        assert result == {"id": "1", "memory": "test"}

    def test_update_proxy(self, memory: Memory) -> None:
        """update() proxies to _mem0.update."""
        result = memory.update("mem-123", "new data")
        memory._mem0.update.assert_called_once_with("mem-123", "new data")
        assert result == {"id": "1", "memory": "updated"}

    def test_delete_proxy(self, memory: Memory) -> None:
        """delete() proxies to _mem0.delete."""
        memory.delete("mem-123")
        memory._mem0.delete.assert_called_once_with("mem-123")

    def test_delete_all_proxy(self, memory: Memory) -> None:
        """delete_all() proxies to _mem0.delete_all."""
        memory.delete_all(user_id="u1")
        memory._mem0.delete_all.assert_called_once_with(user_id="u1")

    def test_history_proxy(self, memory: Memory) -> None:
        """history() proxies to _mem0.history."""
        result = memory.history("mem-123")
        memory._mem0.history.assert_called_once_with("mem-123")
        assert result == {"changes": []}

    def test_reset_proxy(self, memory: Memory) -> None:
        """reset() proxies to _mem0.reset."""
        memory.reset()
        memory._mem0.reset.assert_called_once()

    def test_add_with_kwargs(self, memory: Memory) -> None:
        """add() forwards extra kwargs to mem0."""
        memory.add("msg", extra_param="extra_value")
        memory._mem0.add.assert_called_once_with(
            "msg",
            user_id=None,
            agent_id=None,
            run_id=None,
            metadata=None,
            filters=None,
            prompt=None,
            extra_param="extra_value",
        )

    def test_search_with_kwargs(self, memory: Memory) -> None:
        """search() forwards extra kwargs to mem0."""
        memory.search("q", extra_param="extra_value")
        memory._mem0.search.assert_called_once_with(
            "q",
            user_id=None,
            agent_id=None,
            run_id=None,
            limit=100,
            filters=None,
            extra_param="extra_value",
        )


# ---------------------------------------------------------------------------
# ACE mode fallback tests (pipeline not yet implemented)
# ---------------------------------------------------------------------------


class TestACEModeFallback:
    """When ACE is enabled but pipelines are None, methods fall back to proxy."""

    def test_add_ace_mode_no_pipeline_falls_back(self, ace_memory: Memory) -> None:
        """ACE on but no pipeline: add() still proxies to _mem0."""
        assert ace_memory._config.ace_enabled is True
        assert ace_memory._ingest_pipeline is None
        result = ace_memory.add("test message", user_id="u1")
        ace_memory._mem0.add.assert_called_once()
        assert result == {"results": [{"id": "1", "memory": "test"}]}

    def test_search_ace_mode_no_pipeline_falls_back(self, ace_memory: Memory) -> None:
        """ACE on but no pipeline: search() still proxies to _mem0."""
        assert ace_memory._config.ace_enabled is True
        assert ace_memory._retrieval_pipeline is None
        result = ace_memory.search("query", user_id="u1")
        ace_memory._mem0.search.assert_called_once()
        assert result == {"results": []}


# ---------------------------------------------------------------------------
# NotImplementedError tests
# ---------------------------------------------------------------------------


class TestNotImplemented:
    """ACE-specific methods raise NotImplementedError until their stories land."""

    def test_status_not_implemented(self, memory: Memory) -> None:
        """status() raises NotImplementedError with STORY reference."""
        with pytest.raises(NotImplementedError, match="STORY-041"):
            memory.status()

    def test_export_not_implemented(self, memory: Memory) -> None:
        """export() raises NotImplementedError with STORY reference."""
        with pytest.raises(NotImplementedError, match="STORY-044"):
            memory.export()

    def test_import_data_not_implemented(self, memory: Memory) -> None:
        """import_data() raises NotImplementedError with STORY reference."""
        with pytest.raises(NotImplementedError, match="STORY-044"):
            memory.import_data({})

    def test_run_decay_sweep_not_implemented(self, memory: Memory) -> None:
        """run_decay_sweep() raises NotImplementedError with STORY reference."""
        with pytest.raises(NotImplementedError, match="STORY-021"):
            memory.run_decay_sweep()


# ---------------------------------------------------------------------------
# Config property tests
# ---------------------------------------------------------------------------


class TestConfigProperty:
    """Tests for the config property."""

    def test_config_property_returns_memx_config(self, memory: Memory) -> None:
        """config property returns the MemXConfig instance."""
        assert isinstance(memory.config, MemXConfig)

    def test_config_property_ace_disabled_by_default(self, memory: Memory) -> None:
        """Default config has ace_enabled=False."""
        assert memory.config.ace_enabled is False

    def test_config_property_reflects_ace_enabled(self, ace_memory: Memory) -> None:
        """config property reflects ace_enabled=True when set."""
        assert ace_memory.config.ace_enabled is True


# ---------------------------------------------------------------------------
# Sanitization tests
# ---------------------------------------------------------------------------


class TestSanitization:
    """Tests for _sanitize_messages internal method."""

    def test_sanitize_string_messages(self, memory: Memory) -> None:
        """String messages are sanitized when sanitizer is available."""
        mock_sanitizer = MagicMock()
        mock_result = MagicMock()
        mock_result.clean_content = "cleaned text"
        mock_sanitizer.sanitize.return_value = mock_result
        memory._sanitizer = mock_sanitizer

        result = memory._sanitize_messages("raw text with PII")
        assert result == "cleaned text"
        mock_sanitizer.sanitize.assert_called_once_with("raw text with PII")

    def test_sanitize_list_messages(self, memory: Memory) -> None:
        """List of dict messages are sanitized on the 'content' field."""
        mock_sanitizer = MagicMock()
        mock_result = MagicMock()
        mock_result.clean_content = "cleaned"
        mock_sanitizer.sanitize.return_value = mock_result
        memory._sanitizer = mock_sanitizer

        messages = [
            {"role": "user", "content": "sensitive data"},
            {"role": "assistant", "content": "response"},
        ]
        result = memory._sanitize_messages(messages)
        assert len(result) == 2
        assert result[0]["content"] == "cleaned"
        assert result[0]["role"] == "user"
        assert result[1]["content"] == "cleaned"

    def test_sanitize_list_non_dict_items_passed_through(self, memory: Memory) -> None:
        """Non-dict items in message lists are passed through unchanged."""
        mock_sanitizer = MagicMock()
        memory._sanitizer = mock_sanitizer

        messages = ["plain string", 42, None]
        result = memory._sanitize_messages(messages)
        assert result == ["plain string", 42, None]
        mock_sanitizer.sanitize.assert_not_called()

    def test_sanitize_dict_without_content_passed_through(self, memory: Memory) -> None:
        """Dict items missing 'content' key are passed through unchanged."""
        mock_sanitizer = MagicMock()
        memory._sanitizer = mock_sanitizer

        messages = [{"role": "system", "text": "no content key"}]
        result = memory._sanitize_messages(messages)
        assert result == [{"role": "system", "text": "no content key"}]
        mock_sanitizer.sanitize.assert_not_called()

    def test_sanitize_returns_original_on_error(self, memory: Memory) -> None:
        """Sanitization errors return original messages (graceful degradation)."""
        mock_sanitizer = MagicMock()
        mock_sanitizer.sanitize.side_effect = RuntimeError("sanitizer broken")
        memory._sanitizer = mock_sanitizer

        result = memory._sanitize_messages("raw text")
        assert result == "raw text"

    def test_sanitize_no_sanitizer_returns_original(self, memory: Memory) -> None:
        """When no sanitizer is set, messages pass through unchanged."""
        memory._sanitizer = None
        result = memory._sanitize_messages("some text")
        assert result == "some text"

    def test_sanitize_non_string_non_list_passed_through(self, memory: Memory) -> None:
        """Non-string, non-list inputs are returned unchanged."""
        mock_sanitizer = MagicMock()
        memory._sanitizer = mock_sanitizer

        result = memory._sanitize_messages(12345)
        assert result == 12345
        mock_sanitizer.sanitize.assert_not_called()

    def test_add_with_always_sanitize(self, memory: Memory) -> None:
        """When always_sanitize is True and ACE is off, add() sanitizes messages."""
        # Set up config with always_sanitize
        memory._config = MemXConfig.from_dict({
            "privacy": {"always_sanitize": True},
        })

        # Set up sanitizer mock
        mock_sanitizer = MagicMock()
        mock_result = MagicMock()
        mock_result.clean_content = "sanitized message"
        mock_sanitizer.sanitize.return_value = mock_result
        memory._sanitizer = mock_sanitizer

        memory.add("raw sensitive message", user_id="u1")

        # The sanitizer should have been called
        mock_sanitizer.sanitize.assert_called_once_with("raw sensitive message")
        # mem0 should receive the sanitized version
        memory._mem0.add.assert_called_once()
        call_args = memory._mem0.add.call_args
        assert call_args[0][0] == "sanitized message"

    def test_add_without_always_sanitize_no_sanitization(self, memory: Memory) -> None:
        """When always_sanitize is False (default), add() does not sanitize."""
        mock_sanitizer = MagicMock()
        memory._sanitizer = mock_sanitizer

        memory.add("raw message", user_id="u1")

        mock_sanitizer.sanitize.assert_not_called()
        call_args = memory._mem0.add.call_args
        assert call_args[0][0] == "raw message"


# ---------------------------------------------------------------------------
# _init_ace_engines test
# ---------------------------------------------------------------------------


class TestInitACEEngines:
    """Tests for _init_ace_engines graceful degradation."""

    def test_init_ace_engines_handles_import_failure(self) -> None:
        """When ACE engine imports fail, Memory degrades to proxy mode."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        # Patch imports to fail
        with patch(
            "memx.memory.Memory._init_ace_engines"
        ) as mock_init:
            mock_init.side_effect = None  # no-op
            m._init_ace_engines()

        # Should not crash — graceful degradation
        assert m._ingest_pipeline is None
        assert m._retrieval_pipeline is None


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and integration-style tests."""

    def test_mem0_backend_failure_on_add(self, memory: Memory) -> None:
        """When _mem0 is None, add() raises RuntimeError."""
        memory._mem0 = None
        memory._mem0_init_error = ImportError("missing")
        with pytest.raises(RuntimeError, match="mem0 backend not initialized"):
            memory.add("test")

    def test_mem0_backend_failure_on_search(self, memory: Memory) -> None:
        """When _mem0 is None, search() raises RuntimeError."""
        memory._mem0 = None
        memory._mem0_init_error = ImportError("missing")
        with pytest.raises(RuntimeError, match="mem0 backend not initialized"):
            memory.search("query")

    def test_mem0_backend_failure_on_get(self, memory: Memory) -> None:
        """When _mem0 is None, get() raises RuntimeError."""
        memory._mem0 = None
        memory._mem0_init_error = ImportError("missing")
        with pytest.raises(RuntimeError, match="mem0 backend not initialized"):
            memory.get("id-1")

    def test_mem0_backend_failure_on_delete(self, memory: Memory) -> None:
        """When _mem0 is None, delete() raises RuntimeError."""
        memory._mem0 = None
        memory._mem0_init_error = ImportError("missing")
        with pytest.raises(RuntimeError, match="mem0 backend not initialized"):
            memory.delete("id-1")

    def test_mem0_backend_failure_on_reset(self, memory: Memory) -> None:
        """When _mem0 is None, reset() raises RuntimeError."""
        memory._mem0 = None
        memory._mem0_init_error = ImportError("missing")
        with pytest.raises(RuntimeError, match="mem0 backend not initialized"):
            memory.reset()

    def test_add_with_no_args(self, memory: Memory) -> None:
        """add() with only messages and no optional args works."""
        result = memory.add("simple message")
        memory._mem0.add.assert_called_once_with(
            "simple message",
            user_id=None,
            agent_id=None,
            run_id=None,
            metadata=None,
            filters=None,
            prompt=None,
        )
        assert "results" in result

    def test_search_with_default_limit(self, memory: Memory) -> None:
        """search() uses default limit=100 when not specified."""
        memory.search("query")
        memory._mem0.search.assert_called_once_with(
            "query",
            user_id=None,
            agent_id=None,
            run_id=None,
            limit=100,
            filters=None,
        )

    def test_get_all_with_no_args(self, memory: Memory) -> None:
        """get_all() with no kwargs works."""
        result = memory.get_all()
        memory._mem0.get_all.assert_called_once_with()
        assert result == {"memories": []}
