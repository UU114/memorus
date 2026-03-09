"""Unit tests for memx.integration.cli_hooks.

Covers CLIPreInferenceHook (STORY-033), CLIPostActionHook,
CLISessionEndHook, and setup_signal_handlers (STORY-034).
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest

from memx.core.config import IntegrationConfig
from memx.core.integration.cli_hooks import (
    CLIPostActionHook,
    CLIPreInferenceHook,
    CLISessionEndHook,
    setup_signal_handlers,
)
from memx.core.integration.hooks import (
    ContextInjection,
    PostActionHook,
    PreInferenceHook,
    SessionEndHook,
    ToolEvent,
)


# ---------------------------------------------------------------------------
# Helpers — mock Memory
# ---------------------------------------------------------------------------


def _make_mock_memory(
    search_results: Optional[dict[str, Any]] = None,
    search_side_effect: Optional[Exception] = None,
) -> MagicMock:
    """Create a mock Memory with configurable search() behavior."""
    mock = MagicMock()
    if search_side_effect is not None:
        mock.search.side_effect = search_side_effect
    else:
        mock.search.return_value = search_results or {"results": []}
    return mock


def _sample_results(count: int = 2) -> dict[str, Any]:
    """Generate sample Memory.search() response with *count* results."""
    results = []
    for i in range(count):
        results.append({
            "id": f"mem_{i:03d}",
            "memory": f"Sample memory content {i}.",
            "score": round(0.9 - i * 0.1, 2),
            "metadata": {"memx_knowledge_type": "preference" if i == 0 else "tool_pattern"},
        })
    return {"results": results}


# ===========================================================================
# Inheritance / interface tests
# ===========================================================================


class TestCLIPreInferenceHookInterface:
    def test_inherits_pre_inference_hook(self) -> None:
        mock_mem = _make_mock_memory()
        hook = CLIPreInferenceHook(mock_mem)
        assert isinstance(hook, PreInferenceHook)

    def test_name_property(self) -> None:
        hook = CLIPreInferenceHook(_make_mock_memory())
        assert hook.name == "cli_pre_inference"

    def test_enabled_default_true(self) -> None:
        hook = CLIPreInferenceHook(_make_mock_memory())
        assert hook.enabled is True

    def test_enabled_respects_auto_recall_false(self) -> None:
        config = IntegrationConfig(auto_recall=False)
        hook = CLIPreInferenceHook(_make_mock_memory(), config=config)
        assert hook.enabled is False

    def test_enabled_respects_auto_recall_true(self) -> None:
        config = IntegrationConfig(auto_recall=True)
        hook = CLIPreInferenceHook(_make_mock_memory(), config=config)
        assert hook.enabled is True


# ===========================================================================
# on_user_input — normal flow
# ===========================================================================


class TestOnUserInputNormal:
    def test_returns_context_injection(self) -> None:
        mock_mem = _make_mock_memory(_sample_results(2))
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("How do I run tests?"))
        assert isinstance(result, ContextInjection)
        assert len(result.memories) == 2
        assert result.format == "xml"
        assert result.rendered != ""
        mock_mem.search.assert_called_once_with("How do I run tests?")

    def test_calls_memory_search_with_input(self) -> None:
        mock_mem = _make_mock_memory(_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem)
        asyncio.run(hook.on_user_input("dark mode preference"))
        mock_mem.search.assert_called_once_with("dark mode preference")


# ===========================================================================
# on_user_input — empty / edge cases
# ===========================================================================


class TestOnUserInputEmpty:
    def test_empty_string_skips_search(self) -> None:
        mock_mem = _make_mock_memory()
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input(""))
        assert result.memories == []
        assert result.rendered == ""
        mock_mem.search.assert_not_called()

    def test_whitespace_only_skips_search(self) -> None:
        mock_mem = _make_mock_memory()
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("   \t\n  "))
        assert result.memories == []
        assert result.rendered == ""
        mock_mem.search.assert_not_called()

    def test_no_results_returns_empty_injection(self) -> None:
        mock_mem = _make_mock_memory({"results": []})
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("unknown topic"))
        assert result.memories == []
        assert result.rendered == ""
        mock_mem.search.assert_called_once()

    def test_missing_results_key_returns_empty(self) -> None:
        mock_mem = _make_mock_memory({"something_else": True})
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("query"))
        assert result.memories == []
        assert result.rendered == ""

    def test_non_dict_search_response_returns_empty(self) -> None:
        mock_mem = _make_mock_memory()
        mock_mem.search.return_value = "unexpected string"
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("query"))
        assert result.memories == []
        assert result.rendered == ""


# ===========================================================================
# on_user_input — exception handling
# ===========================================================================


class TestOnUserInputExceptions:
    def test_search_exception_returns_empty_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_mem = _make_mock_memory(search_side_effect=RuntimeError("db down"))
        hook = CLIPreInferenceHook(mock_mem)
        with caplog.at_level(logging.WARNING, logger="memx.core.integration.cli_hooks"):
            result = asyncio.run(hook.on_user_input("test query"))
        assert result.memories == []
        assert result.rendered == ""
        assert "Memory.search() failed" in caplog.text

    def test_search_exception_preserves_format(self) -> None:
        config = IntegrationConfig(context_template="markdown")
        mock_mem = _make_mock_memory(search_side_effect=ValueError("bad"))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("test"))
        assert result.format == "markdown"


# ===========================================================================
# XML template formatting
# ===========================================================================


class TestFormatXML:
    def test_xml_default_format(self) -> None:
        mock_mem = _make_mock_memory(_sample_results(2))
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("query"))
        assert result.format == "xml"
        assert "<memx-context>" in result.rendered
        assert "</memx-context>" in result.rendered

    def test_xml_contains_memory_elements(self) -> None:
        mock_mem = _make_mock_memory(_sample_results(2))
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("query"))
        assert 'id="mem_000"' in result.rendered
        assert 'score="0.90"' in result.rendered
        assert 'type="preference"' in result.rendered
        assert "Sample memory content 0." in result.rendered
        assert 'id="mem_001"' in result.rendered
        assert 'score="0.80"' in result.rendered
        assert 'type="tool_pattern"' in result.rendered
        assert "Sample memory content 1." in result.rendered

    def test_xml_single_result(self) -> None:
        mock_mem = _make_mock_memory(_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("query"))
        assert result.rendered.count("<memory ") == 1
        assert result.rendered.count("</memory>") == 1

    def test_xml_wraps_with_memx_context(self) -> None:
        mock_mem = _make_mock_memory(_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("query"))
        lines = result.rendered.strip().split("\n")
        assert lines[0] == "<memx-context>"
        assert lines[-1] == "</memx-context>"

    def test_xml_metadata_missing_type_defaults_to_knowledge(self) -> None:
        data = {"results": [{"id": "x", "memory": "content", "score": 0.5, "metadata": {}}]}
        mock_mem = _make_mock_memory(data)
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("query"))
        assert 'type="knowledge"' in result.rendered

    def test_xml_no_metadata_defaults_to_knowledge(self) -> None:
        data = {"results": [{"id": "x", "memory": "content", "score": 0.5}]}
        mock_mem = _make_mock_memory(data)
        hook = CLIPreInferenceHook(mock_mem)
        result = asyncio.run(hook.on_user_input("query"))
        assert 'type="knowledge"' in result.rendered


# ===========================================================================
# Markdown template formatting
# ===========================================================================


class TestFormatMarkdown:
    def test_markdown_format(self) -> None:
        config = IntegrationConfig(context_template="markdown")
        mock_mem = _make_mock_memory(_sample_results(2))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("query"))
        assert result.format == "markdown"
        assert "## MemX Context" in result.rendered

    def test_markdown_contains_scores_and_content(self) -> None:
        config = IntegrationConfig(context_template="markdown")
        mock_mem = _make_mock_memory(_sample_results(2))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("query"))
        assert "- **[0.90]** Sample memory content 0." in result.rendered
        assert "- **[0.80]** Sample memory content 1." in result.rendered

    def test_markdown_single_result(self) -> None:
        config = IntegrationConfig(context_template="markdown")
        mock_mem = _make_mock_memory(_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("query"))
        lines = result.rendered.strip().split("\n")
        assert lines[0] == "## MemX Context"
        assert lines[1].startswith("- **[")


# ===========================================================================
# Plain text template formatting
# ===========================================================================


class TestFormatPlain:
    def test_plain_format(self) -> None:
        config = IntegrationConfig(context_template="plain")
        mock_mem = _make_mock_memory(_sample_results(2))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("query"))
        assert result.format == "plain"

    def test_plain_contains_memx_prefix(self) -> None:
        config = IntegrationConfig(context_template="plain")
        mock_mem = _make_mock_memory(_sample_results(2))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("query"))
        assert "[MemX] Sample memory content 0." in result.rendered
        assert "[MemX] Sample memory content 1." in result.rendered

    def test_plain_no_header(self) -> None:
        config = IntegrationConfig(context_template="plain")
        mock_mem = _make_mock_memory(_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("query"))
        lines = result.rendered.strip().split("\n")
        assert lines[0].startswith("[MemX]")


# ===========================================================================
# Template fallback
# ===========================================================================


class TestFormatFallback:
    def test_unknown_template_falls_back_to_xml(self) -> None:
        config = IntegrationConfig(context_template="json")
        mock_mem = _make_mock_memory(_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("query"))
        assert "<memx-context>" in result.rendered
        assert result.format == "json"  # format field reflects config, not actual


# ===========================================================================
# Integration with IntegrationManager
# ===========================================================================


class TestWithIntegrationManager:
    def test_manager_can_fire_cli_hook(self) -> None:
        from memx.core.integration.manager import IntegrationManager

        mock_mem = _make_mock_memory(_sample_results(2))
        config = IntegrationConfig()
        hook = CLIPreInferenceHook(mock_mem, config=config)
        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])
        result = asyncio.run(mgr.fire_pre_inference("test query"))
        assert result is not None
        assert isinstance(result, ContextInjection)
        assert len(result.memories) == 2

    def test_manager_auto_recall_disabled_skips_cli_hook(self) -> None:
        from memx.core.integration.manager import IntegrationManager

        config = IntegrationConfig(auto_recall=False)
        mock_mem = _make_mock_memory(_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])
        result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is None
        mock_mem.search.assert_not_called()


# ===========================================================================
# STORY-034: CLIPostActionHook tests
# ===========================================================================


def _make_tool_event(
    tool_name: str = "bash",
    output: str = "file.txt\nREADME.md",
    session_id: str = "sess-001",
) -> ToolEvent:
    """Create a sample ToolEvent for testing."""
    return ToolEvent(
        tool_name=tool_name,
        input={"cmd": "ls"},
        output=output,
        session_id=session_id,
    )


def _make_mock_memory_for_add(
    add_side_effect: Optional[Exception] = None,
    get_all_return: Optional[dict[str, Any]] = None,
) -> MagicMock:
    """Create a mock Memory with configurable add() and get_all() behavior."""
    mock = MagicMock()
    if add_side_effect is not None:
        mock.add.side_effect = add_side_effect
    else:
        mock.add.return_value = {"results": []}
    mock.get_all.return_value = get_all_return or {"memories": []}
    mock.search.return_value = {"results": []}
    return mock


class TestCLIPostActionHookInterface:
    def test_inherits_post_action_hook(self) -> None:
        hook = CLIPostActionHook(_make_mock_memory_for_add())
        assert isinstance(hook, PostActionHook)

    def test_name_property(self) -> None:
        hook = CLIPostActionHook(_make_mock_memory_for_add())
        assert hook.name == "cli_post_action"

    def test_enabled_default_true(self) -> None:
        hook = CLIPostActionHook(_make_mock_memory_for_add())
        assert hook.enabled is True

    def test_enabled_respects_auto_reflect_false(self) -> None:
        config = IntegrationConfig(auto_reflect=False)
        hook = CLIPostActionHook(_make_mock_memory_for_add(), config=config)
        assert hook.enabled is False

    def test_enabled_respects_auto_reflect_true(self) -> None:
        config = IntegrationConfig(auto_reflect=True)
        hook = CLIPostActionHook(_make_mock_memory_for_add(), config=config)
        assert hook.enabled is True


class TestCLIPostActionHookOnToolResult:
    def test_submits_distillation_to_executor(self) -> None:
        mock_mem = _make_mock_memory_for_add()
        hook = CLIPostActionHook(mock_mem)
        event = _make_tool_event()
        asyncio.run(hook.on_tool_result(event))
        # Give the background thread a moment to execute
        hook.shutdown(wait=True)
        mock_mem.add.assert_called_once()
        args, kwargs = mock_mem.add.call_args
        messages = args[0]
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert "bash" in messages[0]["content"]
        assert messages[1]["role"] == "tool"
        assert "file.txt" in messages[1]["content"]
        assert kwargs["user_id"] == "sess-001"

    def test_does_not_block_caller(self) -> None:
        """on_tool_result should return quickly even if add() is slow."""
        mock_mem = _make_mock_memory_for_add()

        def slow_add(*args: Any, **kwargs: Any) -> dict[str, Any]:
            time.sleep(0.5)
            return {"results": []}

        mock_mem.add.side_effect = slow_add
        hook = CLIPostActionHook(mock_mem)
        event = _make_tool_event()

        start = time.monotonic()
        asyncio.run(hook.on_tool_result(event))
        elapsed = time.monotonic() - start

        # on_tool_result should return in <0.2s even though add() takes 0.5s
        assert elapsed < 0.3
        hook.shutdown(wait=True)

    def test_truncates_long_output(self) -> None:
        mock_mem = _make_mock_memory_for_add()
        hook = CLIPostActionHook(mock_mem)
        long_output = "x" * 15000
        event = _make_tool_event(output=long_output)
        asyncio.run(hook.on_tool_result(event))
        hook.shutdown(wait=True)

        args, kwargs = mock_mem.add.call_args
        messages = args[0]
        tool_content = messages[1]["content"]
        assert len(tool_content) < 15000
        assert "[truncated]" in tool_content

    def test_exception_in_format_does_not_propagate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_mem = _make_mock_memory_for_add()
        hook = CLIPostActionHook(mock_mem)
        # Patch _format_tool_event to raise
        with patch.object(hook, "_format_tool_event", side_effect=RuntimeError("fmt error")):
            with caplog.at_level(logging.WARNING, logger="memx.core.integration.cli_hooks"):
                asyncio.run(hook.on_tool_result(_make_tool_event()))
        assert "CLIPostActionHook failed to submit distillation" in caplog.text
        hook.shutdown(wait=True)

    def test_add_exception_logged_not_propagated(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_mem = _make_mock_memory_for_add(add_side_effect=RuntimeError("db error"))
        hook = CLIPostActionHook(mock_mem)
        with caplog.at_level(logging.WARNING, logger="memx.core.integration.cli_hooks"):
            asyncio.run(hook.on_tool_result(_make_tool_event()))
            hook.shutdown(wait=True)
        assert "Background distillation failed" in caplog.text


class TestCLIPostActionHookShutdown:
    def test_shutdown_completes(self) -> None:
        hook = CLIPostActionHook(_make_mock_memory_for_add())
        hook.shutdown(wait=True)
        # Should not raise

    def test_shutdown_no_wait(self) -> None:
        hook = CLIPostActionHook(_make_mock_memory_for_add())
        hook.shutdown(wait=False)
        # Should not raise


class TestCLIPostActionHookWithManager:
    def test_manager_fires_post_action_hook(self) -> None:
        from memx.core.integration.manager import IntegrationManager

        mock_mem = _make_mock_memory_for_add()
        config = IntegrationConfig()
        hook = CLIPostActionHook(mock_mem, config=config)
        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])
        event = _make_tool_event()
        asyncio.run(mgr.fire_post_action(event))
        hook.shutdown(wait=True)
        mock_mem.add.assert_called_once()

    def test_manager_auto_reflect_disabled_skips(self) -> None:
        from memx.core.integration.manager import IntegrationManager

        config = IntegrationConfig(auto_reflect=False)
        mock_mem = _make_mock_memory_for_add()
        hook = CLIPostActionHook(mock_mem, config=config)
        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])
        asyncio.run(mgr.fire_post_action(_make_tool_event()))
        hook.shutdown(wait=True)
        mock_mem.add.assert_not_called()


# ===========================================================================
# STORY-034: CLISessionEndHook tests
# ===========================================================================


def _make_mock_decay_engine(
    sweep_return: Optional[Any] = None,
    sweep_side_effect: Optional[Exception] = None,
) -> MagicMock:
    """Create a mock DecayEngine with configurable sweep() behavior."""
    mock = MagicMock()
    if sweep_side_effect:
        mock.sweep.side_effect = sweep_side_effect
    elif sweep_return:
        mock.sweep.return_value = sweep_return
    else:
        # Return a minimal DecaySweepResult-like object
        result = MagicMock()
        result.updated = 0
        result.archived = 0
        result.permanent = 0
        mock.sweep.return_value = result
    return mock


def _make_mock_memory_with_bullets(
    bullets: Optional[list[dict[str, Any]]] = None,
) -> MagicMock:
    """Create a mock Memory that returns specified bullets from get_all()."""
    mock = MagicMock()
    mock.add.return_value = {"results": []}
    mock.search.return_value = {"results": []}
    mock.get_all.return_value = {"memories": bullets or []}
    return mock


class TestCLISessionEndHookInterface:
    def test_inherits_session_end_hook(self) -> None:
        hook = CLISessionEndHook(
            _make_mock_memory_for_add(), _make_mock_decay_engine()
        )
        assert isinstance(hook, SessionEndHook)

    def test_name_property(self) -> None:
        hook = CLISessionEndHook(
            _make_mock_memory_for_add(), _make_mock_decay_engine()
        )
        assert hook.name == "cli_session_end"

    def test_enabled_default_true(self) -> None:
        hook = CLISessionEndHook(
            _make_mock_memory_for_add(), _make_mock_decay_engine()
        )
        assert hook.enabled is True

    def test_enabled_respects_sweep_on_exit_false(self) -> None:
        config = IntegrationConfig(sweep_on_exit=False)
        hook = CLISessionEndHook(
            _make_mock_memory_for_add(), _make_mock_decay_engine(), config=config
        )
        assert hook.enabled is False


class TestCLISessionEndHookOnSessionEnd:
    def test_calls_decay_sweep(self) -> None:
        bullets = [
            {
                "id": "b1",
                "memory": "test bullet",
                "metadata": {
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "recall_count": 3,
                    "memx_decay_weight": 0.8,
                },
            }
        ]
        mock_mem = _make_mock_memory_with_bullets(bullets)
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess-001"))

        mock_mem.get_all.assert_called_once_with(user_id="sess-001")
        mock_decay.sweep.assert_called_once()
        sweep_args = mock_decay.sweep.call_args[0]
        assert len(sweep_args[0]) == 1
        assert sweep_args[0][0].bullet_id == "b1"
        assert sweep_args[0][0].recall_count == 3

    def test_no_bullets_skips_sweep(self) -> None:
        mock_mem = _make_mock_memory_with_bullets([])
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess-001"))

        mock_mem.get_all.assert_called_once()
        mock_decay.sweep.assert_not_called()

    def test_idempotent_multiple_calls(self) -> None:
        bullets = [
            {
                "id": "b1",
                "memory": "test",
                "metadata": {"created_at": "2026-01-01T00:00:00+00:00"},
            }
        ]
        mock_mem = _make_mock_memory_with_bullets(bullets)
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess-001"))
        asyncio.run(hook.on_session_end("sess-001"))  # second call is noop

        assert mock_decay.sweep.call_count == 1

    def test_get_all_exception_does_not_propagate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_mem = _make_mock_memory_for_add()
        mock_mem.get_all.side_effect = RuntimeError("db down")
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        with caplog.at_level(logging.WARNING, logger="memx.core.integration.cli_hooks"):
            asyncio.run(hook.on_session_end("sess-001"))

        assert "Decay sweep failed" in caplog.text or "CLISessionEndHook failed" in caplog.text

    def test_sweep_exception_does_not_propagate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        bullets = [
            {
                "id": "b1",
                "memory": "test",
                "metadata": {"created_at": "2026-01-01T00:00:00+00:00"},
            }
        ]
        mock_mem = _make_mock_memory_with_bullets(bullets)
        mock_decay = _make_mock_decay_engine(sweep_side_effect=RuntimeError("sweep error"))
        hook = CLISessionEndHook(mock_mem, mock_decay)

        with caplog.at_level(logging.WARNING, logger="memx.core.integration.cli_hooks"):
            asyncio.run(hook.on_session_end("sess-001"))

        assert "sweep failed" in caplog.text.lower() or "failed" in caplog.text.lower()

    def test_handles_missing_metadata_gracefully(self) -> None:
        bullets = [
            {
                "id": "b1",
                "memory": "test bullet",
                "metadata": {},  # no created_at, no recall_count
            }
        ]
        mock_mem = _make_mock_memory_with_bullets(bullets)
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess-001"))

        mock_decay.sweep.assert_called_once()
        sweep_args = mock_decay.sweep.call_args[0]
        assert sweep_args[0][0].recall_count == 0
        assert sweep_args[0][0].current_weight == 1.0

    def test_handles_invalid_created_at(self) -> None:
        bullets = [
            {
                "id": "b1",
                "memory": "test",
                "metadata": {"created_at": "not-a-date"},
            }
        ]
        mock_mem = _make_mock_memory_with_bullets(bullets)
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        # Should not raise; falls back to now()
        asyncio.run(hook.on_session_end("sess-001"))
        mock_decay.sweep.assert_called_once()

    def test_non_dict_memories_skipped(self) -> None:
        mock_mem = _make_mock_memory_for_add()
        mock_mem.get_all.return_value = {"memories": ["not-a-dict", 42, None]}
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess-001"))

        mock_decay.sweep.assert_not_called()  # no valid bullets


class TestCLISessionEndHookWithManager:
    def test_manager_fires_session_end_hook(self) -> None:
        from memx.core.integration.manager import IntegrationManager

        bullets = [
            {
                "id": "b1",
                "memory": "test",
                "metadata": {"created_at": "2026-01-01T00:00:00+00:00"},
            }
        ]
        mock_mem = _make_mock_memory_with_bullets(bullets)
        mock_decay = _make_mock_decay_engine()
        config = IntegrationConfig()
        hook = CLISessionEndHook(mock_mem, mock_decay, config=config)
        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])

        asyncio.run(mgr.fire_session_end("sess-001"))

        mock_decay.sweep.assert_called_once()

    def test_manager_sweep_on_exit_disabled_skips(self) -> None:
        from memx.core.integration.manager import IntegrationManager

        config = IntegrationConfig(sweep_on_exit=False)
        mock_mem = _make_mock_memory_for_add()
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay, config=config)
        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])

        asyncio.run(mgr.fire_session_end("sess-001"))

        mock_decay.sweep.assert_not_called()


# ===========================================================================
# STORY-034: setup_signal_handlers tests
# ===========================================================================


class TestSetupSignalHandlers:
    def test_registers_sigint(self) -> None:
        from memx.core.integration.manager import IntegrationManager

        mgr = IntegrationManager()
        old_handler = signal.getsignal(signal.SIGINT)
        try:
            setup_signal_handlers(mgr, "sess-001")
            current_handler = signal.getsignal(signal.SIGINT)
            assert current_handler is not old_handler
            assert callable(current_handler)
        finally:
            # Restore original handler
            signal.signal(signal.SIGINT, old_handler)

    def test_signal_handler_calls_fire_session_end(self) -> None:
        from memx.core.integration.manager import IntegrationManager

        mgr = IntegrationManager()
        mock_fire = MagicMock()

        async def fake_fire(session_id: str) -> None:
            mock_fire(session_id)

        mgr.fire_session_end = fake_fire  # type: ignore[assignment]

        old_handler = signal.getsignal(signal.SIGINT)
        try:
            setup_signal_handlers(mgr, "sess-test")
            handler = signal.getsignal(signal.SIGINT)
            # Invoke the handler; it should call fire_session_end then SystemExit
            with pytest.raises(SystemExit) as exc_info:
                handler(signal.SIGINT, None)
            assert exc_info.value.code == 0
            mock_fire.assert_called_once_with("sess-test")
        finally:
            signal.signal(signal.SIGINT, old_handler)

    def test_repeated_signal_ignored(self) -> None:
        """Second signal during shutdown should be ignored (no double fire)."""
        from memx.core.integration.manager import IntegrationManager

        mgr = IntegrationManager()
        call_count = 0

        async def counting_fire(session_id: str) -> None:
            nonlocal call_count
            call_count += 1

        mgr.fire_session_end = counting_fire  # type: ignore[assignment]

        old_handler = signal.getsignal(signal.SIGINT)
        try:
            setup_signal_handlers(mgr, "sess-dup")
            handler = signal.getsignal(signal.SIGINT)

            # First call triggers session end
            with pytest.raises(SystemExit):
                handler(signal.SIGINT, None)
            assert call_count == 1

            # Second call is a no-op (handler already set _shutting_down)
            # Need to re-register since the closure is shared
            # The _shutting_down flag is set, so second call should return None
            result = handler(signal.SIGINT, None)
            assert result is None
            assert call_count == 1  # still 1
        finally:
            signal.signal(signal.SIGINT, old_handler)
