"""Integration tests for memx/integration/ — end-to-end and edge case coverage.

Focuses on scenarios NOT already covered by test_integration_manager.py
and test_cli_hooks.py:
- End-to-end pipelines combining all hook types
- Configuration interaction tests (multiple flags disabled)
- Error recovery and cascading failure across hook types
- Public API re-exports from __init__.py
- Edge cases: re-register after unregister, hooks with same name, etc.
- CLIPreInferenceHook._extract_results edge cases
- Signal handler timeout and error scenarios
- CLISessionEndHook with non-dict get_all response
- Full pipeline: PreInference -> PostAction -> SessionEnd with mock Memory
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
from memx.core.integration import (
    BaseHook,
    CLIPostActionHook,
    CLIPreInferenceHook,
    CLISessionEndHook,
    ContextInjection,
    IntegrationManager,
    PostActionHook,
    PreInferenceHook,
    SessionEndHook,
    ToolEvent,
    setup_signal_handlers,
)
from memx.core.integration.cli_hooks import _MAX_OUTPUT_LENGTH, _VALID_FORMATS
from memx.core.integration.hooks import BaseHook as DirectBaseHook
from memx.core.integration.manager import IntegrationManager as DirectManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_memory(
    search_results: Optional[dict[str, Any]] = None,
    search_side_effect: Optional[Exception] = None,
    add_side_effect: Optional[Exception] = None,
    get_all_return: Optional[dict[str, Any]] = None,
) -> MagicMock:
    """Create a mock Memory with configurable behavior for all methods."""
    mock = MagicMock()
    if search_side_effect is not None:
        mock.search.side_effect = search_side_effect
    else:
        mock.search.return_value = search_results or {"results": []}
    if add_side_effect is not None:
        mock.add.side_effect = add_side_effect
    else:
        mock.add.return_value = {"results": []}
    mock.get_all.return_value = get_all_return or {"memories": []}
    return mock


def _sample_results(count: int = 2) -> dict[str, Any]:
    """Generate sample Memory.search() response with *count* results."""
    results = []
    for i in range(count):
        results.append({
            "id": f"mem_{i:03d}",
            "memory": f"Sample memory content {i}.",
            "score": round(0.9 - i * 0.1, 2),
            "metadata": {
                "memx_knowledge_type": "preference" if i == 0 else "tool_pattern",
            },
        })
    return {"results": results}


def _make_mock_decay_engine(
    sweep_side_effect: Optional[Exception] = None,
) -> MagicMock:
    """Create a mock DecayEngine with configurable sweep() behavior."""
    mock = MagicMock()
    if sweep_side_effect:
        mock.sweep.side_effect = sweep_side_effect
    else:
        result = MagicMock()
        result.updated = 0
        result.archived = 0
        result.permanent = 0
        mock.sweep.return_value = result
    return mock


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


# ===========================================================================
# Public API re-exports (__init__.py)
# ===========================================================================


class TestPublicAPIExports:
    """Verify memx.integration.__init__.py re-exports are correct."""

    def test_base_hook_exported(self) -> None:
        assert BaseHook is DirectBaseHook

    def test_integration_manager_exported(self) -> None:
        assert IntegrationManager is DirectManager

    def test_all_hook_classes_importable(self) -> None:
        # These imports should not raise
        assert PreInferenceHook is not None
        assert PostActionHook is not None
        assert SessionEndHook is not None
        assert CLIPreInferenceHook is not None
        assert CLIPostActionHook is not None
        assert CLISessionEndHook is not None

    def test_data_classes_importable(self) -> None:
        assert ContextInjection is not None
        assert ToolEvent is not None

    def test_setup_signal_handlers_importable(self) -> None:
        assert callable(setup_signal_handlers)


# ===========================================================================
# Module-level constants
# ===========================================================================


class TestModuleConstants:
    def test_valid_formats_contains_expected_values(self) -> None:
        assert "xml" in _VALID_FORMATS
        assert "markdown" in _VALID_FORMATS
        assert "plain" in _VALID_FORMATS
        assert len(_VALID_FORMATS) == 3

    def test_max_output_length_is_positive(self) -> None:
        assert _MAX_OUTPUT_LENGTH > 0
        assert _MAX_OUTPUT_LENGTH == 10000


# ===========================================================================
# Full pipeline: register -> fire all events -> verify
# ===========================================================================


class TestFullPipelineEndToEnd:
    """End-to-end test: register CLI hooks, fire all event types in order."""

    def test_full_lifecycle_with_all_hooks(self) -> None:
        """Register all three CLI hook types, fire in order, verify each stage."""
        mock_mem = _make_mock_memory(
            search_results=_sample_results(2),
            get_all_return={
                "memories": [
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
            },
        )
        mock_decay = _make_mock_decay_engine()
        config = IntegrationConfig()

        pre_hook = CLIPreInferenceHook(mock_mem, config=config)
        post_hook = CLIPostActionHook(mock_mem, config=config)
        session_hook = CLISessionEndHook(mock_mem, mock_decay, config=config)

        mgr = IntegrationManager(config=config)
        mgr.register_hooks([pre_hook, post_hook, session_hook])

        # Step 1: Pre-inference recall
        result = asyncio.run(mgr.fire_pre_inference("How to run tests?"))
        assert result is not None
        assert isinstance(result, ContextInjection)
        assert len(result.memories) == 2
        assert "<memx-context>" in result.rendered
        mock_mem.search.assert_called_once_with("How to run tests?")

        # Step 2: Post-action distillation
        event = _make_tool_event(tool_name="bash", output="pytest passed", session_id="sess-e2e")
        asyncio.run(mgr.fire_post_action(event))
        post_hook.shutdown(wait=True)
        mock_mem.add.assert_called_once()
        add_args = mock_mem.add.call_args
        assert add_args[1]["user_id"] == "sess-e2e"

        # Step 3: Session end -> decay sweep
        asyncio.run(mgr.fire_session_end("sess-e2e"))
        mock_decay.sweep.assert_called_once()

    def test_pipeline_all_disabled(self) -> None:
        """When all config flags are disabled, nothing fires."""
        config = IntegrationConfig(
            auto_recall=False,
            auto_reflect=False,
            sweep_on_exit=False,
        )
        mock_mem = _make_mock_memory(search_results=_sample_results(1))
        mock_decay = _make_mock_decay_engine()

        pre_hook = CLIPreInferenceHook(mock_mem, config=config)
        post_hook = CLIPostActionHook(mock_mem, config=config)
        session_hook = CLISessionEndHook(mock_mem, mock_decay, config=config)

        mgr = IntegrationManager(config=config)
        mgr.register_hooks([pre_hook, post_hook, session_hook])

        # Pre-inference: skipped
        result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is None
        mock_mem.search.assert_not_called()

        # Post-action: skipped
        asyncio.run(mgr.fire_post_action(_make_tool_event()))
        post_hook.shutdown(wait=True)
        mock_mem.add.assert_not_called()

        # Session-end: skipped
        asyncio.run(mgr.fire_session_end("sess"))
        mock_decay.sweep.assert_not_called()

    def test_pipeline_partial_config(self) -> None:
        """Only auto_recall enabled; post-action and session-end disabled."""
        config = IntegrationConfig(
            auto_recall=True,
            auto_reflect=False,
            sweep_on_exit=False,
        )
        mock_mem = _make_mock_memory(search_results=_sample_results(1))
        mock_decay = _make_mock_decay_engine()

        pre_hook = CLIPreInferenceHook(mock_mem, config=config)
        post_hook = CLIPostActionHook(mock_mem, config=config)
        session_hook = CLISessionEndHook(mock_mem, mock_decay, config=config)

        mgr = IntegrationManager(config=config)
        mgr.register_hooks([pre_hook, post_hook, session_hook])

        # Pre-inference: fires
        result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is not None
        assert len(result.memories) == 1

        # Post-action: skipped (auto_reflect=False in manager config)
        asyncio.run(mgr.fire_post_action(_make_tool_event()))
        post_hook.shutdown(wait=True)
        mock_mem.add.assert_not_called()

        # Session-end: skipped (sweep_on_exit=False in manager config)
        asyncio.run(mgr.fire_session_end("sess"))
        mock_decay.sweep.assert_not_called()


# ===========================================================================
# Configuration interaction tests
# ===========================================================================


class TestConfigInteractions:
    """Test that manager-level config and hook-level enabled interact correctly."""

    def test_manager_auto_recall_true_but_hook_disabled(self) -> None:
        """Manager config has auto_recall=True but the hook.enabled=False.
        The hook should be skipped (disabled check happens in fire_pre_inference loop)."""
        config = IntegrationConfig(auto_recall=True)
        # Hook with auto_recall=False in its own config
        hook_config = IntegrationConfig(auto_recall=False)
        mock_mem = _make_mock_memory(search_results=_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem, config=hook_config)
        assert hook.enabled is False

        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])

        # Manager allows pre-inference (auto_recall=True), but hook.enabled=False
        result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is None  # hook skipped due to enabled=False
        mock_mem.search.assert_not_called()

    def test_manager_auto_reflect_true_but_hook_disabled(self) -> None:
        """Manager config has auto_reflect=True but hook.enabled=False."""
        config = IntegrationConfig(auto_reflect=True)
        hook_config = IntegrationConfig(auto_reflect=False)
        mock_mem = _make_mock_memory()
        hook = CLIPostActionHook(mock_mem, config=hook_config)
        assert hook.enabled is False

        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])

        asyncio.run(mgr.fire_post_action(_make_tool_event()))
        hook.shutdown(wait=True)
        mock_mem.add.assert_not_called()

    def test_manager_sweep_true_but_hook_disabled(self) -> None:
        """Manager config has sweep_on_exit=True but hook.enabled=False."""
        config = IntegrationConfig(sweep_on_exit=True)
        hook_config = IntegrationConfig(sweep_on_exit=False)
        mock_mem = _make_mock_memory()
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay, config=hook_config)
        assert hook.enabled is False

        mgr = IntegrationManager(config=config)
        mgr.register_hooks([hook])

        asyncio.run(mgr.fire_session_end("sess"))
        mock_decay.sweep.assert_not_called()

    def test_context_template_respected_through_pipeline(self) -> None:
        """context_template in config flows through to rendered output."""
        for template, marker in [
            ("xml", "<memx-context>"),
            ("markdown", "## MemX Context"),
            ("plain", "[MemX]"),
        ]:
            config = IntegrationConfig(context_template=template)
            mock_mem = _make_mock_memory(search_results=_sample_results(1))
            hook = CLIPreInferenceHook(mock_mem, config=config)
            mgr = IntegrationManager(config=config)
            mgr.register_hooks([hook])

            result = asyncio.run(mgr.fire_pre_inference("query"))
            assert result is not None
            assert marker in result.rendered, f"Expected '{marker}' for template '{template}'"
            assert result.format == template


# ===========================================================================
# Error recovery and cascading failure tests
# ===========================================================================


class StubFailThenSucceedPreHook(PreInferenceHook):
    """Pre-inference hook that fails N times then succeeds."""

    def __init__(self, fail_count: int = 1) -> None:
        self._fail_count = fail_count
        self._calls = 0

    @property
    def name(self) -> str:
        return "fail_then_succeed_pre"

    async def on_user_input(self, input: str) -> ContextInjection:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise RuntimeError(f"Attempt {self._calls} failed")
        return ContextInjection(
            memories=[{"content": "recovered"}],
            format="xml",
            rendered="<recovered />",
        )


class StubFailThenSucceedPostHook(PostActionHook):
    """Post-action hook that fails then later calls succeed."""

    def __init__(self) -> None:
        self._fail_next = True
        self.events: list[ToolEvent] = []

    @property
    def name(self) -> str:
        return "fail_then_succeed_post"

    async def on_tool_result(self, event: ToolEvent) -> None:
        if self._fail_next:
            self._fail_next = False
            raise RuntimeError("First call fails")
        self.events.append(event)


class TestErrorRecovery:
    def test_pre_inference_cascade_to_fallback(self) -> None:
        """Two failing pre-hooks -> fallback success from third hook."""

        class AlwaysFail(PreInferenceHook):
            @property
            def name(self) -> str:
                return "always_fail"

            async def on_user_input(self, input: str) -> ContextInjection:
                raise RuntimeError("I always fail")

        class SuccessHook(PreInferenceHook):
            @property
            def name(self) -> str:
                return "success"

            async def on_user_input(self, input: str) -> ContextInjection:
                return ContextInjection(
                    memories=[{"content": "fallback"}], rendered="ok"
                )

        mgr = IntegrationManager()
        mgr.register_hooks([AlwaysFail(), AlwaysFail(), SuccessHook()])
        result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is not None
        assert result.rendered == "ok"

    def test_post_action_failure_does_not_block_others(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One failing post-action hook does not block the next one."""

        class FailPost(PostActionHook):
            @property
            def name(self) -> str:
                return "fail_post"

            async def on_tool_result(self, event: ToolEvent) -> None:
                raise RuntimeError("post fail")

        class OKPost(PostActionHook):
            def __init__(self) -> None:
                self.fired = False

            @property
            def name(self) -> str:
                return "ok_post"

            async def on_tool_result(self, event: ToolEvent) -> None:
                self.fired = True

        mgr = IntegrationManager()
        ok_hook = OKPost()
        mgr.register_hooks([FailPost(), ok_hook])

        with caplog.at_level(logging.WARNING, logger="memx.core.integration.manager"):
            asyncio.run(mgr.fire_post_action(_make_tool_event()))

        assert ok_hook.fired is True
        assert "PostActionHook 'fail_post' failed" in caplog.text

    def test_session_end_failure_does_not_block_others(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """One failing session-end hook does not block the next one."""

        class FailSession(SessionEndHook):
            @property
            def name(self) -> str:
                return "fail_session"

            async def on_session_end(self, session_id: str) -> None:
                raise RuntimeError("session fail")

        class OKSession(SessionEndHook):
            def __init__(self) -> None:
                self.sessions: list[str] = []

            @property
            def name(self) -> str:
                return "ok_session"

            async def on_session_end(self, session_id: str) -> None:
                self.sessions.append(session_id)

        mgr = IntegrationManager()
        ok_hook = OKSession()
        mgr.register_hooks([FailSession(), ok_hook])

        with caplog.at_level(logging.WARNING, logger="memx.core.integration.manager"):
            asyncio.run(mgr.fire_session_end("s1"))

        assert ok_hook.sessions == ["s1"]
        assert "SessionEndHook 'fail_session' failed" in caplog.text

    def test_mixed_failures_across_event_types(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Pre-inference failure + post-action success + session-end failure."""

        class FailPre(PreInferenceHook):
            @property
            def name(self) -> str:
                return "fail_pre"

            async def on_user_input(self, input: str) -> ContextInjection:
                raise RuntimeError("pre fail")

        class OKPost(PostActionHook):
            def __init__(self) -> None:
                self.fired = False

            @property
            def name(self) -> str:
                return "ok_post"

            async def on_tool_result(self, event: ToolEvent) -> None:
                self.fired = True

        class FailSession(SessionEndHook):
            @property
            def name(self) -> str:
                return "fail_session"

            async def on_session_end(self, session_id: str) -> None:
                raise RuntimeError("session fail")

        mgr = IntegrationManager()
        ok_post = OKPost()
        mgr.register_hooks([FailPre(), ok_post, FailSession()])

        with caplog.at_level(logging.WARNING, logger="memx.core.integration.manager"):
            # Pre-inference fails -> returns None
            result = asyncio.run(mgr.fire_pre_inference("test"))
            assert result is None

            # Post-action succeeds
            asyncio.run(mgr.fire_post_action(_make_tool_event()))
            assert ok_post.fired is True

            # Session-end fails -> logged, no propagation
            asyncio.run(mgr.fire_session_end("s1"))

        assert "PreInferenceHook 'fail_pre' failed" in caplog.text
        assert "SessionEndHook 'fail_session' failed" in caplog.text


# ===========================================================================
# Registration edge cases
# ===========================================================================


class TestRegistrationEdgeCases:
    def test_register_after_unregister(self) -> None:
        """After unregister_all, new hooks can be registered and fired."""

        class SimplePost(PostActionHook):
            def __init__(self) -> None:
                self.fired = False

            @property
            def name(self) -> str:
                return "simple_post"

            async def on_tool_result(self, event: ToolEvent) -> None:
                self.fired = True

        mgr = IntegrationManager()
        h1 = SimplePost()
        mgr.register_hooks([h1])
        mgr.unregister_all()
        assert mgr.get_hooks(BaseHook) == []

        h2 = SimplePost()
        mgr.register_hooks([h2])
        assert mgr.get_hooks(PostActionHook) == [h2]

        asyncio.run(mgr.fire_post_action(_make_tool_event()))
        assert h2.fired is True
        assert h1.fired is False  # h1 was unregistered

    def test_hooks_with_same_name_different_instances(self) -> None:
        """Two hooks with the same name() but different instances are both registered."""

        class NamedPre(PreInferenceHook):
            def __init__(self, tag: str) -> None:
                self.tag = tag
                self.call_count = 0

            @property
            def name(self) -> str:
                return "shared_name"

            async def on_user_input(self, input: str) -> ContextInjection:
                self.call_count += 1
                return ContextInjection(
                    memories=[{"tag": self.tag}], rendered=self.tag
                )

        mgr = IntegrationManager()
        h1 = NamedPre("A")
        h2 = NamedPre("B")
        mgr.register_hooks([h1, h2])

        # Both registered (different instances, despite same name)
        hooks = mgr.get_hooks(PreInferenceHook)
        assert len(hooks) == 2

        # fire_pre_inference returns first successful (h1)
        result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is not None
        assert result.rendered == "A"
        assert h1.call_count == 1
        assert h2.call_count == 0

    def test_register_hooks_called_incrementally(self) -> None:
        """Multiple register_hooks() calls add hooks incrementally."""

        class SimpleSession(SessionEndHook):
            def __init__(self, tag: str) -> None:
                self.tag = tag
                self.sessions: list[str] = []

            @property
            def name(self) -> str:
                return f"session_{self.tag}"

            async def on_session_end(self, session_id: str) -> None:
                self.sessions.append(session_id)

        mgr = IntegrationManager()
        h1 = SimpleSession("A")
        h2 = SimpleSession("B")
        mgr.register_hooks([h1])
        mgr.register_hooks([h2])

        assert len(mgr.get_hooks(SessionEndHook)) == 2

        asyncio.run(mgr.fire_session_end("s1"))
        assert h1.sessions == ["s1"]
        assert h2.sessions == ["s1"]


# ===========================================================================
# CLIPreInferenceHook._extract_results edge cases
# ===========================================================================


class TestExtractResultsEdgeCases:
    """Test the static _extract_results method with various edge-case inputs."""

    def test_results_key_not_a_list(self) -> None:
        assert CLIPreInferenceHook._extract_results({"results": "not-a-list"}) == []

    def test_results_key_is_none(self) -> None:
        assert CLIPreInferenceHook._extract_results({"results": None}) == []

    def test_non_dict_items_in_results_list_filtered(self) -> None:
        raw = {"results": [{"id": "1", "memory": "ok"}, "not-a-dict", 42, None]}
        result = CLIPreInferenceHook._extract_results(raw)
        assert len(result) == 1
        assert result[0]["id"] == "1"

    def test_empty_dict_in_results_list_kept(self) -> None:
        raw = {"results": [{}]}
        result = CLIPreInferenceHook._extract_results(raw)
        assert len(result) == 1
        assert result[0] == {}

    def test_input_is_none(self) -> None:
        assert CLIPreInferenceHook._extract_results(None) == []

    def test_input_is_list(self) -> None:
        assert CLIPreInferenceHook._extract_results([1, 2, 3]) == []

    def test_input_is_integer(self) -> None:
        assert CLIPreInferenceHook._extract_results(42) == []


# ===========================================================================
# CLIPreInferenceHook._format edge cases
# ===========================================================================


class TestFormatEdgeCases:
    def test_format_with_empty_results_list(self) -> None:
        """All templates should produce something even with empty results."""
        for template in ("xml", "markdown", "plain"):
            result = CLIPreInferenceHook._format([], template)
            assert isinstance(result, str)

    def test_xml_with_empty_list(self) -> None:
        result = CLIPreInferenceHook._format([], "xml")
        assert "<memx-context>" in result
        assert "</memx-context>" in result
        assert "<memory" not in result

    def test_markdown_with_empty_list(self) -> None:
        result = CLIPreInferenceHook._format([], "markdown")
        assert "## MemX Context" in result
        assert "- **[" not in result

    def test_plain_with_empty_list(self) -> None:
        result = CLIPreInferenceHook._format([], "plain")
        assert result == ""

    def test_xml_with_missing_fields(self) -> None:
        """Result dict missing id, score, memory should use defaults."""
        results = [{"unrelated_key": "value"}]
        rendered = CLIPreInferenceHook._format(results, "xml")
        assert 'id="unknown"' in rendered
        assert 'score="0.00"' in rendered

    def test_xml_with_non_dict_metadata(self) -> None:
        """When metadata is not a dict, fall back to 'knowledge' type."""
        results = [{"id": "x", "memory": "test", "score": 0.5, "metadata": "not-dict"}]
        rendered = CLIPreInferenceHook._format(results, "xml")
        assert 'type="knowledge"' in rendered

    def test_markdown_with_missing_fields(self) -> None:
        results = [{}]
        rendered = CLIPreInferenceHook._format(results, "markdown")
        assert "- **[0.00]** " in rendered

    def test_plain_with_missing_memory_field(self) -> None:
        results = [{}]
        rendered = CLIPreInferenceHook._format(results, "plain")
        assert "[MemX] " in rendered


# ===========================================================================
# CLIPostActionHook — _format_tool_event edge cases
# ===========================================================================


class TestPostActionFormatEdgeCases:
    def test_format_tool_event_exact_boundary(self) -> None:
        """Output exactly at _MAX_OUTPUT_LENGTH should NOT be truncated."""
        mock_mem = _make_mock_memory()
        hook = CLIPostActionHook(mock_mem)
        exact_output = "x" * _MAX_OUTPUT_LENGTH
        event = ToolEvent(
            tool_name="test", input={}, output=exact_output, session_id="s"
        )
        messages = hook._format_tool_event(event)
        assert "[truncated]" not in messages[1]["content"]
        assert len(messages[1]["content"]) == _MAX_OUTPUT_LENGTH

    def test_format_tool_event_one_over_boundary(self) -> None:
        """Output one character over _MAX_OUTPUT_LENGTH should be truncated."""
        mock_mem = _make_mock_memory()
        hook = CLIPostActionHook(mock_mem)
        over_output = "x" * (_MAX_OUTPUT_LENGTH + 1)
        event = ToolEvent(
            tool_name="test", input={}, output=over_output, session_id="s"
        )
        messages = hook._format_tool_event(event)
        assert "[truncated]" in messages[1]["content"]

    def test_format_tool_event_empty_output(self) -> None:
        mock_mem = _make_mock_memory()
        hook = CLIPostActionHook(mock_mem)
        event = ToolEvent(tool_name="test", input={}, output="", session_id="s")
        messages = hook._format_tool_event(event)
        assert messages[1]["content"] == ""
        assert messages[0]["content"] == "Used tool: test"

    def test_format_tool_event_message_structure(self) -> None:
        mock_mem = _make_mock_memory()
        hook = CLIPostActionHook(mock_mem)
        event = ToolEvent(
            tool_name="Read", input={"path": "/a"}, output="content", session_id="s1"
        )
        messages = hook._format_tool_event(event)
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert "Read" in messages[0]["content"]
        assert messages[1]["role"] == "tool"
        assert messages[1]["content"] == "content"


# ===========================================================================
# CLIPostActionHook — multiple concurrent events
# ===========================================================================


class TestPostActionConcurrency:
    def test_multiple_tool_events_sequential(self) -> None:
        """Fire multiple tool events sequentially; all should be submitted."""
        mock_mem = _make_mock_memory()
        hook = CLIPostActionHook(mock_mem)

        for i in range(5):
            event = _make_tool_event(
                tool_name=f"tool_{i}",
                output=f"output_{i}",
                session_id="sess-multi",
            )
            asyncio.run(hook.on_tool_result(event))

        hook.shutdown(wait=True)
        assert mock_mem.add.call_count == 5


# ===========================================================================
# CLISessionEndHook — _run_sweep edge cases
# ===========================================================================


class TestSessionEndSweepEdgeCases:
    def test_get_all_returns_non_dict(self) -> None:
        """If Memory.get_all() returns a non-dict, no sweep runs."""
        mock_mem = _make_mock_memory()
        mock_mem.get_all.return_value = "unexpected"
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess"))
        mock_decay.sweep.assert_not_called()

    def test_get_all_returns_dict_without_memories_key(self) -> None:
        """If dict has no 'memories' key, no sweep runs."""
        mock_mem = _make_mock_memory()
        mock_mem.get_all.return_value = {"other_key": []}
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess"))
        mock_decay.sweep.assert_not_called()

    def test_multiple_bullets_with_varied_metadata(self) -> None:
        """Sweep with multiple bullets having different metadata completeness."""
        bullets = [
            {
                "id": "b1",
                "memory": "full metadata",
                "metadata": {
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "recall_count": 5,
                    "memx_decay_weight": 0.9,
                },
            },
            {
                "id": "b2",
                "memory": "minimal metadata",
                "metadata": {},
            },
            {
                "id": "b3",
                "memory": "no metadata key",
            },
        ]
        mock_mem = _make_mock_memory(get_all_return={"memories": bullets})
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess"))
        mock_decay.sweep.assert_called_once()
        sweep_args = mock_decay.sweep.call_args[0]
        assert len(sweep_args[0]) == 3

        # Verify b1 has full metadata
        b1_info = sweep_args[0][0]
        assert b1_info.bullet_id == "b1"
        assert b1_info.recall_count == 5
        assert b1_info.current_weight == 0.9

        # Verify b2 has default recall_count and weight
        b2_info = sweep_args[0][1]
        assert b2_info.bullet_id == "b2"
        assert b2_info.recall_count == 0
        assert b2_info.current_weight == 1.0

        # Verify b3 has all defaults (no metadata key)
        b3_info = sweep_args[0][2]
        assert b3_info.bullet_id == "b3"
        assert b3_info.recall_count == 0
        assert b3_info.current_weight == 1.0

    def test_created_at_not_string_uses_default(self) -> None:
        """If created_at is not a string, fall back to datetime.now()."""
        bullets = [
            {
                "id": "b1",
                "memory": "test",
                "metadata": {"created_at": 12345},  # integer, not string
            }
        ]
        mock_mem = _make_mock_memory(get_all_return={"memories": bullets})
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        asyncio.run(hook.on_session_end("sess"))
        mock_decay.sweep.assert_called_once()

    def test_sweep_internal_error_still_marks_completed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_run_sweep catches exceptions internally, so on_session_end
        considers the session end completed even when sweep() fails."""
        bullets = [
            {
                "id": "b1",
                "memory": "test",
                "metadata": {"created_at": "2026-01-01T00:00:00+00:00"},
            }
        ]
        mock_mem = _make_mock_memory(get_all_return={"memories": bullets})
        mock_decay = _make_mock_decay_engine(sweep_side_effect=RuntimeError("sweep fail"))
        hook = CLISessionEndHook(mock_mem, mock_decay)

        with caplog.at_level(logging.WARNING, logger="memx.core.integration.cli_hooks"):
            asyncio.run(hook.on_session_end("sess"))

        # _run_sweep catches exception internally; on_session_end marks completed
        assert hook._completed is True
        assert "Decay sweep failed" in caplog.text

        # Subsequent call is idempotent (no-op)
        mock_decay.sweep.reset_mock()
        asyncio.run(hook.on_session_end("sess"))
        mock_decay.sweep.assert_not_called()

    def test_on_session_end_exception_outside_run_sweep_keeps_incomplete(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """If _run_sweep itself raises (bypassing its internal catch),
        on_session_end catches it and _completed stays False."""
        mock_mem = _make_mock_memory(get_all_return={"memories": []})
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)

        # Patch _run_sweep to raise directly (bypassing internal try/except)
        with patch.object(hook, "_run_sweep", side_effect=RuntimeError("outer fail")):
            with caplog.at_level(logging.WARNING, logger="memx.core.integration.cli_hooks"):
                asyncio.run(hook.on_session_end("sess"))

        assert hook._completed is False
        assert "CLISessionEndHook failed" in caplog.text


# ===========================================================================
# Signal handler edge cases
# ===========================================================================


class TestSignalHandlerEdgeCases:
    def test_handler_timeout_logs_warning(self) -> None:
        """When fire_session_end times out, warning is logged."""
        mgr = IntegrationManager()

        async def slow_fire(session_id: str) -> None:
            await asyncio.sleep(10)  # longer than 5s timeout

        mgr.fire_session_end = slow_fire  # type: ignore[assignment]

        old_handler = signal.getsignal(signal.SIGINT)
        try:
            setup_signal_handlers(mgr, "sess-timeout")
            handler = signal.getsignal(signal.SIGINT)
            # The handler uses wait_for with 5s timeout internally,
            # but we cannot actually wait 5s in a test.
            # We patch asyncio.wait_for to raise TimeoutError immediately
            with patch("memx.core.integration.cli_hooks.asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                with pytest.raises(SystemExit) as exc_info:
                    handler(signal.SIGINT, None)
                assert exc_info.value.code == 0
        finally:
            signal.signal(signal.SIGINT, old_handler)

    def test_handler_exception_logs_warning(self) -> None:
        """When fire_session_end raises, warning is logged and SystemExit(0)."""
        mgr = IntegrationManager()

        async def broken_fire(session_id: str) -> None:
            raise RuntimeError("fire broken")

        mgr.fire_session_end = broken_fire  # type: ignore[assignment]

        old_handler = signal.getsignal(signal.SIGINT)
        try:
            setup_signal_handlers(mgr, "sess-error")
            handler = signal.getsignal(signal.SIGINT)
            with pytest.raises(SystemExit) as exc_info:
                handler(signal.SIGINT, None)
            assert exc_info.value.code == 0
        finally:
            signal.signal(signal.SIGINT, old_handler)


# ===========================================================================
# IntegrationManager default config
# ===========================================================================


class TestManagerDefaultConfig:
    def test_manager_none_config_uses_defaults(self) -> None:
        mgr = IntegrationManager(config=None)
        assert mgr.config.auto_recall is True
        assert mgr.config.auto_reflect is True
        assert mgr.config.sweep_on_exit is True
        assert mgr.config.context_template == "xml"

    def test_manager_config_is_immutable_reference(self) -> None:
        config = IntegrationConfig(auto_recall=False)
        mgr = IntegrationManager(config=config)
        assert mgr.config is config


# ===========================================================================
# CLIPreInferenceHook with IntegrationConfig defaults
# ===========================================================================


class TestCLIPreInferenceConfigDefaults:
    def test_default_config_when_none_passed(self) -> None:
        mock_mem = _make_mock_memory(search_results=_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem, config=None)
        assert hook.enabled is True
        result = asyncio.run(hook.on_user_input("query"))
        assert result is not None
        assert result.format == "xml"

    def test_hook_config_preserved(self) -> None:
        config = IntegrationConfig(context_template="plain", auto_recall=True)
        mock_mem = _make_mock_memory(search_results=_sample_results(1))
        hook = CLIPreInferenceHook(mock_mem, config=config)
        result = asyncio.run(hook.on_user_input("query"))
        assert result.format == "plain"
        assert "[MemX]" in result.rendered


# ===========================================================================
# CLIPostActionHook with default config
# ===========================================================================


class TestCLIPostActionConfigDefaults:
    def test_default_config_when_none_passed(self) -> None:
        mock_mem = _make_mock_memory()
        hook = CLIPostActionHook(mock_mem, config=None)
        assert hook.enabled is True
        assert hook.name == "cli_post_action"


# ===========================================================================
# CLISessionEndHook with default config
# ===========================================================================


class TestCLISessionEndConfigDefaults:
    def test_default_config_when_none_passed(self) -> None:
        mock_mem = _make_mock_memory()
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay, config=None)
        assert hook.enabled is True
        assert hook.name == "cli_session_end"


# ===========================================================================
# Data class detailed tests
# ===========================================================================


class TestContextInjectionDetails:
    def test_equality(self) -> None:
        a = ContextInjection(memories=[{"x": 1}], format="xml", rendered="r")
        b = ContextInjection(memories=[{"x": 1}], format="xml", rendered="r")
        assert a == b

    def test_inequality(self) -> None:
        a = ContextInjection(memories=[], format="xml", rendered="a")
        b = ContextInjection(memories=[], format="xml", rendered="b")
        assert a != b


class TestToolEventDetails:
    def test_equality(self) -> None:
        a = ToolEvent(tool_name="t", input={"k": "v"}, output="o", session_id="s")
        b = ToolEvent(tool_name="t", input={"k": "v"}, output="o", session_id="s")
        assert a == b

    def test_inequality(self) -> None:
        a = ToolEvent(tool_name="t1")
        b = ToolEvent(tool_name="t2")
        assert a != b


# ===========================================================================
# Logging verification tests
# ===========================================================================


class TestLoggingBehavior:
    def test_register_hooks_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        mgr = IntegrationManager()

        class SimpleHook(PreInferenceHook):
            @property
            def name(self) -> str:
                return "log_test_hook"

            async def on_user_input(self, input: str) -> ContextInjection:
                return ContextInjection()

        with caplog.at_level(logging.INFO, logger="memx.core.integration.manager"):
            mgr.register_hooks([SimpleHook()])
        assert "Registered hook: log_test_hook" in caplog.text

    def test_unregister_all_logs_count(self, caplog: pytest.LogCaptureFixture) -> None:
        mgr = IntegrationManager()

        class SimpleHook(PreInferenceHook):
            @property
            def name(self) -> str:
                return "log_test"

            async def on_user_input(self, input: str) -> ContextInjection:
                return ContextInjection()

        mgr.register_hooks([SimpleHook(), SimpleHook()])
        with caplog.at_level(logging.INFO, logger="memx.core.integration.manager"):
            mgr.unregister_all()
        assert "Unregistered all hooks (2 removed)" in caplog.text

    def test_duplicate_registration_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mgr = IntegrationManager()

        class SimpleHook(PreInferenceHook):
            @property
            def name(self) -> str:
                return "dup_test"

            async def on_user_input(self, input: str) -> ContextInjection:
                return ContextInjection()

        hook = SimpleHook()
        mgr.register_hooks([hook])
        with caplog.at_level(logging.DEBUG, logger="memx.core.integration.manager"):
            mgr.register_hooks([hook])
        assert "already registered" in caplog.text

    def test_disabled_hook_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        class DisabledPre(PreInferenceHook):
            @property
            def name(self) -> str:
                return "disabled_pre"

            @property
            def enabled(self) -> bool:
                return False

            async def on_user_input(self, input: str) -> ContextInjection:
                return ContextInjection()

        mgr = IntegrationManager()
        mgr.register_hooks([DisabledPre()])
        with caplog.at_level(logging.DEBUG, logger="memx.core.integration.manager"):
            result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is None
        assert "disabled_pre disabled" in caplog.text

    def test_auto_recall_disabled_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = IntegrationConfig(auto_recall=False)
        mgr = IntegrationManager(config=config)
        with caplog.at_level(logging.DEBUG, logger="memx.core.integration.manager"):
            result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is None
        assert "auto_recall disabled" in caplog.text

    def test_auto_reflect_disabled_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = IntegrationConfig(auto_reflect=False)
        mgr = IntegrationManager(config=config)
        with caplog.at_level(logging.DEBUG, logger="memx.core.integration.manager"):
            asyncio.run(mgr.fire_post_action(_make_tool_event()))
        assert "auto_reflect disabled" in caplog.text

    def test_sweep_on_exit_disabled_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = IntegrationConfig(sweep_on_exit=False)
        mgr = IntegrationManager(config=config)
        with caplog.at_level(logging.DEBUG, logger="memx.core.integration.manager"):
            asyncio.run(mgr.fire_session_end("s"))
        assert "sweep_on_exit disabled" in caplog.text

    def test_session_end_completed_logs_info(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_mem = _make_mock_memory(get_all_return={"memories": []})
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)
        with caplog.at_level(logging.INFO, logger="memx.core.integration.cli_hooks"):
            asyncio.run(hook.on_session_end("sess-log"))
        assert "Session end completed" in caplog.text

    def test_session_end_already_completed_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        mock_mem = _make_mock_memory(get_all_return={"memories": []})
        mock_decay = _make_mock_decay_engine()
        hook = CLISessionEndHook(mock_mem, mock_decay)
        asyncio.run(hook.on_session_end("sess"))
        with caplog.at_level(logging.DEBUG, logger="memx.core.integration.cli_hooks"):
            asyncio.run(hook.on_session_end("sess"))
        assert "Session end already completed" in caplog.text
