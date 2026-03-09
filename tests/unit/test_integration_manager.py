"""Unit tests for memx.integration — hooks, manager, data classes."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import pytest

from memx.core.config import IntegrationConfig
from memx.core.integration.hooks import (
    BaseHook,
    ContextInjection,
    PostActionHook,
    PreInferenceHook,
    SessionEndHook,
    ToolEvent,
)
from memx.core.integration.manager import IntegrationManager


# ---------------------------------------------------------------------------
# Concrete hook stubs for testing
# ---------------------------------------------------------------------------


class StubPreInferenceHook(PreInferenceHook):
    """Test stub that returns a fixed ContextInjection."""

    def __init__(self, hook_name: str = "stub_pre", *, enabled: bool = True) -> None:
        self._name = hook_name
        self._enabled = enabled
        self.call_count = 0
        self.last_input: Optional[str] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def on_user_input(self, input: str) -> ContextInjection:
        self.call_count += 1
        self.last_input = input
        return ContextInjection(
            memories=[{"content": "test memory"}],
            format="xml",
            rendered="<memory>test memory</memory>",
        )


class FailingPreInferenceHook(PreInferenceHook):
    """Pre-inference hook that always raises."""

    @property
    def name(self) -> str:
        return "failing_pre"

    async def on_user_input(self, input: str) -> ContextInjection:
        raise RuntimeError("pre-inference exploded")


class StubPostActionHook(PostActionHook):
    """Test stub that records tool events."""

    def __init__(self, hook_name: str = "stub_post", *, enabled: bool = True) -> None:
        self._name = hook_name
        self._enabled = enabled
        self.events: list[ToolEvent] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def on_tool_result(self, event: ToolEvent) -> None:
        self.events.append(event)


class FailingPostActionHook(PostActionHook):
    """Post-action hook that always raises."""

    @property
    def name(self) -> str:
        return "failing_post"

    async def on_tool_result(self, event: ToolEvent) -> None:
        raise RuntimeError("post-action exploded")


class StubSessionEndHook(SessionEndHook):
    """Test stub that records session end calls."""

    def __init__(self, hook_name: str = "stub_session", *, enabled: bool = True) -> None:
        self._name = hook_name
        self._enabled = enabled
        self.sessions: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def on_session_end(self, session_id: str) -> None:
        self.sessions.append(session_id)


class FailingSessionEndHook(SessionEndHook):
    """Session-end hook that always raises."""

    @property
    def name(self) -> str:
        return "failing_session"

    async def on_session_end(self, session_id: str) -> None:
        raise RuntimeError("session-end exploded")


# ===========================================================================
# Data class tests
# ===========================================================================


class TestContextInjection:
    def test_frozen_dataclass(self) -> None:
        ci = ContextInjection(
            memories=[{"content": "x"}],
            format="markdown",
            rendered="# x",
        )
        assert ci.memories == [{"content": "x"}]
        assert ci.format == "markdown"
        assert ci.rendered == "# x"
        with pytest.raises(AttributeError):
            ci.format = "plain"  # type: ignore[misc]

    def test_defaults(self) -> None:
        ci = ContextInjection()
        assert ci.memories == []
        assert ci.format == "xml"
        assert ci.rendered == ""


class TestToolEvent:
    def test_frozen_dataclass(self) -> None:
        te = ToolEvent(
            tool_name="search",
            input={"q": "test"},
            output="result",
            session_id="s1",
        )
        assert te.tool_name == "search"
        assert te.input == {"q": "test"}
        assert te.output == "result"
        assert te.session_id == "s1"
        with pytest.raises(AttributeError):
            te.tool_name = "other"  # type: ignore[misc]

    def test_defaults(self) -> None:
        te = ToolEvent()
        assert te.tool_name == ""
        assert te.input == {}
        assert te.output == ""
        assert te.session_id == ""


# ===========================================================================
# BaseHook ABC tests
# ===========================================================================


class TestBaseHook:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseHook()  # type: ignore[abstract]

    def test_enabled_default_true(self) -> None:
        hook = StubPreInferenceHook()
        assert hook.enabled is True

    def test_enabled_override(self) -> None:
        hook = StubPreInferenceHook(enabled=False)
        assert hook.enabled is False


# ===========================================================================
# IntegrationManager — Registration
# ===========================================================================


class TestRegistration:
    def test_register_single_hook(self) -> None:
        mgr = IntegrationManager()
        hook = StubPreInferenceHook()
        mgr.register_hooks([hook])
        assert mgr.get_hooks(PreInferenceHook) == [hook]

    def test_register_multiple_hooks(self) -> None:
        mgr = IntegrationManager()
        pre = StubPreInferenceHook()
        post = StubPostActionHook()
        session = StubSessionEndHook()
        mgr.register_hooks([pre, post, session])
        assert mgr.get_hooks(PreInferenceHook) == [pre]
        assert mgr.get_hooks(PostActionHook) == [post]
        assert mgr.get_hooks(SessionEndHook) == [session]

    def test_register_empty_list(self) -> None:
        mgr = IntegrationManager()
        mgr.register_hooks([])
        assert mgr.get_hooks(BaseHook) == []

    def test_duplicate_registration_skipped(self) -> None:
        mgr = IntegrationManager()
        hook = StubPreInferenceHook()
        mgr.register_hooks([hook])
        mgr.register_hooks([hook])  # same instance
        assert len(mgr.get_hooks(PreInferenceHook)) == 1

    def test_get_hooks_returns_registration_order(self) -> None:
        mgr = IntegrationManager()
        h1 = StubPostActionHook("post_a")
        h2 = StubPostActionHook("post_b")
        mgr.register_hooks([h1, h2])
        assert mgr.get_hooks(PostActionHook) == [h1, h2]

    def test_get_hooks_filters_by_type(self) -> None:
        mgr = IntegrationManager()
        pre = StubPreInferenceHook()
        post = StubPostActionHook()
        mgr.register_hooks([pre, post])
        assert mgr.get_hooks(PreInferenceHook) == [pre]
        assert mgr.get_hooks(PostActionHook) == [post]
        assert mgr.get_hooks(SessionEndHook) == []

    def test_get_hooks_base_type_returns_all(self) -> None:
        mgr = IntegrationManager()
        pre = StubPreInferenceHook()
        post = StubPostActionHook()
        session = StubSessionEndHook()
        mgr.register_hooks([pre, post, session])
        all_hooks = mgr.get_hooks(BaseHook)
        assert len(all_hooks) == 3

    def test_unregister_all(self) -> None:
        mgr = IntegrationManager()
        mgr.register_hooks([StubPreInferenceHook(), StubPostActionHook()])
        mgr.unregister_all()
        assert mgr.get_hooks(BaseHook) == []

    def test_unregister_all_empty(self) -> None:
        mgr = IntegrationManager()
        mgr.unregister_all()  # should not raise
        assert mgr.get_hooks(BaseHook) == []


# ===========================================================================
# IntegrationManager — fire_pre_inference
# ===========================================================================


class TestFirePreInference:
    def test_returns_context_injection(self) -> None:
        mgr = IntegrationManager()
        hook = StubPreInferenceHook()
        mgr.register_hooks([hook])
        result = asyncio.run(mgr.fire_pre_inference("hello"))
        assert result is not None
        assert isinstance(result, ContextInjection)
        assert result.memories == [{"content": "test memory"}]
        assert hook.call_count == 1
        assert hook.last_input == "hello"

    def test_returns_first_successful_result(self) -> None:
        """When multiple PreInferenceHook are registered, return the first success."""
        mgr = IntegrationManager()
        h1 = StubPreInferenceHook("pre_a")
        h2 = StubPreInferenceHook("pre_b")
        mgr.register_hooks([h1, h2])
        result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is not None
        assert h1.call_count == 1
        assert h2.call_count == 0  # second hook not called

    def test_skips_disabled_hook(self) -> None:
        mgr = IntegrationManager()
        disabled = StubPreInferenceHook("disabled_pre", enabled=False)
        active = StubPreInferenceHook("active_pre")
        mgr.register_hooks([disabled, active])
        result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is not None
        assert disabled.call_count == 0
        assert active.call_count == 1

    def test_auto_recall_disabled_returns_none(self) -> None:
        config = IntegrationConfig(auto_recall=False)
        mgr = IntegrationManager(config=config)
        hook = StubPreInferenceHook()
        mgr.register_hooks([hook])
        result = asyncio.run(mgr.fire_pre_inference("hello"))
        assert result is None
        assert hook.call_count == 0

    def test_no_hooks_returns_none(self) -> None:
        mgr = IntegrationManager()
        result = asyncio.run(mgr.fire_pre_inference("hello"))
        assert result is None

    def test_failing_hook_isolated(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failing PreInferenceHook logs WARNING and tries next hook."""
        mgr = IntegrationManager()
        failing = FailingPreInferenceHook()
        fallback = StubPreInferenceHook("fallback_pre")
        mgr.register_hooks([failing, fallback])
        with caplog.at_level(logging.WARNING, logger="memx.core.integration.manager"):
            result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is not None
        assert fallback.call_count == 1
        assert "PreInferenceHook 'failing_pre' failed" in caplog.text

    def test_all_hooks_fail_returns_none(self, caplog: pytest.LogCaptureFixture) -> None:
        mgr = IntegrationManager()
        mgr.register_hooks([FailingPreInferenceHook()])
        with caplog.at_level(logging.WARNING, logger="memx.core.integration.manager"):
            result = asyncio.run(mgr.fire_pre_inference("test"))
        assert result is None
        assert "failed" in caplog.text


# ===========================================================================
# IntegrationManager — fire_post_action
# ===========================================================================


class TestFirePostAction:
    def _make_event(self) -> ToolEvent:
        return ToolEvent(
            tool_name="bash",
            input={"cmd": "ls"},
            output="file.txt",
            session_id="sess1",
        )

    def test_calls_all_hooks(self) -> None:
        mgr = IntegrationManager()
        h1 = StubPostActionHook("post_a")
        h2 = StubPostActionHook("post_b")
        mgr.register_hooks([h1, h2])
        event = self._make_event()
        asyncio.run(mgr.fire_post_action(event))
        assert len(h1.events) == 1
        assert len(h2.events) == 1
        assert h1.events[0] is event

    def test_skips_disabled_hook(self) -> None:
        mgr = IntegrationManager()
        disabled = StubPostActionHook("disabled_post", enabled=False)
        active = StubPostActionHook("active_post")
        mgr.register_hooks([disabled, active])
        asyncio.run(
            mgr.fire_post_action(self._make_event())
        )
        assert len(disabled.events) == 0
        assert len(active.events) == 1

    def test_auto_reflect_disabled_skips_all(self) -> None:
        config = IntegrationConfig(auto_reflect=False)
        mgr = IntegrationManager(config=config)
        hook = StubPostActionHook()
        mgr.register_hooks([hook])
        asyncio.run(
            mgr.fire_post_action(self._make_event())
        )
        assert len(hook.events) == 0

    def test_failing_hook_isolated(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failing PostActionHook logs WARNING, subsequent hooks still run."""
        mgr = IntegrationManager()
        failing = FailingPostActionHook()
        ok_hook = StubPostActionHook("ok_post")
        mgr.register_hooks([failing, ok_hook])
        with caplog.at_level(logging.WARNING, logger="memx.core.integration.manager"):
            asyncio.run(
                mgr.fire_post_action(self._make_event())
            )
        assert len(ok_hook.events) == 1
        assert "PostActionHook 'failing_post' failed" in caplog.text

    def test_no_hooks_noop(self) -> None:
        mgr = IntegrationManager()
        asyncio.run(
            mgr.fire_post_action(self._make_event())
        )
        # should not raise


# ===========================================================================
# IntegrationManager — fire_session_end
# ===========================================================================


class TestFireSessionEnd:
    def test_calls_all_hooks(self) -> None:
        mgr = IntegrationManager()
        h1 = StubSessionEndHook("session_a")
        h2 = StubSessionEndHook("session_b")
        mgr.register_hooks([h1, h2])
        asyncio.run(
            mgr.fire_session_end("sess-abc")
        )
        assert h1.sessions == ["sess-abc"]
        assert h2.sessions == ["sess-abc"]

    def test_skips_disabled_hook(self) -> None:
        mgr = IntegrationManager()
        disabled = StubSessionEndHook("disabled_sess", enabled=False)
        active = StubSessionEndHook("active_sess")
        mgr.register_hooks([disabled, active])
        asyncio.run(
            mgr.fire_session_end("sess-1")
        )
        assert disabled.sessions == []
        assert active.sessions == ["sess-1"]

    def test_sweep_on_exit_disabled_skips_all(self) -> None:
        config = IntegrationConfig(sweep_on_exit=False)
        mgr = IntegrationManager(config=config)
        hook = StubSessionEndHook()
        mgr.register_hooks([hook])
        asyncio.run(
            mgr.fire_session_end("sess-1")
        )
        assert hook.sessions == []

    def test_failing_hook_isolated(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failing SessionEndHook logs WARNING, subsequent hooks still run."""
        mgr = IntegrationManager()
        failing = FailingSessionEndHook()
        ok_hook = StubSessionEndHook("ok_session")
        mgr.register_hooks([failing, ok_hook])
        with caplog.at_level(logging.WARNING, logger="memx.core.integration.manager"):
            asyncio.run(
                mgr.fire_session_end("sess-1")
            )
        assert ok_hook.sessions == ["sess-1"]
        assert "SessionEndHook 'failing_session' failed" in caplog.text

    def test_no_hooks_noop(self) -> None:
        mgr = IntegrationManager()
        asyncio.run(
            mgr.fire_session_end("sess-1")
        )
        # should not raise


# ===========================================================================
# IntegrationManager — config property
# ===========================================================================


class TestManagerConfig:
    def test_default_config(self) -> None:
        mgr = IntegrationManager()
        assert mgr.config.auto_recall is True
        assert mgr.config.auto_reflect is True
        assert mgr.config.sweep_on_exit is True

    def test_custom_config(self) -> None:
        config = IntegrationConfig(
            auto_recall=False, auto_reflect=False, sweep_on_exit=False
        )
        mgr = IntegrationManager(config=config)
        assert mgr.config.auto_recall is False
        assert mgr.config.auto_reflect is False
        assert mgr.config.sweep_on_exit is False


# ===========================================================================
# IntegrationManager — mixed hook types
# ===========================================================================


class TestMixedHookTypes:
    def test_multiple_hook_types_coexist(self) -> None:
        mgr = IntegrationManager()
        pre = StubPreInferenceHook()
        post = StubPostActionHook()
        session = StubSessionEndHook()
        mgr.register_hooks([pre, post, session])

        # fire pre-inference
        result = asyncio.run(
            mgr.fire_pre_inference("hello")
        )
        assert result is not None
        assert pre.call_count == 1

        # fire post-action
        event = ToolEvent(tool_name="t", input={}, output="o", session_id="s")
        asyncio.run(mgr.fire_post_action(event))
        assert len(post.events) == 1

        # fire session-end
        asyncio.run(mgr.fire_session_end("s"))
        assert session.sessions == ["s"]

        # other hook types unaffected
        assert post.events[0].tool_name == "t"
        assert pre.call_count == 1  # not called again

    def test_unregister_clears_all_types(self) -> None:
        mgr = IntegrationManager()
        mgr.register_hooks([
            StubPreInferenceHook(),
            StubPostActionHook(),
            StubSessionEndHook(),
        ])
        mgr.unregister_all()
        assert mgr.get_hooks(BaseHook) == []
