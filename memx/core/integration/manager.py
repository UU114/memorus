"""IntegrationManager — registry and executor for MemX hooks.

Manages hook lifecycle: register, unregister, query, and fire.
Respects IntegrationConfig to skip disabled hook categories.
Isolates individual hook failures with WARNING-level logging.
"""

from __future__ import annotations

import logging
from typing import Optional

from memx.core.config import IntegrationConfig
from memx.core.integration.hooks import (
    BaseHook,
    ContextInjection,
    PostActionHook,
    PreInferenceHook,
    SessionEndHook,
    ToolEvent,
)

logger = logging.getLogger(__name__)


class IntegrationManager:
    """Central registry for MemX integration hooks.

    Hooks are stored in registration order and executed sequentially.
    IntegrationConfig controls which hook categories are active.
    """

    def __init__(self, config: Optional[IntegrationConfig] = None) -> None:
        self._config = config or IntegrationConfig()
        self._hooks: list[BaseHook] = []

    @property
    def config(self) -> IntegrationConfig:
        """Return the current IntegrationConfig."""
        return self._config

    # -- Registration -------------------------------------------------------

    def register_hooks(self, hooks: list[BaseHook]) -> None:
        """Register a list of hooks. Duplicates (by identity) are skipped."""
        for hook in hooks:
            if hook in self._hooks:
                logger.debug("Hook %r already registered, skipping", hook.name)
                continue
            self._hooks.append(hook)
            logger.info("Registered hook: %s (%s)", hook.name, type(hook).__name__)

    def unregister_all(self) -> None:
        """Remove all registered hooks."""
        count = len(self._hooks)
        self._hooks.clear()
        logger.info("Unregistered all hooks (%d removed)", count)

    def get_hooks(self, hook_type: type) -> list[BaseHook]:
        """Return all registered hooks matching *hook_type* (including subclasses)."""
        return [h for h in self._hooks if isinstance(h, hook_type)]

    # -- Fire methods -------------------------------------------------------

    async def fire_pre_inference(self, input: str) -> Optional[ContextInjection]:
        """Execute all PreInferenceHook instances if auto_recall is enabled.

        Returns the result from the first successful hook, or None if
        auto_recall is disabled or no hooks succeed.
        """
        if not self._config.auto_recall:
            logger.debug("auto_recall disabled, skipping pre-inference hooks")
            return None

        hooks = self.get_hooks(PreInferenceHook)
        for hook in hooks:
            if not hook.enabled:
                logger.debug("Hook %s disabled, skipping", hook.name)
                continue
            try:
                # Type narrowing: hook is PreInferenceHook here
                assert isinstance(hook, PreInferenceHook)
                result = await hook.on_user_input(input)
                return result
            except Exception:
                logger.warning(
                    "PreInferenceHook %r failed, trying next",
                    hook.name,
                    exc_info=True,
                )
        return None

    async def fire_post_action(self, event: ToolEvent) -> None:
        """Execute all PostActionHook instances if auto_reflect is enabled.

        Each hook runs independently; failures are logged but do not
        prevent subsequent hooks from executing.
        """
        if not self._config.auto_reflect:
            logger.debug("auto_reflect disabled, skipping post-action hooks")
            return

        hooks = self.get_hooks(PostActionHook)
        for hook in hooks:
            if not hook.enabled:
                logger.debug("Hook %s disabled, skipping", hook.name)
                continue
            try:
                assert isinstance(hook, PostActionHook)
                await hook.on_tool_result(event)
            except Exception:
                logger.warning(
                    "PostActionHook %r failed, continuing",
                    hook.name,
                    exc_info=True,
                )

    async def fire_session_end(self, session_id: str) -> None:
        """Execute all SessionEndHook instances if sweep_on_exit is enabled.

        Each hook runs independently; failures are logged but do not
        prevent subsequent hooks from executing.
        """
        if not self._config.sweep_on_exit:
            logger.debug("sweep_on_exit disabled, skipping session-end hooks")
            return

        hooks = self.get_hooks(SessionEndHook)
        for hook in hooks:
            if not hook.enabled:
                logger.debug("Hook %s disabled, skipping", hook.name)
                continue
            try:
                assert isinstance(hook, SessionEndHook)
                await hook.on_session_end(session_id)
            except Exception:
                logger.warning(
                    "SessionEndHook %r failed, continuing",
                    hook.name,
                    exc_info=True,
                )
