"""Hook abstractions for MemX integration layer.

Defines BaseHook ABC and three concrete hook interfaces:
- PreInferenceHook: inject recalled memories before LLM inference
- PostActionHook: distill tool results after execution
- SessionEndHook: sweep decay + finalize on session end

Also defines ContextInjection and ToolEvent data transfer objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContextInjection:
    """Result of a pre-inference hook: recalled memories formatted for injection."""

    memories: list[dict] = field(default_factory=list)
    format: str = "xml"  # "xml" | "markdown" | "plain"
    rendered: str = ""  # formatted context string ready for prompt injection


@dataclass(frozen=True)
class ToolEvent:
    """Payload describing a tool invocation result for post-action hooks."""

    tool_name: str = ""
    input: dict = field(default_factory=dict)
    output: str = ""
    session_id: str = ""


class BaseHook(ABC):
    """Abstract base class for all MemX integration hooks.

    Subclasses must implement the ``name`` property.
    The ``enabled`` property defaults to True and can be overridden.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this hook instance."""
        ...

    @property
    def enabled(self) -> bool:
        """Whether this hook is active. Override to disable dynamically."""
        return True


class PreInferenceHook(BaseHook):
    """Hook fired before LLM inference to inject recalled memories."""

    @abstractmethod
    async def on_user_input(self, input: str) -> ContextInjection:
        """Recall relevant memories for the given user input.

        Returns a ContextInjection with formatted memory context.
        """
        ...


class PostActionHook(BaseHook):
    """Hook fired after a tool/action execution to distill results."""

    @abstractmethod
    async def on_tool_result(self, event: ToolEvent) -> None:
        """Process a tool execution result for memory distillation."""
        ...


class SessionEndHook(BaseHook):
    """Hook fired when a session ends for cleanup and decay sweep."""

    @abstractmethod
    async def on_session_end(self, session_id: str) -> None:
        """Finalize session: flush pending distillation, run decay sweep."""
        ...
