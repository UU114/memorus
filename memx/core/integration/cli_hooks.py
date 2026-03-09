"""CLI hook implementations for MemX integration.

Provides concrete hooks for Claude Code CLI integration:
- CLIPreInferenceHook: recall relevant memories before LLM inference
- CLIPostActionHook: async distillation after tool execution
- CLISessionEndHook: final distillation + decay sweep on session end
- setup_signal_handlers: SIGTERM/SIGINT -> graceful session end
"""

from __future__ import annotations

import asyncio
import logging
import platform
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Optional

from memx.core.config import IntegrationConfig
from memx.core.integration.hooks import (
    ContextInjection,
    PostActionHook,
    PreInferenceHook,
    SessionEndHook,
    ToolEvent,
)
from memx.core.memory import Memory

if TYPE_CHECKING:
    from memx.core.engines.decay.engine import DecayEngine
    from memx.core.integration.manager import IntegrationManager

logger = logging.getLogger(__name__)

# Supported context template formats
_VALID_FORMATS = frozenset({"xml", "markdown", "plain"})


class CLIPreInferenceHook(PreInferenceHook):
    """Pre-inference hook for CLI scenarios.

    Searches the Memory instance for relevant memories based on
    user input and formats them according to the configured template.
    """

    def __init__(self, memory: Memory, config: Optional[IntegrationConfig] = None) -> None:
        self._memory = memory
        self._config = config or IntegrationConfig()

    @property
    def name(self) -> str:
        return "cli_pre_inference"

    @property
    def enabled(self) -> bool:
        return self._config.auto_recall

    async def on_user_input(self, input: str) -> ContextInjection:
        """Recall relevant memories for the given user input.

        Returns a ContextInjection with formatted memory context.
        Returns empty ContextInjection when input is empty or no results found.
        Catches all exceptions and returns empty ContextInjection with WARNING log.
        """
        if not input or not input.strip():
            return ContextInjection(memories=[], format=self._config.context_template, rendered="")

        try:
            raw = self._memory.search(input)
        except Exception:
            logger.warning(
                "CLIPreInferenceHook: Memory.search() failed",
                exc_info=True,
            )
            return ContextInjection(memories=[], format=self._config.context_template, rendered="")

        # Extract results list from search response
        results = self._extract_results(raw)

        if not results:
            return ContextInjection(memories=[], format=self._config.context_template, rendered="")

        rendered = self._format(results, self._config.context_template)
        return ContextInjection(
            memories=results,
            format=self._config.context_template,
            rendered=rendered,
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_results(raw: object) -> list[dict]:
        """Extract the results list from Memory.search() response.

        Memory.search() returns {"results": [...], ...}.
        Each result dict has: id, memory, score, metadata.
        """
        if not isinstance(raw, dict):
            return []
        results = raw.get("results", [])
        if not isinstance(results, list):
            return []
        return [r for r in results if isinstance(r, dict)]

    @staticmethod
    def _format(results: list[dict], template: str) -> str:
        """Format search results using the specified template.

        Supported templates: xml (default), markdown, plain.
        Falls back to xml for unrecognized templates.
        """
        if template == "markdown":
            return CLIPreInferenceHook._format_markdown(results)
        elif template == "plain":
            return CLIPreInferenceHook._format_plain(results)
        else:
            # Default to xml (also handles unknown template values)
            return CLIPreInferenceHook._format_xml(results)

    @staticmethod
    def _format_xml(results: list[dict]) -> str:
        """Format results as XML context block.

        Output:
            <memx-context>
              <memory id="abc123" score="0.85" type="preference">
                User prefers dark mode in all applications.
              </memory>
            </memx-context>
        """
        lines = ["<memx-context>"]
        for r in results:
            mem_id = r.get("id", "unknown")
            score = r.get("score", 0.0)
            metadata = r.get("metadata", {})
            mem_type = metadata.get("memx_knowledge_type", "knowledge") if isinstance(metadata, dict) else "knowledge"
            content = r.get("memory", "")
            lines.append(
                f'  <memory id="{mem_id}" score="{score:.2f}" type="{mem_type}">'
            )
            lines.append(f"    {content}")
            lines.append("  </memory>")
        lines.append("</memx-context>")
        return "\n".join(lines)

    @staticmethod
    def _format_markdown(results: list[dict]) -> str:
        """Format results as Markdown context block.

        Output:
            ## MemX Context
            - **[0.85]** User prefers dark mode in all applications.
        """
        lines = ["## MemX Context"]
        for r in results:
            score = r.get("score", 0.0)
            content = r.get("memory", "")
            lines.append(f"- **[{score:.2f}]** {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_plain(results: list[dict]) -> str:
        """Format results as plain text context block.

        Output:
            [MemX] User prefers dark mode in all applications.
            [MemX] When using pytest, always run with -v flag.
        """
        lines = []
        for r in results:
            content = r.get("memory", "")
            lines.append(f"[MemX] {content}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# STORY-034: PostActionHook + SessionEndHook
# ---------------------------------------------------------------------------

# Maximum characters for tool output before truncation
_MAX_OUTPUT_LENGTH = 10000


class CLIPostActionHook(PostActionHook):
    """Post-action hook that distills tool results asynchronously.

    Receives ToolEvent from CLI tool calls (e.g. Bash, Read, Edit),
    formats them as chat messages, and submits to Memory.add() via a
    background ThreadPoolExecutor so that distillation never blocks
    the main interaction flow.
    """

    def __init__(
        self,
        memory: Memory,
        config: Optional[IntegrationConfig] = None,
    ) -> None:
        self._memory = memory
        self._config = config or IntegrationConfig()
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def name(self) -> str:
        return "cli_post_action"

    @property
    def enabled(self) -> bool:
        return self._config.auto_reflect

    async def on_tool_result(self, event: ToolEvent) -> None:
        """Format ToolEvent and submit distillation to background thread.

        Exceptions are caught and logged at WARNING level; they never
        propagate to the caller.
        """
        try:
            messages = self._format_tool_event(event)
            self._executor.submit(
                self._safe_add, messages, event.session_id
            )
        except Exception:
            logger.warning(
                "CLIPostActionHook failed to submit distillation",
                exc_info=True,
            )

    def _format_tool_event(self, event: ToolEvent) -> list[dict[str, str]]:
        """Convert a ToolEvent into message dicts for Memory.add().

        Truncates tool output that exceeds _MAX_OUTPUT_LENGTH.
        """
        output = event.output
        if len(output) > _MAX_OUTPUT_LENGTH:
            output = output[:_MAX_OUTPUT_LENGTH] + "\n... [truncated]"

        return [
            {"role": "assistant", "content": f"Used tool: {event.tool_name}"},
            {"role": "tool", "content": output},
        ]

    def _safe_add(self, messages: list[dict[str, str]], session_id: str) -> None:
        """Call memory.add() with error isolation."""
        try:
            self._memory.add(messages, user_id=session_id)
        except Exception:
            logger.warning(
                "Background distillation failed for session %r",
                session_id,
                exc_info=True,
            )

    def shutdown(self, wait: bool = True, timeout: float = 5.0) -> None:
        """Shut down the background executor.

        Args:
            wait: Whether to wait for pending tasks.
            timeout: Maximum seconds to wait (Python 3.9+).
        """
        try:
            if sys.version_info >= (3, 9):
                self._executor.shutdown(wait=wait, cancel_futures=not wait)
            else:
                self._executor.shutdown(wait=wait)
        except Exception:
            logger.warning("Executor shutdown error", exc_info=True)


class CLISessionEndHook(SessionEndHook):
    """Session-end hook that runs decay sweep on session end.

    Executed when the CLI session ends (normal exit, SIGTERM, or SIGINT).
    Runs DecayEngine.sweep() on all bullets for the session.
    """

    def __init__(
        self,
        memory: Memory,
        decay_engine: "DecayEngine",
        config: Optional[IntegrationConfig] = None,
    ) -> None:
        self._memory = memory
        self._decay = decay_engine
        self._config = config or IntegrationConfig()
        self._completed = False

    @property
    def name(self) -> str:
        return "cli_session_end"

    @property
    def enabled(self) -> bool:
        return self._config.sweep_on_exit

    async def on_session_end(self, session_id: str) -> None:
        """Run decay sweep on session end.

        Idempotent: if called multiple times (e.g. repeated signals),
        subsequent calls are no-ops.  Exceptions are caught and logged
        at WARNING level; they never propagate.
        """
        if self._completed:
            logger.debug("Session end already completed, skipping")
            return

        try:
            self._run_sweep(session_id)
            self._completed = True
            logger.info("Session end completed for %r", session_id)
        except Exception:
            logger.warning(
                "CLISessionEndHook failed for session %r",
                session_id,
                exc_info=True,
            )

    def _run_sweep(self, session_id: str) -> None:
        """Execute decay sweep on existing bullets.

        Loads all bullets for the session and runs DecayEngine.sweep().
        """
        from datetime import datetime, timezone

        from memx.core.engines.decay.engine import BulletDecayInfo

        try:
            raw = self._memory.get_all(user_id=session_id)
            memories = raw.get("memories", []) if isinstance(raw, dict) else []

            bullets: list[BulletDecayInfo] = []
            for mem in memories:
                if not isinstance(mem, dict):
                    continue
                meta = mem.get("metadata", {})

                created_str = meta.get("created_at")
                if created_str and isinstance(created_str, str):
                    try:
                        created_at = datetime.fromisoformat(created_str)
                    except ValueError:
                        created_at = datetime.now(timezone.utc)
                else:
                    created_at = datetime.now(timezone.utc)

                bullets.append(
                    BulletDecayInfo(
                        bullet_id=mem.get("id", ""),
                        created_at=created_at,
                        recall_count=meta.get("recall_count", 0),
                        current_weight=meta.get("memx_decay_weight", 1.0),
                    )
                )

            if bullets:
                result = self._decay.sweep(bullets)
                logger.info(
                    "Decay sweep: %d updated, %d archived, %d permanent",
                    result.updated,
                    result.archived,
                    result.permanent,
                )
        except Exception:
            logger.warning(
                "Decay sweep failed for session %r",
                session_id,
                exc_info=True,
            )


def setup_signal_handlers(
    manager: "IntegrationManager",
    session_id: str,
) -> None:
    """Register SIGTERM and SIGINT handlers to trigger graceful session end.

    On signal receipt:
    1. Fire all SessionEndHook instances via IntegrationManager
    2. Wait up to 5 seconds for completion
    3. Raise SystemExit(0)

    On Windows, only SIGINT is registered (SIGTERM is not supported).
    Repeated signals during shutdown are ignored.
    """
    _shutting_down = False

    def _handler(signum: int, frame: Any) -> None:
        nonlocal _shutting_down
        if _shutting_down:
            return
        _shutting_down = True

        sig_name = (
            signal.Signals(signum).name
            if hasattr(signal, "Signals")
            else str(signum)
        )
        logger.info(
            "Received %s, triggering session end for %r",
            sig_name,
            session_id,
        )

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                asyncio.wait_for(
                    manager.fire_session_end(session_id),
                    timeout=5.0,
                )
            )
        except asyncio.TimeoutError:
            logger.warning("Session end timed out after 5s")
        except Exception:
            logger.warning(
                "Session end failed during signal handling", exc_info=True
            )
        finally:
            loop.close()

        raise SystemExit(0)

    # SIGINT works on all platforms (Ctrl+C)
    signal.signal(signal.SIGINT, _handler)

    # SIGTERM is not available on Windows
    if platform.system() != "Windows":
        signal.signal(signal.SIGTERM, _handler)
