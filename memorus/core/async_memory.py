"""AsyncMemorusMemory — async drop-in replacement for mem0.AsyncMemory."""

from __future__ import annotations

import logging
import warnings
from typing import Any

logger = logging.getLogger(__name__)


class AsyncMemory:
    """Async Memorus Memory — drop-in replacement for mem0.AsyncMemory.

    ACE OFF (default): direct async proxy to mem0.AsyncMemory.
    ACE ON: async pipeline processing with graceful degradation.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        from memorus.core.config import MemorusConfig

        self._config = MemorusConfig.from_dict(config or {})

        # Lazy import mem0 async
        self._mem0: Any = None
        self._mem0_init_error: Exception | None = None
        try:
            from mem0.configs.base import MemoryConfig

            from mem0 import AsyncMemory as Mem0AsyncMemory

            # mem0 >=1.0 takes a MemoryConfig object. AsyncMemory.from_config
            # is itself a coroutine in 1.0.x, so we build the config directly.
            self._mem0 = Mem0AsyncMemory(
                config=MemoryConfig(**self._config.to_mem0_config())
            )
        except Exception as e:
            self._mem0 = None
            self._mem0_init_error = e
            logger.warning("mem0 AsyncMemory initialization failed: %s", e)

        # Async ACE pipelines are not implemented yet; this class is a mem0
        # proxy. These stay None so add()/search() fall through to mem0.
        self._ingest_pipeline: Any = None
        self._retrieval_pipeline: Any = None
        self._sanitizer: Any = None

        if self._config.ace_enabled:
            self._init_ace_engines()
        elif self._config.privacy.always_sanitize:
            # Sanitizer is needed even without the full ACE pipeline so that
            # always_sanitize is honored on the async add() path.
            self._init_sanitizer()

        # Be honest about the contract: the async ACE pipeline is not built,
        # so anything beyond sanitization (reflection, curation, retrieval
        # shaping) is silently a no-op. Warn loudly instead of pretending.
        if self._config.ace_enabled:
            msg = (
                "AsyncMemory: ace_enabled=True but the async ACE pipeline is "
                "not implemented; running as a mem0 proxy. "
                + (
                    "PII is still sanitized before storage."
                    if self._sanitizer is not None
                    else "Sanitizer is unavailable, so PII is NOT redacted."
                )
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            logger.warning(msg)
        elif self._config.privacy.always_sanitize and self._sanitizer is None:
            msg = (
                "AsyncMemory: always_sanitize=True but the sanitizer failed "
                "to initialize; PII will NOT be redacted before storage."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
            logger.warning(msg)

    def _ensure_mem0(self) -> Any:
        """Raise if mem0 async backend not available."""
        if self._mem0 is None:
            raise RuntimeError(
                f"mem0 async backend not initialized: "
                f"{self._mem0_init_error or 'unknown'}"
            )
        return self._mem0

    def _init_sanitizer(self) -> None:
        """Initialize the sanitizer only (for always_sanitize without ACE)."""
        try:
            from memorus.core.privacy.sanitizer import PrivacySanitizer

            self._sanitizer = PrivacySanitizer(
                custom_patterns=self._config.privacy.custom_patterns
            )
        except Exception as e:
            logger.warning("Sanitizer init failed: %s", e)

    def _init_ace_engines(self) -> None:
        """Initialize ACE engines. Failures degrade to proxy mode.

        The async ACE pipeline (reflector/curator/retrieval) is not yet
        implemented, so only the sanitizer is wired here to keep PII
        redaction working on the proxy add() path.
        """
        self._init_sanitizer()

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> AsyncMemory:
        """Create AsyncMemory from a config dict."""
        return cls(config=config_dict)

    # ---- Core Async API ----------------------------------------------------

    async def add(
        self,
        messages: Any,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add memories. ACE mode processes through IngestPipeline."""
        # Sanitize before storage whenever a sanitizer is wired (ACE on, or
        # always_sanitize without ACE). The async ACE pipeline is not built,
        # so this proxy path is the only place PII can be redacted.
        if self._sanitizer is not None and (
            self._config.ace_enabled or self._config.privacy.always_sanitize
        ):
            messages = self._sanitize_messages(messages)

        # mem0 >=1.0 dropped `filters` from add(); keep for search/get_all only.
        mem0 = self._ensure_mem0()
        return await mem0.add(
            messages,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            metadata=metadata,
            prompt=prompt,
            **kwargs,
        )

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        filters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search memories. ACE mode uses RetrievalPipeline."""
        if not self._config.ace_enabled or self._retrieval_pipeline is None:
            mem0 = self._ensure_mem0()
            return await mem0.search(
                query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
                filters=filters,
                **kwargs,
            )

        # ACE path placeholder
        mem0 = self._ensure_mem0()
        return await mem0.search(
            query,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            limit=limit,
            filters=filters,
            **kwargs,
        )

    async def get_all(self, **kwargs: Any) -> dict[str, Any]:
        """Get all memories."""
        return await self._ensure_mem0().get_all(**kwargs)

    async def get(self, memory_id: str) -> dict[str, Any]:
        """Get a single memory by ID."""
        return await self._ensure_mem0().get(memory_id)

    async def update(self, memory_id: str, data: str) -> dict[str, Any]:
        """Update a memory by ID."""
        return await self._ensure_mem0().update(memory_id, data)

    async def delete(self, memory_id: str) -> None:
        """Delete a single memory by ID."""
        return await self._ensure_mem0().delete(memory_id)

    async def delete_all(self, **kwargs: Any) -> None:
        """Delete all memories matching kwargs."""
        return await self._ensure_mem0().delete_all(**kwargs)

    async def history(self, memory_id: str) -> dict[str, Any]:
        """Get modification history of a memory."""
        return await self._ensure_mem0().history(memory_id)

    async def reset(self) -> None:
        """Reset all memories."""
        return await self._ensure_mem0().reset()

    # ---- Privacy helpers ---------------------------------------------------

    def _sanitize_messages(self, messages: Any) -> Any:
        """Run privacy sanitizer on messages (mirrors sync Memory)."""
        if self._sanitizer is None:
            return messages
        try:
            if isinstance(messages, str):
                result = self._sanitizer.sanitize(messages)
                return result.clean_content
            elif isinstance(messages, list):
                sanitized = []
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        result = self._sanitizer.sanitize(msg["content"])
                        sanitized.append({**msg, "content": result.clean_content})
                    else:
                        sanitized.append(msg)
                return sanitized
            return messages
        except Exception as e:
            logger.warning("Sanitization failed: %s", e)
            return messages

    # ---- ACE-specific (placeholder) ----------------------------------------

    async def status(self) -> dict[str, Any]:
        """Get Memorus status info."""
        raise NotImplementedError("status() will be implemented in STORY-041")

    async def export(self, format: str = "json") -> Any:
        """Export memories in the specified format."""
        raise NotImplementedError("export() will be implemented in STORY-044")

    async def import_data(self, data: Any, format: str = "json") -> Any:
        """Import memories from the specified format."""
        raise NotImplementedError("import_data() will be implemented in STORY-044")

    async def run_decay_sweep(self) -> Any:
        """Run a temporal decay sweep across all memories."""
        raise NotImplementedError("run_decay_sweep() will be implemented in STORY-021")

    # ---- Properties --------------------------------------------------------

    @property
    def config(self) -> Any:
        """Access the Memorus configuration."""
        return self._config
