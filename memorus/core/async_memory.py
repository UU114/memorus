"""AsyncMemorusMemory — async drop-in replacement for mem0.AsyncMemory."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class AsyncMemory:
    """Async Memorus Memory — drop-in replacement for mem0.AsyncMemory.

    ACE OFF (default): direct async proxy to mem0.AsyncMemory.
    ACE ON: async pipeline processing with graceful degradation.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        from memorus.core.config import MemorusConfig

        self._config = MemorusConfig.from_dict(config or {})

        # Lazy import mem0 async
        self._mem0: Any = None
        self._mem0_init_error: Optional[Exception] = None
        try:
            from mem0 import AsyncMemory as Mem0AsyncMemory
            from mem0.configs.base import MemoryConfig

            # mem0 >=1.0 takes a MemoryConfig object. AsyncMemory.from_config
            # is itself a coroutine in 1.0.x, so we build the config directly.
            self._mem0 = Mem0AsyncMemory(
                config=MemoryConfig(**self._config.to_mem0_config())
            )
        except Exception as e:
            self._mem0 = None
            self._mem0_init_error = e
            logger.warning("mem0 AsyncMemory initialization failed: %s", e)

        self._ingest_pipeline: Any = None
        self._retrieval_pipeline: Any = None
        self._sanitizer: Any = None

        if self._config.ace_enabled:
            self._init_ace_engines()

    def _ensure_mem0(self) -> Any:
        """Raise if mem0 async backend not available."""
        if self._mem0 is None:
            raise RuntimeError(
                f"mem0 async backend not initialized: "
                f"{self._mem0_init_error or 'unknown'}"
            )
        return self._mem0

    def _init_ace_engines(self) -> None:
        """Initialize ACE engines. Failures degrade to proxy mode."""
        try:
            from memorus.core.privacy.sanitizer import PrivacySanitizer

            self._sanitizer = PrivacySanitizer(
                custom_patterns=self._config.privacy.custom_patterns
            )
        except Exception as e:
            logger.warning("Sanitizer init failed: %s", e)

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> AsyncMemory:
        """Create AsyncMemory from a config dict."""
        return cls(config=config_dict)

    # ---- Core Async API ----------------------------------------------------

    async def add(
        self,
        messages: Any,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None,
        prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add memories. ACE mode processes through IngestPipeline."""
        # mem0 >=1.0 dropped `filters` from add(); keep for search/get_all only.
        if not self._config.ace_enabled or self._ingest_pipeline is None:
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

        # ACE async path placeholder
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
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 100,
        filters: Optional[dict[str, Any]] = None,
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
