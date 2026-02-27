"""MemXMemory — drop-in replacement for mem0.Memory with ACE capabilities."""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class Memory:
    """MemX Memory — drop-in replacement for mem0.Memory.

    ACE OFF (default): direct proxy to mem0.Memory (zero overhead).
    ACE ON: pipeline processing with graceful degradation.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        from memx.config import MemXConfig

        self._config = MemXConfig.from_dict(config or {})

        # Lazy import mem0 to avoid import errors when testing
        self._mem0: Any = None
        self._mem0_init_error: Optional[Exception] = None
        try:
            from mem0 import Memory as Mem0Memory

            self._mem0 = Mem0Memory(config=self._config.to_mem0_config())
        except Exception as e:
            # If mem0 can't initialize (e.g., no API key), store the error.
            # Users must handle this if they need actual mem0 functionality.
            self._mem0_init_error = e
            logger.warning("mem0 initialization failed: %s", e)

        # Pipeline (lazy init when ACE is enabled)
        self._ingest_pipeline: Any = None
        self._retrieval_pipeline: Any = None
        self._sanitizer: Any = None

        if self._config.ace_enabled:
            self._init_ace_engines()
        elif self._config.privacy.always_sanitize:
            # Sanitizer needed even without full ACE pipeline
            self._init_sanitizer()

    def _ensure_mem0(self) -> Any:
        """Raise if mem0 backend not available."""
        if self._mem0 is None:
            raise RuntimeError(
                f"mem0 backend not initialized: {self._mem0_init_error or 'unknown'}"
            )
        return self._mem0

    def _init_sanitizer(self) -> None:
        """Initialize the sanitizer only (for always_sanitize without ACE)."""
        try:
            from memx.privacy.sanitizer import PrivacySanitizer

            self._sanitizer = PrivacySanitizer(
                custom_patterns=self._config.privacy.custom_patterns
            )
        except Exception as e:
            logger.warning("Sanitizer init failed: %s", e)

    def _init_ace_engines(self) -> None:
        """Initialize ACE engines. Failures degrade to proxy mode."""
        self._init_sanitizer()

        # Ingest pipeline
        try:
            from memx.engines.reflector.engine import ReflectorEngine
            from memx.pipeline.ingest import IngestPipeline

            reflector = ReflectorEngine(
                config=self._config.reflector,
                sanitizer=self._sanitizer,
            )

            # Optional Curator engine for deduplication
            curator = None
            try:
                from memx.engines.curator.engine import CuratorEngine

                curator = CuratorEngine(config=self._config.curator)
            except Exception as e:
                logger.warning("CuratorEngine init failed (dedup disabled): %s", e)

            self._ingest_pipeline = IngestPipeline(
                reflector=reflector,
                sanitizer=self._sanitizer,
                curator=curator,
                mem0_add_fn=self._mem0.add if self._mem0 else None,
                mem0_get_all_fn=self._mem0.get_all if self._mem0 else None,
                mem0_update_fn=self._mem0.update if self._mem0 else None,
            )
        except Exception as e:
            logger.warning("ACE ingest pipeline init failed, proxy mode: %s", e)

        # Retrieval pipeline
        try:
            from memx.config import RetrievalConfig
            from memx.engines.decay.engine import DecayEngine
            from memx.engines.generator.engine import GeneratorEngine
            from memx.engines.generator.vector_searcher import VectorSearcher
            from memx.pipeline.retrieval import RetrievalPipeline
            from memx.utils.token_counter import TokenBudgetTrimmer

            retrieval_cfg = self._config.retrieval
            generator = GeneratorEngine(
                config=retrieval_cfg,
                vector_searcher=VectorSearcher(),
            )
            trimmer = TokenBudgetTrimmer(
                token_budget=retrieval_cfg.token_budget,
                max_results=retrieval_cfg.max_results,
            )
            decay_engine = DecayEngine(config=self._config.decay)

            self._retrieval_pipeline = RetrievalPipeline(
                generator=generator,
                trimmer=trimmer,
                decay_engine=decay_engine,
                mem0_search_fn=self._mem0.search if self._mem0 else None,
            )
        except Exception as e:
            logger.warning("ACE retrieval pipeline init failed, proxy mode: %s", e)

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> Memory:
        """Create Memory from a config dict (mem0-compatible)."""
        return cls(config=config_dict)

    # ---- Core API (mem0-compatible) ----------------------------------------

    def add(
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
        # Sanitize if always_sanitize is on, even in proxy mode
        if (
            not self._config.ace_enabled
            and self._config.privacy.always_sanitize
            and self._sanitizer
        ):
            messages = self._sanitize_messages(messages)

        if not self._config.ace_enabled or self._ingest_pipeline is None:
            mem0 = self._ensure_mem0()
            return mem0.add(
                messages,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                metadata=metadata,
                filters=filters,
                prompt=prompt,
                **kwargs,
            )

        # ACE path - delegate to IngestPipeline
        from memx.pipeline.ingest import IngestResult  # noqa: F811

        ingest_result: IngestResult = self._ingest_pipeline.process(
            messages,
            metadata=metadata,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
        )
        return {
            "results": [],
            "ace_ingest": {
                "bullets_added": ingest_result.bullets_added,
                "raw_fallback": ingest_result.raw_fallback,
                "errors": ingest_result.errors,
            },
        }

    def search(
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
            return mem0.search(
                query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
                filters=filters,
                **kwargs,
            )

        # ACE path: load bullets and run through RetrievalPipeline
        try:
            bullets = self._load_bullets_for_search(user_id, agent_id)
            from memx.pipeline.retrieval import SearchResult

            search_result: SearchResult = self._retrieval_pipeline.search(
                query=query,
                bullets=bullets,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
                filters=filters,
            )
            return {
                "results": [
                    {
                        "id": b.bullet_id,
                        "memory": b.content,
                        "score": b.final_score,
                        "metadata": b.metadata,
                    }
                    for b in search_result.results
                ],
                "ace_search": {
                    "mode": search_result.mode,
                    "total_candidates": search_result.total_candidates,
                },
            }
        except Exception as e:
            logger.warning("ACE search failed, falling back to mem0: %s", e)
            mem0 = self._ensure_mem0()
            return mem0.search(
                query,
                user_id=user_id,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
                filters=filters,
                **kwargs,
            )

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        """Get all memories."""
        return self._ensure_mem0().get_all(**kwargs)

    def get(self, memory_id: str) -> dict[str, Any]:
        """Get a single memory by ID."""
        return self._ensure_mem0().get(memory_id)

    def update(self, memory_id: str, data: str) -> dict[str, Any]:
        """Update a memory by ID."""
        return self._ensure_mem0().update(memory_id, data)

    def delete(self, memory_id: str) -> None:
        """Delete a single memory by ID."""
        return self._ensure_mem0().delete(memory_id)

    def delete_all(self, **kwargs: Any) -> None:
        """Delete all memories matching kwargs."""
        return self._ensure_mem0().delete_all(**kwargs)

    def history(self, memory_id: str) -> dict[str, Any]:
        """Get modification history of a memory."""
        return self._ensure_mem0().history(memory_id)

    def reset(self) -> None:
        """Reset all memories."""
        return self._ensure_mem0().reset()

    # ---- ACE-specific methods ----------------------------------------------

    def status(self) -> dict[str, Any]:
        """Get MemX status info."""
        raise NotImplementedError("status() will be implemented in STORY-041")

    def export(self, format: str = "json") -> Any:
        """Export memories in the specified format."""
        raise NotImplementedError("export() will be implemented in STORY-044")

    def import_data(self, data: Any, format: str = "json") -> Any:
        """Import memories from the specified format."""
        raise NotImplementedError("import_data() will be implemented in STORY-044")

    def run_decay_sweep(self) -> Any:
        """Run a temporal decay sweep across all memories."""
        raise NotImplementedError("run_decay_sweep() will be implemented in STORY-021")

    # ---- Internal ----------------------------------------------------------

    def _load_bullets_for_search(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> list[Any]:
        """Load all bullets from mem0 and convert to BulletForSearch for GeneratorEngine.

        Returns an empty list if mem0 is not available or get_all fails.
        """
        from memx.engines.generator.engine import BulletForSearch
        from memx.engines.generator.metadata_matcher import MetadataInfo
        from memx.utils.bullet_factory import BulletFactory

        mem0 = self._ensure_mem0()
        kwargs: dict[str, Any] = {}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id

        raw = mem0.get_all(**kwargs)
        memories = raw.get("memories", []) if isinstance(raw, dict) else []

        bullets: list[BulletForSearch] = []
        for mem in memories:
            if not isinstance(mem, dict):
                continue
            bullet_meta = BulletFactory.from_mem0_payload(mem)
            bullets.append(
                BulletForSearch(
                    bullet_id=mem.get("id", ""),
                    content=mem.get("memory", ""),
                    metadata=MetadataInfo(
                        related_tools=bullet_meta.related_tools,
                        key_entities=bullet_meta.key_entities,
                        tags=bullet_meta.tags,
                    ),
                    created_at=bullet_meta.created_at,
                    decay_weight=bullet_meta.decay_weight,
                )
            )
        return bullets

    def _sanitize_messages(self, messages: Any) -> Any:
        """Run privacy sanitizer on messages."""
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

    @property
    def config(self) -> Any:
        """Access the MemX configuration."""
        return self._config
