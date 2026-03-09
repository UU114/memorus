"""MemXMemory — drop-in replacement for mem0.Memory with ACE capabilities."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _validate_scope(scope: Optional[str]) -> str:
    """Validate and normalize a scope string.

    Returns "global" for None or empty string.
    Raises ValueError for malformed scopes like "project:" without a name.
    """
    if scope is None or scope == "":
        return "global"
    if scope.startswith("project:") and len(scope) <= len("project:"):
        raise ValueError(
            "Scope 'project:' requires a name, e.g., 'project:myapp'"
        )
    return scope


class Memory:
    """MemX Memory — drop-in replacement for mem0.Memory.

    ACE OFF (default): direct proxy to mem0.Memory (zero overhead).
    ACE ON: pipeline processing with graceful degradation.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        from memx.core.config import MemXConfig

        logger.debug("Memory.__init__ config=%s", config)
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

        # Daemon fallback manager (lazy init when daemon is enabled)
        self._daemon_fallback: Any = None
        self._init_daemon_fallback()

        if self._config.ace_enabled:
            self._init_ace_engines()
        elif self._config.privacy.always_sanitize:
            # Sanitizer needed even without full ACE pipeline
            self._init_sanitizer()

        # Conditionally bootstrap Team Layer (always safe to call)
        self._team_enabled = self._try_team_bootstrap()

    def _try_team_bootstrap(self) -> bool:
        """Attempt Team Layer bootstrap via ext module. Never raises."""
        try:
            from memx.ext.team_bootstrap import try_bootstrap_team

            return try_bootstrap_team(self)
        except Exception as e:
            logger.debug("Team bootstrap import/call failed: %s", e)
            return False

    def _ensure_mem0(self) -> Any:
        """Raise if mem0 backend not available."""
        if self._mem0 is None:
            raise RuntimeError(
                f"mem0 backend not initialized: {self._mem0_init_error or 'unknown'}"
            )
        return self._mem0

    def _init_daemon_fallback(self) -> None:
        """Initialize daemon fallback manager if daemon is enabled.

        Pings the daemon to determine initial availability.  If the daemon
        is unreachable, logs a warning and operates in direct mode.
        """
        if not self._config.daemon.enabled:
            return
        try:
            from memx.core.daemon.fallback import DaemonFallbackManager

            self._daemon_fallback = DaemonFallbackManager(
                config=self._config.daemon,
            )
            self._daemon_fallback.check_initial_availability()
        except Exception as e:
            logger.warning("Daemon fallback init failed: %s", e)
            self._daemon_fallback = None

    def _init_sanitizer(self) -> None:
        """Initialize the sanitizer only (for always_sanitize without ACE)."""
        try:
            from memx.core.privacy.sanitizer import PrivacySanitizer

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
            from memx.core.engines.reflector.engine import ReflectorEngine
            from memx.core.pipeline.ingest import IngestPipeline

            reflector = ReflectorEngine(
                config=self._config.reflector,
                sanitizer=self._sanitizer,
            )

            # Optional Curator engine for deduplication
            curator = None
            try:
                from memx.core.engines.curator.engine import CuratorEngine

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
            from memx.core.config import RetrievalConfig
            from memx.core.engines.decay.engine import DecayEngine
            from memx.core.engines.generator.engine import GeneratorEngine
            from memx.core.engines.generator.vector_searcher import VectorSearcher
            from memx.core.pipeline.retrieval import RetrievalPipeline
            from memx.core.utils.token_counter import TokenBudgetTrimmer

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
        scope: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add memories. ACE mode processes through IngestPipeline.

        Args:
            scope: Hierarchical scope for the memory (e.g., "project:myapp").
                   Defaults to "global" when None or empty.

        When daemon is enabled and available, routes through DaemonClient.
        Falls back to direct mode transparently on daemon failure.
        """
        effective_scope = _validate_scope(scope)
        logger.debug(
            "Memory.add user_id=%s scope=%r ace=%s messages_type=%s",
            user_id, effective_scope, self._config.ace_enabled,
            type(messages).__name__,
        )
        # Sanitize if always_sanitize is on, even in proxy mode
        if (
            not self._config.ace_enabled
            and self._config.privacy.always_sanitize
            and self._sanitizer
        ):
            messages = self._sanitize_messages(messages)

        # Try daemon path if available
        _fb = getattr(self, "_daemon_fallback", None)
        if _fb is not None and _fb.is_available:
            result = asyncio.run(
                _fb.try_curate(messages, user_id=user_id or "default")
            )
            if result is not None:
                return result
            # Daemon failed mid-request, fall through to direct mode

        if not self._config.ace_enabled or self._ingest_pipeline is None:
            logger.debug("Memory.add -> proxy mode (ACE off or pipeline=None)")
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
        logger.debug("Memory.add -> ACE ingest pipeline")
        from memx.core.pipeline.ingest import IngestResult  # noqa: F811

        logger.debug("Memory.add -> IngestPipeline.process scope=%s", effective_scope)
        ingest_result: IngestResult = self._ingest_pipeline.process(
            messages,
            metadata=metadata,
            user_id=user_id,
            agent_id=agent_id,
            run_id=run_id,
            scope=effective_scope,
        )
        logger.debug(
            "Memory.add result: added=%d merged=%d skipped=%d fallback=%s errors=%s",
            ingest_result.bullets_added, ingest_result.bullets_merged,
            ingest_result.bullets_skipped, ingest_result.raw_fallback,
            ingest_result.errors,
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
        scope: Optional[str] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search memories. ACE mode uses RetrievalPipeline.

        Args:
            scope: Target scope for search filtering and boosting.
                   None -> only "global" (backward compatible).
                   "project:myapp" -> "project:myapp" + "global" merged.

        When daemon is enabled and available, routes through DaemonClient.
        Falls back to direct mode transparently on daemon failure.
        """
        logger.debug(
            "Memory.search query=%r user_id=%s scope=%r ace=%s limit=%d",
            query[:60], user_id, scope, self._config.ace_enabled, limit,
        )

        # Try daemon path if available
        _fb = getattr(self, "_daemon_fallback", None)
        if _fb is not None and _fb.is_available:
            result = asyncio.run(
                _fb.try_recall(query, user_id=user_id or "default", limit=limit)
            )
            if result is not None:
                # DaemonClient.recall returns list[dict], wrap to match search format
                return {"results": result}
            # Daemon failed mid-request, fall through to direct mode

        if not self._config.ace_enabled or self._retrieval_pipeline is None:
            logger.debug("Memory.search -> proxy mode")
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
        logger.debug("Memory.search -> ACE retrieval pipeline")
        try:
            bullets = self._load_bullets_for_search(user_id, agent_id)
            logger.debug("Memory.search loaded %d bullets for search", len(bullets))
            from memx.core.pipeline.retrieval import SearchResult

            search_result: SearchResult = self._retrieval_pipeline.search(
                query=query,
                bullets=bullets,
                user_id=user_id,
                agent_id=agent_id,
                limit=limit,
                filters=filters,
                scope=scope,
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

    def status(self, user_id: Optional[str] = None) -> dict[str, Any]:
        """Get MemX knowledge base statistics.

        Returns a dict with total count, section/knowledge_type distributions,
        average decay_weight, and ACE enabled flag.
        """
        kwargs: dict[str, Any] = {}
        if user_id:
            kwargs["user_id"] = user_id

        raw = self._ensure_mem0().get_all(**kwargs)
        memories = raw.get("memories", []) if isinstance(raw, dict) else []

        # Count distribution by memx_section
        sections: dict[str, int] = {}
        knowledge_types: dict[str, int] = {}
        total_decay = 0.0
        count = 0

        for mem in memories:
            if not isinstance(mem, dict):
                continue
            count += 1
            meta = mem.get("metadata", {})
            section = meta.get("memx_section", "general")
            ktype = meta.get("memx_knowledge_type", "knowledge")
            decay = meta.get("memx_decay_weight", 1.0)

            sections[section] = sections.get(section, 0) + 1
            knowledge_types[ktype] = knowledge_types.get(ktype, 0) + 1
            total_decay += float(decay) if decay is not None else 1.0

        avg_decay = round(total_decay / count, 2) if count > 0 else 0.0

        return {
            "total": count,
            "ace_enabled": self._config.ace_enabled,
            "sections": sections,
            "knowledge_types": knowledge_types,
            "avg_decay_weight": avg_decay,
        }

    def detect_conflicts(
        self, user_id: str | None = None
    ) -> list[Any]:
        """Detect contradictory memories in the knowledge base.

        Loads all memories, converts them to ExistingBullet format, and runs
        the ConflictDetector. Returns a list of Conflict objects.

        Returns an empty list if mem0 is unavailable or no conflicts exist.
        """
        from memx.core.engines.curator.conflict import ConflictDetector
        from memx.core.engines.curator.engine import ExistingBullet

        kwargs: dict[str, Any] = {}
        if user_id:
            kwargs["user_id"] = user_id

        try:
            raw = self._ensure_mem0().get_all(**kwargs)
        except RuntimeError:
            logger.warning("mem0 unavailable; cannot detect conflicts")
            return []

        memories_raw = raw.get("memories", []) if isinstance(raw, dict) else []

        existing: list[ExistingBullet] = []
        for mem in memories_raw:
            if not isinstance(mem, dict):
                continue
            meta = mem.get("metadata", {})
            existing.append(
                ExistingBullet(
                    bullet_id=mem.get("id", ""),
                    content=mem.get("memory", ""),
                    scope=meta.get("memx_scope", "global")
                    if isinstance(meta, dict)
                    else "global",
                    metadata=meta if isinstance(meta, dict) else {},
                )
            )

        detector = ConflictDetector(self._config.curator)
        result = detector.detect(existing)
        return result.conflicts

    def export(
        self,
        format: str = "json",
        scope: Optional[str] = None,
    ) -> dict[str, Any] | str:
        """Export all memories in the specified format.

        Args:
            format: Output format — "json" or "markdown".
            scope:  If given, only export memories matching this scope.

        Returns:
            A dict (JSON envelope) or a markdown string depending on *format*.
        Raises:
            ValueError: If *format* is unsupported.
        """
        logger.debug("Memory.export format=%r scope=%r", format, scope)
        raw = self.get_all()
        memories: list[dict[str, Any]] = raw.get("results", [])
        logger.debug("Memory.export loaded %d raw memories", len(memories))

        if scope is not None:
            memories = [
                m
                for m in memories
                if isinstance(m, dict)
                and m.get("metadata", {}).get("memx_scope", "global") == scope
            ]
            logger.debug("Memory.export after scope filter: %d memories", len(memories))

        if format == "json":
            return {
                "version": "1.0",
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "total": len(memories),
                "memories": memories,
            }
        elif format == "markdown":
            return self._export_markdown(memories)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def import_data(
        self,
        data: dict[str, Any] | str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Import memories from an export payload.

        Args:
            data:   A JSON dict/string produced by ``export(format="json")``.
            format: Only ``"json"`` is supported for import.

        Returns:
            Summary dict ``{"imported": N, "skipped": N, "merged": N}``.
        Raises:
            ValueError: On unsupported format or unparseable JSON.
        """
        logger.debug("Memory.import_data format=%r data_type=%s", format, type(data).__name__)
        if format != "json":
            raise ValueError(f"Unsupported import format: {format}")

        # Parse string payloads
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ValueError("Import data must be a JSON object")

        memories: list[dict[str, Any]] = data.get("memories", [])

        imported = 0
        skipped = 0
        merged = 0

        # Process in batches of 500 for large imports
        batch_size = 500
        for batch_start in range(0, max(len(memories), 1), batch_size):
            batch = memories[batch_start : batch_start + batch_size]
            if not batch:
                break
            for mem in batch:
                if not isinstance(mem, dict):
                    skipped += 1
                    continue

                content = mem.get("memory", "")
                if not content or not content.strip():
                    skipped += 1
                    continue

                # Reconstruct metadata; use defaults for legacy payloads
                raw_meta = mem.get("metadata", {})
                if not isinstance(raw_meta, dict):
                    raw_meta = {}

                # Build CandidateBullet for Curator dedup (ACE path)
                if self._config.ace_enabled and self._ingest_pipeline is not None:
                    from memx.core.types import (
                        CandidateBullet,
                        SourceType,
                    )
                    from memx.core.utils.bullet_factory import BulletFactory

                    bullet_meta = BulletFactory.from_export_payload(mem)
                    meta_obj = bullet_meta["metadata"]

                    candidate = CandidateBullet(
                        content=content,
                        distilled_rule=meta_obj.distilled_rule,
                        section=meta_obj.section,
                        knowledge_type=meta_obj.knowledge_type,
                        source_type=SourceType.IMPORT,
                        instructivity_score=meta_obj.instructivity_score,
                        key_entities=list(meta_obj.key_entities),
                        related_tools=list(meta_obj.related_tools),
                        related_files=list(meta_obj.related_files),
                        tags=list(meta_obj.tags),
                        scope=meta_obj.scope,
                    )

                    # Run through Curator if available
                    curator = getattr(self._ingest_pipeline, "_curator", None)
                    if curator is not None:
                        existing_bullets = self._load_existing_for_import()
                        curate_result = curator.curate([candidate], existing_bullets)

                        if curate_result.to_skip:
                            skipped += 1
                            continue
                        if curate_result.to_merge:
                            # Update existing memory with new content
                            merge_info = curate_result.to_merge[0]
                            try:
                                self.update(
                                    merge_info.existing.bullet_id,
                                    merge_info.candidate.content,
                                )
                                merged += 1
                            except Exception:
                                skipped += 1
                            continue

                    # Insert via add()
                    try:
                        mem0_meta = BulletFactory.to_mem0_metadata(meta_obj)
                        self._ensure_mem0().add(
                            content,
                            metadata=mem0_meta,
                        )
                        imported += 1
                    except Exception:
                        skipped += 1
                else:
                    # Non-ACE path: direct insert via mem0
                    try:
                        self._ensure_mem0().add(content, metadata=raw_meta)
                        imported += 1
                    except Exception:
                        skipped += 1

        logger.debug(
            "Memory.import_data result: imported=%d skipped=%d merged=%d",
            imported, skipped, merged,
        )
        return {"imported": imported, "skipped": skipped, "merged": merged}

    def run_decay_sweep(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        archive: bool = True,
    ) -> dict[str, Any]:
        """Run a temporal decay sweep across all memories.

        Computes new decay weights for every bullet and persists changes
        back to the mem0 backend.  Bullets that fall below the archive
        threshold are optionally deleted.

        Args:
            user_id:  Scope the sweep to a specific user.
            agent_id: Scope the sweep to a specific agent.
            archive:  If True (default), delete bullets below archive threshold.

        Returns:
            Summary dict with ``updated``, ``archived``, ``permanent``,
            ``unchanged``, and ``errors`` counts.
        """
        from memx.core.engines.decay.engine import BulletDecayInfo, DecayEngine

        if not self._config.ace_enabled:
            logger.debug("run_decay_sweep: ACE disabled, nothing to do")
            return {"updated": 0, "archived": 0, "permanent": 0, "unchanged": 0, "errors": []}

        mem0 = self._ensure_mem0()
        decay_engine = DecayEngine(config=self._config.decay)

        # Step 1: Load all memories
        kwargs: dict[str, Any] = {}
        if user_id:
            kwargs["user_id"] = user_id
        if agent_id:
            kwargs["agent_id"] = agent_id

        raw = mem0.get_all(**kwargs)
        memories = raw.get("memories", []) if isinstance(raw, dict) else []
        logger.debug("run_decay_sweep: loaded %d memories", len(memories))

        if not memories:
            return {"updated": 0, "archived": 0, "permanent": 0, "unchanged": 0, "errors": []}

        # Step 2: Build BulletDecayInfo list
        bullets: list[BulletDecayInfo] = []
        for mem in memories:
            if not isinstance(mem, dict):
                continue
            meta = mem.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}

            created_str = meta.get("memx_created_at")
            if created_str and isinstance(created_str, str):
                try:
                    created_at = datetime.fromisoformat(created_str)
                except ValueError:
                    created_at = datetime.now(timezone.utc)
            else:
                created_at = datetime.now(timezone.utc)

            recall_count = meta.get("memx_recall_count", 0)
            if not isinstance(recall_count, int):
                try:
                    recall_count = int(recall_count)
                except (TypeError, ValueError):
                    recall_count = 0

            current_weight = meta.get("memx_decay_weight", 1.0)
            if not isinstance(current_weight, (int, float)):
                try:
                    current_weight = float(current_weight)
                except (TypeError, ValueError):
                    current_weight = 1.0

            last_recall_str = meta.get("memx_last_recall")
            last_recall = None
            if last_recall_str and isinstance(last_recall_str, str):
                try:
                    last_recall = datetime.fromisoformat(last_recall_str)
                except ValueError:
                    pass

            bullets.append(BulletDecayInfo(
                bullet_id=mem.get("id", ""),
                created_at=created_at,
                recall_count=recall_count,
                last_recall=last_recall,
                current_weight=current_weight,
            ))

        # Step 3: Run decay sweep
        sweep_result = decay_engine.sweep(bullets)
        logger.debug(
            "run_decay_sweep: sweep done — updated=%d archived=%d permanent=%d unchanged=%d",
            sweep_result.updated, sweep_result.archived,
            sweep_result.permanent, sweep_result.unchanged,
        )

        # Step 4: Persist weight changes and archive
        errors: list[str] = list(sweep_result.errors)
        actual_archived = 0
        actual_updated = 0

        for bullet_id, decay_result in sweep_result.details.items():
            if not bullet_id:
                continue

            # Find the original memory to compare
            original = next(
                (b for b in bullets if b.bullet_id == bullet_id), None,
            )
            if original is None:
                continue

            # Archive (delete) if below threshold
            if archive and decay_result.should_archive:
                try:
                    mem0.delete(bullet_id)
                    actual_archived += 1
                    logger.debug("run_decay_sweep: archived (deleted) %s", bullet_id)
                except Exception as e:
                    errors.append(f"archive {bullet_id}: {e}")
                continue

            # Update weight if changed
            if abs(decay_result.weight - original.current_weight) > 0.0001:
                try:
                    # mem0 update() takes (memory_id, data_str) — we update
                    # the metadata via the underlying mem0 API
                    mem0_mem = mem0.get(bullet_id)
                    if isinstance(mem0_mem, dict):
                        existing_meta = mem0_mem.get("metadata", {})
                        if isinstance(existing_meta, dict):
                            existing_meta["memx_decay_weight"] = round(decay_result.weight, 6)
                            # Persist: re-add with updated metadata
                            content = mem0_mem.get("memory", "")
                            mem0.update(bullet_id, content)
                    actual_updated += 1
                    logger.debug(
                        "run_decay_sweep: updated %s weight %.4f -> %.4f",
                        bullet_id, original.current_weight, decay_result.weight,
                    )
                except Exception as e:
                    errors.append(f"update {bullet_id}: {e}")

        summary = {
            "updated": actual_updated,
            "archived": actual_archived,
            "permanent": sweep_result.permanent,
            "unchanged": sweep_result.unchanged,
            "errors": errors,
        }
        logger.info("run_decay_sweep complete: %s", summary)
        return summary

    # ---- Internal ----------------------------------------------------------

    def _export_markdown(self, memories: list[dict[str, Any]]) -> str:
        """Render memories as a human-readable Markdown document.

        Groups memories by their ``memx_section`` metadata field and formats
        each entry with score, knowledge type, content snippet, and short ID.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        total = len(memories)
        lines: list[str] = [
            "# MemX Knowledge Export",
            f"> Exported: {now_iso} | Total: {total} memories",
            "",
        ]

        # Group by section
        section_map: dict[str, list[dict[str, Any]]] = {}
        for mem in memories:
            if not isinstance(mem, dict):
                continue
            meta = mem.get("metadata", {})
            if not isinstance(meta, dict):
                meta = {}
            section = meta.get("memx_section", "general")
            section_map.setdefault(section, []).append(mem)

        for section_name, section_mems in sorted(section_map.items()):
            title = section_name.replace("_", " ").title()
            lines.append(f"## {title} ({len(section_mems)})")
            for mem in section_mems:
                meta = mem.get("metadata", {})
                if not isinstance(meta, dict):
                    meta = {}
                score = meta.get("memx_instructivity_score", 50.0)
                ktype = meta.get("memx_knowledge_type", "knowledge")
                content = mem.get("memory", "")
                mid = mem.get("id", "")
                short_id = mid[:6] if mid else "------"
                lines.append(
                    f"- [{score:.2f}] **{ktype}** | {content} `{short_id}`"
                )
            lines.append("")

        return "\n".join(lines)

    def _load_existing_for_import(self) -> list[Any]:
        """Load existing bullets from mem0 for Curator dedup during import.

        Returns a list of ExistingBullet objects. Returns empty list
        if mem0 get_all is not available or fails.
        """
        from memx.core.engines.curator.engine import ExistingBullet

        try:
            raw = self._ensure_mem0().get_all()
            all_mems = raw.get("results", []) if isinstance(raw, dict) else []
            existing: list[ExistingBullet] = []
            for mem in all_mems:
                if not isinstance(mem, dict):
                    continue
                mem_meta = mem.get("metadata", {})
                existing.append(
                    ExistingBullet(
                        bullet_id=mem.get("id", ""),
                        content=mem.get("memory", ""),
                        scope=mem_meta.get("memx_scope", "global")
                        if isinstance(mem_meta, dict)
                        else "global",
                        metadata=mem_meta if isinstance(mem_meta, dict) else {},
                    )
                )
            return existing
        except Exception as e:
            logger.warning("Failed to load existing bullets for import: %s", e)
            return []

    def _load_bullets_for_search(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> list[Any]:
        """Load all bullets from mem0 and convert to BulletForSearch for GeneratorEngine.

        Returns an empty list if mem0 is not available or get_all fails.
        """
        from memx.core.engines.generator.engine import BulletForSearch
        from memx.core.engines.generator.metadata_matcher import MetadataInfo
        from memx.core.utils.bullet_factory import BulletFactory

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
                    scope=bullet_meta.scope,
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

    @property
    def daemon_available(self) -> bool:
        """Whether the daemon is currently available for IPC calls."""
        _fb = getattr(self, "_daemon_fallback", None)
        if _fb is None:
            return False
        return _fb.is_available
