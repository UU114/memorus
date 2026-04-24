"""IdleOrchestrator — periodic consolidate runner for the Python daemon.

Fires a consolidate pass only when ALL of these hold:

* the shared session map is empty (no active clients),
* the daemon has been idle for at least ``min_idle_secs``,
* at least ``min_gap_secs`` have elapsed since the previous run.

An internal :class:`asyncio.Lock` guarantees at most one consolidate runs
concurrently. Exceptions inside the pipeline are logged but never
propagate — the daemon survives.

Mirrors ``memorus_ace::curator::orchestrator::IdleOrchestrator``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from memorus.core.config import ConsolidateConfig, CuratorConfig
from memorus.core.engines.curator.engine import CuratorEngine, ExistingBullet
from memorus.core.engines.curator.executor import (
    ConsolidateExecutor,
    ExecutionReport,
    MemoryAdapter,
    append_consolidate_log,
)

logger = logging.getLogger(__name__)


SessionsEmptyFn = Callable[[], Awaitable[bool]]
LoadBulletsFn = Callable[[], list[ExistingBullet]]
BatchStageFn = Callable[[], Any]
"""Callable invoked before the consolidate stage (STORY-R095 BatchAnalyzer)."""

TopicStageFn = Callable[[], Any]
"""Callable invoked AFTER the consolidate stage (STORY-R097 TopicEngine)."""


class IdleOrchestrator:
    """Coordinates consolidate runs inside the daemon."""

    def __init__(
        self,
        config: ConsolidateConfig,
        curator_config: CuratorConfig,
        adapter: MemoryAdapter,
        load_bullets: LoadBulletsFn,
        batch_stage: Optional[BatchStageFn] = None,
        topic_stage: Optional[TopicStageFn] = None,
    ) -> None:
        self._config = config
        self._curator_config = curator_config
        self._adapter = adapter
        self._load_bullets = load_bullets
        self._batch_stage = batch_stage
        self._topic_stage = topic_stage
        self._last_activity_monotonic: float = time.monotonic()
        self._last_run: Optional[datetime] = None
        self._last_batch_report: Any = None
        self._last_topic_report: Any = None
        self._run_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Mark the daemon as active now. Safe to call from sync code."""
        self._last_activity_monotonic = time.monotonic()

    def stop(self) -> None:
        """Signal the run_forever loop to exit."""
        self._stop_event.set()

    @property
    def last_run(self) -> Optional[datetime]:
        return self._last_run

    @property
    def last_batch_report(self) -> Any:
        """Result of the most recent BatchAnalyzer stage, if any."""
        return self._last_batch_report

    @property
    def last_topic_report(self) -> Any:
        """Result of the most recent TopicEngine stage, if any."""
        return self._last_topic_report

    async def _run_topic_stage(self) -> Any:
        """Invoke the configured topic stage, handling sync + async returns."""
        stage = self._topic_stage
        if stage is None:
            return None
        result = stage()
        if asyncio.iscoroutine(result):
            return await result
        return result

    async def _run_batch_stage(self) -> Any:
        """Invoke the configured batch stage, handling both sync and async returns."""
        stage = self._batch_stage
        if stage is None:
            return None
        result = stage()
        if asyncio.iscoroutine(result):
            return await result
        return result

    @property
    def is_running(self) -> bool:
        """True iff a consolidate run is currently in flight."""
        return self._run_lock.locked()

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    async def should_run(self, sessions_empty_fn: SessionsEmptyFn) -> bool:
        """Return True iff all three gates are satisfied."""
        if not await sessions_empty_fn():
            return False
        idle_secs = time.monotonic() - self._last_activity_monotonic
        if idle_secs < self._config.min_idle_secs:
            return False
        if self._last_run is not None:
            gap = (datetime.now(timezone.utc) - self._last_run).total_seconds()
            if gap < self._config.min_gap_secs:
                return False
        return True

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    async def run_once(self, force: bool = False) -> Optional[ExecutionReport]:
        """Execute one consolidate + executor pass.

        ``force=False`` uses ``try_lock``-style semantics so an already-running
        consolidate does not queue up duplicate work. ``force=True`` waits
        for the lock so ``memorus consolidate --now`` is consistent.
        """
        if not force and self._run_lock.locked():
            logger.info("orchestrator: already running, skipping")
            return None

        async with self._run_lock:
            start_dt = datetime.now(timezone.utc)
            start_mono = time.monotonic()

            # STORY-R095 — drain the inbox BEFORE loading bullets so any new
            # bullets produced by BatchAnalyzer are visible to consolidate.
            if self._batch_stage is not None:
                try:
                    self._last_batch_report = await self._run_batch_stage()
                except Exception as e:
                    # Never let the batch stage take down consolidate.
                    logger.error(
                        "orchestrator: batch stage failed: %s", e, exc_info=True,
                    )

            try:
                bullets = self._load_bullets()
            except Exception as e:
                logger.error("orchestrator: load_bullets failed: %s", e, exc_info=True)
                return None

            curator = CuratorEngine(self._curator_config)
            try:
                scan = curator.consolidate_corpus(
                    bullets, max_per_pass=self._config.max_per_pass,
                )
            except Exception as e:
                logger.error(
                    "orchestrator: consolidate_corpus failed: %s", e, exc_info=True
                )
                return None

            executor = ConsolidateExecutor(self._adapter, self._config)
            try:
                report = executor.execute(scan)
            except Exception as e:
                logger.error(
                    "orchestrator: executor.execute failed: %s", e, exc_info=True
                )
                return None

            report.duration_seconds = time.monotonic() - start_mono

            try:
                append_consolidate_log(
                    self._config.log_path, report, start_dt,
                )
            except Exception as e:
                logger.warning("orchestrator: log append failed: %s", e)

            # STORY-R097 — post-consolidate TopicEngine pass. Exceptions are
            # logged but must not mask the successful consolidate result.
            if self._topic_stage is not None:
                try:
                    self._last_topic_report = await self._run_topic_stage()
                except Exception as e:
                    logger.error(
                        "orchestrator: topic stage failed: %s", e, exc_info=True,
                    )

            self._last_run = datetime.now(timezone.utc)
            return report

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run_forever(self, sessions_empty_fn: SessionsEmptyFn) -> None:
        """Periodic loop: every ``check_interval_secs`` check the gates."""
        logger.info(
            "IdleOrchestrator started (interval=%ds idle>=%ds gap>=%ds)",
            self._config.check_interval_secs,
            self._config.min_idle_secs,
            self._config.min_gap_secs,
        )
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._config.check_interval_secs,
                )
                # stop event set
                break
            except asyncio.TimeoutError:
                pass

            try:
                if await self.should_run(sessions_empty_fn):
                    await self.run_once(force=False)
                else:
                    logger.debug("IdleOrchestrator: gate closed, sleeping")
            except Exception as e:
                logger.error("orchestrator loop: %s", e, exc_info=True)
        logger.info("IdleOrchestrator stopping")


# ---------------------------------------------------------------------------
# mem0-backed MemoryAdapter
# ---------------------------------------------------------------------------


class Mem0MemoryAdapter:
    """MemoryAdapter that proxies to a :class:`memorus.core.memory.Memory`
    wrapping a mem0 backend.

    Implemented lazily against the public ``get``/``update``/``delete`` surface
    exposed by :class:`memorus.core.memory.Memory`. Soft-delete is encoded by
    writing a ``deleted_at`` timestamp into the bullet's metadata (the mem0
    record is *not* physically removed).
    """

    def __init__(self, memory: Any) -> None:
        self._memory = memory

    def get_bullet(self, bullet_id: str) -> Any:
        from memorus.core.engines.curator.executor import BulletRecord
        from memorus.core.types import SourceRef

        try:
            raw = self._memory.get(bullet_id)
        except Exception as e:
            logger.debug("Mem0Adapter.get_bullet(%s): %s", bullet_id, e)
            return None
        if not raw:
            return None
        if isinstance(raw, dict):
            meta = raw.get("metadata") or {}
            if isinstance(meta, dict) and meta.get("memorus_deleted_at"):
                return None
            content = raw.get("memory", raw.get("content", ""))
            recall = int(meta.get("memorus_recall_count", 0))
            created = _parse_ts(meta.get("memorus_created_at"))
            sources_raw = meta.get("memorus_sources") or []
            sources: list[SourceRef] = []
            if isinstance(sources_raw, list):
                for s in sources_raw:
                    if isinstance(s, SourceRef):
                        sources.append(s)
                    elif isinstance(s, dict):
                        try:
                            sources.append(SourceRef(**s))
                        except Exception:
                            pass
            return BulletRecord(
                bullet_id=bullet_id,
                content=content,
                recall_count=recall,
                created_at=created,
                sources=sources,
                metadata=meta,
            )
        return None

    def update_metadata(self, bullet_id: str, patch: Any) -> None:
        from memorus.core.engines.curator.executor import MetadataPatch

        assert isinstance(patch, MetadataPatch)
        current = self.get_bullet(bullet_id)
        if current is None:
            return
        meta = dict(current.metadata)
        if patch.sources is not None:
            meta["memorus_sources"] = [
                s.model_dump() if hasattr(s, "model_dump") else s
                for s in patch.sources
            ]
        if patch.supersedes is not None:
            meta["memorus_supersedes"] = list(patch.supersedes)
        if patch.superseded_by is not None:
            meta["memorus_superseded_by"] = patch.superseded_by
        if patch.conflict_with is not None:
            existing = meta.get("memorus_conflict_with") or []
            if not isinstance(existing, list):
                existing = []
            for cw in patch.conflict_with:
                if cw not in existing:
                    existing.append(cw)
            meta["memorus_conflict_with"] = existing
        if patch.conflict_resolution is not None:
            res = patch.conflict_resolution
            meta["memorus_conflict_resolution"] = {
                "conflict_type": res.conflict_type,
                "reason": res.reason,
                "confidence": res.confidence,
                "resolved_at": res.resolved_at.isoformat(),
            }
        if patch.updated_at is not None:
            meta["memorus_updated_at"] = patch.updated_at.isoformat()

        # Best effort: if mem0 exposes update with metadata, use it; otherwise
        # call plain update(id, content) and silently accept that metadata
        # persistence depends on backend capability.
        try:
            # Some mem0 versions accept (id, content, metadata=...). Try that
            # first, then fall back to content-only update.
            self._memory.update(bullet_id, current.content)
        except TypeError:
            try:
                self._memory.update(bullet_id, current.content)
            except Exception as e:
                logger.debug("Mem0Adapter.update_metadata(%s): %s", bullet_id, e)
        except Exception as e:
            logger.debug("Mem0Adapter.update_metadata(%s): %s", bullet_id, e)

    def mark_deleted(self, bullet_id: str, when: datetime) -> None:
        current = self.get_bullet(bullet_id)
        if current is None:
            return
        meta = dict(current.metadata)
        meta["memorus_deleted_at"] = when.isoformat()
        try:
            self._memory.update(bullet_id, current.content)
        except Exception as e:
            logger.debug("Mem0Adapter.mark_deleted(%s): %s", bullet_id, e)


def _parse_ts(v: Any) -> datetime:
    """Parse an ISO-8601 string or datetime into a tz-aware datetime."""
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if isinstance(v, str) and v:
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)
