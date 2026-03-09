"""IngestPipeline -- orchestrates the add() processing flow.

Flow: Raw Input -> Sanitize -> Reflector -> (Curator) -> mem0.add

Each stage has independent error handling; failure in any stage triggers
graceful fallback rather than crashing the pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from memx.core.types import BulletMetadata, CandidateBullet, InteractionEvent
from memx.core.utils.bullet_factory import BulletFactory

logger = logging.getLogger(__name__)


@dataclass
class IngestResult:
    """Result of an ingest pipeline run."""

    bullets_added: int = 0
    bullets_merged: int = 0
    bullets_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    raw_fallback: bool = False


class IngestPipeline:
    """Pipeline for processing add() operations through Reflector and Curator.

    Flow: Raw Input -> Sanitize -> Reflector -> (Curator) -> mem0.add

    When Reflector produces no candidates (e.g. trivial conversation) or
    when Reflector fails, the pipeline falls back to a raw mem0.add so
    that no data is silently lost.
    """

    def __init__(
        self,
        reflector: Any,  # ReflectorEngine
        sanitizer: Any = None,  # PrivacySanitizer
        curator: Any = None,  # CuratorEngine (Sprint 2)
        mem0_add_fn: Optional[Callable[..., Any]] = None,
        mem0_get_all_fn: Optional[Callable[..., Any]] = None,
        mem0_update_fn: Optional[Callable[..., Any]] = None,
    ):
        self._reflector = reflector
        self._sanitizer = sanitizer
        self._curator = curator
        self._mem0_add = mem0_add_fn
        self._mem0_get_all = mem0_get_all_fn
        self._mem0_update = mem0_update_fn

    def process(
        self,
        messages: Any,
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        run_id: Optional[str] = None,
        scope: Optional[str] = None,
        **kwargs: Any,
    ) -> IngestResult:
        """Process messages through the ingest pipeline.

        Args:
            messages:  Raw input messages.
            metadata:  Optional metadata to attach.
            user_id:   Optional user ID.
            agent_id:  Optional agent ID.
            run_id:    Optional run ID.
            scope:     Hierarchical scope (default "global").

        Returns an IngestResult summarising what happened.  Never raises --
        all errors are captured in IngestResult.errors.
        """
        result = IngestResult()
        effective_scope = scope or "global"
        logger.debug(
            "IngestPipeline.process scope=%r user_id=%s messages_type=%s",
            effective_scope, user_id, type(messages).__name__,
        )

        if not messages:
            logger.debug("IngestPipeline.process -> empty messages, returning early")
            return result

        # Step 0: Privacy sanitisation (independent of Reflector)
        logger.debug("IngestPipeline step 0: sanitization (sanitizer=%s)", self._sanitizer is not None)
        sanitized_messages = self._run_sanitizer(messages)

        # Step 1: Parse to InteractionEvent
        event = self._parse_event(sanitized_messages, metadata or {})
        logger.debug(
            "IngestPipeline step 1: parsed event user_msg_len=%d asst_msg_len=%d",
            len(event.user_message), len(event.assistant_message),
        )

        # Step 2: Reflector
        logger.debug("IngestPipeline step 2: reflector.reflect()")
        try:
            candidates = self._reflector.reflect(event)
            logger.debug("IngestPipeline step 2: reflector produced %d candidate(s)", len(candidates))
        except Exception as e:
            logger.warning("Reflector failed, falling back to raw add: %s", e)
            self._raw_add(messages, metadata, user_id, agent_id, run_id, **kwargs)
            result.raw_fallback = True
            return result

        if not candidates:
            # No patterns detected -- do raw add
            logger.debug("IngestPipeline step 2: 0 candidates -> raw_fallback")
            self._raw_add(
                sanitized_messages, metadata, user_id, agent_id, run_id, **kwargs
            )
            result.raw_fallback = True
            return result

        # Apply scope to all candidates
        for candidate in candidates:
            candidate.scope = effective_scope

        # Step 3: Curator deduplication (skip if not available)
        if self._curator:
            logger.debug("IngestPipeline step 3: curator dedup (candidates=%d)", len(candidates))
            try:
                existing = self._load_existing_bullets(user_id, agent_id)
                logger.debug("IngestPipeline step 3: loaded %d existing bullets", len(existing))
                curate_result = self._curator.curate(candidates, existing)
                # Replace candidates with curated partition
                logger.debug(
                    "IngestPipeline step 3: curate -> add=%d merge=%d skip=%d",
                    len(curate_result.to_add), len(curate_result.to_merge),
                    len(curate_result.to_skip),
                )
                candidates = curate_result.to_add
                result.bullets_merged += len(curate_result.to_merge)
                result.bullets_skipped += len(curate_result.to_skip)
                # Handle merge candidates: use merger strategy via mem0 update
                for merge in curate_result.to_merge:
                    try:
                        self._handle_merge(merge, metadata, user_id, agent_id, run_id)
                    except Exception as merge_err:
                        logger.warning("Merge failed for bullet: %s", merge_err)
                        result.errors.append(str(merge_err))
            except Exception as e:
                logger.warning("Curator failed, inserting all candidates: %s", e)

        # Step 4: Write candidates to mem0
        logger.debug("IngestPipeline step 4: writing %d candidate(s) to mem0", len(candidates))
        for bullet in candidates:
            try:
                bullet_meta = BulletMetadata(
                    section=bullet.section,
                    knowledge_type=bullet.knowledge_type,
                    instructivity_score=bullet.instructivity_score,
                    source_type=bullet.source_type,
                    related_tools=bullet.related_tools,
                    key_entities=bullet.key_entities,
                    related_files=bullet.related_files,
                    tags=bullet.tags,
                    scope=bullet.scope,
                )
                mem0_meta = BulletFactory.to_mem0_metadata(bullet_meta)
                merged_meta = {**(metadata or {}), **mem0_meta}

                if self._mem0_add:
                    logger.debug(
                        "IngestPipeline step 4: mem0.add content=%r section=%s score=%.1f",
                        bullet.content[:60], bullet.section.value, bullet.instructivity_score,
                    )
                    self._mem0_add(
                        bullet.content,
                        user_id=user_id,
                        agent_id=agent_id,
                        run_id=run_id,
                        metadata=merged_meta,
                        **kwargs,
                    )
                result.bullets_added += 1
            except Exception as e:
                logger.warning("Failed to add bullet: %s", e)
                result.errors.append(str(e))

        logger.debug(
            "IngestPipeline.process done: added=%d merged=%d skipped=%d fallback=%s errors=%s",
            result.bullets_added, result.bullets_merged, result.bullets_skipped,
            result.raw_fallback, result.errors,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_sanitizer(self, messages: Any) -> Any:
        """Run privacy sanitizer on messages. Never raises."""
        if not self._sanitizer:
            logger.debug("IngestPipeline._run_sanitizer: no sanitizer, pass-through")
            return messages
        try:
            if isinstance(messages, str):
                res = self._sanitizer.sanitize(messages)
                logger.debug("IngestPipeline._run_sanitizer: str modified=%s", res.was_modified)
                return res.clean_content
            elif isinstance(messages, list):
                sanitized = []
                modified_count = 0
                for msg in messages:
                    if isinstance(msg, dict) and "content" in msg:
                        res = self._sanitizer.sanitize(msg["content"])
                        if res.was_modified:
                            modified_count += 1
                        sanitized.append({**msg, "content": res.clean_content})
                    else:
                        sanitized.append(msg)
                logger.debug(
                    "IngestPipeline._run_sanitizer: list(%d msgs), %d modified",
                    len(messages), modified_count,
                )
                return sanitized
            return messages
        except Exception as e:
            logger.warning("Sanitizer failed: %s", e)
            return messages

    @staticmethod
    def _parse_event(messages: Any, metadata: dict[str, Any]) -> InteractionEvent:
        """Parse add() input into InteractionEvent.

        Supports:
        - str: treated as user_message
        - list[dict]: split by role into user/assistant messages
        - other: str() coerced as user_message
        """
        if isinstance(messages, str):
            return InteractionEvent(
                user_message=messages,
                assistant_message="",
                metadata=metadata,
            )
        elif isinstance(messages, list):
            user_msgs: list[str] = []
            assistant_msgs: list[str] = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        user_msgs.append(content)
                    elif role == "assistant":
                        assistant_msgs.append(content)
            return InteractionEvent(
                user_message="\n".join(user_msgs),
                assistant_message="\n".join(assistant_msgs),
                metadata=metadata,
            )
        return InteractionEvent(
            user_message=str(messages),
            assistant_message="",
            metadata=metadata,
        )

    def _load_existing_bullets(
        self,
        user_id: Optional[str],
        agent_id: Optional[str],
    ) -> list[Any]:
        """Load existing bullets from mem0 for Curator comparison.

        Returns a list of ExistingBullet objects. Returns empty list
        if mem0 get_all is not available or fails.
        """
        from memx.core.engines.curator.engine import ExistingBullet

        if not hasattr(self, "_mem0_get_all") or self._mem0_get_all is None:
            return []

        try:
            kwargs: dict[str, Any] = {}
            if user_id:
                kwargs["user_id"] = user_id
            if agent_id:
                kwargs["agent_id"] = agent_id
            raw = self._mem0_get_all(**kwargs)
            memories = raw.get("memories", []) if isinstance(raw, dict) else []
            existing: list[ExistingBullet] = []
            for mem in memories:
                if isinstance(mem, dict):
                    mem_meta = mem.get("metadata", {})
                    existing.append(
                        ExistingBullet(
                            bullet_id=mem.get("id", ""),
                            content=mem.get("memory", ""),
                            scope=mem_meta.get("memx_scope", "global")
                            if isinstance(mem_meta, dict) else "global",
                            metadata=mem_meta,
                        )
                    )
            return existing
        except Exception as e:
            logger.warning("Failed to load existing bullets for Curator: %s", e)
            return []

    def _handle_merge(
        self,
        merge: Any,
        metadata: Optional[dict[str, Any]],
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
    ) -> None:
        """Handle a merge candidate by updating the existing bullet in mem0.

        Uses the candidate's improved content to update the existing memory entry.
        """
        if not hasattr(self, "_mem0_update") or self._mem0_update is None:
            # No update function available; log and skip
            logger.debug("No mem0 update function; merge skipped for %s", merge.existing.bullet_id)
            return

        self._mem0_update(merge.existing.bullet_id, merge.candidate.content)

    def _raw_add(
        self,
        messages: Any,
        metadata: Optional[dict[str, Any]],
        user_id: Optional[str],
        agent_id: Optional[str],
        run_id: Optional[str],
        **kwargs: Any,
    ) -> None:
        """Fallback: direct mem0 add without ACE processing."""
        logger.debug("IngestPipeline._raw_add: fallback mem0 add")
        if self._mem0_add:
            try:
                self._mem0_add(
                    messages,
                    user_id=user_id,
                    agent_id=agent_id,
                    run_id=run_id,
                    metadata=metadata,
                    **kwargs,
                )
            except Exception as e:
                logger.warning("Raw add failed: %s", e)
