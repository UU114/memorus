"""Rust-backed Memory adapters.

These classes present the SAME public API as
``memorus.core.memory.Memory`` / ``memorus.core.async_memory.AsyncMemory`` but
delegate all storage and ACE work to the compiled ``memorus_r`` extension
(``memorus_r.Memory`` / ``memorus_r.AsyncMemory``).

They are selected at import time via ``MEMORUS_BACKEND=rust`` (see
``memorus/__init__.py``); the default backend stays the pure-Python engine, so
importing this module has zero effect unless the env var is set.

The adapters are intentionally thin: argument names, defaults, and return
shapes mirror the pure-Python contract so existing callers do not need to
change. Where the binding's surface diverges from the pure-Python engine the
adapter normalizes the difference (message serialization, dict vs. string
config, ``filters`` parity) and any irreducible gap is documented inline.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _serialize_messages(messages: Any) -> str:
    """Flatten ``add()`` input into a single content string.

    Mirrors ``IngestPipeline._parse_event`` (the pure-Python distillation
    parse) so the Rust ``add(content, ...)`` receives the same text the
    Python path would have distilled:

    - ``str``: used verbatim.
    - ``list[dict]``: user-role contents joined by newlines, then
      assistant-role contents joined by newlines, the two blocks joined by a
      newline (empty blocks are dropped so a user-only list has no trailing
      separator).
    - anything else: coerced via ``str()``.
    """
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
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
        # Join each role block, then the two blocks; drop empties so a
        # user-only conversation does not gain a trailing newline.
        blocks = [
            block
            for block in ("\n".join(user_msgs), "\n".join(assistant_msgs))
            if block
        ]
        return "\n".join(blocks)
    return str(messages)


# mem0-native provider sub-config field maps. Inner ``config`` dicts use
# mem0-style keys; we project the known ones onto the Rust struct field names
# and route unknowns into ``extra`` (vector_store) or drop with a warning.
_VECTOR_STORE_FIELDS = {
    "provider",
    "collection_name",
    "url",
    "api_key",
    "path",
    "embedding_dims",
}
_LLM_FIELDS = {
    "provider",
    "model",
    "api_key",
    "base_url",
    "temperature",
    "max_tokens",
}
_EMBEDDING_FIELDS = {
    "provider",
    "model",
    "api_key",
    "base_url",
    "dimensions",
}


def _map_provider_block(
    block: dict[str, Any], known: set[str], collect_extra: bool
) -> dict[str, Any]:
    """Flatten a mem0-style ``{"provider": P, "config": {...}}`` block.

    Known inner-config keys are lifted to top level; unknown keys are collected
    into ``extra`` when ``collect_extra`` is True, otherwise dropped.
    """
    out: dict[str, Any] = {}
    if "provider" in block:
        out["provider"] = block["provider"]
    inner = block.get("config", {})
    extra: dict[str, Any] = {}
    if isinstance(inner, dict):
        for k, v in inner.items():
            if k in known:
                out[k] = v
            elif collect_extra:
                extra[k] = v
    if collect_extra and extra:
        out["extra"] = extra
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins on leaves)."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _translate_config(config: dict[str, Any] | str | None) -> dict[str, Any] | str | None:
    """Translate a public config into the nested Rust-backend schema.

    * ``None`` -> ``None`` (binding defaults).
    * ``str`` (file path) -> passed through unchanged; the Rust ``from_file``
      path plus serde aliases handle Python-era TOML key names. The flat Python
      dict format is a code-only shape, never serialized to a file.
    * ``dict`` -> built via ``MemorusConfig.from_dict(...).to_rust_dict()`` for
      the ACE/privacy/daemon surface, plus mem0-native ``memory`` keys:
      a top-level ``"memory"`` dict passes through verbatim, while mem0-style
      ``vector_store`` / ``llm`` / ``embedder`` blocks are mapped onto
      ``memory.vector_store`` / ``memory.llm`` / ``memory.embedding``. Any other
      unknown top-level key is logged at WARNING level (never silently dropped).
    """
    from memorus.core.config import MemorusConfig

    if config is None or isinstance(config, str):
        return config

    cfg = MemorusConfig.from_dict(config)
    rust: dict[str, Any] = cfg.to_rust_dict()

    # mem0-native memory surface. Recognized keys are mapped; unknowns warn.
    memory: dict[str, Any] = {}
    _ace_keys = {
        "ace_enabled",
        "reflector",
        "curator",
        "decay",
        "retrieval",
        "privacy",
        "integration",
        "daemon",
        "consolidate",
        "topics",
        "verification",
    }
    for key, value in config.items():
        if key in _ace_keys:
            continue
        if key == "ace" and isinstance(value, dict):
            # Already-canonical nested ace block: deep-merge over the
            # translated defaults so explicit input wins.
            rust["ace"] = _deep_merge(rust.get("ace") or {}, value)
        elif key == "memory" and isinstance(value, dict):
            # Pass a native Rust memory block through verbatim.
            memory.update(value)
        elif key == "vector_store" and isinstance(value, dict):
            memory["vector_store"] = _map_provider_block(
                value, _VECTOR_STORE_FIELDS, collect_extra=True
            )
        elif key == "llm" and isinstance(value, dict):
            memory["llm"] = _map_provider_block(
                value, _LLM_FIELDS, collect_extra=False
            )
        elif key == "embedder" and isinstance(value, dict):
            memory["embedding"] = _map_provider_block(
                value, _EMBEDDING_FIELDS, collect_extra=False
            )
        else:
            logger.warning(
                "memorus: config key %r has no Rust-backend equivalent; ignored",
                key,
            )

    if memory:
        rust["memory"] = memory
    return rust


def _build_search_config(
    limit: int | None,
    filters: dict[str, Any] | None,
    team_pool_path: str | None = None,
) -> dict[str, Any]:
    """Build the binding's search-config dict from public ``search()`` args.

    The binding's ``search(query, config=...)`` reads ``top_k``/``limit``,
    ``min_score``/``threshold``, ``filter``, and ``team_pool_path`` from the
    config dict. Only the keys the caller supplied are forwarded so binding
    defaults stay in effect.

    ``team_pool_path`` is the offline team-pool JSONL path used by the
    ``scope="team:NAME"`` shadow-merge read-path. ``MemorusConfig`` has no
    ``[team]`` section, so it is supplied per-search; the binding defaults to
    ``.ace/playbook.jsonl`` (relative to cwd) when it is omitted.
    """
    config: dict[str, Any] = {}
    if limit is not None:
        config["limit"] = limit
    if filters:
        config["filter"] = filters
    if team_pool_path is not None:
        config["team_pool_path"] = team_pool_path
    return config


def _extract_dropped_stale_count(results: list[Any]) -> int:
    """Read the verification policy's ``dropped_stale_count`` off the results.

    STORY-R103 stamps the same count onto every surviving result dict, so the
    first hit is sufficient. The key is absent for the ``flag`` / ``demote``
    policies and verifier-off runs, in which case the drop count is ``0``.
    """
    for result in results:
        if isinstance(result, dict):
            count = result.get("dropped_stale_count")
            if count:
                return int(count)
            return 0
    return 0


def _summarize_ingest(results: list[Any]) -> dict[str, Any]:
    """Derive an ``ace_ingest`` summary from the binding's per-item events.

    The binding's add result tags each item with an ``event`` (ADD / UPDATE /
    DELETE / NOOP). Map them onto the Python contract's ingest counters instead
    of reporting every row as "added": ADD -> added, UPDATE -> merged (curator
    supersede/merge), NOOP -> skipped (deduped). This mirrors the pure-Python
    ``IngestResult`` shape far more faithfully than ``len(results)``.
    """
    added = merged = skipped = 0
    for item in results:
        event = (item.get("event") if isinstance(item, dict) else None) or "ADD"
        event = str(event).upper()
        if event == "UPDATE":
            merged += 1
        elif event == "NOOP":
            skipped += 1
        else:  # ADD / DELETE / unknown -> count as added
            added += 1
    return {
        "bullets_added": added,
        "bullets_merged": merged,
        "bullets_skipped": skipped,
        "raw_fallback": False,
        "errors": [],
    }


class RustBackedMemory:
    """Sync Memory adapter delegating to ``memorus_r.Memory``.

    Presents the public surface of ``memorus.core.memory.Memory`` while the
    compiled Rust engine does the work.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        from memorus.core.config import MemorusConfig

        logger.debug("RustBackedMemory.__init__ config=%s", config)
        # Keep a pure-Python MemorusConfig so the public .config shape matches
        # the Python engine (callers read .config.ace_enabled, etc.).
        self._config = MemorusConfig.from_dict(config or {})

        import memorus_r

        # Translate the flat Python config into the nested Rust schema before
        # handing it to the binding; a file-path string or None passes through.
        self._inner = memorus_r.Memory(_translate_config(config))

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> RustBackedMemory:
        """Create from a config dict (mem0-compatible), mirroring Memory."""
        return cls(config=config_dict)

    # ---- Core API (mem0-compatible) ----------------------------------------

    def add(
        self,
        messages: Any,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        filters: dict[str, Any] | None = None,
        prompt: str | None = None,
        scope: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add memories, delegating to the binding's ``add``.

        ``messages`` is serialized to one content string the same way the
        pure-Python ingest path does. ``infer`` is driven by
        ``config.ace_enabled`` so the ACE distillation pipeline runs only when
        enabled. ``filters``/``prompt`` are accepted for signature parity but
        not forwarded (mem0 1.0.x dropped ``filters`` from ``add()`` and the
        binding has no ``prompt`` parameter).
        """
        content = _serialize_messages(messages)
        infer = self._config.ace_enabled
        result = self._inner.add(
            content,
            metadata,
            user_id,
            agent_id,
            run_id,
            infer,
            scope,
        )
        # The binding returns a dict with a 'results' list. In ACE mode the
        # Python contract also carries an 'ace_ingest' summary; surface the
        # closest equivalent the binding result allows.
        out: dict[str, Any] = dict(result) if isinstance(result, dict) else {"results": []}
        out.setdefault("results", [])
        if infer:
            results = out.get("results") or []
            out["ace_ingest"] = _summarize_ingest(
                results if isinstance(results, list) else []
            )
        return out

    def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        filters: dict[str, Any] | None = None,
        scope: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search memories, delegating to the binding's ``search``.

        ``limit`` and ``filters`` are mapped into the binding's search-config
        dict; ``scope`` is forwarded as-is. For a non-team scope the binding
        AND-combines it into the local filter; for ``scope="team:NAME"`` the
        binding runs a READ-ONLY shadow merge of the local store with the
        offline team pool (the local store is never written), tagging each
        result dict with ``source`` ("local" / "git-fallback") and
        ``is_mandatory``. The optional ``team_pool_path`` kwarg overrides the
        default ``.ace/playbook.jsonl`` team-pool location. The binding returns
        a list of result dicts; we wrap it in the ``{"results": [...]}``
        envelope the Python contract uses, adding an ``ace_search`` summary when
        ACE is enabled.
        """
        config = _build_search_config(limit, filters, kwargs.get("team_pool_path"))
        results = self._inner.search(query, config, None, scope)
        results_list = list(results) if results is not None else []
        # The binding stamps verified_status / trust_score (STORY-R102) and
        # dropped_stale_count (STORY-R103) onto each result dict when a
        # verifier is active; they pass through this envelope untouched.
        out: dict[str, Any] = {"results": results_list}
        if self._config.ace_enabled:
            ace_search: dict[str, Any] = {
                "mode": "rust",
                "total_candidates": len(results_list),
            }
            # STORY-R103 — surface the policy's drop count. The binding stamps
            # the same value on every surviving result; lift it from the first
            # one so callers see it without scanning the list. Absent (flag /
            # demote / verifier-off) means zero.
            dropped = _extract_dropped_stale_count(results_list)
            if dropped:
                ace_search["dropped_stale_count"] = dropped
            out["ace_search"] = ace_search
        return out

    def get_all(self, **kwargs: Any) -> dict[str, Any]:
        """Get all memories, wrapped in the ``{"memories": [...]}`` envelope."""
        user_id = kwargs.get("user_id")
        agent_id = kwargs.get("agent_id")
        run_id = kwargs.get("run_id")
        limit = kwargs.get("limit")
        results = self._inner.get_all(user_id, agent_id, run_id, limit)
        return {"memories": list(results) if results is not None else []}

    def get(self, memory_id: str) -> dict[str, Any]:
        """Get a single memory by ID."""
        return self._inner.get(memory_id)

    def update(self, memory_id: str, data: str) -> Any:
        """Update a memory's content by ID."""
        return self._inner.update(memory_id, data)

    def delete(self, memory_id: str) -> None:
        """Delete a single memory by ID."""
        return self._inner.delete(memory_id)

    def delete_all(self, **kwargs: Any) -> None:
        """Delete all memories matching the given scoping kwargs."""
        return self._inner.delete_all(
            kwargs.get("user_id"),
            kwargs.get("agent_id"),
            kwargs.get("run_id"),
        )

    def history(self, memory_id: str) -> Any:
        """Get modification history of a memory."""
        return self._inner.history(memory_id)

    def reset(self) -> None:
        """Reset all memories."""
        return self._inner.reset()

    # ---- ACE-specific methods ----------------------------------------------

    def status(self, user_id: str | None = None) -> dict[str, Any]:
        """Knowledge-base statistics (shape matches the binding directly)."""
        return self._inner.status(user_id)

    def detect_conflicts(self, user_id: str | None = None) -> list[Any]:
        """Detect contradictory memories (binding shape passes through)."""
        return self._inner.detect_conflicts(user_id)

    def consolidate_scan(
        self,
        max_per_pass: int = 5000,
        *,
        user_id: str | None = None,
    ) -> Any:
        """Scan the corpus for merge/conflict candidates."""
        return self._inner.consolidate_scan(max_per_pass, user_id)

    def export(
        self,
        format: str = "json",
        scope: str | None = None,
    ) -> dict[str, Any] | str:
        """Export the corpus (binding returns the same shapes)."""
        return self._inner.export(format, scope)

    def import_data(
        self,
        data: dict[str, Any] | str,
        format: str = "json",
    ) -> dict[str, Any]:
        """Import memories from an export payload.

        The binding's ``import_data`` accepts a STRING only, so a dict payload
        (the in-memory ``export(format="json")`` envelope) is JSON-serialized
        first. The returned ``{imported, skipped, merged}`` shape passes
        through unchanged.
        """
        if isinstance(data, dict):
            data = json.dumps(data)
        return self._inner.import_data(data, format)

    def run_decay_sweep(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        archive: bool = True,
    ) -> dict[str, Any]:
        """Run a temporal decay sweep (binding shape passes through)."""
        return self._inner.run_decay_sweep(user_id, agent_id, archive)

    # ---- Deep ACE commands (verify / inbox / topics / purpose / review /
    #      consolidate) — STORY: rust-backend CLI parity --------------------

    def verify(
        self,
        rehydrate_anchors: bool = False,
        stale_only: bool = False,
        scope: str | None = None,
        dry_run: bool = False,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Verify bullets against the filesystem (binding `verify`).

        Mirrors the Rust CLI `verify` orchestration. NOTE: the sync binding
        cannot persist (core's sync Memory exposes no apply_verification_patch);
        ``persisted`` will be ``False`` regardless of ``dry_run``. Use the
        async adapter's ``verify`` for write-back. The returned dict carries the
        ``verified/stale/unverifiable/not_applicable/anchors_added/elapsed_ms``
        counters the CLI report formats, plus ``persisted`` / ``per_bullet``.
        """
        return self._inner.verify(
            rehydrate_anchors, stale_only, scope, dry_run, user_id
        )

    def inbox_status(self) -> dict[str, Any]:
        """Inbox counts (binding shape: {path, pending, in_progress, ...})."""
        return self._inner.inbox_status()

    def inbox_flush(self) -> dict[str, Any]:
        """Recover crashed inbox entries (offline-safe).

        Full LLM distillation needs a daemon-wired BatchAnalyzer; the binding
        performs only the crash-recovery step and returns ``distilled=False``
        with an explanatory ``message``.
        """
        return self._inner.inbox_flush()

    def topics_list(self) -> list[Any]:
        """List persisted TopicPages (binding read-only shape)."""
        return self._inner.topics_list()

    def topics_show(self, slug: str) -> dict[str, Any]:
        """Show one TopicPage's markdown + metadata; KeyError if unknown."""
        return self._inner.topics_show(slug)

    def topics_regen(self, force: bool = False) -> dict[str, Any]:
        """Run the offline TopicEngine pass (binding `topics_regen`).

        Real LLM summarization needs a daemon-provided corpus + LLMSummarizer;
        the binding runs the FallbackSummarizer/empty-corpus pass and returns
        ``llm_regen=False``.
        """
        return self._inner.topics_regen(force)

    def purpose_show(self, scope: str = "effective") -> dict[str, Any] | None:
        """Show the effective/project/global purpose (None when unset)."""
        return self._inner.purpose_show(scope)

    def purpose_init(
        self,
        scope: str,
        force: bool = False,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Write a template purpose.md; FileExistsError if present, no force."""
        return self._inner.purpose_init(scope, force, name)

    def purpose_check(
        self,
        bullet_id: str,
        content: str = "",
        base_score: float = 50.0,
    ) -> dict[str, Any]:
        """Preview how the effective purpose scores a bullet."""
        return self._inner.purpose_check(bullet_id, content, base_score)

    def review_list(self) -> list[Any]:
        """List the consolidate review queue (raw JSONL rows)."""
        return self._inner.review_list()

    def consolidate_run(
        self,
        max_per_pass: int = 5000,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the offline curator consolidate scan (binding `consolidate_run`).

        Durable application (supersede / soft-delete / review-queue) needs a
        daemon-backed MemoryAdapter; the binding performs the pure scan only
        and returns ``executed=False`` with a ``message``.
        """
        return self._inner.consolidate_run(max_per_pass, user_id)

    # ---- Properties --------------------------------------------------------

    @property
    def config(self) -> Any:
        """Access the Memorus configuration (pure-Python MemorusConfig)."""
        return self._config


class RustBackedAsyncMemory:
    """Async Memory adapter delegating to ``memorus_r.AsyncMemory``.

    Presents the public surface of
    ``memorus.core.async_memory.AsyncMemory`` while awaiting the binding's
    coroutine-like futures.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        from memorus.core.config import MemorusConfig

        logger.debug("RustBackedAsyncMemory.__init__ config=%s", config)
        self._config = MemorusConfig.from_dict(config or {})

        import memorus_r

        # Translate the flat Python config into the nested Rust schema before
        # handing it to the binding; a file-path string or None passes through.
        self._inner = memorus_r.AsyncMemory(_translate_config(config))

    @classmethod
    def from_config(cls, config_dict: dict[str, Any]) -> RustBackedAsyncMemory:
        """Create from a config dict, mirroring AsyncMemory."""
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
        scope: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add memories (async). See ``RustBackedMemory.add`` for mappings."""
        content = _serialize_messages(messages)
        infer = self._config.ace_enabled
        result = await self._inner.add(
            content,
            metadata,
            user_id,
            agent_id,
            run_id,
            infer,
            scope,
        )
        out: dict[str, Any] = dict(result) if isinstance(result, dict) else {"results": []}
        out.setdefault("results", [])
        if infer:
            results = out.get("results") or []
            out["ace_ingest"] = _summarize_ingest(
                results if isinstance(results, list) else []
            )
        return out

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        filters: dict[str, Any] | None = None,
        scope: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Search memories (async). See ``RustBackedMemory.search``."""
        config = _build_search_config(limit, filters, kwargs.get("team_pool_path"))
        results = await self._inner.search(query, config, None, scope)
        results_list = list(results) if results is not None else []
        # See sync search(): trust fields + dropped_stale_count pass through.
        out: dict[str, Any] = {"results": results_list}
        if self._config.ace_enabled:
            ace_search: dict[str, Any] = {
                "mode": "rust",
                "total_candidates": len(results_list),
            }
            dropped = _extract_dropped_stale_count(results_list)
            if dropped:
                ace_search["dropped_stale_count"] = dropped
            out["ace_search"] = ace_search
        return out

    async def get_all(self, **kwargs: Any) -> dict[str, Any]:
        """Get all memories (async), wrapped in ``{"memories": [...]}``."""
        results = await self._inner.get_all(
            kwargs.get("user_id"),
            kwargs.get("agent_id"),
            kwargs.get("run_id"),
            kwargs.get("limit"),
        )
        return {"memories": list(results) if results is not None else []}

    async def get(self, memory_id: str) -> dict[str, Any]:
        """Get a single memory by ID (async)."""
        return await self._inner.get(memory_id)

    async def update(self, memory_id: str, data: str) -> Any:
        """Update a memory's content by ID (async)."""
        return await self._inner.update(memory_id, data)

    async def delete(self, memory_id: str) -> None:
        """Delete a single memory by ID (async)."""
        return await self._inner.delete(memory_id)

    async def delete_all(self, **kwargs: Any) -> None:
        """Delete all memories matching the scoping kwargs (async)."""
        return await self._inner.delete_all(
            kwargs.get("user_id"),
            kwargs.get("agent_id"),
            kwargs.get("run_id"),
        )

    async def history(self, memory_id: str) -> Any:
        """Get modification history of a memory (async)."""
        return await self._inner.history(memory_id)

    async def reset(self) -> None:
        """Reset all memories (async)."""
        return await self._inner.reset()

    # ---- ACE-specific methods ----------------------------------------------

    async def status(self, user_id: str | None = None) -> dict[str, Any]:
        """Knowledge-base statistics (async)."""
        return await self._inner.status(user_id)

    async def detect_conflicts(self, user_id: str | None = None) -> list[Any]:
        """Detect contradictory memories (async)."""
        return await self._inner.detect_conflicts(user_id)

    async def consolidate_scan(
        self,
        max_per_pass: int = 5000,
        *,
        user_id: str | None = None,
    ) -> Any:
        """Scan the corpus for merge/conflict candidates (async)."""
        return await self._inner.consolidate_scan(max_per_pass, user_id)

    async def export(
        self,
        format: str = "json",
        scope: str | None = None,
    ) -> Any:
        """Export the corpus (async)."""
        return await self._inner.export(format, scope)

    async def import_data(
        self,
        data: dict[str, Any] | str,
        format: str = "json",
    ) -> Any:
        """Import memories from an export payload (async).

        Dict payloads are JSON-serialized first since the binding accepts a
        string only.
        """
        if isinstance(data, dict):
            data = json.dumps(data)
        return await self._inner.import_data(data, format)

    async def run_decay_sweep(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        archive: bool = True,
    ) -> Any:
        """Run a temporal decay sweep (async)."""
        return await self._inner.run_decay_sweep(user_id, agent_id, archive)

    # ---- Deep ACE commands (verify / inbox / topics / purpose / review /
    #      consolidate) — STORY: rust-backend CLI parity --------------------

    async def verify(
        self,
        rehydrate_anchors: bool = False,
        stale_only: bool = False,
        scope: str | None = None,
        dry_run: bool = False,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Verify bullets against the filesystem (async binding `verify`).

        Unlike the sync adapter, the async binding PERSISTS: when ``dry_run`` is
        False it writes ``verified_status`` / ``trust_score`` / anchors back via
        ``apply_verification_patch`` and returns ``persisted=True`` (no
        ``persist_note``); on ``dry_run`` it returns ``persisted=False`` and a
        ``persist_note``. See ``RustBackedMemory.verify`` for the counter shape.
        """
        return await self._inner.verify(
            rehydrate_anchors, stale_only, scope, dry_run, user_id
        )

    async def inbox_status(self) -> dict[str, Any]:
        """Inbox counts (async)."""
        return await self._inner.inbox_status()

    async def inbox_flush(self) -> dict[str, Any]:
        """Recover crashed inbox entries (async, offline-safe).

        Full LLM distillation needs a daemon-wired BatchAnalyzer; returns
        ``distilled=False`` with a ``message``.
        """
        return await self._inner.inbox_flush()

    async def topics_list(self) -> list[Any]:
        """List persisted TopicPages (async, read-only)."""
        return await self._inner.topics_list()

    async def topics_show(self, slug: str) -> dict[str, Any]:
        """Show one TopicPage's markdown + metadata (async); KeyError if unknown."""
        return await self._inner.topics_show(slug)

    async def topics_regen(self, force: bool = False) -> dict[str, Any]:
        """Run the offline TopicEngine pass (async); returns ``llm_regen=False``."""
        return await self._inner.topics_regen(force)

    async def purpose_show(
        self, scope: str = "effective"
    ) -> dict[str, Any] | None:
        """Show the effective/project/global purpose (async; None when unset)."""
        return await self._inner.purpose_show(scope)

    async def purpose_init(
        self,
        scope: str,
        force: bool = False,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Write a template purpose.md (async); FileExistsError if present."""
        return await self._inner.purpose_init(scope, force, name)

    async def purpose_check(
        self,
        bullet_id: str,
        content: str = "",
        base_score: float = 50.0,
    ) -> dict[str, Any]:
        """Preview how the effective purpose scores a bullet (async)."""
        return await self._inner.purpose_check(bullet_id, content, base_score)

    async def review_list(self) -> list[Any]:
        """List the consolidate review queue (async, raw JSONL rows)."""
        return await self._inner.review_list()

    async def consolidate_run(
        self,
        max_per_pass: int = 5000,
        *,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Run the offline curator consolidate scan (async); ``executed=False``."""
        return await self._inner.consolidate_run(max_per_pass, user_id)

    # ---- Properties --------------------------------------------------------

    @property
    def config(self) -> Any:
        """Access the Memorus configuration (pure-Python MemorusConfig)."""
        return self._config
