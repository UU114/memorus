"""Version-tolerant compatibility seam for the mem0 storage backend.

memorus is written against the mem0 1.x call contract (entity IDs as top-level
kwargs, ``limit`` for result count, ``threshold`` default = no filtering). mem0
2.x changed that contract:

* ``search`` / ``get_all`` REJECT top-level ``user_id`` / ``agent_id`` /
  ``run_id`` (a ``ValueError`` via ``_reject_top_level_entity_params``); entity
  scoping must go through ``filters={...}``.
* ``limit`` was renamed to ``top_k`` (default dropped 100 -> 20); a stray
  ``limit`` is swallowed by ``**kwargs`` and silently ignored -> recall collapses
  to 20.
* ``threshold`` default changed ``None`` -> ``0.1`` (low-score rows now dropped).
* ``rerank`` default changed ``True`` -> ``False``.

Rather than rewrite all 38 call sites, we wrap the mem0 instance ONCE at init.
On mem0 1.x ``wrap_mem0`` returns the instance unchanged (zero overhead, zero
behavior change). On 2.x it returns a thin proxy that translates ``search`` /
``get_all`` arguments back to 1.x semantics and delegates everything else
(``add`` / ``get`` / ``update`` / ``delete`` / ...) verbatim.
"""

from __future__ import annotations

from typing import Any


def mem0_major() -> int | None:
    """Return the installed mem0 major version, or None if undeterminable."""
    try:
        import mem0

        raw = getattr(mem0, "__version__", "") or ""
        return int(raw.split(".")[0])
    except Exception:
        return None


def _to_filters(
    filters: dict[str, Any] | None,
    user_id: Any,
    agent_id: Any,
    run_id: Any,
) -> dict[str, Any] | None:
    """Fold top-level entity IDs into a 2.x ``filters`` dict.

    Existing ``filters`` keys win (we only fill what the caller did not set).
    Returns None when there is nothing to scope by, so 2.x sees an absent
    filter rather than an empty dict.
    """
    out = dict(filters or {})
    for key, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
        if val is not None:
            out.setdefault(key, val)
    return out or None


def _translate_search(query: str, kw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Translate 1.x-style search kwargs to the 2.x signature.

    Maps entity IDs -> filters, ``limit`` -> ``top_k``, and restores the 1.x
    "no score floor" behavior with ``threshold=0.0`` (2.x defaults to 0.1).
    ``rerank`` is forwarded only when the caller set it explicitly: forcing
    ``rerank=True`` on 2.x can raise when no reranker is configured, and without
    a reranker 1.x's ``True`` and 2.x's ``False`` are both no-ops — so leaving it
    alone preserves observable behavior. (Validate in the 2.x integration pass.)
    """
    user_id = kw.pop("user_id", None)
    agent_id = kw.pop("agent_id", None)
    run_id = kw.pop("run_id", None)
    limit = kw.pop("limit", None)
    threshold = kw.pop("threshold", None)
    rerank = kw.pop("rerank", None)
    filters = kw.pop("filters", None)

    call = dict(kw)
    call["filters"] = _to_filters(filters, user_id, agent_id, run_id)
    if limit is not None:
        call["top_k"] = limit
    call["threshold"] = threshold if threshold is not None else 0.0
    if rerank is not None:
        call["rerank"] = rerank
    return query, call


def _translate_get_all(kw: dict[str, Any]) -> dict[str, Any]:
    """Translate 1.x-style get_all kwargs to the 2.x signature."""
    user_id = kw.pop("user_id", None)
    agent_id = kw.pop("agent_id", None)
    run_id = kw.pop("run_id", None)
    limit = kw.pop("limit", None)
    filters = kw.pop("filters", None)

    call = dict(kw)
    call["filters"] = _to_filters(filters, user_id, agent_id, run_id)
    if limit is not None:
        call["top_k"] = limit
    return call


class _Mem0Compat:
    """Synchronous 2.x-compatibility proxy around a mem0 Memory instance."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def search(self, query: str, **kwargs: Any) -> Any:
        q, call = _translate_search(query, kwargs)
        return self._inner.search(q, **call)

    def get_all(self, **kwargs: Any) -> Any:
        return self._inner.get_all(**_translate_get_all(kwargs))

    def __getattr__(self, name: str) -> Any:
        # Only reached for names not defined on this class -> delegate verbatim
        # (add/get/update/delete/reset/...). _inner is a real instance attr, so
        # this never recurses.
        return getattr(self._inner, name)


class _AsyncMem0Compat:
    """Asynchronous 2.x-compatibility proxy around a mem0 AsyncMemory instance."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    async def search(self, query: str, **kwargs: Any) -> Any:
        q, call = _translate_search(query, kwargs)
        return await self._inner.search(q, **call)

    async def get_all(self, **kwargs: Any) -> Any:
        return await self._inner.get_all(**_translate_get_all(kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def wrap_mem0(inner: Any) -> Any:
    """Wrap a sync mem0 instance for 2.x; return it unchanged on 1.x / None."""
    if inner is None:
        return None
    return _Mem0Compat(inner) if (mem0_major() or 0) >= 2 else inner


def wrap_async_mem0(inner: Any) -> Any:
    """Wrap an async mem0 instance for 2.x; return it unchanged on 1.x / None."""
    if inner is None:
        return None
    return _AsyncMem0Compat(inner) if (mem0_major() or 0) >= 2 else inner
