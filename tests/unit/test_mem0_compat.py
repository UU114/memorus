"""Unit tests for the mem0 1.x<->2.x compatibility seam.

The 2.x translation is exercised by constructing the proxies directly with a
mock inner (so these run on any installed mem0). The 1.x passthrough is asserted
against the actually-installed version.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from memorus.core._mem0_compat import (
    _AsyncMem0Compat,
    _Mem0Compat,
    mem0_major,
    wrap_async_mem0,
    wrap_mem0,
)


class TestVersionGate:
    def test_wrap_is_noop_on_installed_1x(self) -> None:
        """On mem0 1.x (the pinned reality) wrap returns the instance unchanged."""
        if (mem0_major() or 0) >= 2:
            pytest.skip("installed mem0 is 2.x; passthrough assertion is for 1.x")
        inner = MagicMock()
        assert wrap_mem0(inner) is inner
        assert wrap_async_mem0(inner) is inner

    def test_wrap_handles_none(self) -> None:
        assert wrap_mem0(None) is None
        assert wrap_async_mem0(None) is None


class TestSearchTranslation:
    def test_entity_ids_folded_into_filters(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).search("q", user_id="u1", agent_id="a1", run_id="r1")
        _, kwargs = inner.search.call_args
        assert kwargs["filters"] == {"user_id": "u1", "agent_id": "a1", "run_id": "r1"}
        # top-level entity ids must NOT leak (2.x rejects them)
        assert "user_id" not in kwargs and "agent_id" not in kwargs

    def test_limit_mapped_to_top_k(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).search("q", user_id="u1", limit=50)
        _, kwargs = inner.search.call_args
        assert kwargs["top_k"] == 50
        assert "limit" not in kwargs

    def test_threshold_defaults_to_zero_for_1x_recall(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).search("q")
        _, kwargs = inner.search.call_args
        # 2.x default 0.1 would silently drop low-score rows; restore 1.x behavior
        assert kwargs["threshold"] == 0.0

    def test_explicit_threshold_and_rerank_preserved(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).search("q", threshold=0.3, rerank=True)
        _, kwargs = inner.search.call_args
        assert kwargs["threshold"] == 0.3
        assert kwargs["rerank"] is True

    def test_rerank_omitted_when_unset(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).search("q")
        _, kwargs = inner.search.call_args
        # not forced -> avoids requiring a reranker on 2.x
        assert "rerank" not in kwargs

    def test_existing_filters_win_over_entity_ids(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).search("q", user_id="u1", filters={"user_id": "explicit"})
        _, kwargs = inner.search.call_args
        assert kwargs["filters"]["user_id"] == "explicit"

    def test_query_passed_positionally(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).search("find me")
        args, _ = inner.search.call_args
        assert args[0] == "find me"


class TestGetAllTranslation:
    def test_entity_ids_and_limit_translated(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).get_all(user_id="u1", limit=200)
        _, kwargs = inner.get_all.call_args
        assert kwargs["filters"] == {"user_id": "u1"}
        assert kwargs["top_k"] == 200
        assert "user_id" not in kwargs and "limit" not in kwargs

    def test_no_scope_yields_none_filters(self) -> None:
        inner = MagicMock()
        _Mem0Compat(inner).get_all()
        _, kwargs = inner.get_all.call_args
        assert kwargs["filters"] is None


class TestDelegation:
    def test_other_methods_delegate_verbatim(self) -> None:
        inner = MagicMock()
        proxy = _Mem0Compat(inner)
        proxy.add("content", user_id="u1", infer=False)
        proxy.get("mid")
        proxy.update("mid", "data")
        proxy.delete("mid")
        inner.add.assert_called_once_with("content", user_id="u1", infer=False)
        inner.get.assert_called_once_with("mid")
        inner.update.assert_called_once_with("mid", "data")
        inner.delete.assert_called_once_with("mid")


class TestAsyncProxy:
    @pytest.mark.asyncio
    async def test_async_search_translates_and_awaits(self) -> None:
        inner = MagicMock()
        inner.search = AsyncMock(return_value={"results": []})
        out = await _AsyncMem0Compat(inner).search("q", user_id="u1", limit=10)
        assert out == {"results": []}
        _, kwargs = inner.search.call_args
        assert kwargs["filters"] == {"user_id": "u1"}
        assert kwargs["top_k"] == 10
        assert kwargs["threshold"] == 0.0

    @pytest.mark.asyncio
    async def test_async_get_all_translates(self) -> None:
        inner = MagicMock()
        inner.get_all = AsyncMock(return_value={"results": []})
        await _AsyncMem0Compat(inner).get_all(user_id="u1")
        _, kwargs = inner.get_all.call_args
        assert kwargs["filters"] == {"user_id": "u1"}


class TestMem0Major:
    def test_returns_int_or_none(self) -> None:
        v = mem0_major()
        assert v is None or isinstance(v, int)
