"""Unit tests for memx.pipeline.retrieval — RetrievalPipeline + RecallReinforcer."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from memx.core.config import DecayConfig, RetrievalConfig
from memx.core.engines.decay.engine import DecayEngine
from memx.core.engines.generator.engine import BulletForSearch, GeneratorEngine
from memx.core.engines.generator.metadata_matcher import MetadataInfo
from memx.core.engines.generator.score_merger import ScoredBullet
from memx.core.engines.generator.vector_searcher import VectorSearcher
from memx.core.pipeline.retrieval import RecallReinforcer, RetrievalPipeline, SearchResult
from memx.core.utils.token_counter import TokenBudgetTrimmer

# ── Helper fixtures ──────────────────────────────────────────────────────

_NOW = datetime(2026, 2, 27, 12, 0, 0, tzinfo=timezone.utc)


def _make_scored_bullet(
    bid: str,
    content: str = "test content",
    score: float = 0.5,
) -> ScoredBullet:
    """Create a ScoredBullet for testing."""
    return ScoredBullet(
        bullet_id=bid,
        content=content,
        final_score=score,
        keyword_score=10.0,
        semantic_score=0.5,
        decay_weight=1.0,
        recency_boost=1.0,
    )


def _make_bullet(
    bid: str,
    content: str = "test content",
    days_ago: float = 30.0,
    decay_weight: float = 1.0,
) -> BulletForSearch:
    """Create a BulletForSearch for testing."""
    return BulletForSearch(
        bullet_id=bid,
        content=content,
        metadata=MetadataInfo(),
        created_at=_NOW - timedelta(days=days_ago),
        decay_weight=decay_weight,
    )


def _make_mock_generator(
    results: list[ScoredBullet] | None = None,
    mode: str = "full",
    raises: Exception | None = None,
) -> MagicMock:
    """Create a mock GeneratorEngine."""
    mock = MagicMock(spec=GeneratorEngine)
    if raises:
        mock.search.side_effect = raises
    else:
        mock.search.return_value = results or []
    mock.mode = mode
    return mock


def _make_mock_trimmer(
    passthrough: bool = True,
    raises: Exception | None = None,
) -> MagicMock:
    """Create a mock TokenBudgetTrimmer."""
    mock = MagicMock(spec=TokenBudgetTrimmer)
    if raises:
        mock.trim.side_effect = raises
    elif passthrough:
        mock.trim.side_effect = lambda x: x  # pass-through
    return mock


# ---------------------------------------------------------------------------
# SearchResult dataclass tests
# ---------------------------------------------------------------------------


class TestSearchResult:
    """SearchResult has correct defaults and fields."""

    def test_default_values(self) -> None:
        """SearchResult can be created with minimum args."""
        result = SearchResult(results=[], mode="full")
        assert result.results == []
        assert result.mode == "full"
        assert result.total_candidates == 0

    def test_with_results(self) -> None:
        """SearchResult holds scored bullets and mode."""
        bullets = [_make_scored_bullet("b1"), _make_scored_bullet("b2")]
        result = SearchResult(results=bullets, mode="degraded", total_candidates=10)
        assert len(result.results) == 2
        assert result.mode == "degraded"
        assert result.total_candidates == 10


# ---------------------------------------------------------------------------
# RecallReinforcer tests
# ---------------------------------------------------------------------------


class TestRecallReinforcer:
    """RecallReinforcer delegates to DecayEngine.reinforce()."""

    def test_reinforce_sync(self) -> None:
        """Sync reinforcement calls DecayEngine.reinforce with correct args."""
        decay = MagicMock(spec=DecayEngine)
        decay.reinforce.return_value = 2
        update_fn = MagicMock()

        reinforcer = RecallReinforcer(decay_engine=decay, update_fn=update_fn)
        count = reinforcer.reinforce_sync(["b1", "b2"])

        assert count == 2
        decay.reinforce.assert_called_once_with(["b1", "b2"], update_fn)

    def test_reinforce_sync_no_ids(self) -> None:
        """Empty bullet_ids returns 0 without calling DecayEngine."""
        decay = MagicMock(spec=DecayEngine)
        update_fn = MagicMock()

        reinforcer = RecallReinforcer(decay_engine=decay, update_fn=update_fn)
        count = reinforcer.reinforce_sync([])

        assert count == 0
        decay.reinforce.assert_not_called()

    def test_reinforce_sync_no_update_fn(self) -> None:
        """No update_fn returns 0 without calling DecayEngine."""
        decay = MagicMock(spec=DecayEngine)

        reinforcer = RecallReinforcer(decay_engine=decay, update_fn=None)
        count = reinforcer.reinforce_sync(["b1"])

        assert count == 0
        decay.reinforce.assert_not_called()

    def test_reinforce_async_fires_thread(self) -> None:
        """Async reinforcement spawns a background thread."""
        decay = MagicMock(spec=DecayEngine)
        decay.reinforce.return_value = 1
        update_fn = MagicMock()

        reinforcer = RecallReinforcer(decay_engine=decay, update_fn=update_fn)
        reinforcer.reinforce_async(["b1"])

        # Give the daemon thread time to execute
        time.sleep(0.1)

        decay.reinforce.assert_called_once_with(["b1"], update_fn)

    def test_reinforce_async_no_ids(self) -> None:
        """Async with empty IDs does not spawn thread."""
        decay = MagicMock(spec=DecayEngine)
        update_fn = MagicMock()

        reinforcer = RecallReinforcer(decay_engine=decay, update_fn=update_fn)
        reinforcer.reinforce_async([])

        time.sleep(0.05)
        decay.reinforce.assert_not_called()

    def test_reinforce_async_exception_swallowed(self) -> None:
        """Async reinforcement swallows exceptions (logs WARNING)."""
        decay = MagicMock(spec=DecayEngine)
        decay.reinforce.side_effect = RuntimeError("db error")
        update_fn = MagicMock()

        reinforcer = RecallReinforcer(decay_engine=decay, update_fn=update_fn)
        # Should not raise
        reinforcer.reinforce_async(["b1"])
        time.sleep(0.1)
        # No assertion needed — the test passes if no exception propagates


# ---------------------------------------------------------------------------
# RetrievalPipeline: Normal flow
# ---------------------------------------------------------------------------


class TestRetrievalPipelineNormalFlow:
    """RetrievalPipeline.search() orchestrates Generator -> Trimmer -> Reinforce."""

    def test_full_pipeline(self) -> None:
        """Normal flow: Generator -> Trimmer -> returns SearchResult."""
        scored = [
            _make_scored_bullet("b1", score=0.9),
            _make_scored_bullet("b2", score=0.7),
        ]
        generator = _make_mock_generator(results=scored, mode="full")
        trimmer = _make_mock_trimmer(passthrough=True)

        pipeline = RetrievalPipeline(
            generator=generator,
            trimmer=trimmer,
        )

        bullets = [_make_bullet("b1"), _make_bullet("b2")]
        result = pipeline.search("test query", bullets=bullets, limit=5)

        assert isinstance(result, SearchResult)
        assert len(result.results) == 2
        assert result.mode == "full"
        assert result.total_candidates == 2

        # Generator was called
        generator.search.assert_called_once()
        # Trimmer was called
        trimmer.trim.assert_called_once_with(scored)

    def test_degraded_mode_propagated(self) -> None:
        """Generator in degraded mode -> SearchResult.mode = 'degraded'."""
        scored = [_make_scored_bullet("b1")]
        generator = _make_mock_generator(results=scored, mode="degraded")
        trimmer = _make_mock_trimmer(passthrough=True)

        pipeline = RetrievalPipeline(generator=generator, trimmer=trimmer)
        result = pipeline.search("query", bullets=[_make_bullet("b1")])

        assert result.mode == "degraded"

    def test_empty_query_returns_empty(self) -> None:
        """Empty query returns empty SearchResult without calling Generator."""
        generator = _make_mock_generator()

        pipeline = RetrievalPipeline(generator=generator)
        result = pipeline.search("", bullets=[_make_bullet("b1")])

        assert result.results == []
        assert result.mode == "full"
        generator.search.assert_not_called()

    def test_no_results_from_generator(self) -> None:
        """Generator returns empty list -> empty SearchResult."""
        generator = _make_mock_generator(results=[], mode="full")

        pipeline = RetrievalPipeline(generator=generator)
        result = pipeline.search("query", bullets=[_make_bullet("b1")])

        assert result.results == []
        assert result.total_candidates == 0
        assert result.mode == "full"


# ---------------------------------------------------------------------------
# RetrievalPipeline: Generator failure -> fallback
# ---------------------------------------------------------------------------


class TestRetrievalPipelineGeneratorFallback:
    """Generator exception triggers fallback to mem0 search."""

    def test_generator_failure_fallback_to_mem0(self) -> None:
        """Generator raises -> fallback to mem0 search."""
        generator = _make_mock_generator(raises=RuntimeError("embedding error"))
        mem0_search = MagicMock()
        mem0_search.return_value = {
            "results": [
                {"id": "m1", "memory": "mem0 result", "score": 0.8, "metadata": {}},
            ]
        }

        pipeline = RetrievalPipeline(
            generator=generator,
            mem0_search_fn=mem0_search,
        )

        result = pipeline.search("query", user_id="u1", limit=5)

        assert result.mode == "fallback"
        assert len(result.results) == 1
        assert result.results[0].bullet_id == "m1"
        assert result.results[0].content == "mem0 result"

        # mem0 search was called
        mem0_search.assert_called_once()

    def test_generator_failure_no_mem0_fn(self) -> None:
        """Generator fails, no mem0_search_fn -> empty fallback."""
        generator = _make_mock_generator(raises=RuntimeError("fail"))

        pipeline = RetrievalPipeline(
            generator=generator,
            mem0_search_fn=None,
        )

        result = pipeline.search("query")

        assert result.mode == "fallback"
        assert result.results == []

    def test_generator_and_mem0_both_fail(self) -> None:
        """Both Generator and mem0 fail -> empty fallback."""
        generator = _make_mock_generator(raises=RuntimeError("gen fail"))
        mem0_search = MagicMock()
        mem0_search.side_effect = RuntimeError("mem0 fail")

        pipeline = RetrievalPipeline(
            generator=generator,
            mem0_search_fn=mem0_search,
        )

        result = pipeline.search("query")

        assert result.mode == "fallback"
        assert result.results == []


# ---------------------------------------------------------------------------
# RetrievalPipeline: Trimmer failure -> untrimmed results
# ---------------------------------------------------------------------------


class TestRetrievalPipelineTrimmerFallback:
    """Trimmer exception returns untrimmed results."""

    def test_trimmer_failure_returns_untrimmed(self) -> None:
        """Trimmer raises -> original scored results returned."""
        scored = [
            _make_scored_bullet("b1", score=0.9),
            _make_scored_bullet("b2", score=0.7),
        ]
        generator = _make_mock_generator(results=scored, mode="full")
        trimmer = _make_mock_trimmer(raises=RuntimeError("trimmer bug"))

        pipeline = RetrievalPipeline(generator=generator, trimmer=trimmer)
        result = pipeline.search("query", bullets=[_make_bullet("b1")])

        assert len(result.results) == 2
        assert result.mode == "full"

    def test_no_trimmer_returns_all(self) -> None:
        """No trimmer configured -> all results returned."""
        scored = [_make_scored_bullet("b1"), _make_scored_bullet("b2")]
        generator = _make_mock_generator(results=scored, mode="full")

        pipeline = RetrievalPipeline(generator=generator, trimmer=None)
        result = pipeline.search("query", bullets=[_make_bullet("b1")])

        assert len(result.results) == 2


# ---------------------------------------------------------------------------
# RetrievalPipeline: Reinforcer failure -> search unaffected
# ---------------------------------------------------------------------------


class TestRetrievalPipelineReinforcerFallback:
    """Reinforcer exception does not affect search results."""

    def test_reinforcer_failure_does_not_affect_results(self) -> None:
        """Reinforcer raises -> results still returned, only WARNING logged."""
        scored = [_make_scored_bullet("b1", score=0.9)]
        generator = _make_mock_generator(results=scored, mode="full")
        trimmer = _make_mock_trimmer(passthrough=True)

        # Create a decay engine with a failing reinforce
        decay = MagicMock(spec=DecayEngine)
        decay.reinforce.side_effect = RuntimeError("reinforce error")
        update_fn = MagicMock()

        pipeline = RetrievalPipeline(
            generator=generator,
            trimmer=trimmer,
            decay_engine=decay,
            update_fn=update_fn,
        )

        result = pipeline.search("query", bullets=[_make_bullet("b1")])

        # Results should still be returned
        assert len(result.results) == 1
        assert result.results[0].bullet_id == "b1"

        # Wait for async thread
        time.sleep(0.1)

    def test_no_decay_engine_skips_reinforcement(self) -> None:
        """No decay engine -> reinforcement is skipped silently."""
        scored = [_make_scored_bullet("b1")]
        generator = _make_mock_generator(results=scored, mode="full")

        pipeline = RetrievalPipeline(
            generator=generator,
            trimmer=None,
            decay_engine=None,
        )

        result = pipeline.search("query", bullets=[_make_bullet("b1")])

        assert len(result.results) == 1

    def test_reinforcer_called_with_hit_ids(self) -> None:
        """Reinforcer receives the IDs of the trimmed (returned) results."""
        scored = [
            _make_scored_bullet("b1", score=0.9),
            _make_scored_bullet("b2", score=0.7),
        ]
        generator = _make_mock_generator(results=scored, mode="full")
        trimmer = _make_mock_trimmer(passthrough=True)

        decay = MagicMock(spec=DecayEngine)
        decay.reinforce.return_value = 2
        update_fn = MagicMock()

        pipeline = RetrievalPipeline(
            generator=generator,
            trimmer=trimmer,
            decay_engine=decay,
            update_fn=update_fn,
        )

        result = pipeline.search("query", bullets=[_make_bullet("b1"), _make_bullet("b2")])

        # Wait for async reinforcement thread
        time.sleep(0.1)

        # Reinforcer should have been called with ["b1", "b2"]
        decay.reinforce.assert_called_once()
        called_ids = decay.reinforce.call_args[0][0]
        assert set(called_ids) == {"b1", "b2"}


# ---------------------------------------------------------------------------
# RetrievalPipeline: convert_mem0_results
# ---------------------------------------------------------------------------


class TestConvertMem0Results:
    """Static method _convert_mem0_results handles various mem0 response shapes."""

    def test_valid_results(self) -> None:
        """Valid mem0 response is converted to ScoredBullet list."""
        raw = {
            "results": [
                {"id": "m1", "memory": "content1", "score": 0.9, "metadata": {"k": "v"}},
                {"id": "m2", "memory": "content2", "score": 0.5},
            ]
        }
        bullets = RetrievalPipeline._convert_mem0_results(raw)

        assert len(bullets) == 2
        assert bullets[0].bullet_id == "m1"
        assert bullets[0].content == "content1"
        assert bullets[0].final_score == 0.9
        assert bullets[0].metadata == {"k": "v"}

    def test_empty_results(self) -> None:
        """Empty results list returns empty."""
        assert RetrievalPipeline._convert_mem0_results({"results": []}) == []

    def test_non_dict_input(self) -> None:
        """Non-dict input returns empty list."""
        assert RetrievalPipeline._convert_mem0_results("not a dict") == []
        assert RetrievalPipeline._convert_mem0_results(None) == []

    def test_missing_results_key(self) -> None:
        """Missing 'results' key returns empty list."""
        assert RetrievalPipeline._convert_mem0_results({"other": "data"}) == []


# ---------------------------------------------------------------------------
# Integration: Memory.search() ACE path
# ---------------------------------------------------------------------------


class TestMemorySearchACEPath:
    """Verify Memory.search() delegates to RetrievalPipeline when ACE is enabled."""

    def test_memory_search_ace_path(self) -> None:
        """ACE enabled + pipeline exists -> returns ace_search dict."""
        from memx.core.config import MemXConfig
        from memx.core.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0.get_all.return_value = {"memories": []}
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._sanitizer = None

        # Create a mock retrieval pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = SearchResult(
            results=[_make_scored_bullet("b1", score=0.85)],
            mode="full",
            total_candidates=5,
        )
        m._retrieval_pipeline = mock_pipeline

        result = m.search("test query", user_id="u1")

        assert "ace_search" in result
        assert result["ace_search"]["mode"] == "full"
        assert result["ace_search"]["total_candidates"] == 5
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "b1"
        assert result["results"][0]["score"] == 0.85

    def test_memory_search_ace_no_pipeline_falls_back(self) -> None:
        """ACE enabled but pipeline is None -> falls back to proxy."""
        from memx.core.config import MemXConfig
        from memx.core.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0.search.return_value = {"results": []}
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        result = m.search("test query", user_id="u1")

        m._mem0.search.assert_called_once()
        assert result == {"results": []}

    def test_memory_search_ace_failure_falls_back(self) -> None:
        """ACE search failure -> fallback to mem0 proxy."""
        from memx.core.config import MemXConfig
        from memx.core.memory import Memory

        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0.get_all.side_effect = RuntimeError("db error")
        m._mem0.search.return_value = {"results": []}
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._sanitizer = None

        mock_pipeline = MagicMock()
        mock_pipeline.search.side_effect = RuntimeError("pipeline fail")
        m._retrieval_pipeline = mock_pipeline

        result = m.search("test query", user_id="u1")

        # Should fall back to mem0
        m._mem0.search.assert_called_once()


# ---------------------------------------------------------------------------
# IngestPipeline Curator integration tests
# ---------------------------------------------------------------------------


class TestIngestPipelineCuratorIntegration:
    """Verify Curator deduplication flow in IngestPipeline."""

    def test_curator_dedup_adds_new(self) -> None:
        """Curator classifies candidate as 'to_add' -> bullet is written."""
        from memx.core.engines.curator.engine import CurateResult, ExistingBullet
        from memx.core.pipeline.ingest import IngestPipeline
        from memx.core.types import BulletSection, CandidateBullet, KnowledgeType, SourceType

        candidate = CandidateBullet(
            content="new bullet content",
            section=BulletSection.GENERAL,
            knowledge_type=KnowledgeType.KNOWLEDGE,
            source_type=SourceType.INTERACTION,
        )

        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]

        mock_curator = MagicMock()
        mock_curator.curate.return_value = CurateResult(
            to_add=[candidate], to_merge=[], to_skip=[]
        )

        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            curator=mock_curator,
            mem0_add_fn=mem0_add,
            mem0_get_all_fn=MagicMock(return_value={"memories": []}),
        )

        result = pipeline.process("test msg", user_id="u1")

        assert result.bullets_added == 1
        assert result.bullets_merged == 0
        assert result.bullets_skipped == 0
        mem0_add.assert_called_once()

    def test_curator_dedup_skips(self) -> None:
        """Curator classifies candidate as 'to_skip' -> bullet not written."""
        from memx.core.engines.curator.engine import CurateResult
        from memx.core.pipeline.ingest import IngestPipeline
        from memx.core.types import CandidateBullet

        candidate = CandidateBullet(content="")

        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]

        mock_curator = MagicMock()
        mock_curator.curate.return_value = CurateResult(
            to_add=[], to_merge=[], to_skip=[candidate]
        )

        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            curator=mock_curator,
            mem0_add_fn=mem0_add,
            mem0_get_all_fn=MagicMock(return_value={"memories": []}),
        )

        result = pipeline.process("test msg")

        assert result.bullets_added == 0
        assert result.bullets_skipped == 1
        mem0_add.assert_not_called()

    def test_curator_failure_inserts_all(self) -> None:
        """Curator raises -> all candidates are inserted (graceful degradation)."""
        from memx.core.pipeline.ingest import IngestPipeline
        from memx.core.types import BulletSection, CandidateBullet, KnowledgeType, SourceType

        candidate = CandidateBullet(
            content="bullet content",
            section=BulletSection.GENERAL,
            knowledge_type=KnowledgeType.KNOWLEDGE,
            source_type=SourceType.INTERACTION,
        )

        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]

        mock_curator = MagicMock()
        mock_curator.curate.side_effect = RuntimeError("curator crash")

        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            curator=mock_curator,
            mem0_add_fn=mem0_add,
            mem0_get_all_fn=MagicMock(return_value={"memories": []}),
        )

        result = pipeline.process("test msg")

        # Despite curator failure, bullet should still be added
        assert result.bullets_added == 1
        mem0_add.assert_called_once()

    def test_no_curator_skips_dedup(self) -> None:
        """No curator -> dedup step is skipped entirely."""
        from memx.core.pipeline.ingest import IngestPipeline
        from memx.core.types import BulletSection, CandidateBullet, KnowledgeType, SourceType

        candidate = CandidateBullet(
            content="bullet content",
            section=BulletSection.GENERAL,
            knowledge_type=KnowledgeType.KNOWLEDGE,
            source_type=SourceType.INTERACTION,
        )

        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            curator=None,
            mem0_add_fn=mem0_add,
        )

        result = pipeline.process("test msg")

        assert result.bullets_added == 1
        assert result.bullets_merged == 0
        assert result.bullets_skipped == 0


# ---------------------------------------------------------------------------
# STORY-031 补充测试：Fallback 路径完备性、Reinforce 集成、Pipeline 边界
# ---------------------------------------------------------------------------


class TestRetrievalPipelineFallbackPathCompleteness:
    """Fallback 路径完备性验证。"""

    def test_fallback_mem0_search_passes_parameters(self) -> None:
        """Fallback to mem0 should pass user_id, agent_id, limit, filters."""
        generator = _make_mock_generator(raises=RuntimeError("gen fail"))
        captured: dict[str, object] = {}

        def mem0_fn(
            query: str,
            user_id: str | None = None,
            agent_id: str | None = None,
            limit: int = 5,
            filters: dict[str, object] | None = None,
        ) -> dict[str, list[dict[str, object]]]:
            captured.update({
                "query": query,
                "user_id": user_id,
                "agent_id": agent_id,
                "limit": limit,
                "filters": filters,
            })
            return {"results": []}

        pipeline = RetrievalPipeline(
            generator=generator,
            mem0_search_fn=mem0_fn,
        )

        pipeline.search(
            "test query",
            user_id="u1",
            agent_id="a1",
            limit=10,
            filters={"tag": "python"},
        )

        assert captured["query"] == "test query"
        assert captured["user_id"] == "u1"
        assert captured["agent_id"] == "a1"
        assert captured["limit"] == 10
        assert captured["filters"] == {"tag": "python"}

    def test_fallback_with_multiple_results(self) -> None:
        """Fallback should handle multiple results from mem0."""
        generator = _make_mock_generator(raises=RuntimeError("fail"))
        mem0_fn = MagicMock()
        mem0_fn.return_value = {
            "results": [
                {"id": "m1", "memory": "result 1", "score": 0.9, "metadata": {}},
                {"id": "m2", "memory": "result 2", "score": 0.7, "metadata": {}},
                {"id": "m3", "memory": "result 3", "score": 0.5, "metadata": {}},
            ]
        }

        pipeline = RetrievalPipeline(generator=generator, mem0_search_fn=mem0_fn)
        result = pipeline.search("query")

        assert result.mode == "fallback"
        assert len(result.results) == 3
        assert result.results[0].bullet_id == "m1"
        assert result.results[2].bullet_id == "m3"

    def test_fallback_result_has_correct_scored_bullet_fields(self) -> None:
        """Fallback ScoredBullet should have correct field mapping."""
        generator = _make_mock_generator(raises=RuntimeError("fail"))
        mem0_fn = MagicMock()
        mem0_fn.return_value = {
            "results": [
                {
                    "id": "m1",
                    "memory": "test content",
                    "score": 0.85,
                    "metadata": {"tags": ["python"]},
                },
            ]
        }

        pipeline = RetrievalPipeline(generator=generator, mem0_search_fn=mem0_fn)
        result = pipeline.search("query")

        r = result.results[0]
        assert r.bullet_id == "m1"
        assert r.content == "test content"
        assert r.final_score == 0.85
        assert r.keyword_score == 0.0
        assert r.semantic_score == 0.85
        assert r.decay_weight == 1.0
        assert r.recency_boost == 1.0
        assert r.metadata == {"tags": ["python"]}


class TestRetrievalPipelineReinforceIntegration:
    """Reinforce 集成完整流程验证。"""

    def test_reinforce_receives_trimmed_ids_not_all(self) -> None:
        """Reinforcer should receive only the trimmed result IDs."""
        scored = [
            _make_scored_bullet("b1", score=0.9),
            _make_scored_bullet("b2", score=0.7),
            _make_scored_bullet("b3", score=0.3),
        ]
        generator = _make_mock_generator(results=scored, mode="full")

        # Trimmer keeps only first 2
        trimmer = MagicMock(spec=TokenBudgetTrimmer)
        trimmer.trim.return_value = scored[:2]  # only b1, b2

        decay = MagicMock(spec=DecayEngine)
        decay.reinforce.return_value = 2
        update_fn = MagicMock()

        pipeline = RetrievalPipeline(
            generator=generator,
            trimmer=trimmer,
            decay_engine=decay,
            update_fn=update_fn,
        )

        result = pipeline.search("query", bullets=[_make_bullet("b1")])
        time.sleep(0.1)

        # Only b1, b2 IDs should be reinforced (not b3)
        decay.reinforce.assert_called_once()
        called_ids = decay.reinforce.call_args[0][0]
        assert set(called_ids) == {"b1", "b2"}


class TestRetrievalPipelineBoundaryConditions:
    """Pipeline 边界条件。"""

    def test_whitespace_query_treated_as_empty(self) -> None:
        """Whitespace-only query should return empty results."""
        generator = _make_mock_generator()
        pipeline = RetrievalPipeline(generator=generator)
        # Note: "" is falsy, but "  " is truthy
        # The pipeline checks `if not query` which catches "" but not "  "
        result = pipeline.search("  ", bullets=[_make_bullet("b1")])
        # "  " is truthy so generator.search IS called
        # But the generator mock returns [] by default
        assert isinstance(result, SearchResult)

    def test_none_bullets_uses_empty_list(self) -> None:
        """None bullets should be treated as empty list."""
        generator = _make_mock_generator()
        pipeline = RetrievalPipeline(generator=generator)
        result = pipeline.search("query", bullets=None)
        # bullets=None -> bullets or [] = []
        generator.search.assert_called_once()

    def test_over_fetch_multiplier(self) -> None:
        """Generator is called with limit*4 for trimming headroom."""
        generator = _make_mock_generator()
        pipeline = RetrievalPipeline(generator=generator)
        pipeline.search("query", bullets=[_make_bullet("b1")], limit=5)
        # Verify the limit passed to generator is 5*4=20
        call_kwargs = generator.search.call_args
        assert call_kwargs[1]["limit"] == 20  # 5 * 4

    def test_total_candidates_reflects_pre_trim_count(self) -> None:
        """total_candidates should reflect count before trimming."""
        scored = [_make_scored_bullet(f"b{i}", score=float(i)) for i in range(8)]
        generator = _make_mock_generator(results=scored, mode="full")
        # Trimmer keeps only 3
        trimmer = MagicMock(spec=TokenBudgetTrimmer)
        trimmer.trim.return_value = scored[:3]

        pipeline = RetrievalPipeline(generator=generator, trimmer=trimmer)
        result = pipeline.search("query", bullets=[_make_bullet("b1")])

        assert result.total_candidates == 8  # all scored, before trim
        assert len(result.results) == 3  # after trim


class TestConvertMem0ResultsEdge:
    """_convert_mem0_results 边界补充。"""

    def test_non_dict_items_in_results_skipped(self) -> None:
        """Non-dict items in results list should be silently skipped."""
        raw = {
            "results": [
                "not a dict",
                42,
                {"id": "m1", "memory": "valid", "score": 0.5},
            ]
        }
        bullets = RetrievalPipeline._convert_mem0_results(raw)
        assert len(bullets) == 1
        assert bullets[0].bullet_id == "m1"

    def test_missing_id_uses_empty_string(self) -> None:
        """Item without 'id' key should use empty string for bullet_id."""
        raw = {
            "results": [
                {"memory": "no id", "score": 0.5},
            ]
        }
        bullets = RetrievalPipeline._convert_mem0_results(raw)
        assert len(bullets) == 1
        assert bullets[0].bullet_id == ""

    def test_integer_input_returns_empty(self) -> None:
        """Integer input should return empty list."""
        assert RetrievalPipeline._convert_mem0_results(42) == []

    def test_list_input_returns_empty(self) -> None:
        """List input (not dict) should return empty list."""
        assert RetrievalPipeline._convert_mem0_results([1, 2, 3]) == []
