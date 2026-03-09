# mypy: disable-error-code="untyped-decorator"
"""Benchmark: IngestPipeline + RetrievalPipeline end-to-end.

Measures the overhead of pipeline orchestration on top of individual engines.
No strict threshold — this is a regression tracking benchmark.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from memx.core.engines.decay.engine import DecayEngine
from memx.core.engines.generator.engine import BulletForSearch, GeneratorEngine
from memx.core.engines.reflector.engine import ReflectorEngine
from memx.core.pipeline.ingest import IngestPipeline
from memx.core.pipeline.retrieval import RetrievalPipeline
from memx.core.types import InteractionEvent
from memx.core.utils.token_counter import TokenBudgetTrimmer


@pytest.mark.benchmark(group="pipeline")
def test_ingest_pipeline_process(
    benchmark: Any,
    sample_interaction_event: InteractionEvent,
) -> None:
    """IngestPipeline.process() end-to-end with mocked mem0 backend."""
    reflector = ReflectorEngine()
    mock_add = MagicMock(return_value={"id": "test-id"})

    pipeline = IngestPipeline(
        reflector=reflector,
        mem0_add_fn=mock_add,
    )

    messages = [
        {"role": "user", "content": sample_interaction_event.user_message},
        {"role": "assistant", "content": sample_interaction_event.assistant_message},
    ]

    result = benchmark.pedantic(
        pipeline.process,
        args=(messages,),
        kwargs={"user_id": "bench-user"},
        rounds=20,
        warmup_rounds=3,
    )

    # Sanity: pipeline should produce a result
    assert result is not None


@pytest.mark.benchmark(group="pipeline")
def test_retrieval_pipeline_search_5000(
    benchmark: Any,
    generate_bullets: Any,
) -> None:
    """RetrievalPipeline.search() with 5000 bullets end-to-end."""
    bullets: list[BulletForSearch] = generate_bullets(5000)
    generator = GeneratorEngine()
    trimmer = TokenBudgetTrimmer(token_budget=2000, max_results=10)
    decay_engine = DecayEngine()

    pipeline = RetrievalPipeline(
        generator=generator,
        trimmer=trimmer,
        decay_engine=decay_engine,
    )

    result = benchmark.pedantic(
        pipeline.search,
        args=("git rebase tips",),
        kwargs={"bullets": bullets, "limit": 10},
        rounds=10,
        warmup_rounds=2,
    )

    # Sanity: should return search results
    assert result is not None
    assert len(result.results) > 0
    assert result.mode in ("full", "degraded")


@pytest.mark.benchmark(group="pipeline")
def test_retrieval_pipeline_search_1000(
    benchmark: Any,
    generate_bullets: Any,
) -> None:
    """RetrievalPipeline.search() with 1000 bullets (lighter workload)."""
    bullets: list[BulletForSearch] = generate_bullets(1000)
    generator = GeneratorEngine()
    trimmer = TokenBudgetTrimmer(token_budget=2000, max_results=5)

    pipeline = RetrievalPipeline(
        generator=generator,
        trimmer=trimmer,
    )

    result = benchmark.pedantic(
        pipeline.search,
        args=("docker compose setup",),
        kwargs={"bullets": bullets, "limit": 5},
        rounds=20,
        warmup_rounds=3,
    )

    assert result is not None
    assert len(result.results) > 0


@pytest.mark.benchmark(group="pipeline")
def test_ingest_pipeline_with_curator(
    benchmark: Any,
    sample_interaction_event: InteractionEvent,
) -> None:
    """IngestPipeline.process() with Curator enabled (no existing bullets)."""
    from memx.core.engines.curator.engine import CuratorEngine

    reflector = ReflectorEngine()
    curator = CuratorEngine()
    mock_add = MagicMock(return_value={"id": "test-id"})
    mock_get_all = MagicMock(return_value={"memories": []})

    pipeline = IngestPipeline(
        reflector=reflector,
        curator=curator,
        mem0_add_fn=mock_add,
        mem0_get_all_fn=mock_get_all,
    )

    messages = [
        {"role": "user", "content": sample_interaction_event.user_message},
        {"role": "assistant", "content": sample_interaction_event.assistant_message},
    ]

    result = benchmark.pedantic(
        pipeline.process,
        args=(messages,),
        kwargs={"user_id": "bench-user"},
        rounds=20,
        warmup_rounds=3,
    )

    assert result is not None


@pytest.mark.benchmark(group="pipeline")
def test_retrieval_pipeline_empty_query(
    benchmark: Any,
) -> None:
    """Edge case: empty query returns instantly."""
    generator = GeneratorEngine()
    pipeline = RetrievalPipeline(generator=generator)

    result = benchmark(pipeline.search, "")

    assert result is not None
    assert len(result.results) == 0
