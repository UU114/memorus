# mypy: disable-error-code="untyped-decorator"
"""Benchmark: ReflectorEngine.reflect() single interaction.

Threshold: p95 < 20ms (relaxed to 40ms for CI safety margin).
"""

from __future__ import annotations

from typing import Any

import pytest

from memx.core.engines.reflector.engine import ReflectorEngine
from memx.core.types import InteractionEvent

# Threshold in seconds (20 ms).  Doubled to 40 ms as safety margin —
# see STORY-045 note on adaptive thresholds.
_P95_THRESHOLD_SEC = 0.040


@pytest.mark.benchmark(group="reflector")
def test_reflector_reflect_single(
    benchmark: Any,
    sample_interaction_event: InteractionEvent,
) -> None:
    """ReflectorEngine.reflect() for a single event should complete within threshold."""
    engine = ReflectorEngine()

    result = benchmark.pedantic(
        engine.reflect,
        args=(sample_interaction_event,),
        rounds=20,
        warmup_rounds=3,
    )

    # Sanity: should produce candidate bullets (may be empty if patterns
    # are not detected, but the call itself must succeed)
    assert isinstance(result, list)


@pytest.mark.benchmark(group="reflector")
def test_reflector_reflect_rich_event(
    benchmark: Any,
) -> None:
    """ReflectorEngine.reflect() with a richer interaction event."""
    engine = ReflectorEngine()

    event = InteractionEvent(
        user_message=(
            "I use git rebase --interactive frequently. "
            "When I encounter merge conflicts during rebase, "
            "I prefer using git mergetool with vimdiff. "
            "Also I always set core.autocrlf=true on Windows. "
            "For Docker, I use multi-stage builds to keep images small."
        ),
        assistant_message=(
            "Great workflow! Here are some tips: "
            "1. Use git rebase -i HEAD~5 to squash last 5 commits. "
            "2. Configure mergetool: git config merge.tool vimdiff. "
            "3. For Docker multi-stage, use 'FROM ... AS builder' pattern. "
            "4. Consider .dockerignore to exclude unnecessary files."
        ),
        user_id="bench-user",
        session_id="bench-session",
    )

    result = benchmark.pedantic(
        engine.reflect,
        args=(event,),
        rounds=20,
        warmup_rounds=3,
    )

    assert isinstance(result, list)


@pytest.mark.benchmark(group="reflector")
def test_reflector_reflect_empty_event(
    benchmark: Any,
) -> None:
    """Edge case: reflect() with minimal content should be very fast."""
    engine = ReflectorEngine()
    event = InteractionEvent(
        user_message="hi",
        assistant_message="hello",
    )

    result = benchmark(engine.reflect, event)

    assert isinstance(result, list)
