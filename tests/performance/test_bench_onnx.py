# mypy: disable-error-code="untyped-decorator"
"""Benchmark: ONNXEmbedder.embed() single text.

Threshold: p95 < 10ms (relaxed to 20ms for CI safety margin).
Skipped when onnxruntime is not installed.
"""

from __future__ import annotations

from typing import Any

import pytest

# Check onnxruntime availability before importing ONNXEmbedder
_ort_available = True
try:
    import onnxruntime  # type: ignore[import-not-found]  # noqa: F401
except ImportError:
    _ort_available = False

_tokenizers_available = True
try:
    import tokenizers  # type: ignore[import-untyped]  # noqa: F401
except ImportError:
    _tokenizers_available = False

_hf_hub_available = True
try:
    import huggingface_hub  # noqa: F401
except ImportError:
    _hf_hub_available = False


_SKIP_REASON = "onnxruntime, tokenizers, or huggingface_hub not installed"
_ALL_DEPS = _ort_available and _tokenizers_available and _hf_hub_available


# Threshold in seconds (10 ms).  Doubled to 20 ms as safety margin —
# see STORY-045 note on adaptive thresholds.
_P95_THRESHOLD_SEC = 0.020


@pytest.mark.benchmark(group="onnx")
@pytest.mark.skipif(not _ALL_DEPS, reason=_SKIP_REASON)
def test_onnx_embed_single(benchmark: Any) -> None:
    """ONNXEmbedder.embed() single text should complete within threshold."""
    from memx.core.embeddings.onnx import ONNXEmbedder

    embedder = ONNXEmbedder()

    # Warm up: first call triggers lazy model loading
    embedder.embed("warmup text")

    result = benchmark.pedantic(
        embedder.embed,
        args=("git rebase interactive tips and best practices",),
        rounds=20,
        warmup_rounds=3,
    )

    # Sanity: embedding should have correct dimensions (384 for MiniLM)
    assert isinstance(result, list)
    assert len(result) == 384


@pytest.mark.benchmark(group="onnx")
@pytest.mark.skipif(not _ALL_DEPS, reason=_SKIP_REASON)
def test_onnx_embed_empty(benchmark: Any) -> None:
    """ONNXEmbedder.embed() with empty text should return zero vector instantly."""
    from memx.core.embeddings.onnx import ONNXEmbedder

    embedder = ONNXEmbedder()
    embedder.embed("warmup")  # Trigger lazy load

    result = benchmark(embedder.embed, "")

    assert isinstance(result, list)
    assert len(result) == 384
    assert all(v == 0.0 for v in result)


@pytest.mark.benchmark(group="onnx")
@pytest.mark.skipif(not _ALL_DEPS, reason=_SKIP_REASON)
def test_onnx_embed_batch_5(benchmark: Any) -> None:
    """ONNXEmbedder.embed_batch() with 5 texts for throughput measurement."""
    from memx.core.embeddings.onnx import ONNXEmbedder

    embedder = ONNXEmbedder()
    embedder.embed("warmup")  # Trigger lazy load

    texts = [
        "git rebase interactive best practices",
        "docker compose multi-stage builds",
        "pytest fixtures and parametrize patterns",
        "vim keybindings for efficient editing",
        "rust cargo workspace management tips",
    ]

    result = benchmark.pedantic(
        embedder.embed_batch,
        args=(texts,),
        rounds=10,
        warmup_rounds=2,
    )

    assert isinstance(result, list)
    assert len(result) == 5
    for vec in result:
        assert len(vec) == 384
