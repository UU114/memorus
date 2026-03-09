"""Unit tests for memx.embeddings.onnx — ONNXEmbedder.

All tests mock onnxruntime, tokenizers, and huggingface_hub so they
run fast without any model files or optional dependencies.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from mem0.configs.embeddings.base import BaseEmbedderConfig


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

DIMS = 384
MAX_LEN = 256


class FakeEncoding:
    """Mimics tokenizers.Encoding returned by Tokenizer.encode()."""

    def __init__(self, length: int = 10) -> None:
        self.ids = list(range(length))
        self.attention_mask = [1] * length


class FakeTokenizer:
    """Mimics tokenizers.Tokenizer."""

    def __init__(self) -> None:
        self.truncation = None
        self._truncation_enabled = False
        self._padding_enabled = False

    def encode(self, text: str) -> FakeEncoding:
        # Return consistent token length for testing
        return FakeEncoding(length=min(len(text.split()) + 2, MAX_LEN))

    def enable_truncation(self, max_length: int = MAX_LEN) -> None:
        self._truncation_enabled = True

    def enable_padding(self, **kwargs: Any) -> None:
        self._padding_enabled = True

    @classmethod
    def from_file(cls, path: str) -> "FakeTokenizer":
        return cls()


class FakeSession:
    """Mimics ort.InferenceSession.run() output."""

    def __init__(self, model_path: str, **kwargs: Any) -> None:
        self.model_path = model_path

    def run(
        self, output_names: Any, input_feed: dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        # Return token embeddings: (1, seq_len, DIMS)
        seq_len = input_feed["input_ids"].shape[1]
        # Use deterministic but non-trivial values
        rng = np.random.RandomState(42)
        token_embeddings = rng.randn(1, seq_len, DIMS).astype(np.float32)
        return [token_embeddings]


@pytest.fixture()
def tmp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary model directory with dummy files."""
    model_dir = tmp_path / "models" / "all-MiniLM-L6-v2"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"fake-onnx-model")
    (model_dir / "tokenizer.json").write_text('{"fake": true}')
    return tmp_path / "models"


@pytest.fixture()
def empty_model_dir(tmp_path: Path) -> Path:
    """Return a temporary model directory with NO model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True)
    return model_dir


# We need to patch at the module level where the names are used
_ORT_MODULE = "memx.core.embeddings.onnx"


def _patch_onnx_deps():
    """Return a stack of patches for all ONNX optional dependencies."""
    import memx.core.embeddings.onnx as mod

    patches = [
        patch.object(mod, "_ORT_AVAILABLE", True),
        patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
        patch.object(mod, "_HF_HUB_AVAILABLE", True),
        patch.object(mod, "ort", create=True),
        patch.object(mod, "Tokenizer", FakeTokenizer),
        patch.object(mod, "hf_hub_download", MagicMock(return_value="/fake/path")),
    ]
    return patches


def _create_embedder(
    tmp_model_dir: Path, **kwargs: Any
) -> "memx.core.embeddings.onnx.ONNXEmbedder":
    """Create an ONNXEmbedder with mocked dependencies."""
    from memx.core.embeddings.onnx import ONNXEmbedder

    config = BaseEmbedderConfig(
        model="all-MiniLM-L6-v2",
        embedding_dims=DIMS,
    )
    return ONNXEmbedder(
        config=config,
        model_dir=str(tmp_model_dir),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests: Construction & defaults
# ---------------------------------------------------------------------------


class TestONNXEmbedderConstruction:
    """Tests for embedder initialization and default config values."""

    def test_default_config_values(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        assert embedder.config.model == "all-MiniLM-L6-v2"
        assert embedder.config.embedding_dims == DIMS
        assert embedder._session is None  # Lazy load

    def test_custom_config(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        config = BaseEmbedderConfig(model="custom-model", embedding_dims=768)
        embedder = ONNXEmbedder(config=config, model_dir=str(tmp_model_dir))
        assert embedder.config.model == "custom-model"
        assert embedder.config.embedding_dims == 768

    def test_dimensions_property(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        assert embedder.dimensions == DIMS

    def test_repr(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        r = repr(embedder)
        assert "ONNXEmbedder" in r
        assert "all-MiniLM-L6-v2" in r
        assert "loaded=False" in r


# ---------------------------------------------------------------------------
# Tests: embed()
# ---------------------------------------------------------------------------


class TestONNXEmbedderEmbed:
    """Tests for the embed() method with mocked ONNX runtime."""

    def _setup_embedder(self, tmp_model_dir: Path) -> Any:
        """Create an embedder and manually inject mocked session/tokenizer."""
        from memx.core.embeddings.onnx import ONNXEmbedder

        config = BaseEmbedderConfig(model="all-MiniLM-L6-v2", embedding_dims=DIMS)
        embedder = ONNXEmbedder(config=config, model_dir=str(tmp_model_dir))
        # Inject mocked session and tokenizer directly
        embedder._session = FakeSession(str(tmp_model_dir / "all-MiniLM-L6-v2" / "model.onnx"))
        embedder._tokenizer = FakeTokenizer()
        return embedder

    def test_embed_returns_list_of_floats(self, tmp_model_dir: Path) -> None:
        embedder = self._setup_embedder(tmp_model_dir)
        result = embedder.embed("Hello world")
        assert isinstance(result, list)
        assert len(result) == DIMS
        assert all(isinstance(x, float) for x in result)

    def test_embed_empty_text_returns_zero_vector(self, tmp_model_dir: Path) -> None:
        embedder = self._setup_embedder(tmp_model_dir)
        result = embedder.embed("")
        assert result == [0.0] * DIMS

    def test_embed_whitespace_only_returns_zero_vector(self, tmp_model_dir: Path) -> None:
        embedder = self._setup_embedder(tmp_model_dir)
        result = embedder.embed("   ")
        assert result == [0.0] * DIMS

    def test_embed_is_normalized(self, tmp_model_dir: Path) -> None:
        """Embedding vectors should be L2-normalized (unit length)."""
        embedder = self._setup_embedder(tmp_model_dir)
        result = embedder.embed("The quick brown fox")
        vec = np.array(result)
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5, f"Vector norm = {norm}, expected ~1.0"

    def test_embed_deterministic(self, tmp_model_dir: Path) -> None:
        """Same input should produce same output."""
        embedder = self._setup_embedder(tmp_model_dir)
        a = embedder.embed("Deterministic test")
        b = embedder.embed("Deterministic test")
        np.testing.assert_array_almost_equal(a, b)

    def test_embed_different_texts_differ(self, tmp_model_dir: Path) -> None:
        """Different inputs should produce different embeddings."""
        embedder = self._setup_embedder(tmp_model_dir)
        a = embedder.embed("Hello world")
        b = embedder.embed("Goodbye moon planet stars")
        # They should not be identical (different token counts -> different embeddings)
        assert a != b

    def test_memory_action_is_accepted(self, tmp_model_dir: Path) -> None:
        """memory_action parameter should be accepted without error."""
        embedder = self._setup_embedder(tmp_model_dir)
        for action in ("add", "search", "update", None):
            result = embedder.embed("test", memory_action=action)
            assert len(result) == DIMS


# ---------------------------------------------------------------------------
# Tests: embed_batch()
# ---------------------------------------------------------------------------


class TestONNXEmbedderBatch:
    """Tests for batch embedding."""

    def _setup_embedder(self, tmp_model_dir: Path) -> Any:
        from memx.core.embeddings.onnx import ONNXEmbedder

        config = BaseEmbedderConfig(model="all-MiniLM-L6-v2", embedding_dims=DIMS)
        embedder = ONNXEmbedder(config=config, model_dir=str(tmp_model_dir))
        embedder._session = FakeSession(str(tmp_model_dir / "all-MiniLM-L6-v2" / "model.onnx"))
        embedder._tokenizer = FakeTokenizer()
        return embedder

    def test_batch_returns_correct_count(self, tmp_model_dir: Path) -> None:
        embedder = self._setup_embedder(tmp_model_dir)
        texts = ["Hello", "World", "Test input"]
        results = embedder.embed_batch(texts)
        assert len(results) == 3
        for vec in results:
            assert len(vec) == DIMS

    def test_batch_empty_list(self, tmp_model_dir: Path) -> None:
        embedder = self._setup_embedder(tmp_model_dir)
        results = embedder.embed_batch([])
        assert results == []

    def test_batch_matches_individual(self, tmp_model_dir: Path) -> None:
        """Batch results should match individual embed() calls."""
        embedder = self._setup_embedder(tmp_model_dir)
        texts = ["Alpha", "Beta"]
        batch_results = embedder.embed_batch(texts)
        for i, text in enumerate(texts):
            individual = embedder.embed(text)
            np.testing.assert_array_almost_equal(batch_results[i], individual)


# ---------------------------------------------------------------------------
# Tests: Tokenization
# ---------------------------------------------------------------------------


class TestTokenization:
    """Tests for the internal _tokenize method."""

    def test_tokenize_returns_expected_keys(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        embedder._tokenizer = FakeTokenizer()
        result = embedder._tokenize("Hello world")
        assert "input_ids" in result
        assert "attention_mask" in result
        assert "token_type_ids" in result

    def test_tokenize_returns_numpy_arrays(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        embedder._tokenizer = FakeTokenizer()
        result = embedder._tokenize("Hello world")
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            assert isinstance(result[key], np.ndarray)
            assert result[key].dtype == np.int64

    def test_tokenize_batch_dim(self, tmp_model_dir: Path) -> None:
        """All tensors should have batch dimension of 1."""
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        embedder._tokenizer = FakeTokenizer()
        result = embedder._tokenize("Test")
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            assert result[key].shape[0] == 1


# ---------------------------------------------------------------------------
# Tests: Mean pooling
# ---------------------------------------------------------------------------


class TestMeanPooling:
    """Tests for _mean_pooling."""

    def test_mean_pooling_shape(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        token_emb = np.random.randn(1, 5, DIMS).astype(np.float32)
        mask = np.ones((1, 5), dtype=np.int64)
        result = embedder._mean_pooling([token_emb], mask)
        assert result.shape == (DIMS,)

    def test_mean_pooling_normalized(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        token_emb = np.random.randn(1, 5, DIMS).astype(np.float32)
        mask = np.ones((1, 5), dtype=np.int64)
        result = embedder._mean_pooling([token_emb], mask)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_mean_pooling_respects_mask(self, tmp_model_dir: Path) -> None:
        """Padding tokens (mask=0) should not affect the result."""
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        # 3 real tokens + 2 padding
        token_emb = np.array([
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [99, 99, 99], [99, 99, 99]]
        ], dtype=np.float32)
        mask = np.array([[1, 1, 1, 0, 0]], dtype=np.int64)
        result = embedder._mean_pooling([token_emb], mask)
        # Mean of [[1,0,0],[0,1,0],[0,0,1]] = [1/3, 1/3, 1/3], then normalized
        expected_raw = np.array([1 / 3, 1 / 3, 1 / 3])
        expected_norm = expected_raw / np.linalg.norm(expected_raw)
        np.testing.assert_array_almost_equal(result, expected_norm, decimal=5)


# ---------------------------------------------------------------------------
# Tests: Lazy loading (_ensure_loaded)
# ---------------------------------------------------------------------------


class TestLazyLoading:
    """Tests for model lazy loading behavior."""

    def test_ensure_loaded_creates_session(self, tmp_model_dir: Path) -> None:
        """_ensure_loaded should create session and tokenizer."""
        import memx.core.embeddings.onnx as mod

        # Create mock ort module with required attributes
        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.InferenceSession.return_value = FakeSession("dummy")

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
            patch.object(mod, "_HF_HUB_AVAILABLE", True),
            patch.object(mod, "ort", mock_ort),
            patch.object(mod, "Tokenizer", FakeTokenizer),
        ):
            embedder = _create_embedder(tmp_model_dir)
            assert embedder._session is None
            embedder._ensure_loaded()
            assert embedder._session is not None
            assert embedder._tokenizer is not None

    def test_ensure_loaded_is_idempotent(self, tmp_model_dir: Path) -> None:
        """Calling _ensure_loaded twice should not reload."""
        import memx.core.embeddings.onnx as mod

        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.InferenceSession.return_value = FakeSession("dummy")

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
            patch.object(mod, "_HF_HUB_AVAILABLE", True),
            patch.object(mod, "ort", mock_ort),
            patch.object(mod, "Tokenizer", FakeTokenizer),
        ):
            embedder = _create_embedder(tmp_model_dir)
            embedder._ensure_loaded()
            first_session = embedder._session
            embedder._ensure_loaded()
            assert embedder._session is first_session


# ---------------------------------------------------------------------------
# Tests: Model download
# ---------------------------------------------------------------------------


class TestModelDownload:
    """Tests for model download behavior."""

    def test_download_called_when_files_missing(self, empty_model_dir: Path) -> None:
        """Should attempt download when model files don't exist."""
        import memx.core.embeddings.onnx as mod

        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.InferenceSession.return_value = FakeSession("dummy")

        mock_download = MagicMock()

        # After download, create the files so the load succeeds
        def fake_download(repo_id: str, filename: str, **kwargs: Any) -> str:
            local_dir = Path(kwargs.get("local_dir", empty_model_dir))
            target = local_dir / Path(filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fake")
            return str(target)

        mock_download.side_effect = fake_download

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
            patch.object(mod, "_HF_HUB_AVAILABLE", True),
            patch.object(mod, "ort", mock_ort),
            patch.object(mod, "Tokenizer", FakeTokenizer),
            patch.object(mod, "hf_hub_download", mock_download),
        ):
            embedder = _create_embedder(empty_model_dir)
            embedder._ensure_loaded()
            assert mock_download.call_count >= 1

    def test_no_download_when_files_exist(self, tmp_model_dir: Path) -> None:
        """Should NOT download when model files already exist."""
        import memx.core.embeddings.onnx as mod

        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock_ort.InferenceSession.return_value = FakeSession("dummy")

        mock_download = MagicMock()

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
            patch.object(mod, "_HF_HUB_AVAILABLE", True),
            patch.object(mod, "ort", mock_ort),
            patch.object(mod, "Tokenizer", FakeTokenizer),
            patch.object(mod, "hf_hub_download", mock_download),
        ):
            embedder = _create_embedder(tmp_model_dir)
            embedder._ensure_loaded()
            mock_download.assert_not_called()

    def test_auto_download_disabled_raises(self, empty_model_dir: Path) -> None:
        """Should raise RuntimeError when auto_download=False and files missing."""
        import memx.core.embeddings.onnx as mod

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
            patch.object(mod, "_HF_HUB_AVAILABLE", True),
        ):
            embedder = _create_embedder(empty_model_dir, auto_download=False)
            with pytest.raises(RuntimeError, match="auto_download is disabled"):
                embedder._ensure_loaded()

    def test_download_failure_raises_runtime_error(self, empty_model_dir: Path) -> None:
        """Failed download should raise RuntimeError with clear message."""
        import memx.core.embeddings.onnx as mod

        mock_download = MagicMock(side_effect=Exception("Network error"))

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
            patch.object(mod, "_HF_HUB_AVAILABLE", True),
            patch.object(mod, "hf_hub_download", mock_download),
        ):
            embedder = _create_embedder(empty_model_dir)
            with pytest.raises(RuntimeError, match="Failed to download ONNX model"):
                embedder._download_model()


# ---------------------------------------------------------------------------
# Tests: Corrupted model recovery
# ---------------------------------------------------------------------------


class TestCorruptedModelRecovery:
    """Tests for corrupted model file detection and re-download."""

    def test_redownload_on_session_load_failure(self, tmp_model_dir: Path) -> None:
        """Should attempt re-download when ONNX session fails to load."""
        import memx.core.embeddings.onnx as mod

        call_count = 0
        mock_ort = MagicMock()
        mock_ort.SessionOptions.return_value = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

        def session_factory(path: str, **kwargs: Any) -> FakeSession:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Corrupted model")
            return FakeSession(path)

        mock_ort.InferenceSession.side_effect = session_factory

        def fake_download(repo_id: str, filename: str, **kwargs: Any) -> str:
            local_dir = Path(kwargs.get("local_dir", ""))
            target = local_dir / Path(filename).name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixed-model")
            return str(target)

        mock_download = MagicMock(side_effect=fake_download)

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
            patch.object(mod, "_HF_HUB_AVAILABLE", True),
            patch.object(mod, "ort", mock_ort),
            patch.object(mod, "Tokenizer", FakeTokenizer),
            patch.object(mod, "hf_hub_download", mock_download),
        ):
            embedder = _create_embedder(tmp_model_dir)
            embedder._ensure_loaded()
            # First attempt failed, then re-downloaded and loaded
            assert call_count == 2
            assert mock_download.call_count >= 1


# ---------------------------------------------------------------------------
# Tests: Dependency checking
# ---------------------------------------------------------------------------


class TestDependencyChecking:
    """Tests for optional dependency handling."""

    def test_import_memx_embeddings_always_succeeds(self) -> None:
        """Importing the module should never fail, even without onnxruntime."""
        # This test verifies the module-level import is safe
        import memx.core.embeddings.onnx  # noqa: F401

    def test_missing_onnxruntime_raises_on_use(self, tmp_model_dir: Path) -> None:
        """Should raise ImportError with helpful message when ort is missing."""
        import memx.core.embeddings.onnx as mod

        with patch.object(mod, "_ORT_AVAILABLE", False):
            embedder = _create_embedder(tmp_model_dir)
            with pytest.raises(ImportError, match="onnxruntime"):
                embedder._ensure_loaded()

    def test_missing_tokenizers_raises_on_use(self, tmp_model_dir: Path) -> None:
        """Should raise ImportError when tokenizers is missing."""
        import memx.core.embeddings.onnx as mod

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", False),
        ):
            embedder = _create_embedder(tmp_model_dir)
            with pytest.raises(ImportError, match="tokenizers"):
                embedder._ensure_loaded()

    def test_missing_huggingface_hub_raises_on_use(self, tmp_model_dir: Path) -> None:
        """Should raise ImportError when huggingface_hub is missing."""
        import memx.core.embeddings.onnx as mod

        with (
            patch.object(mod, "_ORT_AVAILABLE", True),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", True),
            patch.object(mod, "_HF_HUB_AVAILABLE", False),
        ):
            embedder = _create_embedder(tmp_model_dir)
            with pytest.raises(ImportError, match="huggingface_hub"):
                embedder._ensure_loaded()

    def test_missing_multiple_deps_lists_all(self, tmp_model_dir: Path) -> None:
        """ImportError should list all missing dependencies."""
        import memx.core.embeddings.onnx as mod

        with (
            patch.object(mod, "_ORT_AVAILABLE", False),
            patch.object(mod, "_TOKENIZERS_AVAILABLE", False),
            patch.object(mod, "_HF_HUB_AVAILABLE", False),
        ):
            embedder = _create_embedder(tmp_model_dir)
            with pytest.raises(ImportError, match="onnxruntime.*tokenizers.*huggingface_hub"):
                embedder._ensure_loaded()


# ---------------------------------------------------------------------------
# Tests: File path management
# ---------------------------------------------------------------------------


class TestFilePathManagement:
    """Tests for model path resolution."""

    def test_model_subdir(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        subdir = embedder._model_subdir()
        assert subdir == tmp_model_dir / "all-MiniLM-L6-v2"

    def test_resolve_model_path(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        path = embedder._resolve_model_path()
        assert path == tmp_model_dir / "all-MiniLM-L6-v2" / "model.onnx"

    def test_resolve_tokenizer_path(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        path = embedder._resolve_tokenizer_path()
        assert path == tmp_model_dir / "all-MiniLM-L6-v2" / "tokenizer.json"

    def test_cleanup_model_dir(self, tmp_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(tmp_model_dir))
        subdir = embedder._model_subdir()
        assert subdir.exists()
        embedder._cleanup_model_dir()
        assert not subdir.exists()

    def test_cleanup_nonexistent_dir_is_safe(self, empty_model_dir: Path) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = ONNXEmbedder(model_dir=str(empty_model_dir))
        # Should not raise
        embedder._cleanup_model_dir()


# ---------------------------------------------------------------------------
# Tests: EmbedderFactory registration
# ---------------------------------------------------------------------------


class TestEmbedderFactoryRegistration:
    """Tests that ONNXEmbedder is properly registered in mem0's factory.

    Registration happens at import time when ``memx.embeddings`` is loaded.
    """

    def test_onnx_in_factory_provider_map(self) -> None:
        # Trigger registration by importing the package
        import memx.core.embeddings  # noqa: F401
        from mem0.utils.factory import EmbedderFactory

        assert "onnx" in EmbedderFactory.provider_to_class

    def test_onnx_factory_class_path(self) -> None:
        import memx.core.embeddings  # noqa: F401
        from mem0.utils.factory import EmbedderFactory

        assert EmbedderFactory.provider_to_class["onnx"] == "memx.core.embeddings.onnx.ONNXEmbedder"

    def test_register_is_idempotent(self) -> None:
        """Calling register_onnx_provider multiple times should not error."""
        from memx.core.embeddings import register_onnx_provider

        register_onnx_provider()
        register_onnx_provider()  # Second call should be harmless

        from mem0.utils.factory import EmbedderFactory

        assert "onnx" in EmbedderFactory.provider_to_class

    def test_factory_create_onnx(self, tmp_model_dir: Path) -> None:
        """Factory.create('onnx', ...) should instantiate ONNXEmbedder."""
        import memx.core.embeddings  # noqa: F401
        from mem0.utils.factory import EmbedderFactory
        from memx.core.embeddings.onnx import ONNXEmbedder

        embedder = EmbedderFactory.create(
            "onnx",
            {"model": "all-MiniLM-L6-v2", "embedding_dims": DIMS},
            vector_config=None,
        )
        assert isinstance(embedder, ONNXEmbedder)


# ---------------------------------------------------------------------------
# Tests: EmbeddingBase interface compliance
# ---------------------------------------------------------------------------


class TestEmbeddingBaseInterface:
    """Verify ONNXEmbedder is a proper subclass of EmbeddingBase."""

    def test_is_subclass_of_embedding_base(self) -> None:
        from mem0.embeddings.base import EmbeddingBase
        from memx.core.embeddings.onnx import ONNXEmbedder

        assert issubclass(ONNXEmbedder, EmbeddingBase)

    def test_has_embed_method(self) -> None:
        from memx.core.embeddings.onnx import ONNXEmbedder

        assert hasattr(ONNXEmbedder, "embed")
        assert callable(getattr(ONNXEmbedder, "embed"))

    def test_embed_signature_compatible(self) -> None:
        """embed() should accept text and optional memory_action."""
        import inspect
        from memx.core.embeddings.onnx import ONNXEmbedder

        sig = inspect.signature(ONNXEmbedder.embed)
        params = list(sig.parameters.keys())
        assert "text" in params
        assert "memory_action" in params
