"""ONNXEmbedder — local embedding using ONNX Runtime.

Uses all-MiniLM-L6-v2 by default (384 dimensions). Model files are
auto-downloaded to ~/.memx/models/ on first use and cached for offline
operation thereafter.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import numpy as np

from mem0.configs.embeddings.base import BaseEmbedderConfig
from mem0.embeddings.base import EmbeddingBase

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for optional dependencies
# ---------------------------------------------------------------------------

_ORT_AVAILABLE = False
_TOKENIZERS_AVAILABLE = False
_HF_HUB_AVAILABLE = False

try:
    import onnxruntime as ort  # type: ignore[import-untyped]

    _ORT_AVAILABLE = True
except ImportError:
    ort = None  # type: ignore[assignment]

try:
    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    _TOKENIZERS_AVAILABLE = True
except ImportError:
    Tokenizer = None  # type: ignore[assignment,misc]

try:
    from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

    _HF_HUB_AVAILABLE = True
except ImportError:
    hf_hub_download = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIMS = 384
_DEFAULT_MAX_LENGTH = 256
_DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), ".memx", "models")

# HuggingFace Hub repo for all-MiniLM-L6-v2 ONNX
_HF_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"
_ONNX_FILENAME = "onnx/model.onnx"
_TOKENIZER_FILENAME = "tokenizer.json"


def _check_onnx_deps() -> None:
    """Raise ImportError with a helpful message if optional deps are missing."""
    missing: list[str] = []
    if not _ORT_AVAILABLE:
        missing.append("onnxruntime")
    if not _TOKENIZERS_AVAILABLE:
        missing.append("tokenizers")
    if not _HF_HUB_AVAILABLE:
        missing.append("huggingface_hub")
    if missing:
        raise ImportError(
            f"ONNXEmbedder requires: {', '.join(missing)}. "
            f"Install with: pip install {' '.join(missing)}"
        )


# ---------------------------------------------------------------------------
# ONNXEmbedder
# ---------------------------------------------------------------------------


class ONNXEmbedder(EmbeddingBase):
    """Local embedding provider using ONNX Runtime.

    Implements the mem0 ``EmbeddingBase`` interface so it can be used as a
    drop-in replacement for cloud-based embedders.

    Parameters
    ----------
    config : BaseEmbedderConfig, optional
        Standard mem0 embedder config.  ``model`` defaults to
        ``all-MiniLM-L6-v2`` and ``embedding_dims`` to 384.
    model_dir : str, optional
        Directory for cached model files.  Defaults to ``~/.memx/models/``.
    max_length : int, optional
        Maximum token length for input text.  Defaults to 256.
    auto_download : bool, optional
        Whether to download the model automatically.  Defaults to True.
    """

    def __init__(
        self,
        config: Optional[BaseEmbedderConfig] = None,
        model_dir: Optional[str] = None,
        max_length: int = _DEFAULT_MAX_LENGTH,
        auto_download: bool = True,
    ) -> None:
        # Defer dependency check until actual usage, not import time
        super().__init__(config)

        # Apply defaults
        self.config.model = self.config.model or _DEFAULT_MODEL
        self.config.embedding_dims = self.config.embedding_dims or _DEFAULT_DIMS

        self._model_dir = Path(model_dir or _DEFAULT_MODEL_DIR)
        self._max_length = max_length
        self._auto_download = auto_download

        # Lazy-loaded resources
        self._session: Any = None  # ort.InferenceSession
        self._tokenizer: Any = None  # tokenizers.Tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        text: str,
        memory_action: Optional[Literal["add", "search", "update"]] = None,
    ) -> List[float]:
        """Embed a single text string into a dense vector.

        Args:
            text: Input text to embed.
            memory_action: Unused; kept for interface compatibility.

        Returns:
            List of floats with ``embedding_dims`` dimensions.
        """
        self._ensure_loaded()

        if not text or not text.strip():
            return [0.0] * self.config.embedding_dims

        tokens = self._tokenize(text)
        outputs = self._session.run(None, tokens)
        vector = self._mean_pooling(outputs, tokens["attention_mask"])
        return vector.tolist()

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed multiple texts.

        Currently processes sequentially; batched ONNX inference can be
        added later for throughput improvements.

        Args:
            texts: Sequence of input strings.

        Returns:
            List of embedding vectors.
        """
        return [self.embed(t) for t in texts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load model and tokenizer on first call (lazy init)."""
        if self._session is not None:
            return

        _check_onnx_deps()

        model_path = self._resolve_model_path()
        tokenizer_path = self._resolve_tokenizer_path()

        # Download if needed
        if not model_path.exists() or not tokenizer_path.exists():
            if not self._auto_download:
                raise RuntimeError(
                    f"Model files not found at {self._model_dir / self.config.model} "
                    "and auto_download is disabled."
                )
            self._download_model()

        # Load ONNX session
        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:
            # Model file corrupted — re-download
            logger.warning("Failed to load ONNX model, re-downloading: %s", exc)
            self._cleanup_model_dir()
            self._download_model()
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

        # Load tokenizer
        try:
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        except Exception as exc:
            logger.warning("Failed to load tokenizer, re-downloading: %s", exc)
            self._cleanup_model_dir()
            self._download_model()
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Configure truncation
        if self._tokenizer.truncation is None:
            self._tokenizer.enable_truncation(max_length=self._max_length)

        # Configure padding
        self._tokenizer.enable_padding(
            length=self._max_length,
            pad_id=0,
            pad_token="[PAD]",
        )

        logger.info(
            "ONNXEmbedder loaded: model=%s, dims=%d",
            self.config.model,
            self.config.embedding_dims,
        )

    def _tokenize(self, text: str) -> Dict[str, np.ndarray]:
        """Tokenize text and return numpy arrays for ONNX input."""
        encoded = self._tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def _mean_pooling(
        self, model_output: List[np.ndarray], attention_mask: np.ndarray
    ) -> np.ndarray:
        """Apply mean pooling to token embeddings using the attention mask.

        Args:
            model_output: Raw ONNX model outputs. First element is
                token embeddings with shape (1, seq_len, hidden_dim).
            attention_mask: Binary mask of shape (1, seq_len).

        Returns:
            Normalized embedding vector of shape (hidden_dim,).
        """
        token_embeddings = model_output[0]  # (1, seq_len, hidden_dim)
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask  # (1, hidden_dim)

        # L2 normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        norm = np.clip(norm, a_min=1e-9, a_max=None)
        normalized = pooled / norm

        return normalized[0]

    # ------------------------------------------------------------------
    # Model file management
    # ------------------------------------------------------------------

    def _model_subdir(self) -> Path:
        """Return the per-model subdirectory."""
        return self._model_dir / self.config.model

    def _resolve_model_path(self) -> Path:
        """Return the full path to the ONNX model file."""
        return self._model_subdir() / "model.onnx"

    def _resolve_tokenizer_path(self) -> Path:
        """Return the full path to the tokenizer JSON file."""
        return self._model_subdir() / "tokenizer.json"

    def _cleanup_model_dir(self) -> None:
        """Remove the model subdirectory for a clean re-download."""
        subdir = self._model_subdir()
        if subdir.exists():
            shutil.rmtree(subdir, ignore_errors=True)

    def _download_model(self) -> None:
        """Download model files from Hugging Face Hub.

        Downloads the ONNX model and tokenizer.json into the local
        model cache directory.
        """
        _check_onnx_deps()

        subdir = self._model_subdir()
        subdir.mkdir(parents=True, exist_ok=True)

        model_target = self._resolve_model_path()
        tokenizer_target = self._resolve_tokenizer_path()

        logger.info("Downloading ONNX model '%s' to %s ...", self.config.model, subdir)

        try:
            # Download ONNX model
            if not model_target.exists():
                downloaded = hf_hub_download(
                    repo_id=_HF_REPO_ID,
                    filename=_ONNX_FILENAME,
                    local_dir=str(subdir),
                    local_dir_use_symlinks=False,
                )
                # hf_hub_download may place the file in a subdirectory
                downloaded_path = Path(downloaded)
                if downloaded_path != model_target:
                    shutil.copy2(str(downloaded_path), str(model_target))

            # Download tokenizer
            if not tokenizer_target.exists():
                downloaded = hf_hub_download(
                    repo_id=_HF_REPO_ID,
                    filename=_TOKENIZER_FILENAME,
                    local_dir=str(subdir),
                    local_dir_use_symlinks=False,
                )
                downloaded_path = Path(downloaded)
                if downloaded_path != tokenizer_target:
                    shutil.copy2(str(downloaded_path), str(tokenizer_target))

        except Exception as exc:
            raise RuntimeError(
                f"Failed to download ONNX model '{self.config.model}': {exc}"
            ) from exc

        logger.info("Model download complete: %s", subdir)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def dimensions(self) -> int:
        """Return the output embedding dimensionality."""
        return self.config.embedding_dims

    def __repr__(self) -> str:
        return (
            f"ONNXEmbedder(model={self.config.model!r}, "
            f"dims={self.config.embedding_dims}, "
            f"loaded={self._session is not None})"
        )
