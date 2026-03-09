"""MemX embedding providers.

Currently supports:
- ``ONNXEmbedder``: Local embedding via ONNX Runtime (all-MiniLM-L6-v2).

On import, this module registers the "onnx" provider with mem0's
``EmbedderFactory`` so that ``provider="onnx"`` works out of the box.
"""

from __future__ import annotations

import logging

# Lazy import: ONNXEmbedder requires optional onnxruntime dep.
# Importing the module is safe; the class checks deps on first use.
from memx.core.embeddings.onnx import ONNXEmbedder

logger = logging.getLogger(__name__)

__all__ = ["ONNXEmbedder", "register_onnx_provider"]


def register_onnx_provider() -> None:
    """Register ONNXEmbedder as provider='onnx' in mem0's EmbedderFactory.

    This modifies the class-level ``provider_to_class`` dict at runtime
    so that ``EmbedderFactory.create("onnx", ...)`` resolves to
    ``memx.embeddings.onnx.ONNXEmbedder``.

    Also patches the ``EmbedderConfig`` validator to accept "onnx".

    Safe to call multiple times (idempotent).
    """
    try:
        from mem0.utils.factory import EmbedderFactory

        if "onnx" not in EmbedderFactory.provider_to_class:
            EmbedderFactory.provider_to_class["onnx"] = (
                "memx.core.embeddings.onnx.ONNXEmbedder"
            )
            logger.debug("Registered 'onnx' provider in EmbedderFactory")
    except ImportError:
        logger.debug("mem0 not available; skipping EmbedderFactory registration")

    try:
        from mem0.embeddings.configs import EmbedderConfig

        # Patch the validator to accept "onnx" by updating the validator function.
        # The simplest approach: check if "onnx" is already handled by the existing
        # validator. If not, we monkey-patch the validate_config to accept it.
        # Since the validator uses a static list, we wrap it.
        original_validator = EmbedderConfig.__pydantic_validator__
        # We test if it already accepts "onnx"
        try:
            EmbedderConfig(provider="onnx", config={})
            # Already accepted — nothing to do
        except Exception:
            # Need to patch — simplest way is to modify the field_validator
            # by rebuilding the model with updated validation
            _patch_embedder_config_validator()
    except ImportError:
        pass


def _patch_embedder_config_validator() -> None:
    """Patch EmbedderConfig to accept 'onnx' as a valid provider.

    This replaces the validate_config field validator to include 'onnx'
    in the accepted providers list.
    """
    try:
        import mem0.embeddings.configs as configs_mod

        _VALID_PROVIDERS = [
            "openai",
            "ollama",
            "huggingface",
            "azure_openai",
            "gemini",
            "vertexai",
            "together",
            "lmstudio",
            "langchain",
            "aws_bedrock",
            "fastembed",
            "onnx",
        ]

        from pydantic import BaseModel, Field, field_validator
        from typing import Optional

        class PatchedEmbedderConfig(BaseModel):
            provider: str = Field(
                description="Provider of the embedding model",
                default="openai",
            )
            config: Optional[dict] = Field(
                description="Configuration for the specific embedding model",
                default={},
            )

            @field_validator("config")
            @classmethod
            def validate_config(cls, v, values):  # type: ignore[no-untyped-def]
                provider = values.data.get("provider")
                if provider in _VALID_PROVIDERS:
                    return v
                else:
                    raise ValueError(f"Unsupported embedding provider: {provider}")

        # Replace the class in the module
        configs_mod.EmbedderConfig = PatchedEmbedderConfig  # type: ignore[attr-defined]
        logger.debug("Patched EmbedderConfig to accept 'onnx' provider")
    except Exception as exc:
        logger.debug("Failed to patch EmbedderConfig: %s", exc)


# Auto-register on import
register_onnx_provider()
