"""MemX configuration models.

Defines all ACE sub-configs and the top-level MemXConfig.
Uses Pydantic v2 for validation with sensible defaults so that
``MemXConfig()`` always succeeds with zero arguments.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from memx.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Valid reflector operating modes
_VALID_REFLECTOR_MODES = frozenset({"rules", "llm", "hybrid"})


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------


class ReflectorConfig(BaseModel):
    """Configuration for the Reflector engine."""

    mode: str = "rules"  # "rules" | "llm" | "hybrid"
    min_score: float = Field(default=30.0, ge=0.0, le=100.0)
    max_content_length: int = Field(default=500, gt=0)
    max_code_lines: int = Field(default=3, gt=0)

    @model_validator(mode="after")
    def _validate_mode(self) -> ReflectorConfig:
        if self.mode not in _VALID_REFLECTOR_MODES:
            raise ValueError(
                f"reflector.mode must be one of {sorted(_VALID_REFLECTOR_MODES)}, "
                f"got {self.mode!r}"
            )
        return self


class CuratorConfig(BaseModel):
    """Configuration for the Curator (dedup / merge) engine."""

    similarity_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    merge_strategy: str = "keep_best"  # "keep_best" | "merge_content"
    conflict_detection: bool = False


class DecayConfig(BaseModel):
    """Configuration for temporal decay of Bullet weights."""

    half_life_days: float = Field(default=30.0, gt=0)
    boost_factor: float = Field(default=0.1, ge=0)
    protection_days: int = Field(default=7, ge=0)
    permanent_threshold: int = Field(default=15, ge=1)
    archive_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    sweep_on_session_end: bool = True


class RetrievalConfig(BaseModel):
    """Configuration for search / retrieval scoring."""

    keyword_weight: float = Field(default=0.6, ge=0.0)
    semantic_weight: float = Field(default=0.4, ge=0.0)
    recency_boost_days: int = Field(default=7, ge=0)
    recency_boost_factor: float = Field(default=1.2, ge=1.0)
    max_results: int = Field(default=5, gt=0)
    token_budget: int = Field(default=2000, gt=0)


class PrivacyConfig(BaseModel):
    """Configuration for PII / secret sanitisation."""

    custom_patterns: list[str] = Field(default_factory=list)
    sanitize_paths: bool = True
    always_sanitize: bool = False


class IntegrationConfig(BaseModel):
    """Configuration for Claude Code integration hooks."""

    auto_recall: bool = True
    auto_reflect: bool = True
    sweep_on_exit: bool = True


class DaemonConfig(BaseModel):
    """Configuration for the optional background daemon."""

    enabled: bool = False
    idle_timeout_seconds: int = Field(default=300, gt=0)
    socket_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class MemXConfig(BaseModel):
    """Root configuration for the MemX Adaptive Context Engine.

    All fields carry defaults so ``MemXConfig()`` always succeeds.
    Use ``from_dict`` to build from a flat dictionary that mixes ACE
    and mem0-native keys.
    """

    ace_enabled: bool = False
    reflector: ReflectorConfig = Field(default_factory=ReflectorConfig)
    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    decay: DecayConfig = Field(default_factory=DecayConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    daemon: DaemonConfig = Field(default_factory=DaemonConfig)
    mem0_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_weights(self) -> MemXConfig:
        """Warn (but do not error) if retrieval weights do not sum to 1.0."""
        kw = self.retrieval.keyword_weight
        sw = self.retrieval.semantic_weight
        total = kw + sw
        if abs(total - 1.0) > 0.01:
            warnings.warn(
                f"keyword_weight ({kw}) + semantic_weight ({sw}) = {total}, not 1.0. "
                "ScoreMerger will normalize at runtime.",
                stacklevel=2,
            )
        return self

    # -- Factory helpers ----------------------------------------------------

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> MemXConfig:
        """Create config from *config_dict*, separating ACE fields from mem0 fields.

        Keys that belong to the ACE sub-systems are extracted; everything
        else is collected under ``mem0_config`` and forwarded to the mem0
        backend as-is.
        """
        ace_keys = {
            "ace_enabled",
            "reflector",
            "curator",
            "decay",
            "retrieval",
            "privacy",
            "integration",
            "daemon",
        }
        ace_fields: dict[str, Any] = {}
        mem0_fields: dict[str, Any] = {}
        for key, value in config_dict.items():
            if key in ace_keys:
                ace_fields[key] = value
            else:
                mem0_fields[key] = value
        ace_fields["mem0_config"] = mem0_fields
        try:
            return cls.model_validate(ace_fields)
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e

    def to_mem0_config(self) -> dict[str, Any]:
        """Return a *copy* of the mem0-compatible config dict."""
        return dict(self.mem0_config)
