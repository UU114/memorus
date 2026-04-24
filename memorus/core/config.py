"""Memorus configuration models.

Defines all ACE sub-configs and the top-level MemorusConfig.
Uses Pydantic v2 for validation with sensible defaults so that
``MemorusConfig()`` always succeeds with zero arguments.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from memorus.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Valid reflector operating modes
_VALID_REFLECTOR_MODES = frozenset({"rules", "llm", "hybrid"})


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------


class ReflectorBatchConfig(BaseModel):
    """Configuration for delayed batch distillation (STORY-R095).

    When ``batch_enabled`` is ``True`` and the reflector runs in a mode that
    uses the LLM ("llm" or "hybrid"), non-correction turns are appended to
    ``inbox_path`` instead of being distilled immediately. A BatchAnalyzer
    stage drains the inbox during the IdleOrchestrator pass.
    """

    # Default OFF — delayed batching is opt-in per deployment so that the
    # vanilla ReflectorEngine remains backwards-compatible with callers that
    # expect realtime bullets. Enable via config or the daemon bootstrap.
    batch_enabled: bool = False
    inbox_path: str = ".ace/inbox.jsonl"
    # Triggers
    batch_min_turns: int = Field(default=10, ge=1)
    batch_max_age_seconds: int = Field(default=4 * 3600, ge=0)
    # Batch sizing
    min_batch_size: int = Field(default=10, ge=1)
    max_batch_size: int = Field(default=50, ge=1)
    # Token budget per single prompt (Analyze or Generate)
    prompt_token_budget: int = Field(default=8000, gt=0)
    # Retention
    consumed_retention_seconds: int = Field(default=7 * 24 * 3600, ge=0)
    # Failure handling
    max_batch_retries: int = Field(default=3, ge=1)
    # Provisional search
    provisional_score_factor: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_sizes(self) -> ReflectorBatchConfig:
        if self.max_batch_size < self.min_batch_size:
            raise ValueError(
                f"max_batch_size ({self.max_batch_size}) must be >= "
                f"min_batch_size ({self.min_batch_size})"
            )
        return self


class ReflectorConfig(BaseModel):
    """Configuration for the Reflector engine."""

    mode: str = "hybrid"  # "rules" | "llm" | "hybrid"
    min_score: float = Field(default=30.0, ge=0.0, le=100.0)
    max_content_length: int = Field(default=500, gt=0)
    max_code_lines: int = Field(default=3, gt=0)

    # LLM settings (used by "llm" and "hybrid" modes)
    llm_model: str = "openai/gpt-4o-mini"
    llm_api_base: Optional[str] = None
    llm_api_key: Optional[str] = None
    max_eval_tokens: int = Field(default=512, gt=0)
    max_distill_tokens: int = Field(default=256, gt=0)
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # STORY-R095 — delayed batch distillation
    batch: ReflectorBatchConfig = Field(default_factory=ReflectorBatchConfig)

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
    conflict_min_similarity: float = Field(default=0.5, ge=0.0, le=1.0)
    conflict_max_similarity: float = Field(default=0.8, ge=0.0, le=1.0)


class DecayConfig(BaseModel):
    """Configuration for temporal decay of Bullet weights."""

    half_life_days: float = Field(default=30.0, gt=0)
    boost_factor: float = Field(default=0.1, ge=0)
    protection_days: int = Field(default=7, ge=0)
    permanent_threshold: int = Field(default=15, ge=1)
    archive_threshold: float = Field(default=0.02, ge=0.0, le=1.0)
    sweep_on_session_end: bool = True


class GraphExpansionConfig(BaseModel):
    """Configuration for Generator graph-aware retrieval (STORY-R096).

    Mirrors ``memorus_core::config::GraphExpansionConfig`` on the Rust side so
    TOML files can be shared. When ``enabled`` is ``False`` the Generator
    behaves byte-identically to its pre-R096 form.
    """

    enabled: bool = True
    expand_hops: int = Field(default=1, ge=1, le=2)
    k_expand: int = Field(default=5, ge=0)
    graph_score_weight: float = Field(default=0.3, ge=0.0)
    co_recall_delta: float = Field(default=0.1, ge=0.0, le=1.0)
    co_recall_max_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    # Consumed by R094's IdleOrchestrator; Generator never schedules decay.
    monthly_decay_factor: float = Field(default=0.95, ge=0.0, le=1.0)


class RetrievalConfig(BaseModel):
    """Configuration for search / retrieval scoring."""

    keyword_weight: float = Field(default=0.6, ge=0.0)
    semantic_weight: float = Field(default=0.4, ge=0.0)
    recency_boost_days: int = Field(default=7, ge=0)
    recency_boost_factor: float = Field(default=1.2, ge=1.0)
    scope_boost: float = Field(default=1.3, ge=1.0)
    max_results: int = Field(default=5, gt=0)
    token_budget: int = Field(default=2000, gt=0)
    graph_expansion: GraphExpansionConfig = Field(default_factory=GraphExpansionConfig)


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
    context_template: str = "xml"  # "xml" | "markdown" | "plain"


class DaemonConfig(BaseModel):
    """Configuration for the optional background daemon."""

    enabled: bool = False
    idle_timeout_seconds: int = Field(default=300, gt=0)
    socket_path: Optional[str] = None


class ConsolidateConflictConfig(BaseModel):
    """Thresholds for the three-tier conflict triage in ConsolidateExecutor."""

    auto_supersede_min_confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    review_queue_min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)


class ConsolidateConfig(BaseModel):
    """IdleOrchestrator + ConsolidateExecutor configuration (STORY-R094).

    Mirrors the Rust ``memorus_core::config::ConsolidateConfig``.
    """

    check_interval_secs: int = Field(default=600, gt=0)
    min_idle_secs: int = Field(default=180, ge=0)
    min_gap_secs: int = Field(default=1800, ge=0)
    max_per_pass: int = Field(default=5000, gt=0)
    log_path: str = ".ace/log.md"
    review_queue_path: str = ".ace/review_queue.jsonl"
    conflict: ConsolidateConflictConfig = Field(
        default_factory=ConsolidateConflictConfig
    )


class VerificationConfig(BaseModel):
    """Memory Trust / Verification Layer configuration (STORY-R099).

    Governs the behavior of the FileSystemVerifier (STORY-R101) and the
    retrieval-time verification policy (STORY-R102/R103). This Story ships
    only the data shape — engines plug in later.

    * ``enabled``: master switch; when ``False`` the verifier never runs.
    * ``ttl_seconds``: cache lifetime for a single bullet's verification
      result before re-checking against the filesystem.
    * ``project_root``: root directory that ``Anchor.file_path`` is relative
      to. If ``None`` at runtime the engine will resolve from the workspace.
    * ``verified_trust_score`` / ``stale_trust_score`` /
      ``unverifiable_trust_score``: per-state trust multipliers applied by
      the scoring pipeline.
    * ``policy``: how a stale bullet is handled — ``flag`` keeps it with a
      warning, ``demote`` lowers its rank, ``drop`` removes it entirely.

    Mirrors ``memorus_core::config::VerificationConfig`` on the Rust side.
    """

    enabled: bool = True
    ttl_seconds: int = Field(default=60, ge=0)
    project_root: str | None = None
    verified_trust_score: float = Field(default=1.0, ge=0.0, le=1.0)
    stale_trust_score: float = Field(default=0.3, ge=0.0, le=1.0)
    unverifiable_trust_score: float = Field(default=0.7, ge=0.0, le=1.0)
    policy: Literal["flag", "demote", "drop"] = "flag"


class TopicsConfig(BaseModel):
    """TopicPage aggregation layer configuration (STORY-R097).

    Opt-in — defaults keep the Generator byte-identical to its pre-R097
    behaviour. Mirrors the Rust ``memorus_core::config::TopicsConfig``.
    """

    enabled: bool = False
    min_cluster_size: int = Field(default=3, ge=1)
    similarity_edge_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    page_summary_max_tokens: int = Field(default=800, gt=0)
    page_regen_drift_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    topic_match_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    llm_model: str = "gpt-4o-mini"
    sqlite_path: str = ".ace/topics.db"
    pages_dir: str = ".ace/pages"
    # Weight split applied when a topic hit wins over the bullet path.
    page_score_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    backing_score_weight: float = Field(default=0.3, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Top-level configuration
# ---------------------------------------------------------------------------


class MemorusConfig(BaseModel):
    """Root configuration for the Memorus Adaptive Context Engine.

    All fields carry defaults so ``MemorusConfig()`` always succeeds.
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
    consolidate: ConsolidateConfig = Field(default_factory=ConsolidateConfig)
    topics: TopicsConfig = Field(default_factory=TopicsConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    mem0_config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_weights(self) -> MemorusConfig:
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
    def from_dict(cls, config_dict: dict[str, Any]) -> MemorusConfig:
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
            "consolidate",
            "topics",
            "verification",
        }
        ace_fields: dict[str, Any] = {}
        mem0_fields: dict[str, Any] = {}
        for key, value in config_dict.items():
            if key in ace_keys:
                ace_fields[key] = value
            else:
                mem0_fields[key] = value
        ace_fields["mem0_config"] = mem0_fields
        logger.debug(
            "MemorusConfig.from_dict: ace_keys=%s mem0_keys=%s",
            sorted(ace_fields.keys() - {"mem0_config"}),
            sorted(mem0_fields.keys()),
        )
        try:
            cfg = cls.model_validate(ace_fields)
            logger.debug(
                "MemorusConfig created: ace_enabled=%s reflector.mode=%s "
                "curator.threshold=%.2f decay.half_life=%.1f",
                cfg.ace_enabled, cfg.reflector.mode,
                cfg.curator.similarity_threshold, cfg.decay.half_life_days,
            )
            return cfg
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}") from e

    def to_mem0_config(self) -> dict[str, Any]:
        """Return a *copy* of the mem0-compatible config dict."""
        return dict(self.mem0_config)
