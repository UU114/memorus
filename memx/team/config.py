"""TeamConfig — independent configuration model for Team Memory layer.

Completely separate from MemXConfig (no inheritance, no nesting).
Supports loading from YAML files and environment variables (MEMX_TEAM_* prefix).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# File search paths in priority order
_CONFIG_FILENAMES = ("team_config.yaml",)
_CONFIG_SEARCH_PATHS = (
    Path("."),
    Path.home() / ".ace",
)


# ---------------------------------------------------------------------------
# Sub-configuration models
# ---------------------------------------------------------------------------


class LayerBoostConfig(BaseModel):
    """Weight multipliers for local vs team bullets during retrieval."""

    local_boost: float = 1.5
    team_boost: float = 1.0


class AutoNominateConfig(BaseModel):
    """Controls automatic nomination of local bullets to the team pool."""

    min_recall_count: int = 3
    min_score: float = 70.0
    max_prompts_per_session: int = 1
    silent: bool = False


class RedactorConfig(BaseModel):
    """PII / sensitive-data redaction before sharing to team."""

    llm_generalize: bool = False
    custom_patterns: list[str] = Field(default_factory=list)


_MAX_OVERRIDE_DAYS = 90


class MandatoryOverride(BaseModel):
    """Project-level override that temporarily bypasses a team mandatory bullet.

    Validation rules:
      - reason must be non-empty (whitespace-only is rejected).
      - expires must be set and at most 90 days from now.
    """

    bullet_id: str
    reason: str  # required — why this override exists
    expires: datetime  # required — when the override expires

    @model_validator(mode="after")
    def _validate_override(self) -> MandatoryOverride:
        """Ensure reason is non-empty and expires is within 90 days."""
        if not self.reason or not self.reason.strip():
            raise ValueError("MandatoryOverride.reason must be non-empty")
        now = datetime.now(timezone.utc)
        # Normalize naive datetimes to UTC for comparison
        exp = self.expires
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        max_expiry = now + timedelta(days=_MAX_OVERRIDE_DAYS)
        if exp > max_expiry:
            raise ValueError(
                f"MandatoryOverride.expires must be at most {_MAX_OVERRIDE_DAYS} days "
                f"from now (got {exp.isoformat()}, max {max_expiry.isoformat()})"
            )
        return self

    def is_active(self, *, now: datetime | None = None) -> bool:
        """Return True if this override has not yet expired."""
        if now is None:
            now = datetime.now(timezone.utc)
        exp = self.expires
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now < exp


# ---------------------------------------------------------------------------
# Top-level TeamConfig
# ---------------------------------------------------------------------------


class TeamConfig(BaseModel):
    """Root configuration for the Team Memory layer.

    Independent from MemXConfig — no inheritance, no nesting.
    All fields carry sensible defaults so ``TeamConfig()`` always succeeds.
    """

    enabled: bool = False
    server_url: Optional[str] = None
    team_id: Optional[str] = None
    subscribed_tags: list[str] = Field(default_factory=list)
    cache_max_bullets: int = 2000
    cache_ttl_minutes: int = 60
    layer_boost: LayerBoostConfig = Field(default_factory=LayerBoostConfig)
    auto_nominate: AutoNominateConfig = Field(default_factory=AutoNominateConfig)
    redactor: RedactorConfig = Field(default_factory=RedactorConfig)
    mandatory_overrides: list[MandatoryOverride] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _find_config_file() -> Optional[Path]:
    """Locate the first existing team_config.yaml in search paths."""
    for search_dir in _CONFIG_SEARCH_PATHS:
        for filename in _CONFIG_FILENAMES:
            candidate = search_dir / filename
            if candidate.is_file():
                logger.debug("Found team config file: %s", candidate)
                return candidate
    return None


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load YAML file and return as dict. Returns empty dict on failure."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("PyYAML not installed; cannot load %s", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.warning("Failed to load team config from %s: %s", path, e)
        return {}


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Override top-level fields with MEMX_TEAM_* environment variables.

    Supported env vars:
      MEMX_TEAM_ENABLED        -> enabled (bool)
      MEMX_TEAM_SERVER_URL     -> server_url (str)
      MEMX_TEAM_TEAM_ID        -> team_id (str)
      MEMX_TEAM_SUBSCRIBED_TAGS -> subscribed_tags (comma-separated)
      MEMX_TEAM_CACHE_MAX_BULLETS -> cache_max_bullets (int)
      MEMX_TEAM_CACHE_TTL_MINUTES -> cache_ttl_minutes (int)
    """
    env_map: dict[str, tuple[str, type]] = {
        "MEMX_TEAM_ENABLED": ("enabled", bool),
        "MEMX_TEAM_SERVER_URL": ("server_url", str),
        "MEMX_TEAM_TEAM_ID": ("team_id", str),
        "MEMX_TEAM_SUBSCRIBED_TAGS": ("subscribed_tags", list),
        "MEMX_TEAM_CACHE_MAX_BULLETS": ("cache_max_bullets", int),
        "MEMX_TEAM_CACHE_TTL_MINUTES": ("cache_ttl_minutes", int),
    }

    for env_key, (field_name, field_type) in env_map.items():
        raw = os.environ.get(env_key)
        if raw is None:
            continue

        if field_type is bool:
            data[field_name] = raw.lower() in ("true", "1", "yes")
        elif field_type is int:
            try:
                data[field_name] = int(raw)
            except ValueError:
                logger.warning("Invalid int for %s: %r", env_key, raw)
        elif field_type is list:
            data[field_name] = [t.strip() for t in raw.split(",") if t.strip()]
        else:
            data[field_name] = raw

    return data


def load_team_config(config_path: Optional[Path] = None) -> TeamConfig:
    """Load TeamConfig from file + environment variable overrides.

    Priority: env vars > file values > defaults.

    Args:
        config_path: Explicit path to YAML config. If None, auto-discovers.

    Returns:
        Validated TeamConfig instance.
    """
    # Load from file
    path = config_path or _find_config_file()
    if path is not None:
        data = _load_yaml_file(path)
        logger.info("Loaded team config from %s", path)
    else:
        data = {}
        logger.debug("No team config file found; using defaults + env vars")

    # Apply environment variable overrides
    data = _apply_env_overrides(data)

    return TeamConfig.model_validate(data)
