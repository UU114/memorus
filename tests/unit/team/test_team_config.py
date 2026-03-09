"""Unit tests for memx.team.config — TeamConfig and sub-models."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from memx.team.config import (
    AutoNominateConfig,
    LayerBoostConfig,
    MandatoryOverride,
    RedactorConfig,
    TeamConfig,
    load_team_config,
)


# ---------------------------------------------------------------------------
# TeamConfig independence check
# ---------------------------------------------------------------------------


class TestTeamConfigIndependence:
    """Verify TeamConfig does not depend on MemXConfig."""

    def test_no_inheritance_from_memx_config(self) -> None:
        from memx.core.config import MemXConfig

        assert not issubclass(TeamConfig, MemXConfig)

    def test_no_nesting_of_memx_config(self) -> None:
        tc = TeamConfig()
        for field_name, field_info in TeamConfig.model_fields.items():
            assert "MemXConfig" not in str(field_info.annotation)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestDefaults:
    """All models should have sensible defaults."""

    def test_team_config_defaults(self) -> None:
        tc = TeamConfig()
        assert tc.enabled is False
        assert tc.server_url is None
        assert tc.team_id is None
        assert tc.subscribed_tags == []
        assert tc.cache_max_bullets == 2000
        assert tc.cache_ttl_minutes == 60
        assert tc.mandatory_overrides == []

    def test_layer_boost_defaults(self) -> None:
        lb = LayerBoostConfig()
        assert lb.local_boost == 1.5
        assert lb.team_boost == 1.0

    def test_auto_nominate_defaults(self) -> None:
        an = AutoNominateConfig()
        assert an.min_recall_count == 3
        assert an.min_score == 70.0
        assert an.max_prompts_per_session == 1
        assert an.silent is False

    def test_redactor_defaults(self) -> None:
        r = RedactorConfig()
        assert r.llm_generalize is False
        assert r.custom_patterns == []

    def test_nested_sub_models_defaults(self) -> None:
        tc = TeamConfig()
        assert tc.layer_boost.local_boost == 1.5
        assert tc.auto_nominate.min_recall_count == 3
        assert tc.redactor.llm_generalize is False


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


class TestValidation:
    """Test Pydantic validation on models."""

    def test_mandatory_override_requires_reason_and_expires(self) -> None:
        # Valid
        mo = MandatoryOverride(
            bullet_id="b-123",
            reason="Critical security policy",
            expires=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert mo.bullet_id == "b-123"
        assert mo.reason == "Critical security policy"

    def test_mandatory_override_missing_reason_raises(self) -> None:
        with pytest.raises(Exception):
            MandatoryOverride(
                bullet_id="b-123",
                expires=datetime.now(timezone.utc) + timedelta(days=30),
            )  # type: ignore[call-arg]

    def test_mandatory_override_missing_expires_raises(self) -> None:
        with pytest.raises(Exception):
            MandatoryOverride(
                bullet_id="b-123",
                reason="test",
            )  # type: ignore[call-arg]

    def test_team_config_with_custom_values(self) -> None:
        tc = TeamConfig(
            enabled=True,
            server_url="https://team.example.com",
            team_id="team-alpha",
            subscribed_tags=["python", "devops"],
            cache_max_bullets=500,
            cache_ttl_minutes=30,
        )
        assert tc.enabled is True
        assert tc.server_url == "https://team.example.com"
        assert tc.team_id == "team-alpha"
        assert tc.subscribed_tags == ["python", "devops"]
        assert tc.cache_max_bullets == 500
        assert tc.cache_ttl_minutes == 30

    def test_team_config_with_mandatory_overrides(self) -> None:
        tc = TeamConfig(
            mandatory_overrides=[
                MandatoryOverride(
                    bullet_id="b-1",
                    reason="Policy A",
                    expires=datetime(2026, 6, 1),
                ),
            ]
        )
        assert len(tc.mandatory_overrides) == 1
        assert tc.mandatory_overrides[0].bullet_id == "b-1"


# ---------------------------------------------------------------------------
# Environment variable overrides
# ---------------------------------------------------------------------------


class TestEnvOverrides:
    """Test loading config with environment variable overrides."""

    def test_env_enabled(self) -> None:
        with patch.dict(os.environ, {"MEMX_TEAM_ENABLED": "true"}, clear=False):
            tc = load_team_config()
        assert tc.enabled is True

    def test_env_enabled_false(self) -> None:
        with patch.dict(os.environ, {"MEMX_TEAM_ENABLED": "false"}, clear=False):
            tc = load_team_config()
        assert tc.enabled is False

    def test_env_server_url(self) -> None:
        with patch.dict(
            os.environ,
            {"MEMX_TEAM_SERVER_URL": "https://my-server.io"},
            clear=False,
        ):
            tc = load_team_config()
        assert tc.server_url == "https://my-server.io"

    def test_env_team_id(self) -> None:
        with patch.dict(os.environ, {"MEMX_TEAM_TEAM_ID": "team-42"}, clear=False):
            tc = load_team_config()
        assert tc.team_id == "team-42"

    def test_env_subscribed_tags(self) -> None:
        with patch.dict(
            os.environ,
            {"MEMX_TEAM_SUBSCRIBED_TAGS": "python, rust, go"},
            clear=False,
        ):
            tc = load_team_config()
        assert tc.subscribed_tags == ["python", "rust", "go"]

    def test_env_cache_max_bullets(self) -> None:
        with patch.dict(
            os.environ, {"MEMX_TEAM_CACHE_MAX_BULLETS": "500"}, clear=False
        ):
            tc = load_team_config()
        assert tc.cache_max_bullets == 500

    def test_env_cache_ttl_minutes(self) -> None:
        with patch.dict(
            os.environ, {"MEMX_TEAM_CACHE_TTL_MINUTES": "120"}, clear=False
        ):
            tc = load_team_config()
        assert tc.cache_ttl_minutes == 120

    def test_env_invalid_int_keeps_default(self) -> None:
        with patch.dict(
            os.environ, {"MEMX_TEAM_CACHE_MAX_BULLETS": "not_a_number"}, clear=False
        ):
            tc = load_team_config()
        assert tc.cache_max_bullets == 2000  # default preserved


# ---------------------------------------------------------------------------
# YAML file loading
# ---------------------------------------------------------------------------


class TestYamlLoading:
    """Test loading config from YAML files."""

    def test_load_from_explicit_yaml(self, tmp_path: Path) -> None:
        yaml_content = (
            "enabled: true\n"
            "server_url: https://yaml-server.io\n"
            "team_id: yaml-team\n"
            "subscribed_tags:\n"
            "  - tag1\n"
            "  - tag2\n"
            "cache_max_bullets: 1000\n"
        )
        config_file = tmp_path / "team_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        tc = load_team_config(config_path=config_file)
        assert tc.enabled is True
        assert tc.server_url == "https://yaml-server.io"
        assert tc.team_id == "yaml-team"
        assert tc.subscribed_tags == ["tag1", "tag2"]
        assert tc.cache_max_bullets == 1000
        assert tc.cache_ttl_minutes == 60  # default

    def test_env_overrides_yaml(self, tmp_path: Path) -> None:
        yaml_content = "enabled: false\ncache_max_bullets: 999\n"
        config_file = tmp_path / "team_config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        with patch.dict(os.environ, {"MEMX_TEAM_ENABLED": "true"}, clear=False):
            tc = load_team_config(config_path=config_file)
        assert tc.enabled is True  # env overrides file
        assert tc.cache_max_bullets == 999  # from file

    def test_load_no_file_returns_defaults(self) -> None:
        # Point to a non-existent path so no file is found
        tc = load_team_config(config_path=Path("/nonexistent/team_config.yaml"))
        # Falls back to defaults since file doesn't exist
        assert tc.enabled is False
        assert tc.cache_max_bullets == 2000


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    """Ensure model_dump / model_validate round-trip works."""

    def test_round_trip(self) -> None:
        original = TeamConfig(
            enabled=True,
            server_url="https://rt.io",
            team_id="rt-1",
            subscribed_tags=["a"],
            layer_boost=LayerBoostConfig(local_boost=2.0, team_boost=0.5),
            mandatory_overrides=[
                MandatoryOverride(
                    bullet_id="b-1",
                    reason="test",
                    expires=datetime.now(timezone.utc) + timedelta(days=30),
                ),
            ],
        )
        dumped = original.model_dump()
        restored = TeamConfig.model_validate(dumped)
        assert restored == original
