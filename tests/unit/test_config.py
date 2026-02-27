"""Unit tests for memx.config — MemXConfig and all sub-config models."""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from memx.config import (
    CuratorConfig,
    DaemonConfig,
    DecayConfig,
    IntegrationConfig,
    MemXConfig,
    PrivacyConfig,
    ReflectorConfig,
    RetrievalConfig,
)
from memx.exceptions import ConfigurationError


# ── Default construction ───────────────────────────────────────────────


class TestMemXConfigDefaults:
    """MemXConfig() with no args must succeed and carry sensible defaults."""

    def test_zero_arg_construction(self) -> None:
        cfg = MemXConfig()
        assert cfg.ace_enabled is False
        assert isinstance(cfg.reflector, ReflectorConfig)
        assert isinstance(cfg.curator, CuratorConfig)
        assert isinstance(cfg.decay, DecayConfig)
        assert isinstance(cfg.retrieval, RetrievalConfig)
        assert isinstance(cfg.privacy, PrivacyConfig)
        assert isinstance(cfg.integration, IntegrationConfig)
        assert isinstance(cfg.daemon, DaemonConfig)
        assert cfg.mem0_config == {}

    def test_ace_enabled_default_false(self) -> None:
        assert MemXConfig().ace_enabled is False


# ── Sub-config defaults ───────────────────────────────────────────────


class TestReflectorConfigDefaults:
    def test_defaults(self) -> None:
        r = ReflectorConfig()
        assert r.mode == "rules"
        assert r.min_score == 30.0
        assert r.max_content_length == 500
        assert r.max_code_lines == 3


class TestCuratorConfigDefaults:
    def test_defaults(self) -> None:
        c = CuratorConfig()
        assert c.similarity_threshold == 0.8
        assert c.merge_strategy == "keep_best"
        assert c.conflict_detection is False


class TestDecayConfigDefaults:
    def test_defaults(self) -> None:
        d = DecayConfig()
        assert d.half_life_days == 30.0
        assert d.boost_factor == 0.1
        assert d.protection_days == 7
        assert d.permanent_threshold == 15
        assert d.archive_threshold == 0.02
        assert d.sweep_on_session_end is True


class TestRetrievalConfigDefaults:
    def test_defaults(self) -> None:
        r = RetrievalConfig()
        assert r.keyword_weight == 0.6
        assert r.semantic_weight == 0.4
        assert r.recency_boost_days == 7
        assert r.recency_boost_factor == 1.2
        assert r.max_results == 5
        assert r.token_budget == 2000


class TestPrivacyConfigDefaults:
    def test_defaults(self) -> None:
        p = PrivacyConfig()
        assert p.custom_patterns == []
        assert p.sanitize_paths is True
        assert p.always_sanitize is False

    def test_list_isolation(self) -> None:
        a = PrivacyConfig()
        b = PrivacyConfig()
        a.custom_patterns.append(r"\bSECRET\b")
        assert b.custom_patterns == []


class TestIntegrationConfigDefaults:
    def test_defaults(self) -> None:
        i = IntegrationConfig()
        assert i.auto_recall is True
        assert i.auto_reflect is True
        assert i.sweep_on_exit is True


class TestDaemonConfigDefaults:
    def test_defaults(self) -> None:
        d = DaemonConfig()
        assert d.enabled is False
        assert d.idle_timeout_seconds == 300
        assert d.socket_path is None


# ── Field-level validation errors ─────────────────────────────────────


class TestDecayValidation:
    def test_negative_half_life_days_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecayConfig(half_life_days=-1.0)

    def test_zero_half_life_days_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DecayConfig(half_life_days=0)

    def test_positive_half_life_days_accepted(self) -> None:
        assert DecayConfig(half_life_days=0.5).half_life_days == 0.5


class TestCuratorValidation:
    def test_similarity_threshold_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CuratorConfig(similarity_threshold=1.01)

    def test_similarity_threshold_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CuratorConfig(similarity_threshold=-0.1)

    def test_similarity_threshold_boundary_values(self) -> None:
        assert CuratorConfig(similarity_threshold=0.0).similarity_threshold == 0.0
        assert CuratorConfig(similarity_threshold=1.0).similarity_threshold == 1.0


class TestReflectorValidation:
    def test_min_score_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReflectorConfig(min_score=-1.0)

    def test_min_score_above_100_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReflectorConfig(min_score=100.1)

    def test_min_score_boundary_values(self) -> None:
        assert ReflectorConfig(min_score=0.0).min_score == 0.0
        assert ReflectorConfig(min_score=100.0).min_score == 100.0

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValidationError, match="reflector.mode must be one of"):
            ReflectorConfig(mode="invalid_mode")

    def test_valid_modes_accepted(self) -> None:
        for mode in ("rules", "llm", "hybrid"):
            r = ReflectorConfig(mode=mode)
            assert r.mode == mode

    def test_max_content_length_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReflectorConfig(max_content_length=0)

    def test_max_code_lines_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ReflectorConfig(max_code_lines=0)


class TestRetrievalValidation:
    def test_negative_keyword_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(keyword_weight=-0.1)

    def test_negative_semantic_weight_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(semantic_weight=-0.1)

    def test_recency_boost_factor_below_1_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(recency_boost_factor=0.9)

    def test_max_results_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(max_results=0)

    def test_token_budget_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(token_budget=0)


class TestDaemonValidation:
    def test_idle_timeout_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DaemonConfig(idle_timeout_seconds=0)

    def test_idle_timeout_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DaemonConfig(idle_timeout_seconds=-10)


# ── from_dict ─────────────────────────────────────────────────────────


class TestFromDict:
    def test_empty_dict(self) -> None:
        cfg = MemXConfig.from_dict({})
        assert cfg.ace_enabled is False
        assert cfg.mem0_config == {}

    def test_ace_enabled_true(self) -> None:
        cfg = MemXConfig.from_dict({"ace_enabled": True})
        assert cfg.ace_enabled is True
        assert cfg.mem0_config == {}

    def test_mem0_fields_separated(self) -> None:
        cfg = MemXConfig.from_dict({
            "vector_store": {"provider": "qdrant", "config": {"host": "localhost"}},
            "llm": {"provider": "openai", "config": {"model": "gpt-4"}},
        })
        assert "vector_store" in cfg.mem0_config
        assert "llm" in cfg.mem0_config
        assert cfg.mem0_config["vector_store"]["provider"] == "qdrant"

    def test_mixed_ace_and_mem0_fields(self) -> None:
        cfg = MemXConfig.from_dict({
            "ace_enabled": True,
            "reflector": {"mode": "hybrid", "min_score": 50.0},
            "decay": {"half_life_days": 14.0},
            "vector_store": {"provider": "chroma"},
            "embedder": {"provider": "openai"},
        })
        assert cfg.ace_enabled is True
        assert cfg.reflector.mode == "hybrid"
        assert cfg.reflector.min_score == 50.0
        assert cfg.decay.half_life_days == 14.0
        # mem0 fields captured
        assert cfg.mem0_config["vector_store"]["provider"] == "chroma"
        assert cfg.mem0_config["embedder"]["provider"] == "openai"
        # ACE fields NOT in mem0_config
        assert "ace_enabled" not in cfg.mem0_config
        assert "reflector" not in cfg.mem0_config

    def test_invalid_values_raise_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            MemXConfig.from_dict({"decay": {"half_life_days": -5}})

    def test_invalid_reflector_mode_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="Invalid configuration"):
            MemXConfig.from_dict({"reflector": {"mode": "turbo"}})

    def test_all_ace_keys_recognized(self) -> None:
        """Every known ACE key should be extracted, not put into mem0_config."""
        ace_keys = [
            "ace_enabled",
            "reflector",
            "curator",
            "decay",
            "retrieval",
            "privacy",
            "integration",
            "daemon",
        ]
        input_dict = {k: {} if k != "ace_enabled" else True for k in ace_keys}
        cfg = MemXConfig.from_dict(input_dict)
        assert cfg.mem0_config == {}


# ── to_mem0_config ────────────────────────────────────────────────────


class TestToMem0Config:
    def test_returns_mem0_fields(self) -> None:
        cfg = MemXConfig.from_dict({
            "vector_store": {"provider": "qdrant"},
            "embedder": {"provider": "openai"},
        })
        m0 = cfg.to_mem0_config()
        assert m0 == {
            "vector_store": {"provider": "qdrant"},
            "embedder": {"provider": "openai"},
        }

    def test_returns_copy(self) -> None:
        cfg = MemXConfig.from_dict({"vector_store": {"provider": "qdrant"}})
        m0 = cfg.to_mem0_config()
        m0["extra"] = "injected"
        # Original should be unaffected
        assert "extra" not in cfg.mem0_config

    def test_empty_when_no_mem0_fields(self) -> None:
        cfg = MemXConfig.from_dict({"ace_enabled": True})
        assert cfg.to_mem0_config() == {}


# ── Weight warning ────────────────────────────────────────────────────


class TestWeightWarning:
    def test_default_weights_no_warning(self) -> None:
        """Default 0.6 + 0.4 = 1.0 should not warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MemXConfig()
        weight_warnings = [x for x in w if "keyword_weight" in str(x.message)]
        assert len(weight_warnings) == 0

    def test_unbalanced_weights_emit_warning(self) -> None:
        """keyword_weight + semantic_weight != 1.0 should warn."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MemXConfig(retrieval=RetrievalConfig(keyword_weight=0.5, semantic_weight=0.3))
        weight_warnings = [x for x in w if "keyword_weight" in str(x.message)]
        assert len(weight_warnings) == 1
        assert "not 1.0" in str(weight_warnings[0].message)
        assert "ScoreMerger will normalize" in str(weight_warnings[0].message)

    def test_unbalanced_weights_still_valid(self) -> None:
        """Unbalanced weights warn but do NOT raise."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            cfg = MemXConfig(
                retrieval=RetrievalConfig(keyword_weight=0.7, semantic_weight=0.7)
            )
        assert cfg.retrieval.keyword_weight == 0.7
        assert cfg.retrieval.semantic_weight == 0.7

    def test_weights_summing_to_1_within_tolerance(self) -> None:
        """0.6 + 0.4 = 1.0 exactly, no warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            MemXConfig(retrieval=RetrievalConfig(keyword_weight=0.6, semantic_weight=0.4))
        weight_warnings = [x for x in w if "keyword_weight" in str(x.message)]
        assert len(weight_warnings) == 0


# ── Reflector mode validation ────────────────────────────────────────


class TestReflectorModeValidation:
    def test_invalid_mode_via_direct_construction(self) -> None:
        with pytest.raises(ValidationError, match="reflector.mode"):
            ReflectorConfig(mode="nonexistent")

    def test_invalid_mode_via_memx_config(self) -> None:
        with pytest.raises(ValidationError, match="reflector.mode"):
            MemXConfig(reflector=ReflectorConfig(mode="bad"))

    def test_invalid_mode_via_from_dict(self) -> None:
        with pytest.raises(ConfigurationError):
            MemXConfig.from_dict({"reflector": {"mode": "bad"}})

    def test_all_valid_modes(self) -> None:
        for mode in ("rules", "llm", "hybrid"):
            cfg = MemXConfig(reflector=ReflectorConfig(mode=mode))
            assert cfg.reflector.mode == mode


# ── Serialization round-trip ──────────────────────────────────────────


class TestConfigSerialization:
    def test_model_dump_round_trip(self) -> None:
        original = MemXConfig(
            ace_enabled=True,
            reflector=ReflectorConfig(mode="hybrid", min_score=45.0),
            decay=DecayConfig(half_life_days=14.0),
        )
        d = original.model_dump()
        restored = MemXConfig.model_validate(d)
        assert restored.ace_enabled is True
        assert restored.reflector.mode == "hybrid"
        assert restored.reflector.min_score == 45.0
        assert restored.decay.half_life_days == 14.0

    def test_json_round_trip(self) -> None:
        original = MemXConfig(
            ace_enabled=True,
            privacy=PrivacyConfig(custom_patterns=[r"\bAPI_KEY\b"]),
        )
        json_str = original.model_dump_json()
        restored = MemXConfig.model_validate_json(json_str)
        assert restored.privacy.custom_patterns == [r"\bAPI_KEY\b"]
