"""Unit tests for the flat-Python -> nested-Rust config translation.

These tests are pure dict assertions over ``MemorusConfig.to_rust_dict`` and
``memorus.core._rust_backend._translate_config``; they do NOT require the
compiled ``memorus_r`` extension.
"""

from __future__ import annotations

import logging

import pytest

from memorus.core._rust_backend import _translate_config
from memorus.core.config import MemorusConfig


# ── to_rust_dict: section-by-section mapping ──────────────────────────────


class TestToRustDictMapping:
    def test_default_nested_shape(self) -> None:
        rust = MemorusConfig().to_rust_dict()
        assert set(rust.keys()) >= {"ace", "privacy", "daemon"}
        ace = rust["ace"]
        assert set(ace.keys()) >= {
            "enabled",
            "reflector",
            "curator",
            "decay",
            "retrieval",
            "consolidate",
            "topics",
            "verification",
        }

    def test_ace_enabled_maps_to_ace_enabled(self) -> None:
        rust = MemorusConfig.from_dict({"ace_enabled": False}).to_rust_dict()
        assert rust["ace"]["enabled"] is False

    def test_reflector_fields(self) -> None:
        cfg = MemorusConfig.from_dict(
            {
                "reflector": {
                    "mode": "rules",
                    "min_score": 40.0,
                    "max_content_length": 300,
                    "max_code_lines": 5,
                }
            }
        )
        refl = cfg.to_rust_dict()["ace"]["reflector"]
        assert refl["mode"] == "rules"
        assert refl["min_score"] == 40.0
        assert refl["max_content_length"] == 300
        assert refl["max_code_lines"] == 5
        # batch sub-config forwarded with identical key names.
        assert "batch_enabled" in refl["batch"]

    def test_curator_field_renames(self) -> None:
        cfg = MemorusConfig.from_dict(
            {
                "curator": {
                    "similarity_threshold": 0.85,
                    "conflict_detection": True,
                    "conflict_min_similarity": 0.4,
                    "conflict_max_similarity": 0.75,
                }
            }
        )
        cur = cfg.to_rust_dict()["ace"]["curator"]
        assert cur["dedup_threshold"] == 0.85
        assert cur["conflict_detection"] is True
        assert cur["conflict_similarity_min"] == 0.4
        assert cur["conflict_similarity_max"] == 0.75

    def test_decay_field_renames_and_protection_int_to_float(self) -> None:
        cfg = MemorusConfig.from_dict(
            {"decay": {"half_life_days": 45.0, "protection_days": 7}}
        )
        decay = cfg.to_rust_dict()["ace"]["decay"]
        assert decay["half_life"] == 45.0
        # int -> float cast for the Rust f64 field.
        assert decay["protection_days"] == 7.0
        assert isinstance(decay["protection_days"], float)

    def test_retrieval_fields(self) -> None:
        cfg = MemorusConfig.from_dict(
            {"retrieval": {"scope_boost": 1.7, "token_budget": 8192}}
        )
        rt = cfg.to_rust_dict()["ace"]["retrieval"]
        assert rt["scope_boost"] == 1.7
        assert rt["token_budget"] == 8192
        assert "graph_expansion" in rt

    def test_privacy_custom_patterns_renamed(self) -> None:
        cfg = MemorusConfig.from_dict(
            {"privacy": {"custom_patterns": [r"\bSECRET\b"]}}
        )
        assert cfg.to_rust_dict()["privacy"]["redaction_patterns"] == [r"\bSECRET\b"]

    def test_daemon_enabled_forwarded(self) -> None:
        cfg = MemorusConfig.from_dict({"daemon": {"enabled": False}})
        assert cfg.to_rust_dict()["daemon"]["enabled"] is False

    def test_python_side_only_keys_not_forwarded(self) -> None:
        # Python-only retrieval weights and reflector llm keys must NOT appear
        # in the Rust dict.
        cfg = MemorusConfig.from_dict(
            {
                "retrieval": {"keyword_weight": 0.6, "semantic_weight": 0.4},
                "reflector": {"llm_model": "openai/gpt-4o-mini"},
            }
        )
        rust = cfg.to_rust_dict()
        assert "keyword_weight" not in rust["ace"]["retrieval"]
        assert "semantic_weight" not in rust["ace"]["retrieval"]
        assert "llm_model" not in rust["ace"]["reflector"]


# ── Warning behavior ──────────────────────────────────────────────────────


class TestWarnings:
    def test_merge_strategy_non_default_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = MemorusConfig.from_dict(
            {"curator": {"merge_strategy": "merge_content"}}
        )
        with caplog.at_level(logging.WARNING, logger="memorus.core.config"):
            cfg.to_rust_dict()
        assert any("merge_strategy" in rec.message for rec in caplog.records)

    def test_default_merge_strategy_no_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = MemorusConfig()
        with caplog.at_level(logging.WARNING, logger="memorus.core.config"):
            cfg.to_rust_dict()
        assert not any("merge_strategy" in rec.message for rec in caplog.records)


# ── _translate_config (full path incl. mem0-native keys) ──────────────────


class TestTranslateConfig:
    def test_none_passes_through(self) -> None:
        assert _translate_config(None) is None

    def test_str_path_passes_through(self) -> None:
        assert _translate_config("/path/to/config.toml") == "/path/to/config.toml"

    def test_dict_produces_nested_schema(self) -> None:
        rust = _translate_config({"ace_enabled": True})
        assert isinstance(rust, dict)
        assert rust["ace"]["enabled"] is True

    def test_native_memory_block_passes_through(self) -> None:
        rust = _translate_config(
            {"memory": {"vector_store": {"provider": "qdrant"}}}
        )
        assert rust["memory"]["vector_store"]["provider"] == "qdrant"

    def test_mem0_vector_store_mapping(self) -> None:
        rust = _translate_config(
            {
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "collection_name": "memories",
                        "url": "http://localhost:6333",
                        "weird_key": 1,
                    },
                }
            }
        )
        vs = rust["memory"]["vector_store"]
        assert vs["provider"] == "qdrant"
        assert vs["collection_name"] == "memories"
        assert vs["url"] == "http://localhost:6333"
        # Unknown inner key lands in extra.
        assert vs["extra"]["weird_key"] == 1

    def test_mem0_llm_and_embedder_mapping(self) -> None:
        rust = _translate_config(
            {
                "llm": {
                    "provider": "openai",
                    "config": {"model": "gpt-4o-mini", "temperature": 0.2},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small", "dimensions": 1536},
                },
            }
        )
        assert rust["memory"]["llm"]["model"] == "gpt-4o-mini"
        assert rust["memory"]["llm"]["temperature"] == 0.2
        assert rust["memory"]["embedding"]["model"] == "text-embedding-3-small"
        assert rust["memory"]["embedding"]["dimensions"] == 1536

    def test_unknown_top_level_key_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="memorus.core._rust_backend"):
            _translate_config({"totally_unknown_section": {"x": 1}})
        assert any(
            "totally_unknown_section" in rec.message for rec in caplog.records
        )
