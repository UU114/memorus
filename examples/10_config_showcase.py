"""Demo 10: Config Showcase — MemorusConfig key separation + validation + defaults."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memorus.config import (
    CuratorConfig,
    DaemonConfig,
    DecayConfig,
    IntegrationConfig,
    MemorusConfig,
    PrivacyConfig,
    ReflectorConfig,
    RetrievalConfig,
)


def main() -> None:
    # --- 1. Default config ---
    cfg = MemorusConfig()
    assert cfg.ace_enabled is False
    assert cfg.reflector.mode == "rules"
    # Aligned to Rust source-of-truth: was 0.8, now 0.9.
    assert cfg.curator.similarity_threshold == 0.9
    assert cfg.decay.half_life_days == 30.0
    assert cfg.retrieval.max_results == 5
    assert cfg.privacy.sanitize_paths is True
    assert cfg.integration.context_template == "xml"
    assert cfg.daemon.enabled is False
    print("[1/5] Default config: all defaults verified")

    # --- 2. ACE key separation ---
    mixed_config = {
        # ACE keys
        "ace_enabled": True,
        "reflector": {"min_score": 25.0, "max_content_length": 300},
        "curator": {"similarity_threshold": 0.75, "conflict_detection": True},
        "decay": {"half_life_days": 14.0, "permanent_threshold": 10},
        "privacy": {"custom_patterns": [r"SSN-\d{3}-\d{2}-\d{4}"]},
        # mem0 keys (should be separated)
        "version": "v1.1",
        "llm": {"provider": "openai", "config": {"model": "gpt-4"}},
        "embedder": {"provider": "openai"},
        "vector_store": {"provider": "chroma"},
    }

    cfg2 = MemorusConfig.from_dict(mixed_config)
    assert cfg2.ace_enabled is True
    assert cfg2.reflector.min_score == 25.0
    assert cfg2.curator.conflict_detection is True
    assert cfg2.decay.half_life_days == 14.0
    assert len(cfg2.privacy.custom_patterns) == 1

    # mem0 keys preserved in mem0_config
    mem0 = cfg2.to_mem0_config()
    assert mem0.get("version") == "v1.1"
    assert "llm" in mem0
    assert "embedder" in mem0
    print("[2/5] Key separation: ACE keys extracted, mem0 keys preserved")
    print(f"       ACE keys: ace_enabled, reflector, curator, decay, privacy")
    print(f"       mem0 keys: {list(mem0.keys())}")

    # --- 3. Sub-config validation ---
    # Test that invalid values are rejected
    try:
        ReflectorConfig(min_score=-10)
        assert False, "Should reject negative min_score"
    except Exception:
        pass

    try:
        CuratorConfig(similarity_threshold=1.5)
        assert False, "Should reject similarity > 1.0"
    except Exception:
        pass

    try:
        DecayConfig(half_life_days=0)
        assert False, "Should reject zero half_life"
    except Exception:
        pass
    print("[3/5] Validation: invalid values rejected correctly")

    # --- 4. All sub-configs instantiate ---
    configs = {
        "ReflectorConfig": ReflectorConfig(),
        "CuratorConfig": CuratorConfig(),
        "DecayConfig": DecayConfig(),
        "RetrievalConfig": RetrievalConfig(),
        "PrivacyConfig": PrivacyConfig(),
        "IntegrationConfig": IntegrationConfig(),
        "DaemonConfig": DaemonConfig(),
    }
    for name, c in configs.items():
        print(f"       {name}: OK")
    print(f"[4/5] All {len(configs)} sub-configs instantiate with defaults")

    # --- 5. Config round-trip ---
    full_config = {
        "ace_enabled": True,
        "reflector": {"mode": "rules", "min_score": 40.0},
        "retrieval": {"keyword_weight": 0.7, "semantic_weight": 0.3, "max_results": 10},
        "integration": {"context_template": "markdown"},
    }
    cfg3 = MemorusConfig.from_dict(full_config)
    assert cfg3.reflector.min_score == 40.0
    assert cfg3.retrieval.keyword_weight == 0.7
    assert cfg3.retrieval.max_results == 10
    assert cfg3.integration.context_template == "markdown"
    print("[5/5] Config round-trip: values preserved correctly")

    print("\nPASS: 10_config_showcase")


if __name__ == "__main__":
    main()
