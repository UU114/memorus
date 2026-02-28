"""Unit tests for memx.engines.curator.conflict — ConflictDetector."""

from __future__ import annotations

from memx.config import CuratorConfig
from memx.engines.curator.conflict import (
    Conflict,
    ConflictDetector,
    ConflictResult,
)
from memx.engines.curator.engine import ExistingBullet

# -- Helper factories -------------------------------------------------------


def _bullet(
    bullet_id: str = "b1",
    content: str = "test content",
    scope: str = "global",
) -> ExistingBullet:
    """Create a minimal ExistingBullet for testing."""
    return ExistingBullet(bullet_id=bullet_id, content=content, scope=scope)


# -- ConflictResult dataclass ------------------------------------------------


class TestConflictResult:
    def test_defaults(self) -> None:
        r = ConflictResult()
        assert r.conflicts == []
        assert r.total_pairs_checked == 0
        assert r.scan_time_ms == 0.0

    def test_independent_instances(self) -> None:
        r1 = ConflictResult()
        r2 = ConflictResult()
        r1.conflicts.append(
            Conflict(
                memory_a_id="a",
                memory_b_id="b",
                memory_a_content="x",
                memory_b_content="y",
                similarity=0.6,
                reason="test",
            )
        )
        assert r2.conflicts == []


# -- Conflict dataclass ------------------------------------------------------


class TestConflict:
    def test_fields(self) -> None:
        c = Conflict(
            memory_a_id="a1",
            memory_b_id="b1",
            memory_a_content="always use tabs",
            memory_b_content="never use tabs",
            similarity=0.7,
            reason="opposing_pair: always vs never",
        )
        assert c.memory_a_id == "a1"
        assert c.memory_b_id == "b1"
        assert c.similarity == 0.7
        assert "opposing_pair" in c.reason


# -- ConflictDetector: empty / edge cases ------------------------------------


class TestDetectEdgeCases:
    def test_empty_list_returns_empty(self) -> None:
        detector = ConflictDetector()
        result = detector.detect([])
        assert result.conflicts == []
        assert result.total_pairs_checked == 0
        assert result.scan_time_ms >= 0.0

    def test_single_memory_returns_empty(self) -> None:
        detector = ConflictDetector()
        result = detector.detect([_bullet()])
        assert result.conflicts == []
        assert result.total_pairs_checked == 0

    def test_no_contradiction_returns_empty(self) -> None:
        """Two similar memories with no negation/opposing words -> no conflict."""
        detector = ConflictDetector(
            CuratorConfig(conflict_min_similarity=0.3, conflict_max_similarity=0.9)
        )
        a = _bullet("a", "use cargo build for compiling")
        b = _bullet("b", "use cargo build for fast compiling")
        result = detector.detect([a, b])
        assert result.conflicts == []
        assert result.total_pairs_checked == 1

    def test_scan_time_populated(self) -> None:
        detector = ConflictDetector()
        result = detector.detect([_bullet("a"), _bullet("b")])
        assert result.scan_time_ms >= 0.0


# -- ConflictDetector: opposing pairs ----------------------------------------


class TestOpposingPairs:
    def test_always_vs_never(self) -> None:
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "always use dark mode in the editor")
        b = _bullet("b", "never use dark mode in the editor")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1
        assert "opposing_pair" in result.conflicts[0].reason
        assert "always" in result.conflicts[0].reason
        assert "never" in result.conflicts[0].reason

    def test_enable_vs_disable(self) -> None:
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "enable auto-save in the editor settings")
        b = _bullet("b", "disable auto-save in the editor settings")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1
        assert "opposing_pair" in result.conflicts[0].reason

    def test_use_vs_avoid(self) -> None:
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "use global variables for config state")
        b = _bullet("b", "avoid global variables for config state")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1
        assert "opposing_pair" in result.conflicts[0].reason

    def test_with_vs_without(self) -> None:
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "deploy with docker container setup")
        b = _bullet("b", "deploy without docker container setup")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1

    def test_reversed_order_still_detected(self) -> None:
        """Opposing pair detected regardless of which memory has which word."""
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "never use dark mode in the editor")
        b = _bullet("b", "always use dark mode in the editor")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1


# -- ConflictDetector: negation asymmetry ------------------------------------


class TestNegationAsymmetry:
    def test_not_in_one_memory(self) -> None:
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "do not use console.log for debugging production code")
        b = _bullet("b", "do use console.log for debugging production code")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1
        assert result.conflicts[0].reason == "negation_asymmetry"

    def test_chinese_negation(self) -> None:
        config = CuratorConfig(
            conflict_min_similarity=0.0, conflict_max_similarity=0.99
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "不要使用 eval 函数处理用户输入")
        b = _bullet("b", "使用 eval 函数处理用户输入")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1
        assert result.conflicts[0].reason == "negation_asymmetry"

    def test_both_have_negation_no_conflict(self) -> None:
        """When both memories contain negation, there is no asymmetry."""
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "don't use eval and avoid exec for security")
        b = _bullet("b", "don't use exec and avoid eval for security")
        result = detector.detect([a, b])
        # Both have negation words, so no asymmetry -> no conflict
        assert len(result.conflicts) == 0


# -- ConflictDetector: similarity filtering ----------------------------------


class TestSimilarityFiltering:
    def test_below_min_similarity_ignored(self) -> None:
        """Pairs with similarity < min are not checked for contradiction."""
        config = CuratorConfig(
            conflict_min_similarity=0.8, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "always use tabs indentation preference")
        b = _bullet("b", "never use tabs indentation preference")
        # Text similarity is moderate but likely below 0.8
        result = detector.detect([a, b])
        # With high min_sim, the pair might not qualify
        # (depends on exact Jaccard; let's verify it doesn't match)
        for c in result.conflicts:
            assert c.similarity >= 0.8

    def test_above_max_similarity_ignored(self) -> None:
        """Near-identical memories (likely duplicates, not conflicts)."""
        config = CuratorConfig(
            conflict_min_similarity=0.5, conflict_max_similarity=0.7
        )
        detector = ConflictDetector(config)
        # These are near-identical -> high similarity -> above max
        a = _bullet("a", "always use dark mode")
        b = _bullet("b", "always use dark mode please")
        result = detector.detect([a, b])
        # Similarity is very high (>0.7), should be excluded
        assert len(result.conflicts) == 0

    def test_within_range_detected(self) -> None:
        """Pair within [min, max] similarity range with contradiction."""
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "always use dark mode in the editor for focus")
        b = _bullet("b", "never use dark mode in the editor for focus")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1
        sim = result.conflicts[0].similarity
        assert 0.2 <= sim <= 0.95


# -- ConflictDetector: scope grouping ----------------------------------------


class TestScopeGrouping:
    def test_different_scopes_not_compared(self) -> None:
        """Memories in different scopes are not compared."""
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "always use dark mode in editor", scope="project:frontend")
        b = _bullet("b", "never use dark mode in editor", scope="project:backend")
        result = detector.detect([a, b])
        assert result.conflicts == []
        assert result.total_pairs_checked == 0

    def test_same_scope_compared(self) -> None:
        """Memories in the same scope are compared."""
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "always use dark mode in editor", scope="global")
        b = _bullet("b", "never use dark mode in editor", scope="global")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1
        assert result.total_pairs_checked == 1


# -- ConflictDetector: multiple memories (>2) --------------------------------


class TestMultipleMemories:
    def test_three_memories_two_conflicting(self) -> None:
        config = CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )
        detector = ConflictDetector(config)
        a = _bullet("a", "always use dark mode in the editor setting")
        b = _bullet("b", "never use dark mode in the editor setting")
        c = _bullet("c", "run pytest with coverage flag enabled")
        result = detector.detect([a, b, c])
        # Only a-b should conflict
        assert len(result.conflicts) == 1
        ids = {result.conflicts[0].memory_a_id, result.conflicts[0].memory_b_id}
        assert ids == {"a", "b"}

    def test_total_pairs_counted_correctly(self) -> None:
        detector = ConflictDetector()
        mems = [_bullet(f"b{i}", f"content {i}") for i in range(4)]
        result = detector.detect(mems)
        # C(4,2) = 6 pairs
        assert result.total_pairs_checked == 6


# -- ConflictDetector: CuratorEngine integration -----------------------------


class TestCuratorEngineIntegration:
    def test_curate_with_conflict_detection_enabled(self) -> None:
        """CuratorEngine.curate() attaches conflicts when enabled."""
        from memx.engines.curator.engine import CuratorEngine
        from memx.types import CandidateBullet

        config = CuratorConfig(
            conflict_detection=True,
            conflict_min_similarity=0.2,
            conflict_max_similarity=0.95,
        )
        engine = CuratorEngine(config)
        candidates = [CandidateBullet(content="new stuff here")]
        existing = [
            _bullet("a", "always use dark mode in the editor for focus"),
            _bullet("b", "never use dark mode in the editor for focus"),
        ]
        result = engine.curate(candidates, existing)
        assert len(result.conflicts) >= 1

    def test_curate_without_conflict_detection(self) -> None:
        """CuratorEngine.curate() does not run conflict detection by default."""
        from memx.engines.curator.engine import CuratorEngine
        from memx.types import CandidateBullet

        config = CuratorConfig(conflict_detection=False)
        engine = CuratorEngine(config)
        candidates = [CandidateBullet(content="new stuff here")]
        existing = [
            _bullet("a", "always use dark mode"),
            _bullet("b", "never use dark mode"),
        ]
        result = engine.curate(candidates, existing)
        assert result.conflicts == []


# -- ConflictDetector: version conflict detection ------------------------------


class TestVersionConflict:
    """Version conflict: same tool/library but different version numbers."""

    def _wide_config(self) -> CuratorConfig:
        return CuratorConfig(
            conflict_min_similarity=0.2, conflict_max_similarity=0.95
        )

    def test_semver_at_syntax(self) -> None:
        """react@17.0.2 vs react@18.2.0 -> version conflict."""
        detector = ConflictDetector(self._wide_config())
        a = _bullet("a", "install react@17.0.2 for the legacy project components")
        b = _bullet("b", "install react@18.2.0 for the legacy project components")
        result = detector.detect([a, b])
        assert len(result.conflicts) == 1
        assert "version_conflict" in result.conflicts[0].reason
        assert "react" in result.conflicts[0].reason

    def test_version_with_space(self) -> None:
        """python 3.9 vs python 3.12 -> version conflict."""
        detector = ConflictDetector(self._wide_config())
        a = _bullet("a", "use python 3.9 for building the data pipeline")
        b = _bullet("b", "use python 3.12 for building the data pipeline")
        result = detector.detect([a, b])
        assert len(result.conflicts) >= 1
        reasons = [c.reason for c in result.conflicts]
        assert any("version_conflict" in r for r in reasons)

    def test_v_prefix(self) -> None:
        """node v18 vs node v20 -> version conflict."""
        detector = ConflictDetector(self._wide_config())
        a = _bullet("a", "run the tests on node v18 runtime environment")
        b = _bullet("b", "run the tests on node v20 runtime environment")
        result = detector.detect([a, b])
        assert len(result.conflicts) >= 1
        reasons = [c.reason for c in result.conflicts]
        assert any("version_conflict" in r for r in reasons)

    def test_same_version_no_conflict(self) -> None:
        """Same tool + same version -> no version conflict."""
        detector = ConflictDetector(self._wide_config())
        a = _bullet("a", "install react@18.2.0 for the project components setup")
        b = _bullet("b", "install react@18.2.0 for the project components config")
        result = detector.detect([a, b])
        # No version conflict (may still detect other types)
        version_conflicts = [c for c in result.conflicts if "version_conflict" in c.reason]
        assert len(version_conflicts) == 0

    def test_different_tools_no_conflict(self) -> None:
        """Different tools with different versions -> no version conflict."""
        detector = ConflictDetector(self._wide_config())
        a = _bullet("a", "install react@18.2.0 for frontend component rendering")
        b = _bullet("b", "install express@5.0.0 for backend server component routing")
        result = detector.detect([a, b])
        version_conflicts = [c for c in result.conflicts if "version_conflict" in c.reason]
        assert len(version_conflicts) == 0

    def test_no_version_in_text(self) -> None:
        """No version numbers at all -> no version conflict."""
        detector = ConflictDetector(self._wide_config())
        a = _bullet("a", "use react hooks for the state management in components")
        b = _bullet("b", "use react classes for the state management in components")
        result = detector.detect([a, b])
        version_conflicts = [c for c in result.conflicts if "version_conflict" in c.reason]
        assert len(version_conflicts) == 0


class TestVersionExtraction:
    """Unit tests for the _extract_versions helper."""

    def test_at_syntax(self) -> None:
        d = ConflictDetector()
        assert d._extract_versions("lodash@4.17.21") == {"lodash": "4.17.21"}

    def test_space_syntax(self) -> None:
        d = ConflictDetector()
        assert d._extract_versions("python 3.11 is great") == {"python": "3.11"}

    def test_v_prefix_syntax(self) -> None:
        d = ConflictDetector()
        assert d._extract_versions("use vue v3.4 for SPA") == {"vue": "3.4"}

    def test_multiple_tools(self) -> None:
        d = ConflictDetector()
        result = d._extract_versions("react@18.2 and express@5.0.1")
        assert result["react"] == "18.2"
        assert result["express"] == "5.0.1"

    def test_stopwords_filtered(self) -> None:
        d = ConflictDetector()
        result = d._extract_versions("version 2.0 and step 3")
        assert result == {}

    def test_empty_text(self) -> None:
        d = ConflictDetector()
        assert d._extract_versions("") == {}

    def test_prerelease_version(self) -> None:
        d = ConflictDetector()
        result = d._extract_versions("next@14.3.0-canary.87")
        assert result["next"] == "14.3.0-canary.87"


# -- ConflictDetector: default config ----------------------------------------


class TestDefaultConfig:
    def test_default_thresholds(self) -> None:
        """ConflictDetector with no config uses sensible defaults."""
        detector = ConflictDetector()
        assert detector._min_sim == 0.5
        assert detector._max_sim == 0.8

    def test_custom_config(self) -> None:
        config = CuratorConfig(
            conflict_min_similarity=0.3,
            conflict_max_similarity=0.9,
        )
        detector = ConflictDetector(config)
        assert detector._min_sim == 0.3
        assert detector._max_sim == 0.9
