"""Unit tests for STORY-043: Hierarchical Scope Management.

Tests scope validation, scope filtering in GeneratorEngine, scope boost
in ScoreMerger, scope-aware deduplication in CuratorEngine, scope
passthrough in pipelines, and CLI --scope option.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memx.core.cli.main import cli
from memx.core.config import CuratorConfig, MemXConfig, RetrievalConfig
from memx.core.engines.curator.engine import CuratorEngine, ExistingBullet
from memx.core.engines.generator.engine import BulletForSearch, GeneratorEngine
from memx.core.engines.generator.metadata_matcher import MetadataInfo
from memx.core.engines.generator.score_merger import BulletInfo, ScoreMerger
from memx.core.memory import Memory, _validate_scope
from memx.core.pipeline.ingest import IngestPipeline, IngestResult
from memx.core.pipeline.retrieval import RetrievalPipeline, SearchResult
from memx.core.types import BulletMetadata, BulletSection, CandidateBullet, KnowledgeType, SourceType

_NOW = datetime(2026, 2, 27, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bullet(
    bid: str,
    content: str = "test content",
    scope: str = "global",
    days_ago: float = 30.0,
    decay_weight: float = 1.0,
    tools: list[str] | None = None,
    tags: list[str] | None = None,
) -> BulletForSearch:
    """Create a BulletForSearch for testing."""
    return BulletForSearch(
        bullet_id=bid,
        content=content,
        metadata=MetadataInfo(
            related_tools=tools or [],
            key_entities=[],
            tags=tags or [],
        ),
        created_at=_NOW - timedelta(days=days_ago),
        decay_weight=decay_weight,
        scope=scope,
    )


def _make_info(
    bid: str,
    content: str = "",
    scope: str = "global",
    days_ago: float = 0.0,
    decay_weight: float = 1.0,
) -> BulletInfo:
    """Create a BulletInfo for testing."""
    created = _NOW - timedelta(days=days_ago)
    return BulletInfo(
        bullet_id=bid,
        content=content,
        created_at=created,
        decay_weight=decay_weight,
        scope=scope,
    )


# ===========================================================================
# 1. _validate_scope tests
# ===========================================================================


class TestValidateScope:
    """Scope validation logic."""

    def test_none_returns_global(self) -> None:
        assert _validate_scope(None) == "global"

    def test_empty_string_returns_global(self) -> None:
        assert _validate_scope("") == "global"

    def test_global_passthrough(self) -> None:
        assert _validate_scope("global") == "global"

    def test_project_scope_valid(self) -> None:
        assert _validate_scope("project:myapp") == "project:myapp"

    def test_project_scope_no_name_raises(self) -> None:
        with pytest.raises(ValueError, match="requires a name"):
            _validate_scope("project:")

    def test_arbitrary_scope_passthrough(self) -> None:
        assert _validate_scope("team:backend") == "team:backend"

    def test_project_scope_with_slashes(self) -> None:
        assert _validate_scope("project:org/repo") == "project:org/repo"


# ===========================================================================
# 2. BulletForSearch scope field
# ===========================================================================


class TestBulletForSearchScope:
    """BulletForSearch.scope defaults and construction."""

    def test_default_scope_is_global(self) -> None:
        b = BulletForSearch(bullet_id="b1")
        assert b.scope == "global"

    def test_custom_scope(self) -> None:
        b = BulletForSearch(bullet_id="b1", scope="project:myapp")
        assert b.scope == "project:myapp"


# ===========================================================================
# 3. RetrievalConfig.scope_boost
# ===========================================================================


class TestRetrievalConfigScopeBoost:
    """RetrievalConfig.scope_boost default and validation."""

    def test_default_scope_boost(self) -> None:
        cfg = RetrievalConfig()
        assert cfg.scope_boost == 1.3

    def test_custom_scope_boost(self) -> None:
        cfg = RetrievalConfig(scope_boost=1.5)
        assert cfg.scope_boost == 1.5

    def test_scope_boost_below_one_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            RetrievalConfig(scope_boost=0.9)

    def test_scope_boost_exactly_one(self) -> None:
        cfg = RetrievalConfig(scope_boost=1.0)
        assert cfg.scope_boost == 1.0


# ===========================================================================
# 4. GeneratorEngine scope filtering
# ===========================================================================


class TestGeneratorEngineScopeFiltering:
    """GeneratorEngine.search() filters bullets by scope."""

    def test_no_scope_returns_all(self) -> None:
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase", scope="global"),
            _make_bullet("b2", "git rebase", scope="project:myapp"),
        ]
        results = engine.search("git rebase", bullets)
        assert len(results) == 2

    def test_scope_filters_to_matching_and_global(self) -> None:
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase", scope="global"),
            _make_bullet("b2", "git rebase", scope="project:myapp"),
            _make_bullet("b3", "git rebase", scope="project:other"),
        ]
        results = engine.search("git rebase", bullets, scope="project:myapp")
        result_ids = {r.bullet_id for r in results}
        assert "b1" in result_ids  # global included
        assert "b2" in result_ids  # matching scope included
        assert "b3" not in result_ids  # different project excluded

    def test_scope_filter_no_matches_returns_empty(self) -> None:
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase", scope="project:other"),
        ]
        results = engine.search("git rebase", bullets, scope="project:myapp")
        # b1 is not "global" and not "project:myapp"
        assert len(results) == 0

    def test_scope_filter_global_only(self) -> None:
        """When all bullets are global, scope filter should keep them."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase", scope="global"),
            _make_bullet("b2", "git merge", scope="global"),
        ]
        results = engine.search("git rebase", bullets, scope="project:myapp")
        assert len(results) == 2  # all global bullets pass


# ===========================================================================
# 5. ScoreMerger scope boost
# ===========================================================================


class TestScoreMergerScopeBoost:
    """ScoreMerger applies scope boost when target_scope is set."""

    def test_no_target_scope_no_boost(self) -> None:
        merger = ScoreMerger()
        infos = {"b1": _make_info("b1", "test", scope="project:myapp", days_ago=30)}
        results = merger.merge({"b1": 35.0}, None, infos, target_scope=None)
        assert len(results) == 1
        # No scope boost: FinalScore = 1.0 * 1.0 * 1.0 * 1.0 = 1.0
        assert abs(results[0].final_score - 1.0) < 1e-9

    def test_matching_scope_gets_boost(self) -> None:
        cfg = RetrievalConfig(scope_boost=1.3)
        merger = ScoreMerger(cfg)
        infos = {
            "b1": _make_info("b1", "project", scope="project:myapp", days_ago=30),
            "b2": _make_info("b2", "global", scope="global", days_ago=30),
        }
        kw = {"b1": 35.0, "b2": 35.0}
        results = merger.merge(kw, None, infos, target_scope="project:myapp")
        b1 = next(r for r in results if r.bullet_id == "b1")
        b2 = next(r for r in results if r.bullet_id == "b2")
        # b1 gets 1.3x boost, b2 gets 1.0x
        assert abs(b1.final_score - 1.3) < 1e-9
        assert abs(b2.final_score - 1.0) < 1e-9
        assert b1.final_score > b2.final_score

    def test_scope_boost_combined_with_recency(self) -> None:
        """ScopeBoost and RecencyBoost should multiply together."""
        cfg = RetrievalConfig(scope_boost=1.3, recency_boost_factor=1.2)
        merger = ScoreMerger(cfg)
        infos = {
            "b1": _make_info("b1", "recent scoped", scope="project:myapp", days_ago=2),
        }
        kw = {"b1": 35.0}
        results = merger.merge(kw, None, infos, target_scope="project:myapp", now=_NOW)
        # FinalScore = 1.0 * 1.0 * 1.2 * 1.3 = 1.56
        expected = 1.0 * 1.0 * 1.2 * 1.3
        assert abs(results[0].final_score - expected) < 1e-9

    def test_custom_scope_boost_value(self) -> None:
        cfg = RetrievalConfig(scope_boost=2.0)
        merger = ScoreMerger(cfg)
        infos = {"b1": _make_info("b1", "test", scope="project:x", days_ago=30)}
        results = merger.merge({"b1": 35.0}, None, infos, target_scope="project:x")
        assert abs(results[0].final_score - 2.0) < 1e-9

    def test_global_scope_never_boosted(self) -> None:
        """Global bullets should NOT receive scope boost even with target_scope."""
        cfg = RetrievalConfig(scope_boost=1.5)
        merger = ScoreMerger(cfg)
        infos = {"b1": _make_info("b1", "global", scope="global", days_ago=30)}
        results = merger.merge({"b1": 35.0}, None, infos, target_scope="project:x")
        # global scope != "project:x" -> no boost
        assert abs(results[0].final_score - 1.0) < 1e-9


# ===========================================================================
# 6. CuratorEngine scope-aware deduplication
# ===========================================================================


class TestCuratorEngineScopeAware:
    """CuratorEngine.curate() only compares within the same scope."""

    def test_same_scope_triggers_merge(self) -> None:
        engine = CuratorEngine()
        c = CandidateBullet(content="use cargo check", scope="project:myapp")
        ex = ExistingBullet(
            bullet_id="b1", content="use cargo check", scope="project:myapp"
        )
        result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1

    def test_different_scope_triggers_insert(self) -> None:
        """Candidate in one scope should not be deduped against different scope."""
        engine = CuratorEngine()
        c = CandidateBullet(content="use cargo check", scope="project:myapp")
        ex = ExistingBullet(
            bullet_id="b1", content="use cargo check", scope="project:other"
        )
        result = engine.curate([c], [ex])
        assert len(result.to_add) == 1
        assert len(result.to_merge) == 0

    def test_global_scope_dedup_within_global(self) -> None:
        engine = CuratorEngine()
        c = CandidateBullet(content="global knowledge", scope="global")
        ex = ExistingBullet(
            bullet_id="b1", content="global knowledge", scope="global"
        )
        result = engine.curate([c], [ex])
        assert len(result.to_merge) == 1

    def test_project_not_deduped_against_global(self) -> None:
        """Project-scoped candidate should not merge with global existing."""
        engine = CuratorEngine()
        c = CandidateBullet(content="use cargo check", scope="project:myapp")
        ex = ExistingBullet(
            bullet_id="b1", content="use cargo check", scope="global"
        )
        result = engine.curate([c], [ex])
        assert len(result.to_add) == 1
        assert len(result.to_merge) == 0

    def test_mixed_scopes_in_existing(self) -> None:
        """When existing bullets have mixed scopes, only same-scope is compared."""
        config = CuratorConfig(similarity_threshold=0.5)
        engine = CuratorEngine(config)
        c = CandidateBullet(content="use cargo check for fast feedback", scope="project:myapp")
        ex_global = ExistingBullet(
            bullet_id="b1", content="use cargo check for fast feedback", scope="global"
        )
        ex_same = ExistingBullet(
            bullet_id="b2", content="use cargo check for fast feedback", scope="project:myapp"
        )
        ex_other = ExistingBullet(
            bullet_id="b3", content="use cargo check for fast feedback", scope="project:other"
        )
        result = engine.curate([c], [ex_global, ex_same, ex_other])
        # Should merge with b2 (same scope)
        assert len(result.to_merge) == 1
        assert result.to_merge[0].existing.bullet_id == "b2"


# ===========================================================================
# 7. ExistingBullet scope field
# ===========================================================================


class TestExistingBulletScope:
    """ExistingBullet.scope defaults and construction."""

    def test_default_scope(self) -> None:
        eb = ExistingBullet(bullet_id="b1", content="hello")
        assert eb.scope == "global"

    def test_custom_scope(self) -> None:
        eb = ExistingBullet(bullet_id="b1", content="hello", scope="project:x")
        assert eb.scope == "project:x"


# ===========================================================================
# 8. IngestPipeline scope passthrough
# ===========================================================================


class TestIngestPipelineScopePassthrough:
    """IngestPipeline.process() passes scope to candidates."""

    def test_scope_set_on_candidates(self) -> None:
        """Scope is applied to all candidates before writing."""
        candidate = CandidateBullet(
            content="test bullet",
            section=BulletSection.GENERAL,
            knowledge_type=KnowledgeType.KNOWLEDGE,
            source_type=SourceType.INTERACTION,
        )
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            mem0_add_fn=mem0_add,
        )

        result = pipeline.process(
            "test message",
            scope="project:myapp",
            user_id="u1",
        )

        assert result.bullets_added == 1
        # Verify memx_scope is in the metadata written to mem0
        meta = mem0_add.call_args[1]["metadata"]
        assert meta["memx_scope"] == "project:myapp"

    def test_scope_default_global(self) -> None:
        """When scope is not passed, default to global."""
        candidate = CandidateBullet(
            content="test bullet",
            section=BulletSection.GENERAL,
            knowledge_type=KnowledgeType.KNOWLEDGE,
            source_type=SourceType.INTERACTION,
        )
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]
        mem0_add = MagicMock()

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            mem0_add_fn=mem0_add,
        )

        result = pipeline.process("test message")

        assert result.bullets_added == 1
        meta = mem0_add.call_args[1]["metadata"]
        assert meta["memx_scope"] == "global"

    def test_scope_on_candidate_object(self) -> None:
        """Verify the candidate object's scope field is set."""
        candidate = CandidateBullet(content="test bullet")
        mock_reflector = MagicMock()
        mock_reflector.reflect.return_value = [candidate]

        pipeline = IngestPipeline(
            reflector=mock_reflector,
            mem0_add_fn=MagicMock(),
        )

        pipeline.process("test", scope="project:abc")
        assert candidate.scope == "project:abc"


# ===========================================================================
# 9. RetrievalPipeline scope passthrough
# ===========================================================================


class TestRetrievalPipelineScopePassthrough:
    """RetrievalPipeline.search() passes scope to GeneratorEngine."""

    def test_scope_passed_to_generator(self) -> None:
        mock_gen = MagicMock(spec=GeneratorEngine)
        mock_gen.search.return_value = []
        mock_gen.mode = "full"

        pipeline = RetrievalPipeline(generator=mock_gen)
        pipeline.search("query", bullets=[], scope="project:myapp")

        mock_gen.search.assert_called_once()
        call_kwargs = mock_gen.search.call_args[1]
        assert call_kwargs["scope"] == "project:myapp"

    def test_no_scope_passed_as_none(self) -> None:
        mock_gen = MagicMock(spec=GeneratorEngine)
        mock_gen.search.return_value = []
        mock_gen.mode = "full"

        pipeline = RetrievalPipeline(generator=mock_gen)
        pipeline.search("query", bullets=[])

        call_kwargs = mock_gen.search.call_args[1]
        assert call_kwargs["scope"] is None


# ===========================================================================
# 10. Memory.add() scope passthrough
# ===========================================================================


class TestMemoryAddScope:
    """Memory.add() passes scope to IngestPipeline."""

    def test_add_with_scope(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = IngestResult(bullets_added=1)
        m._ingest_pipeline = mock_pipeline

        m.add("test", user_id="u1", scope="project:myapp")

        mock_pipeline.process.assert_called_once()
        call_kwargs = mock_pipeline.process.call_args[1]
        assert call_kwargs["scope"] == "project:myapp"

    def test_add_without_scope_defaults_global(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        mock_pipeline = MagicMock()
        mock_pipeline.process.return_value = IngestResult(bullets_added=1)
        m._ingest_pipeline = mock_pipeline

        m.add("test", user_id="u1")

        call_kwargs = mock_pipeline.process.call_args[1]
        assert call_kwargs["scope"] == "global"

    def test_add_with_invalid_scope_raises(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0_init_error = None
        m._retrieval_pipeline = None
        m._sanitizer = None
        m._ingest_pipeline = MagicMock()

        with pytest.raises(ValueError, match="requires a name"):
            m.add("test", scope="project:")


# ===========================================================================
# 11. Memory.search() scope passthrough
# ===========================================================================


class TestMemorySearchScope:
    """Memory.search() passes scope to RetrievalPipeline."""

    def test_search_with_scope(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0.get_all.return_value = {"memories": []}
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._sanitizer = None

        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = SearchResult(
            results=[], mode="full", total_candidates=0
        )
        m._retrieval_pipeline = mock_pipeline

        m.search("query", user_id="u1", scope="project:myapp")

        mock_pipeline.search.assert_called_once()
        call_kwargs = mock_pipeline.search.call_args[1]
        assert call_kwargs["scope"] == "project:myapp"

    def test_search_without_scope_passes_none(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig(ace_enabled=True)
        m._mem0 = MagicMock()
        m._mem0.get_all.return_value = {"memories": []}
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._sanitizer = None

        mock_pipeline = MagicMock()
        mock_pipeline.search.return_value = SearchResult(
            results=[], mode="full", total_candidates=0
        )
        m._retrieval_pipeline = mock_pipeline

        m.search("query", user_id="u1")

        call_kwargs = mock_pipeline.search.call_args[1]
        assert call_kwargs["scope"] is None


# ===========================================================================
# 12. Memory._load_bullets_for_search scope loading
# ===========================================================================


class TestLoadBulletsForSearchScope:
    """Memory._load_bullets_for_search() loads scope from mem0 payload."""

    def test_scope_loaded_from_payload(self) -> None:
        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._mem0 = MagicMock()
        m._mem0.get_all.return_value = {
            "memories": [
                {
                    "id": "b1",
                    "memory": "test content",
                    "metadata": {"memx_scope": "project:myapp"},
                }
            ]
        }
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        bullets = m._load_bullets_for_search()
        assert len(bullets) == 1
        assert bullets[0].scope == "project:myapp"

    def test_missing_scope_defaults_to_global(self) -> None:
        """Legacy payloads without memx_scope default to global."""
        m = Memory.__new__(Memory)
        m._config = MemXConfig()
        m._mem0 = MagicMock()
        m._mem0.get_all.return_value = {
            "memories": [
                {
                    "id": "b1",
                    "memory": "legacy content",
                    "metadata": {},
                }
            ]
        }
        m._mem0_init_error = None
        m._ingest_pipeline = None
        m._retrieval_pipeline = None
        m._sanitizer = None

        bullets = m._load_bullets_for_search()
        assert len(bullets) == 1
        assert bullets[0].scope == "global"


# ===========================================================================
# 13. CLI --scope option
# ===========================================================================


class TestCLIScopeOption:
    """CLI search --scope option."""

    def test_search_with_scope(self) -> None:
        runner = CliRunner()
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}
        with patch("memx.core.cli.main._create_memory", return_value=mock_memory):
            result = runner.invoke(
                cli, ["search", "query", "--scope", "project:myapp"]
            )
        assert result.exit_code == 0
        mock_memory.search.assert_called_once_with(
            "query", user_id=None, limit=5, scope="project:myapp"
        )

    def test_search_without_scope(self) -> None:
        runner = CliRunner()
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}
        with patch("memx.core.cli.main._create_memory", return_value=mock_memory):
            result = runner.invoke(cli, ["search", "query"])
        assert result.exit_code == 0
        mock_memory.search.assert_called_once_with(
            "query", user_id=None, limit=5, scope=None
        )

    def test_search_scope_combined_with_limit(self) -> None:
        runner = CliRunner()
        mock_memory = MagicMock()
        mock_memory.search.return_value = {"results": []}
        with patch("memx.core.cli.main._create_memory", return_value=mock_memory):
            result = runner.invoke(
                cli,
                ["search", "query", "--scope", "project:myapp", "--limit", "10"],
            )
        assert result.exit_code == 0
        mock_memory.search.assert_called_once_with(
            "query", user_id=None, limit=10, scope="project:myapp"
        )


# ===========================================================================
# 14. BulletInfo scope field
# ===========================================================================


class TestBulletInfoScope:
    """BulletInfo.scope defaults and construction."""

    def test_default_scope(self) -> None:
        info = BulletInfo(bullet_id="b1")
        assert info.scope == "global"

    def test_custom_scope(self) -> None:
        info = BulletInfo(bullet_id="b1", scope="project:myapp")
        assert info.scope == "project:myapp"


# ===========================================================================
# 15. BulletMetadata.scope (already existed, verify backward compat)
# ===========================================================================


class TestBulletMetadataScope:
    """BulletMetadata.scope field."""

    def test_default_scope(self) -> None:
        bm = BulletMetadata()
        assert bm.scope == "global"

    def test_project_scope(self) -> None:
        bm = BulletMetadata(scope="project:myapp")
        assert bm.scope == "project:myapp"

    def test_serialization_round_trip(self) -> None:
        bm = BulletMetadata(scope="project:my-app")
        d = bm.model_dump()
        restored = BulletMetadata.model_validate(d)
        assert restored.scope == "project:my-app"


# ===========================================================================
# 16. CandidateBullet.scope (already existed, verify in pipeline context)
# ===========================================================================


class TestCandidateBulletScope:
    """CandidateBullet.scope field."""

    def test_default_scope(self) -> None:
        cb = CandidateBullet()
        assert cb.scope == "global"

    def test_custom_scope(self) -> None:
        cb = CandidateBullet(scope="project:myapp")
        assert cb.scope == "project:myapp"


# ===========================================================================
# 17. End-to-end scope flow integration test
# ===========================================================================


class TestScopeEndToEnd:
    """Integration test for full scope flow through GeneratorEngine."""

    def test_project_scope_ranked_higher_than_global(self) -> None:
        """With scope set, project-scoped bullets should rank above global."""
        cfg = RetrievalConfig(scope_boost=1.3)
        engine = GeneratorEngine(config=cfg)
        bullets = [
            _make_bullet("b_global", "git rebase interactive", scope="global"),
            _make_bullet("b_project", "git rebase interactive", scope="project:myapp"),
        ]
        results = engine.search("git rebase", bullets, scope="project:myapp")
        assert len(results) == 2
        assert results[0].bullet_id == "b_project"
        assert results[0].final_score > results[1].final_score

    def test_no_scope_search_returns_all_equally(self) -> None:
        """Without scope, bullets from all scopes are treated equally."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase", scope="global"),
            _make_bullet("b2", "git rebase", scope="project:myapp"),
        ]
        results = engine.search("git rebase", bullets)
        assert len(results) == 2
        # Without scope, both get the same score (same content, same metadata)
        assert abs(results[0].final_score - results[1].final_score) < 1e-9

    def test_empty_project_scope_only_global_returned(self) -> None:
        """When no bullets match a project scope, only global results remain."""
        engine = GeneratorEngine()
        bullets = [
            _make_bullet("b1", "git rebase", scope="global"),
            _make_bullet("b2", "git rebase", scope="project:other"),
        ]
        results = engine.search("git rebase", bullets, scope="project:nonexistent")
        assert len(results) == 1
        assert results[0].bullet_id == "b1"

    def test_scope_boost_formula_verification(self) -> None:
        """Verify the exact formula: FinalScore = Blended * Decay * Recency * ScopeBoost."""
        cfg = RetrievalConfig(
            keyword_weight=0.6,
            semantic_weight=0.4,
            recency_boost_days=7,
            recency_boost_factor=1.2,
            scope_boost=1.5,
        )
        merger = ScoreMerger(cfg)
        infos = {
            "b1": _make_info("b1", "test", scope="project:x", days_ago=3, decay_weight=0.8),
        }
        kw = {"b1": 25.0}
        results = merger.merge(kw, None, infos, target_scope="project:x", now=_NOW)

        norm_kw = 25.0 / 35.0
        blended = norm_kw * 1.0  # degraded mode kw_weight = 1.0
        expected = blended * 0.8 * 1.2 * 1.5

        assert abs(results[0].final_score - expected) < 1e-9


# ===========================================================================
# 18. IngestPipeline existing bullets scope loading
# ===========================================================================


class TestIngestPipelineExistingBulletsScope:
    """IngestPipeline._load_existing_bullets() loads scope from mem0."""

    def test_existing_bullets_have_scope(self) -> None:
        pipeline = IngestPipeline(
            reflector=MagicMock(),
            mem0_get_all_fn=MagicMock(return_value={
                "memories": [
                    {
                        "id": "b1",
                        "memory": "content",
                        "metadata": {"memx_scope": "project:myapp"},
                    },
                    {
                        "id": "b2",
                        "memory": "content",
                        "metadata": {},
                    },
                ]
            }),
        )

        existing = pipeline._load_existing_bullets(None, None)
        assert len(existing) == 2
        assert existing[0].scope == "project:myapp"
        assert existing[1].scope == "global"  # default for missing memx_scope
