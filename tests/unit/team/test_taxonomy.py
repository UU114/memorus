"""Tests for TagTaxonomy — tag normalization and alignment."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from memx.team.taxonomy import (
    PRESET_ALIASES,
    PRESET_CATEGORIES,
    TagTaxonomy,
    TaxonomyResolver,
    _convert_sync_response,
    _cosine_similarity,
    build_preset_taxonomy,
    load_taxonomy_from_cache,
    load_taxonomy_from_git_fallback,
    normalize_with_vector_fallback,
    save_taxonomy_to_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_taxonomy(
    categories: dict[str, list[str]] | None = None,
    aliases: dict[str, str] | None = None,
) -> TagTaxonomy:
    """Create a TagTaxonomy with given data."""
    return TagTaxonomy(
        version=1,
        updated_at=datetime.now(timezone.utc),
        categories=categories or {"languages": ["python", "rust", "typescript"]},
        aliases=aliases or {"py": "python", "ts": "typescript"},
    )


def _make_embedder(tag_vectors: dict[str, list[float]]) -> MagicMock:
    """Create a mock embedder that returns predefined vectors for tags."""
    embedder = MagicMock()
    embedder.embed.side_effect = lambda text: tag_vectors.get(
        text, [0.0] * len(next(iter(tag_vectors.values())))
    )
    return embedder


# ---------------------------------------------------------------------------
# TagTaxonomy model
# ---------------------------------------------------------------------------


class TestTagTaxonomyModel:
    """TagTaxonomy data model and basic properties."""

    def test_default_construction(self):
        tax = TagTaxonomy()
        assert tax.version == 1
        assert tax.categories == {}
        assert tax.aliases == {}
        assert isinstance(tax.updated_at, datetime)

    def test_all_tags(self):
        tax = _make_taxonomy(
            categories={"lang": ["python", "rust"], "fw": ["react", "vue"]}
        )
        tags = tax.all_tags()
        assert tags == {"python", "rust", "react", "vue"}

    def test_all_tags_cached(self):
        tax = _make_taxonomy()
        tags1 = tax.all_tags()
        tags2 = tax.all_tags()
        assert tags1 is tags2  # same object (cached)

    def test_invalidate_cache(self):
        tax = _make_taxonomy()
        tags1 = tax.all_tags()
        tax.invalidate_cache()
        tags2 = tax.all_tags()
        assert tags1 == tags2
        assert tags1 is not tags2  # different object (rebuilt)


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------


class TestNormalize:
    """Tag normalization via exact alias and case-insensitive match."""

    def test_exact_alias_match(self):
        tax = _make_taxonomy(aliases={"py": "python", "ts": "typescript"})
        assert tax.normalize("py") == "python"
        assert tax.normalize("ts") == "typescript"

    def test_case_insensitive_match(self):
        tax = _make_taxonomy(categories={"lang": ["python", "TypeScript"]})
        assert tax.normalize("Python") == "python"
        assert tax.normalize("PYTHON") == "python"
        assert tax.normalize("typescript") == "TypeScript"

    def test_no_match_returns_original(self):
        tax = _make_taxonomy()
        assert tax.normalize("unknown-tag") == "unknown-tag"

    def test_canonical_tag_unchanged(self):
        tax = _make_taxonomy(categories={"lang": ["python"]})
        assert tax.normalize("python") == "python"

    def test_alias_takes_priority_over_case(self):
        """When a tag matches an alias exactly, alias wins over case match."""
        tax = _make_taxonomy(
            categories={"lang": ["Python3"]},
            aliases={"Python3": "python"},
        )
        assert tax.normalize("Python3") == "python"


class TestNormalizeTags:
    """Batch tag normalization with deduplication."""

    def test_basic_normalization(self):
        tax = _make_taxonomy(aliases={"py": "python", "ts": "typescript"})
        result = tax.normalize_tags(["py", "rust", "ts"])
        assert result == ["python", "rust", "typescript"]

    def test_deduplication(self):
        tax = _make_taxonomy(aliases={"py": "python"})
        result = tax.normalize_tags(["py", "python", "rust"])
        assert result == ["python", "rust"]

    def test_preserves_order(self):
        tax = _make_taxonomy()
        result = tax.normalize_tags(["rust", "python", "typescript"])
        assert result == ["rust", "python", "typescript"]

    def test_empty_list(self):
        tax = _make_taxonomy()
        assert tax.normalize_tags([]) == []


# ---------------------------------------------------------------------------
# merge()
# ---------------------------------------------------------------------------


class TestMerge:
    """Taxonomy merging (project-level overrides team-level)."""

    def test_categories_union_merged(self):
        t1 = _make_taxonomy(categories={"lang": ["python", "rust"]})
        t2 = _make_taxonomy(categories={"lang": ["go"], "fw": ["react"]})
        merged = t1.merge(t2)
        assert "go" in merged.categories["lang"]
        assert "python" in merged.categories["lang"]
        assert merged.categories["fw"] == ["react"]

    def test_aliases_override(self):
        t1 = _make_taxonomy(aliases={"py": "python"})
        t2 = _make_taxonomy(aliases={"py": "python3"})
        merged = t1.merge(t2)
        assert merged.aliases["py"] == "python3"  # t2 overrides

    def test_version_takes_max(self):
        t1 = TagTaxonomy(version=1)
        t2 = TagTaxonomy(version=3)
        merged = t1.merge(t2)
        assert merged.version == 3


# ---------------------------------------------------------------------------
# Preset taxonomy
# ---------------------------------------------------------------------------


class TestPresetTaxonomy:
    """Built-in preset taxonomy templates."""

    def test_build_preset(self):
        tax = build_preset_taxonomy()
        assert "languages" in tax.categories
        assert "frameworks" in tax.categories
        assert "domains" in tax.categories
        assert "practices" in tax.categories

    def test_preset_aliases(self):
        tax = build_preset_taxonomy()
        assert tax.normalize("reactjs") == "react"
        assert tax.normalize("py") == "python"
        assert tax.normalize("k8s") == "kubernetes"
        assert tax.normalize("ts") == "typescript"

    def test_preset_case_insensitive(self):
        tax = build_preset_taxonomy()
        assert tax.normalize("Python") == "python"
        assert tax.normalize("RUST") == "rust"

    def test_preset_has_common_languages(self):
        tax = build_preset_taxonomy()
        langs = tax.categories["languages"]
        for expected in ["python", "rust", "typescript", "go", "java"]:
            assert expected in langs


# ---------------------------------------------------------------------------
# Cache persistence
# ---------------------------------------------------------------------------


class TestCachePersistence:
    """Taxonomy cache save/load to ~/.ace/team_cache/{team_id}/taxonomy.json."""

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        tax = _make_taxonomy(
            categories={"lang": ["python"]},
            aliases={"py": "python"},
        )
        assert save_taxonomy_to_cache(tax, "test-team")

        loaded = load_taxonomy_from_cache("test-team")
        assert loaded is not None
        assert loaded.categories == {"lang": ["python"]}
        assert loaded.aliases == {"py": "python"}

    def test_load_nonexistent_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert load_taxonomy_from_cache("no-such-team") is None

    def test_load_corrupt_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cache_dir = tmp_path / ".ace" / "team_cache" / "bad"
        cache_dir.mkdir(parents=True)
        (cache_dir / "taxonomy.json").write_text("not-json{{{", encoding="utf-8")
        assert load_taxonomy_from_cache("bad") is None

    def test_load_invalid_format_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cache_dir = tmp_path / ".ace" / "team_cache" / "inv"
        cache_dir.mkdir(parents=True)
        (cache_dir / "taxonomy.json").write_text('"just a string"', encoding="utf-8")
        assert load_taxonomy_from_cache("inv") is None


# ---------------------------------------------------------------------------
# Git Fallback
# ---------------------------------------------------------------------------


class TestGitFallback:
    """Load taxonomy from project-level .ace/taxonomy.json."""

    def test_load_from_project_root(self, tmp_path):
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        tax_data = {
            "version": 2,
            "updated_at": "2026-01-01T00:00:00+00:00",
            "categories": {"custom": ["my-tag"]},
            "aliases": {"mt": "my-tag"},
        }
        (ace_dir / "taxonomy.json").write_text(
            json.dumps(tax_data), encoding="utf-8"
        )
        loaded = load_taxonomy_from_git_fallback(project_root=tmp_path)
        assert loaded is not None
        assert loaded.categories == {"custom": ["my-tag"]}
        assert loaded.aliases == {"mt": "my-tag"}

    def test_no_taxonomy_file_returns_none(self, tmp_path):
        assert load_taxonomy_from_git_fallback(project_root=tmp_path) is None

    def test_corrupt_file_returns_none(self, tmp_path):
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        (ace_dir / "taxonomy.json").write_text("!!!bad", encoding="utf-8")
        assert load_taxonomy_from_git_fallback(project_root=tmp_path) is None


# ---------------------------------------------------------------------------
# Server response conversion
# ---------------------------------------------------------------------------


class TestConvertSyncResponse:
    """Convert TaxonomyResponse to TagTaxonomy."""

    def test_basic_conversion(self):
        tags = [
            SimpleNamespace(name="python", aliases=["py", "Python3"], parent="languages"),
            SimpleNamespace(name="react", aliases=["reactjs"], parent="frameworks"),
        ]
        response = SimpleNamespace(tags=tags)
        tax = _convert_sync_response(response)

        assert "python" in tax.categories.get("languages", [])
        assert "react" in tax.categories.get("frameworks", [])
        assert tax.aliases["py"] == "python"
        assert tax.aliases["reactjs"] == "react"

    def test_no_parent_uses_uncategorized(self):
        tags = [SimpleNamespace(name="misc-tag", aliases=[], parent=None)]
        response = SimpleNamespace(tags=tags)
        tax = _convert_sync_response(response)
        assert "misc-tag" in tax.categories.get("uncategorized", [])

    def test_empty_response(self):
        response = SimpleNamespace(tags=[])
        tax = _convert_sync_response(response)
        assert tax.categories == {}
        assert tax.aliases == {}


# ---------------------------------------------------------------------------
# Vector similarity fallback
# ---------------------------------------------------------------------------


class TestVectorFallback:
    """Vector cosine similarity fallback for tag normalization."""

    def test_cosine_similarity_identical(self):
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_cosine_similarity_empty(self):
        assert _cosine_similarity([], []) == 0.0

    def test_cosine_similarity_length_mismatch(self):
        assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_vector_fallback_high_similarity(self):
        """Tag with >= 0.9 similarity to a canonical tag should be normalized."""
        tax = _make_taxonomy(categories={"lang": ["python"]}, aliases={})
        # Vectors are close (cosine ~0.995)
        embedder = _make_embedder({
            "python3": [0.9, 0.1],
            "python": [0.95, 0.1],
        })
        result = normalize_with_vector_fallback("python3", tax, embedder)
        assert result == "python"

    def test_vector_fallback_low_similarity(self):
        """Tag with < 0.9 similarity should be kept as-is."""
        tax = _make_taxonomy(categories={"lang": ["python"]}, aliases={})
        embedder = _make_embedder({
            "unrelated": [1.0, 0.0],
            "python": [0.0, 1.0],
        })
        result = normalize_with_vector_fallback("unrelated", tax, embedder)
        assert result == "unrelated"

    def test_vector_fallback_no_embedder(self):
        """Without an embedder, return original tag."""
        tax = _make_taxonomy(categories={"lang": ["python"]}, aliases={})
        result = normalize_with_vector_fallback("py3", tax, None)
        assert result == "py3"

    def test_vector_fallback_embedder_error(self):
        """Embedder that raises should gracefully return original tag."""
        tax = _make_taxonomy(categories={"lang": ["python"]}, aliases={})
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("model not loaded")
        result = normalize_with_vector_fallback("py3", tax, embedder)
        assert result == "py3"

    def test_exact_match_skips_vector(self):
        """If taxonomy.normalize() finds a match, vector search is skipped."""
        tax = _make_taxonomy(aliases={"py": "python"})
        embedder = MagicMock()
        result = normalize_with_vector_fallback("py", tax, embedder)
        assert result == "python"
        embedder.embed.assert_not_called()


# ---------------------------------------------------------------------------
# TaxonomyResolver
# ---------------------------------------------------------------------------


class TestTaxonomyResolver:
    """High-level taxonomy resolver with multi-source loading."""

    def test_default_loads_preset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        resolver = TaxonomyResolver(team_id="test")
        tax = resolver.taxonomy
        assert "languages" in tax.categories
        assert tax.normalize("py") == "python"

    def test_normalize_delegates(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        resolver = TaxonomyResolver(team_id="test")
        assert resolver.normalize("reactjs") == "react"
        assert resolver.normalize("unknown") == "unknown"

    def test_normalize_tags(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        resolver = TaxonomyResolver(team_id="test")
        result = resolver.normalize_tags(["py", "python", "reactjs"])
        assert result == ["python", "react"]

    def test_loads_from_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Save a custom taxonomy to cache
        custom = TagTaxonomy(
            categories={"custom": ["my-tag"]},
            aliases={"mt": "my-tag"},
        )
        save_taxonomy_to_cache(custom, "cached-team")

        resolver = TaxonomyResolver(team_id="cached-team")
        assert resolver.normalize("mt") == "my-tag"

    def test_loads_git_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        # Create project-level taxonomy
        ace_dir = tmp_path / "project" / ".ace"
        ace_dir.mkdir(parents=True)
        tax_data = {
            "categories": {"project": ["proj-tag"]},
            "aliases": {"pt": "proj-tag"},
        }
        (ace_dir / "taxonomy.json").write_text(
            json.dumps(tax_data), encoding="utf-8"
        )

        resolver = TaxonomyResolver(
            team_id="test",
            project_root=tmp_path / "project",
        )
        assert resolver.normalize("pt") == "proj-tag"

    def test_git_fallback_overrides_cache(self, tmp_path, monkeypatch):
        """Project-level taxonomy aliases override cache aliases."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        # Cache says py -> python
        cached = TagTaxonomy(aliases={"custom-alias": "cached-value"})
        save_taxonomy_to_cache(cached, "override-team")

        # Git fallback overrides
        ace_dir = tmp_path / "proj" / ".ace"
        ace_dir.mkdir(parents=True)
        tax_data = {"aliases": {"custom-alias": "git-value"}}
        (ace_dir / "taxonomy.json").write_text(
            json.dumps(tax_data), encoding="utf-8"
        )

        resolver = TaxonomyResolver(
            team_id="override-team",
            project_root=tmp_path / "proj",
        )
        assert resolver.normalize("custom-alias") == "git-value"

    def test_update_from_server(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        resolver = TaxonomyResolver(team_id="server-team")

        tags = [
            SimpleNamespace(name="go", aliases=["golang"], parent="languages"),
        ]
        response = SimpleNamespace(tags=tags)
        resolver.update_from_server(response)

        assert resolver.normalize("golang") == "go"
        # Cache should have been saved
        loaded = load_taxonomy_from_cache("server-team")
        assert loaded is not None

    def test_reload(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        resolver = TaxonomyResolver(team_id="reload-team")
        tax1 = resolver.taxonomy
        tax2 = resolver.reload()
        assert tax1 is not tax2  # different instances

    def test_vector_fallback_integration(self, tmp_path, monkeypatch):
        """Resolver uses vector fallback when taxonomy has no match."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        embedder = _make_embedder({
            "pythonn": [0.95, 0.1],
            "python": [0.95, 0.1],  # nearly identical
        })
        resolver = TaxonomyResolver(
            team_id="vec-team",
            embedder=embedder,
        )
        # "pythonn" has no alias or case match, but vector similarity should match
        result = resolver.normalize("pythonn")
        assert result == "python"


# ---------------------------------------------------------------------------
# Degradation scenarios
# ---------------------------------------------------------------------------


class TestDegradation:
    """Taxonomy unavailable — tag generation behavior unchanged."""

    def test_empty_taxonomy_passthrough(self):
        tax = TagTaxonomy()
        assert tax.normalize("any-tag") == "any-tag"
        assert tax.normalize_tags(["a", "b", "c"]) == ["a", "b", "c"]

    def test_resolver_without_server_or_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        resolver = TaxonomyResolver(team_id="nonexistent")
        # Should still work with presets
        assert resolver.normalize("unknown-custom-tag") == "unknown-custom-tag"
        # But presets should still work
        assert resolver.normalize("py") == "python"

    def test_normalize_with_none_embedder_and_no_match(self):
        tax = _make_taxonomy(categories={"lang": ["python"]}, aliases={})
        result = normalize_with_vector_fallback("newlang", tax, None)
        assert result == "newlang"
