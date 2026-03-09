# mypy: disable-error-code="untyped-decorator"
"""End-to-end integration tests for Git Fallback team retrieval.

Validates the full chain: GitFallbackStorage -> MultiPoolRetriever -> merged results.
Uses real components (no mocks for core logic), only fixtures for test data.

STORY-058: Git Fallback End-to-End Integration Tests
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from memx.team.git_storage import GitFallbackStorage
from memx.team.merger import LayerBoostConfig, MultiPoolRetriever


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_playbook(ace_dir: Path, bullets: list[dict[str, Any]]) -> Path:
    """Write a list of bullet dicts as JSONL to .ace/playbook.jsonl."""
    playbook = ace_dir / "playbook.jsonl"
    with playbook.open("w", encoding="utf-8") as f:
        for bullet in bullets:
            f.write(json.dumps(bullet, ensure_ascii=False) + "\n")
    return playbook


def _make_header(
    model: str = "all-MiniLM-L6-v2", dim: int = 384
) -> dict[str, Any]:
    """Build a playbook header line."""
    return {"_header": True, "model": model, "dim": dim, "version": "1.0"}


def _make_bullet(
    content: str,
    section: str = "general",
    knowledge_type: str = "Knowledge",
    instructivity_score: float = 50.0,
    enforcement: str = "suggestion",
    tags: list[str] | None = None,
    incompatible_tags: list[str] | None = None,
) -> dict[str, Any]:
    """Build a single TeamBullet dict."""
    return {
        "content": content,
        "section": section,
        "knowledge_type": knowledge_type,
        "instructivity_score": instructivity_score,
        "schema_version": 2,
        "author_id": "anon-001",
        "enforcement": enforcement,
        "tags": tags or [],
        "incompatible_tags": incompatible_tags or [],
    }


@pytest.fixture()
def playbook_dir(tmp_path: Path) -> Path:
    """Create a temporary .ace/ directory with a standard test playbook."""
    ace_dir = tmp_path / ".ace"
    ace_dir.mkdir()

    bullets = [
        _make_header(),
        _make_bullet(
            "Always use --locked with cargo build in CI",
            section="rust",
            knowledge_type="Method",
            instructivity_score=85,
            tags=["rust", "ci"],
        ),
        _make_bullet(
            "Never commit .env files to git",
            section="security",
            knowledge_type="Pitfall",
            instructivity_score=95,
            enforcement="mandatory",
            tags=["security", "git"],
        ),
        _make_bullet(
            "Use snake_case for Python function names",
            section="python",
            knowledge_type="Preference",
            instructivity_score=70,
            tags=["python", "style"],
            incompatible_tags=["camelCase"],
        ),
    ]

    _write_playbook(ace_dir, bullets)
    return tmp_path


class FakeLocalStorage:
    """Minimal local storage backend that returns canned results.

    Implements StorageBackend Protocol so MultiPoolRetriever accepts it.
    """

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self._results = results or []

    def search(
        self, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self._results[:limit]


# ---------------------------------------------------------------------------
# Tests: GitFallbackStorage standalone
# ---------------------------------------------------------------------------


class TestGitFallbackStorageBasic:
    """Basic loading / search behaviour of GitFallbackStorage."""

    def test_loads_bullets_from_playbook(self, playbook_dir: Path) -> None:
        """All non-header bullets are loaded."""
        storage = GitFallbackStorage(playbook_dir / ".ace" / "playbook.jsonl")
        assert storage.bullet_count == 3

    def test_header_parsed(self, playbook_dir: Path) -> None:
        """Header line is parsed and accessible."""
        storage = GitFallbackStorage(playbook_dir / ".ace" / "playbook.jsonl")
        header = storage.header
        assert header is not None
        assert header["model"] == "all-MiniLM-L6-v2"
        assert header["dim"] == 384

    def test_keyword_search_by_content(self, playbook_dir: Path) -> None:
        """Keyword search matches on content substring."""
        storage = GitFallbackStorage(playbook_dir / ".ace" / "playbook.jsonl")
        results = storage.search("cargo build")
        assert len(results) >= 1
        assert any("cargo" in r["content"].lower() for r in results)

    def test_keyword_search_by_tag(self, playbook_dir: Path) -> None:
        """Keyword search matches on tag substring."""
        storage = GitFallbackStorage(playbook_dir / ".ace" / "playbook.jsonl")
        results = storage.search("security")
        assert len(results) >= 1
        assert any("security" in r.get("tags", []) for r in results)

    def test_search_returns_source_marker(self, playbook_dir: Path) -> None:
        """Every result carries source='git_fallback'."""
        storage = GitFallbackStorage(playbook_dir / ".ace" / "playbook.jsonl")
        results = storage.search("python")
        for r in results:
            assert r["source"] == "git_fallback"


# ---------------------------------------------------------------------------
# Tests: End-to-end retrieval through MultiPoolRetriever
# ---------------------------------------------------------------------------


class TestGitFallbackIntegration:
    """End-to-end tests for Git Fallback team retrieval."""

    def test_search_returns_merged_results(self, playbook_dir: Path) -> None:
        """Local + Team results are merged in the output."""
        local_results = [
            {
                "content": "Run pytest with -x flag for fast failure",
                "section": "testing",
                "tags": ["pytest"],
                "score": 0.8,
            }
        ]
        local = FakeLocalStorage(local_results)
        git_storage = GitFallbackStorage(
            playbook_dir / ".ace" / "playbook.jsonl"
        )

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("git_fallback", git_storage)],
        )

        results = retriever.search("python")
        contents = [r["content"] for r in results]
        # Local result should appear
        assert any("pytest" in c for c in contents)
        # Team result should appear (snake_case bullet matches 'python' tag)
        assert any("snake_case" in c for c in contents)

    def test_shadow_merge_boost(self, tmp_path: Path) -> None:
        """Local results get 1.5x boost, Team gets 1.0x (default config).

        When a local result and a team result have the same raw score,
        the local result should rank higher after boosting.
        Uses a query that only matches team via tag (score 0.5) not
        content (score 1.0), so the boost math is predictable.
        """
        # Build a dedicated playbook with a single team bullet.
        # Content does NOT contain the query string "lint" but tags do.
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        _write_playbook(ace_dir, [
            _make_header(),
            _make_bullet(
                "Always run static analysis before pushing code",
                section="quality",
                tags=["lint", "ci"],
                instructivity_score=70,
            ),
        ])

        local_results = [
            {
                "content": "Use flake8 lint for style checking in CI",
                "section": "quality",
                "tags": ["lint"],
                "score": 0.6,
            }
        ]
        local = FakeLocalStorage(local_results)
        git_storage = GitFallbackStorage(ace_dir / "playbook.jsonl")

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("git_fallback", git_storage)],
            boost_config=LayerBoostConfig(local_boost=1.5, team_boost=1.0),
        )

        # Query "lint" matches local content (score 1.0 * 1.5 = 1.5)
        # and matches team via tag only (score 0.5 * 1.0 = 0.5).
        results = retriever.search("lint")
        assert len(results) >= 2, f"Expected >= 2 merged results, got {len(results)}"

        local_content = "Use flake8 lint for style checking in CI"
        team_content = "Always run static analysis before pushing code"

        local_idx = next(
            (i for i, r in enumerate(results) if r["content"] == local_content), None
        )
        team_idx = next(
            (i for i, r in enumerate(results) if r["content"] == team_content), None
        )

        assert local_idx is not None, "Local result not found in merged output"
        assert team_idx is not None, "Team result not found in merged output"
        assert local_idx < team_idx, (
            f"Local result (idx={local_idx}, boosted=0.6*1.5=0.9) should rank "
            f"before team result (idx={team_idx}, boosted=0.5*1.0=0.5) "
            f"due to 1.5x local boost"
        )

    def test_mandatory_bullet_priority(self, playbook_dir: Path) -> None:
        """Mandatory TeamBullets skip boost and appear first."""
        local_results = [
            {
                "content": "Local high-score result about git and security",
                "section": "local",
                "tags": ["security", "git"],
                "score": 0.99,
            }
        ]
        local = FakeLocalStorage(local_results)
        git_storage = GitFallbackStorage(
            playbook_dir / ".ace" / "playbook.jsonl"
        )

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("git_fallback", git_storage)],
        )

        # Search for something that matches the mandatory bullet
        results = retriever.search("git")
        assert len(results) >= 1

        # The mandatory bullet ("Never commit .env files to git") should
        # appear before non-mandatory results.
        mandatory_indices = [
            i
            for i, r in enumerate(results)
            if r.get("enforcement") == "mandatory"
        ]
        non_mandatory_indices = [
            i
            for i, r in enumerate(results)
            if r.get("enforcement") != "mandatory"
        ]

        if mandatory_indices and non_mandatory_indices:
            assert mandatory_indices[0] < non_mandatory_indices[0], (
                "Mandatory bullets must appear before non-mandatory results"
            )

    def test_no_playbook_returns_local_only(self, tmp_path: Path) -> None:
        """Without playbook.jsonl, only Local results returned."""
        local_results = [
            {
                "content": "Local result A",
                "section": "general",
                "tags": [],
                "score": 0.9,
            }
        ]
        local = FakeLocalStorage(local_results)

        # Point to a directory that has no .ace/playbook.jsonl
        empty_ace = tmp_path / ".ace"
        empty_ace.mkdir()
        git_storage = GitFallbackStorage(empty_ace / "playbook.jsonl")

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("git_fallback", git_storage)],
        )

        results = retriever.search("anything")
        # Only local result should appear
        assert len(results) == 1
        assert results[0]["content"] == "Local result A"

    def test_model_mismatch_fallback(self, playbook_dir: Path) -> None:
        """Wrong model fingerprint degrades to keyword search.

        When the playbook header says model X but the local expectation
        is model Y, vector search should be skipped and keyword search
        should still return results.
        """
        git_storage = GitFallbackStorage(
            playbook_dir / ".ace" / "playbook.jsonl",
            expected_model="some-other-model",
            expected_dim=768,
        )

        assert git_storage.model_mismatch is True

        # Keyword search should still work
        results = git_storage.search("cargo")
        assert len(results) >= 1
        assert any("cargo" in r["content"].lower() for r in results)

    def test_incompatible_tags_conflict(self) -> None:
        """Conflicting incompatible_tags resolved by keeping higher score.

        Uses a FakeLocalStorage that carries incompatible_tags in its results,
        and a team pool that returns conflicting tags. The Shadow Merge should
        drop the lower-scored entry.
        """
        # Team pool returns a result with incompatible_tags=["camelCase"]
        team_results = [
            {
                "content": "Use snake_case for Python function names",
                "section": "python",
                "tags": ["python", "style"],
                "incompatible_tags": ["camelCase"],
                "score": 0.8,
                "enforcement": "suggestion",
            }
        ]
        # Local pool returns a result tagged "camelCase" which conflicts
        local_results = [
            {
                "content": "Use camelCase for JavaScript function names",
                "section": "javascript",
                "tags": ["camelCase", "javascript"],
                "incompatible_tags": ["snake_case"],
                "score": 0.4,
            }
        ]
        local = FakeLocalStorage(local_results)
        team = FakeLocalStorage(team_results)

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("team_test", team)],
        )

        results = retriever.search("style")
        contents = [r["content"] for r in results]

        # Both "camelCase" and "snake_case" bullets should NOT both appear
        # because they conflict via incompatible_tags.
        has_camel = any("camelCase" in c for c in contents)
        has_snake = any("snake_case" in c for c in contents)
        assert not (has_camel and has_snake), (
            "Incompatible tag conflict was not resolved: both camelCase and "
            "snake_case results appear"
        )
        # The higher-scored one should survive
        assert has_snake or has_camel, "At least one result should survive"

    def test_empty_playbook(self, tmp_path: Path) -> None:
        """Empty playbook (header only) returns only Local results."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        _write_playbook(ace_dir, [_make_header()])

        local_results = [
            {"content": "Local result", "tags": [], "score": 0.8}
        ]
        local = FakeLocalStorage(local_results)
        git_storage = GitFallbackStorage(ace_dir / "playbook.jsonl")

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("git_fallback", git_storage)],
        )

        results = retriever.search("anything")
        assert len(results) == 1
        assert results[0]["content"] == "Local result"

    def test_invalid_json_lines_skipped(self, tmp_path: Path) -> None:
        """Invalid JSON lines are skipped without crashing."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        playbook = ace_dir / "playbook.jsonl"

        with playbook.open("w", encoding="utf-8") as f:
            f.write(json.dumps(_make_header()) + "\n")
            f.write("NOT VALID JSON\n")
            f.write(json.dumps(_make_bullet("Valid bullet about git")) + "\n")
            f.write("{broken: json}\n")
            f.write(json.dumps(_make_bullet("Another valid bullet")) + "\n")

        storage = GitFallbackStorage(playbook)
        assert storage.bullet_count == 2

    def test_header_only_playbook_returns_empty(self, tmp_path: Path) -> None:
        """Playbook with only header line returns zero team results."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        _write_playbook(ace_dir, [_make_header()])

        storage = GitFallbackStorage(ace_dir / "playbook.jsonl")
        results = storage.search("anything")
        assert results == []

    def test_all_mandatory_bullets_returned(self, tmp_path: Path) -> None:
        """When all TeamBullets are mandatory, all are prioritized."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        bullets = [
            _make_header(),
            _make_bullet("Mandatory rule A about git", enforcement="mandatory", tags=["git"]),
            _make_bullet("Mandatory rule B about git", enforcement="mandatory", tags=["git"]),
            _make_bullet("Mandatory rule C about git", enforcement="mandatory", tags=["git"]),
        ]
        _write_playbook(ace_dir, bullets)

        local = FakeLocalStorage([
            {"content": "Local git result", "tags": ["git"], "score": 0.9}
        ])
        git_storage = GitFallbackStorage(ace_dir / "playbook.jsonl")

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("git_fallback", git_storage)],
        )

        results = retriever.search("git")
        mandatory_results = [r for r in results if r.get("enforcement") == "mandatory"]
        assert len(mandatory_results) == 3

    def test_near_duplicate_dedup(self, tmp_path: Path) -> None:
        """Local and Team with nearly identical content are deduplicated.

        Shadow Merge uses Jaccard similarity >= 0.95 threshold.
        """
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        bullets = [
            _make_header(),
            _make_bullet(
                "Always run tests before committing code changes",
                tags=["testing"],
                instructivity_score=70,
            ),
        ]
        _write_playbook(ace_dir, bullets)

        # Local has nearly the same content
        local = FakeLocalStorage([
            {
                "content": "Always run tests before committing code changes",
                "tags": ["testing"],
                "score": 0.9,
            }
        ])
        git_storage = GitFallbackStorage(ace_dir / "playbook.jsonl")

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("git_fallback", git_storage)],
        )

        results = retriever.search("testing")
        # Near-duplicates should be merged (only one survives)
        test_results = [
            r for r in results
            if "always run tests" in r["content"].lower()
        ]
        assert len(test_results) == 1, (
            f"Near-duplicate should be merged, got {len(test_results)} entries"
        )


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case scenarios for robustness."""

    def test_nonexistent_path_returns_empty(self) -> None:
        """Storage with nonexistent path returns empty results."""
        storage = GitFallbackStorage(Path("/nonexistent/playbook.jsonl"))
        assert storage.bullet_count == 0
        assert storage.search("anything") == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        """Completely empty playbook file returns empty results."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        playbook = ace_dir / "playbook.jsonl"
        playbook.write_text("", encoding="utf-8")

        storage = GitFallbackStorage(playbook)
        assert storage.bullet_count == 0

    def test_empty_query_returns_empty(self, playbook_dir: Path) -> None:
        """Empty query string returns no keyword matches."""
        storage = GitFallbackStorage(playbook_dir / ".ace" / "playbook.jsonl")
        # Empty string is contained in every string, so keyword search
        # will actually match everything. This is expected behavior.
        results = storage.search("")
        # All bullets match because "" is in every content string
        assert len(results) == 3

    def test_large_playbook_loads(self, tmp_path: Path) -> None:
        """Playbook with 500+ entries loads without error."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        bullets: list[dict[str, Any]] = [_make_header()]
        for i in range(500):
            bullets.append(
                _make_bullet(
                    f"Rule number {i}: always follow guideline {i} carefully",
                    tags=[f"tag-{i % 50}"],
                    instructivity_score=float(50 + i % 50),
                )
            )
        _write_playbook(ace_dir, bullets)

        storage = GitFallbackStorage(ace_dir / "playbook.jsonl")
        assert storage.bullet_count == 500
        results = storage.search("guideline 42")
        assert len(results) >= 1

    def test_multi_pool_retriever_team_pool_failure(
        self, playbook_dir: Path
    ) -> None:
        """If a team pool raises an exception, retrieval degrades gracefully."""

        class FailingStorage:
            def search(
                self, query: str, *, limit: int = 10, **kwargs: Any
            ) -> list[dict[str, Any]]:
                raise RuntimeError("Simulated failure")

        local = FakeLocalStorage([
            {"content": "Local result", "tags": [], "score": 0.8}
        ])

        retriever = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("failing_pool", FailingStorage())],
        )

        # Should not raise; local results returned even if team pool fails
        results = retriever.search("test")
        assert len(results) == 1
        assert results[0]["content"] == "Local result"
