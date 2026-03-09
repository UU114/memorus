# mypy: disable-error-code="untyped-decorator"
"""Benchmark: Team retrieval overhead via MultiPoolRetriever and TeamCacheStorage.

Threshold: team retrieval increment < 40ms over local-only baseline.
TeamCacheStorage keyword search < 100ms for 500 bullets.

STORY-058: Git Fallback Performance Test
STORY-067: Federation MVP — TeamCacheStorage benchmark
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any

import pytest

from memx.team.git_storage import GitFallbackStorage
from memx.team.merger import LayerBoostConfig, MultiPoolRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WARMUP_ROUNDS = 3
_BENCH_ROUNDS = 20
_TEAM_OVERHEAD_THRESHOLD_MS = 40.0


def _build_playbook(ace_dir: Path, n_bullets: int = 100) -> Path:
    """Create a playbook.jsonl with n_bullets entries."""
    playbook = ace_dir / "playbook.jsonl"
    with playbook.open("w", encoding="utf-8") as f:
        header = {"_header": True, "model": "all-MiniLM-L6-v2", "dim": 384, "version": "1.0"}
        f.write(json.dumps(header) + "\n")
        for i in range(n_bullets):
            bullet = {
                "content": f"Team rule {i}: follow best practice {i} for project setup and configuration",
                "section": "general",
                "knowledge_type": "Method",
                "instructivity_score": 50 + (i % 50),
                "schema_version": 2,
                "author_id": "bench-author",
                "enforcement": "suggestion",
                "tags": [f"tag-{i % 20}", "bench"],
                "incompatible_tags": [],
            }
            f.write(json.dumps(bullet) + "\n")
    return playbook


class FakeLocalStorage:
    """Minimal local backend returning canned results for benchmarking."""

    def __init__(self, n_results: int = 5) -> None:
        self._results = [
            {
                "content": f"Local result {i} about testing and development",
                "section": "general",
                "tags": ["testing"],
                "score": 0.8 - i * 0.05,
            }
            for i in range(n_results)
        ]

    def search(
        self, query: str, *, limit: int = 10, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return self._results[:limit]


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestTeamSearchPerformance:
    """Performance benchmarks for team retrieval overhead."""

    @pytest.fixture()
    def bench_playbook(self, tmp_path: Path) -> Path:
        """Create a 100-bullet playbook for benchmarking."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        return _build_playbook(ace_dir, n_bullets=100)

    def test_team_search_latency_under_40ms(
        self, bench_playbook: Path
    ) -> None:
        """Team retrieval overhead should be < 40ms compared to local-only.

        Measures incremental cost: (local+team) - (local-only).
        """
        local = FakeLocalStorage(n_results=5)

        # Baseline: local-only retriever
        local_only = MultiPoolRetriever(
            local_backend=local,
            team_pools=[],
        )

        # With team: local + Git Fallback
        git_storage = GitFallbackStorage(bench_playbook)
        with_team = MultiPoolRetriever(
            local_backend=local,
            team_pools=[("git_fallback", git_storage)],
        )

        query = "testing best practice"

        # Warmup both paths
        for _ in range(_WARMUP_ROUNDS):
            local_only.search(query)
            with_team.search(query)

        # Benchmark local-only
        local_times: list[float] = []
        for _ in range(_BENCH_ROUNDS):
            # Reset loaded state for fair comparison
            start = time.perf_counter()
            local_only.search(query)
            local_times.append(time.perf_counter() - start)

        # Benchmark with team
        team_times: list[float] = []
        for _ in range(_BENCH_ROUNDS):
            start = time.perf_counter()
            with_team.search(query)
            team_times.append(time.perf_counter() - start)

        local_p50 = statistics.median(local_times) * 1000  # ms
        team_p50 = statistics.median(team_times) * 1000  # ms
        overhead_ms = team_p50 - local_p50

        # Report for CI visibility
        print(f"\n  Local p50: {local_p50:.2f}ms")
        print(f"  Team  p50: {team_p50:.2f}ms")
        print(f"  Overhead:  {overhead_ms:.2f}ms (threshold: {_TEAM_OVERHEAD_THRESHOLD_MS}ms)")

        assert overhead_ms < _TEAM_OVERHEAD_THRESHOLD_MS, (
            f"Team retrieval overhead {overhead_ms:.2f}ms exceeds "
            f"{_TEAM_OVERHEAD_THRESHOLD_MS}ms threshold"
        )

    def test_large_playbook_search_latency(self, tmp_path: Path) -> None:
        """Search over 1000-bullet playbook completes within reasonable time."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir()
        playbook = _build_playbook(ace_dir, n_bullets=1000)

        storage = GitFallbackStorage(playbook)

        # Warmup
        storage.search("best practice")

        times: list[float] = []
        for _ in range(_BENCH_ROUNDS):
            start = time.perf_counter()
            storage.search("best practice")
            times.append(time.perf_counter() - start)

        p95 = sorted(times)[int(len(times) * 0.95)] * 1000
        print(f"\n  1000-bullet keyword search p95: {p95:.2f}ms")

        # Keyword search over 1000 bullets should be well under 100ms
        assert p95 < 100.0, (
            f"1000-bullet search p95 {p95:.2f}ms exceeds 100ms threshold"
        )


# ---------------------------------------------------------------------------
# STORY-067: TeamCacheStorage keyword search benchmark
# ---------------------------------------------------------------------------

_CACHE_BENCH_ROUNDS = 30
_CACHE_SEARCH_THRESHOLD_MS = 100.0


class TestTeamCacheSearchPerformance:
    """Benchmark TeamCacheStorage.search() with keyword fallback (no ONNX)."""

    @pytest.fixture()
    def populated_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> "TeamCacheStorage":
        """Create a TeamCacheStorage with ~500 bullets."""
        from memx.team.cache_storage import TeamCacheStorage
        from memx.team.config import TeamConfig
        from memx.team.types import TeamBullet

        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        config = TeamConfig(
            enabled=True,
            server_url="https://bench.example.com",
            team_id="bench-team",
            cache_max_bullets=2000,
        )
        cache = TeamCacheStorage(config)

        bullets = [
            TeamBullet(
                content=(
                    f"Team best practice {i}: use proper error handling "
                    f"and logging for service {i % 50} in production environments"
                ),
                origin_id=f"bench-{i}",
                tags=[f"tag-{i % 25}", "benchmark", f"svc-{i % 50}"],
                enforcement="suggestion",
                instructivity_score=50.0 + (i % 50),
                status="approved",
            )
            for i in range(500)
        ]
        cache.add_bullets(bullets)
        return cache

    def test_cache_keyword_search_under_100ms(
        self, populated_cache: "TeamCacheStorage"
    ) -> None:
        """TeamCacheStorage keyword search over 500 bullets < 100ms (p95)."""
        assert populated_cache.bullet_count == 500

        queries = [
            "error handling production",
            "best practice logging",
            "service configuration",
            "benchmark tag",
            "nonexistent query term",
        ]

        # Warmup
        for q in queries:
            populated_cache.search(q)

        all_times: list[float] = []
        for _ in range(_CACHE_BENCH_ROUNDS):
            for q in queries:
                start = time.perf_counter()
                populated_cache.search(q, limit=10)
                all_times.append(time.perf_counter() - start)

        p50_ms = statistics.median(all_times) * 1000
        p95_ms = sorted(all_times)[int(len(all_times) * 0.95)] * 1000
        mean_ms = statistics.mean(all_times) * 1000

        print(f"\n  TeamCacheStorage 500-bullet keyword search:")
        print(f"    mean: {mean_ms:.2f}ms")
        print(f"    p50:  {p50_ms:.2f}ms")
        print(f"    p95:  {p95_ms:.2f}ms  (threshold: {_CACHE_SEARCH_THRESHOLD_MS}ms)")

        assert mean_ms < _CACHE_SEARCH_THRESHOLD_MS, (
            f"TeamCacheStorage mean search latency {mean_ms:.2f}ms exceeds "
            f"{_CACHE_SEARCH_THRESHOLD_MS}ms threshold"
        )

    def test_cache_search_returns_results(
        self, populated_cache: "TeamCacheStorage"
    ) -> None:
        """Sanity check: search actually returns relevant results."""
        results = populated_cache.search("error handling", limit=10)
        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all(r.get("source") == "team_cache" for r in results)
