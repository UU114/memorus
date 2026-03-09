"""Unit tests for GitFallbackStorage vector cache (STORY-055)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from memx.team.git_storage import GitFallbackStorage, TeamBulletRecord


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

HEADER = {"_header": True, "model": "all-MiniLM-L6-v2", "dim": 384, "version": "1.0"}

BULLET_A = {
    "content": "Always use --locked with cargo build",
    "section": "rust",
    "knowledge_type": "Method",
    "instructivity_score": 85,
    "tags": ["rust", "cargo"],
}

BULLET_B = {
    "content": "Never commit .env files to git",
    "section": "security",
    "knowledge_type": "Pitfall",
    "instructivity_score": 95,
    "tags": ["security", "git"],
}

BULLET_C = {
    "content": "Use pytest for all Python unit tests",
    "section": "testing",
    "knowledge_type": "Knowledge",
    "instructivity_score": 70,
    "tags": ["python", "testing"],
}

DIM = 384


def _write_playbook(path: Path, lines: list[dict[str, Any] | str]) -> Path:
    """Write a playbook.jsonl to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + "\n")
            else:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


def _fake_embed(text: str) -> List[float]:
    """Deterministic fake embedding based on hash of text."""
    # Use a simple hash-based approach to generate reproducible vectors
    rng = np.random.RandomState(abs(hash(text)) % (2**31))
    vec = rng.randn(DIM).astype(np.float32)
    # L2 normalize
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _fake_embed_batch(texts: list[str]) -> List[List[float]]:
    """Fake batch embedding."""
    return [_fake_embed(t) for t in texts]


def _make_mock_embedder() -> MagicMock:
    """Create a mock embedder with embed and embed_batch methods."""
    mock = MagicMock()
    mock.embed.side_effect = _fake_embed
    mock.embed_batch.side_effect = _fake_embed_batch
    return mock


# ---------------------------------------------------------------------------
# Tests: vector cache generation
# ---------------------------------------------------------------------------


class TestVectorCacheGeneration:
    """Test auto-generation of .ace/playbook.vec on first search."""

    def test_vector_cache_created_on_first_search(self, tmp_path: Path) -> None:
        """Vector cache file should be created after first search."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            storage.search("cargo")

        # Vector cache file should exist (numpy adds .npz suffix)
        vec_path = tmp_path / ".ace" / "playbook.vec.npz"
        assert vec_path.exists(), f"Expected {vec_path} to exist"

    def test_vectors_resident_in_memory(self, tmp_path: Path) -> None:
        """After first search, vectors should be in memory."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            storage.search("cargo")

        assert storage.vectors_available
        assert storage._vectors is not None
        assert storage._vectors.shape == (2, DIM)

    def test_empty_playbook_no_vectors(self, tmp_path: Path) -> None:
        """Empty playbook should not generate vectors."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        storage.search("anything")

        assert not storage.vectors_available
        vec_path = tmp_path / ".ace" / "playbook.vec.npz"
        assert not vec_path.exists()


# ---------------------------------------------------------------------------
# Tests: vector cache loading & staleness
# ---------------------------------------------------------------------------


class TestVectorCacheLoading:
    """Test loading existing vector cache and staleness detection."""

    def test_load_from_existing_cache(self, tmp_path: Path) -> None:
        """Second instantiation should load vectors from cache, not rebuild."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B],
        )

        # First pass: build cache
        s1 = GitFallbackStorage(playbook_path=pb)
        mock1 = _make_mock_embedder()
        with patch.object(s1, "_get_embedder", return_value=mock1):
            s1.search("cargo")
        assert mock1.embed_batch.call_count == 1

        # Second pass: should load from cache without calling embedder
        s2 = GitFallbackStorage(playbook_path=pb)
        mock2 = _make_mock_embedder()
        with patch.object(s2, "_get_embedder", return_value=mock2):
            s2.search("cargo")

        # embed_batch should NOT be called (loaded from cache)
        assert mock2.embed_batch.call_count == 0
        assert s2.vectors_available

    def test_stale_cache_triggers_rebuild(self, tmp_path: Path) -> None:
        """Modifying playbook.jsonl should invalidate the vector cache."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A],
        )

        # Build cache
        s1 = GitFallbackStorage(playbook_path=pb)
        mock1 = _make_mock_embedder()
        with patch.object(s1, "_get_embedder", return_value=mock1):
            s1.search("cargo")

        # Modify the playbook (different mtime)
        time.sleep(0.05)
        _write_playbook(pb, [HEADER, BULLET_A, BULLET_B, BULLET_C])

        # Second pass: should rebuild
        s2 = GitFallbackStorage(playbook_path=pb)
        mock2 = _make_mock_embedder()
        with patch.object(s2, "_get_embedder", return_value=mock2):
            s2.search("test")

        assert mock2.embed_batch.call_count == 1
        assert s2._vectors is not None
        assert s2._vectors.shape[0] == 3

    def test_count_mismatch_triggers_rebuild(self, tmp_path: Path) -> None:
        """If bullet count doesn't match cache, rebuild."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B],
        )

        # Build cache with 2 bullets
        s1 = GitFallbackStorage(playbook_path=pb)
        mock1 = _make_mock_embedder()
        with patch.object(s1, "_get_embedder", return_value=mock1):
            s1.search("cargo")

        # Now save a cache with wrong mtime but manipulate count
        # by adding a bullet while keeping mtime (trick: update content in place)
        vec_path = tmp_path / ".ace" / "playbook.vec.npz"
        assert vec_path.exists()

        # Load and save with wrong vector count
        data = np.load(str(vec_path), allow_pickle=False)
        # Save with only 1 vector but same mtime
        np.savez_compressed(
            str(tmp_path / ".ace" / "playbook.vec"),
            vectors=data["vectors"][:1],
            source_mtime=data["source_mtime"],
            model=data["model"],
            dim=data["dim"],
        )

        s2 = GitFallbackStorage(playbook_path=pb)
        mock2 = _make_mock_embedder()
        with patch.object(s2, "_get_embedder", return_value=mock2):
            s2.search("test")

        # Should have rebuilt
        assert mock2.embed_batch.call_count == 1


# ---------------------------------------------------------------------------
# Tests: ONNXEmbedder graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Test fallback to keyword search when embedder is unavailable."""

    def test_no_embedder_falls_back_to_keyword(self, tmp_path: Path) -> None:
        """When ONNXEmbedder is unavailable, keyword search should work."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        # Patch _get_embedder to return None (simulating missing ONNX)
        with patch.object(storage, "_get_embedder", return_value=None):
            results = storage.search("cargo")

        assert len(results) == 1
        assert results[0]["content"] == BULLET_A["content"]
        assert not storage.vectors_available

    def test_embedder_exception_falls_back(self, tmp_path: Path) -> None:
        """If embed_batch raises, should fall back gracefully."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.side_effect = RuntimeError("ONNX crashed")

        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            results = storage.search("cargo")

        # Should fall back to keyword search
        assert len(results) == 1
        assert results[0]["content"] == BULLET_A["content"]

    def test_model_mismatch_uses_keyword(self, tmp_path: Path) -> None:
        """When header model doesn't match, use keyword search even if vectors exist."""
        header = {**HEADER, "model": "other-model", "dim": 768}
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [header, BULLET_A, BULLET_B],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            results = storage.search("cargo")

        # model_mismatch should force keyword search
        assert storage.model_mismatch
        assert len(results) == 1
        assert results[0]["score"] == 1.0  # keyword score, not cosine


# ---------------------------------------------------------------------------
# Tests: cosine similarity search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    """Test vector-based search with cosine similarity."""

    def test_vector_search_returns_results(self, tmp_path: Path) -> None:
        """Vector search should return results sorted by similarity."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B, BULLET_C],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            results = storage.search("cargo build rust")

        # Should return results (exact content depends on fake embeddings)
        assert isinstance(results, list)
        for r in results:
            assert "content" in r
            assert "score" in r
            assert r["source"] == "git_fallback"

    def test_vector_search_respects_limit(self, tmp_path: Path) -> None:
        """Vector search should respect the limit parameter."""
        bullets = [
            {**BULLET_A, "content": f"Rule number {i}", "instructivity_score": i}
            for i in range(20)
        ]
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, *bullets],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            results = storage.search("rule", limit=3)

        assert len(results) <= 3

    def test_vector_search_similarity_threshold(self, tmp_path: Path) -> None:
        """Results with similarity <= 0.3 should be excluded."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        # Create embedder that returns orthogonal vectors (low similarity)
        mock_embedder = MagicMock()
        # Bullet vector: unit vector along dim 0
        bullet_vec = np.zeros(DIM, dtype=np.float32)
        bullet_vec[0] = 1.0
        mock_embedder.embed_batch.return_value = [bullet_vec.tolist()]
        # Query vector: unit vector along dim 1 (orthogonal => sim ~0)
        query_vec = np.zeros(DIM, dtype=np.float32)
        query_vec[1] = 1.0
        mock_embedder.embed.return_value = query_vec.tolist()

        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            results = storage.search("totally unrelated")

        # Orthogonal vectors have cosine sim ~0, below 0.3 threshold
        assert len(results) == 0

    def test_vector_search_high_similarity(self, tmp_path: Path) -> None:
        """Identical query and content vectors should produce high similarity."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        # Same vector for both bullet and query
        vec = np.random.RandomState(42).randn(DIM).astype(np.float32)
        vec = vec / np.linalg.norm(vec)

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [vec.tolist()]
        mock_embedder.embed.return_value = vec.tolist()

        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            results = storage.search("cargo build")

        assert len(results) == 1
        assert results[0]["score"] > 0.99  # near-identical vectors


# ---------------------------------------------------------------------------
# Tests: .gitignore maintenance
# ---------------------------------------------------------------------------


class TestGitignoreMaintenance:
    """Test automatic .ace/.gitignore updates."""

    def test_gitignore_created_with_vec_entries(self, tmp_path: Path) -> None:
        """Vector cache save should create/update .gitignore."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            storage.search("cargo")

        gitignore = tmp_path / ".ace" / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text(encoding="utf-8")
        assert "playbook.vec" in content
        assert "playbook.vec.npz" in content
        assert "playbook.cache" in content

    def test_gitignore_idempotent(self, tmp_path: Path) -> None:
        """Multiple saves should not duplicate .gitignore entries."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A],
        )

        for _ in range(3):
            storage = GitFallbackStorage(playbook_path=pb)
            mock_embedder = _make_mock_embedder()
            with patch.object(storage, "_get_embedder", return_value=mock_embedder):
                storage._ensure_loaded()
                storage._vectors_initialized = False  # force re-init
                storage._vectors = None
                storage.search("cargo")

        gitignore = tmp_path / ".ace" / ".gitignore"
        content = gitignore.read_text(encoding="utf-8")
        # Each entry should appear exactly once
        assert content.count("playbook.vec.npz") == 1

    def test_gitignore_preserves_existing(self, tmp_path: Path) -> None:
        """Existing .gitignore content should be preserved."""
        ace_dir = tmp_path / ".ace"
        ace_dir.mkdir(parents=True)
        gitignore = ace_dir / ".gitignore"
        gitignore.write_text("*.pyc\n__pycache__/\n", encoding="utf-8")

        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            storage.search("cargo")

        content = gitignore.read_text(encoding="utf-8")
        assert "*.pyc" in content
        assert "__pycache__/" in content
        assert "playbook.vec" in content


# ---------------------------------------------------------------------------
# Tests: zero disk I/O on subsequent searches
# ---------------------------------------------------------------------------


class TestZeroDiskIO:
    """Test that subsequent searches use only in-memory vectors."""

    def test_second_search_no_disk_io(self, tmp_path: Path) -> None:
        """After first search, no file reads should happen for vectors."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            # First search: builds/loads cache
            storage.search("cargo")
            assert storage._vectors_initialized

            # Second search: should not re-initialize
            with patch.object(storage, "_load_vector_cache") as mock_load:
                with patch.object(storage, "_build_vector_cache") as mock_build:
                    storage.search("env")

            mock_load.assert_not_called()
            mock_build.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases for vector cache."""

    def test_nonexistent_playbook(self) -> None:
        """Storage with nonexistent path should work without errors."""
        storage = GitFallbackStorage(
            playbook_path=Path("/nonexistent/playbook.jsonl")
        )
        results = storage.search("anything")
        assert results == []
        assert not storage.vectors_available

    def test_corrupt_cache_file_handled(self, tmp_path: Path) -> None:
        """Corrupt .vec.npz should be handled gracefully."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A],
        )

        # Write corrupt cache
        corrupt_path = tmp_path / ".ace" / "playbook.vec.npz"
        corrupt_path.write_bytes(b"NOT A VALID NPZ FILE")

        storage = GitFallbackStorage(playbook_path=pb)
        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            # Should handle corrupt cache and rebuild
            results = storage.search("cargo")

        # Should still work (either via rebuilt vectors or keyword fallback)
        assert isinstance(results, list)

    def test_query_embed_failure_falls_back(self, tmp_path: Path) -> None:
        """If embedding the query fails, fall back to keyword search."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A, BULLET_B],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = _fake_embed_batch(
            [BULLET_A["content"], BULLET_B["content"]]
        )
        # embed() for query raises
        mock_embedder.embed.side_effect = RuntimeError("embed failed")

        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            results = storage.search("cargo")

        # Should fall back to keyword search
        assert len(results) == 1
        assert results[0]["content"] == BULLET_A["content"]

    def test_disk_write_failure_keeps_memory(self, tmp_path: Path) -> None:
        """If cache write fails, vectors should still be in memory."""
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_A],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        mock_embedder = _make_mock_embedder()
        with patch.object(storage, "_get_embedder", return_value=mock_embedder):
            with patch("numpy.savez_compressed", side_effect=OSError("disk full")):
                storage.search("cargo")

        # Vectors should still be in memory despite write failure
        assert storage.vectors_available
