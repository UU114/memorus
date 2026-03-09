"""Unit tests for GitFallbackStorage (STORY-054)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memx.team.git_storage import GitFallbackStorage, TeamBulletRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HEADER = {"_header": True, "model": "all-MiniLM-L6-v2", "dim": 384, "version": "1.0"}

BULLET_CARGO = {
    "content": "Always use --locked with cargo build",
    "section": "rust",
    "knowledge_type": "Method",
    "instructivity_score": 85,
    "schema_version": 2,
    "author_id": "anon-abc123",
    "enforcement": "suggestion",
    "tags": ["rust", "cargo"],
}

BULLET_ENV = {
    "content": "Never commit .env files to git",
    "section": "security",
    "knowledge_type": "Pitfall",
    "instructivity_score": 95,
    "schema_version": 2,
    "enforcement": "mandatory",
    "tags": ["security", "git"],
    "incompatible_tags": [],
}


def _write_playbook(path: Path, lines: list[dict | str]) -> Path:
    """Write a playbook.jsonl to the given path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + "\n")
            else:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Tests: normal loading
# ---------------------------------------------------------------------------


class TestNormalLoading:
    def test_load_with_header_and_bullets(self, tmp_path: Path) -> None:
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_CARGO, BULLET_ENV],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        assert storage.bullet_count == 2
        assert storage.header is not None
        assert storage.header["model"] == "all-MiniLM-L6-v2"
        assert not storage.model_mismatch

    def test_search_content_match(self, tmp_path: Path) -> None:
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_CARGO, BULLET_ENV],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        results = storage.search("cargo")
        assert len(results) == 1
        assert results[0]["content"] == BULLET_CARGO["content"]
        assert results[0]["source"] == "git_fallback"
        assert results[0]["score"] == 1.0

    def test_search_tag_match(self, tmp_path: Path) -> None:
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_CARGO, BULLET_ENV],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        # "security" is a tag on BULLET_ENV but also in section field of content
        results = storage.search("rust")
        # Should find BULLET_CARGO via content or tag
        assert len(results) >= 1

    def test_search_limit(self, tmp_path: Path) -> None:
        bullets = [
            {**BULLET_CARGO, "content": f"Rule {i}", "instructivity_score": i}
            for i in range(20)
        ]
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, *bullets],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        results = storage.search("Rule", limit=5)
        assert len(results) == 5

    def test_search_no_match(self, tmp_path: Path) -> None:
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_CARGO],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        results = storage.search("nonexistent_xyz")
        assert results == []

    def test_search_case_insensitive(self, tmp_path: Path) -> None:
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_CARGO],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        results = storage.search("CARGO")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_file_not_exist(self) -> None:
        storage = GitFallbackStorage(
            playbook_path=Path("/nonexistent/playbook.jsonl")
        )
        results = storage.search("anything")
        assert results == []
        assert storage.bullet_count == 0

    def test_empty_file(self, tmp_path: Path) -> None:
        pb = tmp_path / ".ace" / "playbook.jsonl"
        pb.parent.mkdir(parents=True)
        pb.write_text("", encoding="utf-8")

        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 0
        assert storage.header is None

    def test_header_only(self, tmp_path: Path) -> None:
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 0
        assert storage.header is not None

    def test_invalid_json_skipped(self, tmp_path: Path, caplog) -> None:
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, "NOT VALID JSON {{{", BULLET_CARGO],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        with caplog.at_level("WARNING"):
            assert storage.bullet_count == 1

        assert "Invalid JSON" in caplog.text

    def test_empty_content_bullet_excluded(self, tmp_path: Path) -> None:
        empty_bullet = {**BULLET_CARGO, "content": ""}
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, empty_bullet, BULLET_ENV],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 1

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        pb = tmp_path / ".ace" / "playbook.jsonl"
        pb.parent.mkdir(parents=True)
        lines = [
            json.dumps(HEADER),
            "",
            "   ",
            json.dumps(BULLET_CARGO),
        ]
        pb.write_text("\n".join(lines), encoding="utf-8")

        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.bullet_count == 1

    def test_none_path(self) -> None:
        # When path is None (no playbook found), should return empty
        storage = GitFallbackStorage.__new__(GitFallbackStorage)
        storage._path = None
        storage._bullets = []
        storage._header = None
        storage._loaded = False
        storage._model_mismatch = False
        storage._expected_model = "all-MiniLM-L6-v2"
        storage._expected_dim = 384
        storage._vectors = None
        storage._embedder = None
        storage._vectors_initialized = False

        assert storage.search("test") == []


# ---------------------------------------------------------------------------
# Tests: model fingerprint
# ---------------------------------------------------------------------------


class TestModelFingerprint:
    def test_matching_model(self, tmp_path: Path) -> None:
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, BULLET_CARGO],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        storage._ensure_loaded()
        assert not storage.model_mismatch

    def test_mismatched_model(self, tmp_path: Path, caplog) -> None:
        header = {**HEADER, "model": "other-model", "dim": 768}
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [header, BULLET_CARGO],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        with caplog.at_level("WARNING"):
            storage._ensure_loaded()

        assert storage.model_mismatch
        assert "mismatch" in caplog.text.lower()

    def test_mismatched_dim_only(self, tmp_path: Path) -> None:
        header = {**HEADER, "dim": 768}
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [header, BULLET_CARGO],
        )
        storage = GitFallbackStorage(playbook_path=pb)
        assert storage.model_mismatch

    def test_custom_expected_model(self, tmp_path: Path) -> None:
        header = {"_header": True, "model": "custom-model", "dim": 512}
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [header, BULLET_CARGO],
        )
        storage = GitFallbackStorage(
            playbook_path=pb,
            expected_model="custom-model",
            expected_dim=512,
        )
        assert not storage.model_mismatch


# ---------------------------------------------------------------------------
# Tests: read-only guarantee
# ---------------------------------------------------------------------------


class TestReadOnly:
    def test_no_write_methods(self) -> None:
        """Verify that GitFallbackStorage has no write/add/update/delete methods."""
        write_prefixes = ("write", "add", "update", "delete", "remove", "save", "put")
        methods = [m for m in dir(GitFallbackStorage) if not m.startswith("_")]
        for method in methods:
            assert not any(
                method.startswith(p) for p in write_prefixes
            ), f"Found write-like method: {method}"


# ---------------------------------------------------------------------------
# Tests: TeamBulletRecord dataclass
# ---------------------------------------------------------------------------


class TestTeamBulletRecord:
    def test_defaults(self) -> None:
        r = TeamBulletRecord()
        assert r.content == ""
        assert r.section == "general"
        assert not r.is_active  # empty content

    def test_is_active(self) -> None:
        r = TeamBulletRecord(content="something")
        assert r.is_active

    def test_extra_fields(self) -> None:
        r = TeamBulletRecord(content="test", extra={"foo": "bar"})
        assert r.extra["foo"] == "bar"


# ---------------------------------------------------------------------------
# Tests: UTF-8 support
# ---------------------------------------------------------------------------


class TestUTF8:
    def test_chinese_content(self, tmp_path: Path) -> None:
        bullet = {**BULLET_CARGO, "content": "使用 cargo build 时总是加上 --locked"}
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, bullet],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        results = storage.search("cargo")
        assert len(results) == 1
        assert "cargo" in results[0]["content"]

    def test_unicode_tags(self, tmp_path: Path) -> None:
        bullet = {**BULLET_CARGO, "content": "rule", "tags": ["编译", "构建"]}
        pb = _write_playbook(
            tmp_path / ".ace" / "playbook.jsonl",
            [HEADER, bullet],
        )
        storage = GitFallbackStorage(playbook_path=pb)

        results = storage.search("编译")
        assert len(results) == 1
