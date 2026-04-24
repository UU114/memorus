"""Unit tests for the VerificationEngine (STORY-R101).

Coverage per story AC:
1. verified path (anchor hits, hash match)
2. stale path (anchor_text missing from file)
3. unverifiable (file deleted / outside project_root / IO error)
4. not_applicable (no anchors)
5. TTL hit performs zero file reads (monkey-patched ``open()`` counts I/O)
6. Batch with same-file bullets reads the file exactly once
7. _resolve_project_root: config override + cwd walk-up
8. Cross-language parity against ``tests/fixtures/verifier_fixture.json``
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from memorus.core.config import VerificationConfig
from memorus.core.engines.verifier import (
    VerificationEngine,
    VerificationOutcome,
)
from memorus.core.types import Anchor, BulletMetadata, VerifiedStatus


_FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "verifier_fixture.json"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_anchor(
    file_path: str,
    anchor_text: str,
    hash_source: str | None = None,
    created_at: datetime | None = None,
) -> Anchor:
    """Build an anchor whose hash is the sha256 of ``hash_source`` (default:
    ``anchor_text``). Pass a different ``hash_source`` to simulate a tampered
    anchor row."""
    src = hash_source if hash_source is not None else anchor_text
    return Anchor(
        file_path=file_path,
        anchor_text=anchor_text,
        anchor_hash=hashlib.sha256(src.encode("utf-8")).hexdigest(),
        created_at=created_at or _now(),
    )


def _make_bullet(
    anchors: list[Anchor] | None = None,
    verified_at: datetime | None = None,
    verified_status: str | None = None,
    trust_score: float | None = None,
) -> BulletMetadata:
    return BulletMetadata(
        anchors=anchors or [],
        verified_at=verified_at,
        verified_status=verified_status,
        trust_score=trust_score,
    )


def _write_tree(root: Path, tree: dict[str, str]) -> None:
    for rel, content in tree.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _engine(
    root: Path | None = None,
    ttl_seconds: int = 60,
) -> VerificationEngine:
    cfg = VerificationConfig(
        ttl_seconds=ttl_seconds,
        project_root=str(root) if root is not None else None,
    )
    return VerificationEngine(cfg)


# ---------------------------------------------------------------------------
# 1. verified
# ---------------------------------------------------------------------------


class TestVerifiedPath:
    def test_single_anchor_matches_file_content(self, tmp_path: Path) -> None:
        _write_tree(
            tmp_path,
            {"src/a.rs": "fn alpha() { /* hello */ }\n"},
        )
        bullet = _make_bullet(anchors=[_make_anchor("src/a.rs", "fn alpha()")])
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.VERIFIED
        assert out.trust_score == 1.0

    def test_multiple_anchors_all_match(self, tmp_path: Path) -> None:
        _write_tree(tmp_path, {"x.rs": "fn one() {}\nfn two() {}\n"})
        bullet = _make_bullet(
            anchors=[
                _make_anchor("x.rs", "fn one()"),
                _make_anchor("x.rs", "fn two()"),
            ]
        )
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.VERIFIED


# ---------------------------------------------------------------------------
# 2. stale
# ---------------------------------------------------------------------------


class TestStalePath:
    def test_anchor_text_missing_from_file(self, tmp_path: Path) -> None:
        _write_tree(tmp_path, {"utils.rs": "fn alive() {}\n"})
        bullet = _make_bullet(
            anchors=[_make_anchor("utils.rs", "fn deprecated_symbol()")]
        )
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.STALE
        assert out.trust_score == 0.3

    def test_tampered_anchor_hash_flags_stale(self, tmp_path: Path) -> None:
        _write_tree(tmp_path, {"x.rs": "fn real() {}\n"})
        # anchor_text exists in file, but anchor_hash points at a different
        # string — meaning the anchor row itself was tampered with.
        bullet = _make_bullet(
            anchors=[
                _make_anchor(
                    "x.rs",
                    "fn real()",
                    hash_source="something completely different",
                )
            ]
        )
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.STALE

    def test_first_stale_anchor_short_circuits_rest(
        self, tmp_path: Path
    ) -> None:
        # Only the first anchor misses — the second would verify. Engine must
        # still return STALE without silently upgrading to VERIFIED.
        _write_tree(tmp_path, {"x.rs": "fn two() {}\n"})
        bullet = _make_bullet(
            anchors=[
                _make_anchor("x.rs", "fn missing()"),
                _make_anchor("x.rs", "fn two()"),
            ]
        )
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.STALE


# ---------------------------------------------------------------------------
# 3. unverifiable
# ---------------------------------------------------------------------------


class TestUnverifiablePath:
    def test_file_missing(self, tmp_path: Path) -> None:
        bullet = _make_bullet(
            anchors=[_make_anchor("never/existed.rs", "fn x()")]
        )
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.UNVERIFIABLE
        assert out.trust_score == 0.7

    def test_path_outside_project_root_rejected(self, tmp_path: Path) -> None:
        # A `..` escape should resolve outside the project_root canonical
        # prefix and be rejected without I/O.
        outside = tmp_path.parent / "escape.rs"
        outside.write_text("irrelevant", encoding="utf-8")
        bullet = _make_bullet(
            anchors=[_make_anchor("../escape.rs", "irrelevant")]
        )
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.UNVERIFIABLE

    def test_read_error_returns_unverifiable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # File exists — but every attempt to open it raises OSError. The
        # engine must *not* propagate the exception; it must downgrade to
        # UNVERIFIABLE and stay usable for other bullets in the same batch.
        target = tmp_path / "guarded.rs"
        target.write_text("fn guarded() {}\n", encoding="utf-8")

        real_open = builtins.open

        def exploding_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            if os.fspath(path) == str(target):
                raise PermissionError("denied")
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", exploding_open)

        bullet = _make_bullet(
            anchors=[_make_anchor("guarded.rs", "fn guarded()")]
        )
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.UNVERIFIABLE


# ---------------------------------------------------------------------------
# 4. not_applicable
# ---------------------------------------------------------------------------


class TestNotApplicablePath:
    def test_no_anchors_returns_not_applicable_with_none_score(
        self, tmp_path: Path
    ) -> None:
        bullet = _make_bullet(anchors=[])
        out = _engine(tmp_path).verify(bullet)
        assert out.verified_status == VerifiedStatus.NOT_APPLICABLE
        assert out.trust_score is None


# ---------------------------------------------------------------------------
# 5. TTL hit => zero I/O
# ---------------------------------------------------------------------------


class TestTTLCache:
    def test_ttl_hit_performs_zero_file_reads(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_tree(tmp_path, {"a.rs": "fn a() {}\n"})

        # A TTL-hit bullet: verified_status is set AND verified_at is newer
        # than (now - ttl_seconds).
        bullet = _make_bullet(
            anchors=[_make_anchor("a.rs", "fn a()")],
            verified_at=_now() - timedelta(seconds=5),
            verified_status=VerifiedStatus.VERIFIED.value,
            trust_score=1.0,
        )

        call_count = {"n": 0}
        real_open = builtins.open

        def counting_open(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            return real_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting_open)

        out = _engine(tmp_path, ttl_seconds=60).verify(bullet)
        assert out.verified_status == VerifiedStatus.VERIFIED
        assert out.trust_score == 1.0
        assert call_count["n"] == 0  # zero-IO on TTL hit

    def test_ttl_expired_triggers_reread(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_tree(tmp_path, {"a.rs": "fn a() {}\n"})
        bullet = _make_bullet(
            anchors=[_make_anchor("a.rs", "fn a()")],
            # verified_at is well beyond the TTL window — stale cache.
            verified_at=_now() - timedelta(seconds=999),
            verified_status=VerifiedStatus.VERIFIED.value,
            trust_score=1.0,
        )

        call_count = {"n": 0}
        real_open = builtins.open

        def counting_open(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            return real_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting_open)

        _engine(tmp_path, ttl_seconds=10).verify(bullet)
        assert call_count["n"] >= 1  # re-read happened

    def test_zero_ttl_disables_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_tree(tmp_path, {"a.rs": "fn a() {}\n"})
        bullet = _make_bullet(
            anchors=[_make_anchor("a.rs", "fn a()")],
            verified_at=_now(),
            verified_status=VerifiedStatus.VERIFIED.value,
            trust_score=1.0,
        )

        call_count = {"n": 0}
        real_open = builtins.open

        def counting_open(*args, **kwargs):  # type: ignore[no-untyped-def]
            call_count["n"] += 1
            return real_open(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", counting_open)

        _engine(tmp_path, ttl_seconds=0).verify(bullet)
        assert call_count["n"] >= 1


# ---------------------------------------------------------------------------
# 6. Batch — same file read exactly once
# ---------------------------------------------------------------------------


class TestBatchSameFileOneRead:
    def test_same_file_read_only_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_tree(
            tmp_path,
            {"shared.rs": "fn alpha() {}\nfn beta() {}\nfn gamma() {}\n"},
        )

        # Three bullets, each pointing at a DIFFERENT anchor_text but inside
        # the SAME file. The engine must open the file once for the whole batch.
        bullets = [
            _make_bullet(anchors=[_make_anchor("shared.rs", "fn alpha()")]),
            _make_bullet(anchors=[_make_anchor("shared.rs", "fn beta()")]),
            _make_bullet(anchors=[_make_anchor("shared.rs", "fn gamma()")]),
        ]

        opened_paths: list[str] = []
        real_open = builtins.open

        def recording_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Only record reads of our tracked file to stay robust against
            # interpreter-internal opens from pytest/monkeypatch.
            if str(path).endswith("shared.rs"):
                opened_paths.append(str(path))
            return real_open(path, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", recording_open)

        outcomes = _engine(tmp_path).verify_many(bullets)
        assert [o.verified_status for o in outcomes] == [
            VerifiedStatus.VERIFIED,
            VerifiedStatus.VERIFIED,
            VerifiedStatus.VERIFIED,
        ]
        assert len(opened_paths) == 1


# ---------------------------------------------------------------------------
# 7. project_root resolution
# ---------------------------------------------------------------------------


class TestProjectRootResolution:
    def test_config_project_root_takes_priority(self, tmp_path: Path) -> None:
        cfg = VerificationConfig(project_root=str(tmp_path))
        engine = VerificationEngine(cfg)
        root = engine._resolve_project_root()
        assert root == tmp_path.resolve()

    def test_cwd_walkup_finds_git_marker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Simulate a workspace: tmp_path contains .git; cwd is a subdir.
        (tmp_path / ".git").mkdir()
        deeper = tmp_path / "crate" / "src"
        deeper.mkdir(parents=True)
        monkeypatch.chdir(deeper)

        engine = VerificationEngine(VerificationConfig(project_root=None))
        root = engine._resolve_project_root()
        assert root == tmp_path.resolve()


# ---------------------------------------------------------------------------
# 8. Cross-language fixture parity
# ---------------------------------------------------------------------------


class TestCrossLanguageFixture:
    """Verifies the Python engine matches the expected-outcome JSON produced
    by/for the Rust FileSystemVerifier against the same file tree and bullets.
    """

    def test_fixture_matches_expected_outcomes(self, tmp_path: Path) -> None:
        data = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
        _write_tree(tmp_path, data["file_tree"])

        cfg_raw = data["config"]
        cfg = VerificationConfig(
            enabled=cfg_raw["enabled"],
            ttl_seconds=cfg_raw["ttl_seconds"],
            project_root=str(tmp_path),
            verified_trust_score=cfg_raw["verified_trust_score"],
            stale_trust_score=cfg_raw["stale_trust_score"],
            unverifiable_trust_score=cfg_raw["unverifiable_trust_score"],
            policy=cfg_raw["policy"],
        )

        bullets: list[BulletMetadata] = []
        for b in data["bullets"]:
            anchors = [
                Anchor(
                    file_path=a["file_path"],
                    anchor_text=a["anchor_text"],
                    anchor_hash=hashlib.sha256(
                        a["anchor_hash_source"].encode("utf-8")
                    ).hexdigest(),
                    created_at=datetime(
                        2026, 4, 24, tzinfo=timezone.utc
                    ),
                )
                for a in b["anchors"]
            ]
            bullets.append(BulletMetadata(anchors=anchors))

        engine = VerificationEngine(cfg)
        outcomes = engine.verify_many(bullets)

        expected = data["expected"]
        assert len(outcomes) == len(expected)
        for i, (out, exp) in enumerate(zip(outcomes, expected)):
            assert out.verified_status.value == exp["verified_status"], (
                f"bullet[{i}] status mismatch"
            )
            assert out.trust_score == exp["trust_score"], (
                f"bullet[{i}] trust_score mismatch"
            )

    def test_outcome_type_exported(self) -> None:
        # Sanity check: the public surface exposes both symbols.
        assert VerificationOutcome.__module__.endswith(".verifier.engine")
