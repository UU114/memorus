"""Unit tests for STORY-R100 anchor extraction in BulletDistiller.

The Distiller is responsible for capturing region anchors at ingest time.
Each test sets up a fake project root with a real source file on disk so
``_extract_anchors`` can validate paths and read content. Anchor extraction
must be best-effort — these tests verify both happy paths and the edge
cases that should silently produce ``anchors=[]``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memorus.core.config import ReflectorConfig, VerificationConfig
from memorus.core.engines.reflector.distiller import BulletDistiller
from memorus.core.types import (
    Anchor,
    BulletSection,
    DetectedPattern,
    InteractionEvent,
    KnowledgeType,
    ScoredCandidate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a temp directory marked as a project root via a ``.git`` dir."""
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def distiller_factory(project_root: Path):
    """Factory that builds a Distiller bound to the given project_root."""

    def _make() -> BulletDistiller:
        verification = VerificationConfig(project_root=str(project_root))
        return BulletDistiller(ReflectorConfig(), verification=verification)

    return _make


def _write(project_root: Path, rel_path: str, content: str) -> Path:
    """Write *content* into ``project_root / rel_path``, creating dirs."""
    abs_path = project_root / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(content, encoding="utf-8")
    return abs_path


def _candidate(
    content: str,
    *,
    user_message: str = "",
    assistant_message: str = "",
    pattern_type: str = "error_fix",
) -> ScoredCandidate:
    """Build a ScoredCandidate carrying both turn text and bullet content."""
    event = InteractionEvent(
        user_message=user_message,
        assistant_message=assistant_message,
    )
    pattern = DetectedPattern(
        pattern_type=pattern_type,
        content=content,
        confidence=0.8,
        source_event=event,
    )
    return ScoredCandidate(
        pattern=pattern,
        section=BulletSection.GENERAL,
        knowledge_type=KnowledgeType.METHOD,
        instructivity_score=70.0,
    )


# ---------------------------------------------------------------------------
# Acceptance Criteria
# ---------------------------------------------------------------------------


class TestAnchorExtractionAC:
    """Mirror the ``Acceptance Criteria`` table in STORY-R100."""

    def test_code_block_with_path_produces_anchor(
        self, project_root: Path, distiller_factory
    ) -> None:
        """AC: source dialogue contains a code block + path → 1 anchor."""
        file_text = (
            "use std::net::TcpStream;\n\n"
            "fn connect(addr: &str) -> std::io::Result<TcpStream> {\n"
            "    TcpStream::connect(addr)\n"
            "}\n"
        )
        _write(project_root, "src/foo.rs", file_text)

        # Bullet text alone references a file path; the assistant turn
        # carries the code block. The distiller scans both.
        candidate = _candidate(
            content="connect() in src/foo.rs uses TcpStream",
            assistant_message=(
                "Look at src/foo.rs:\n"
                "```rust\nfn connect(addr: &str) -> std::io::Result<TcpStream> {\n"
                "    TcpStream::connect(addr)\n"
                "}\n```"
            ),
        )

        bullet = distiller_factory().distill(candidate)
        assert bullet.anchors, "expected at least 1 anchor"
        # All anchors must point to the in-project file.
        assert {a.file_path for a in bullet.anchors} == {"src/foo.rs"}
        for anchor in bullet.anchors:
            assert anchor.anchor_text in file_text, (
                "anchor_text must be a real slice of the file"
            )
            assert anchor.anchor_hash == Anchor.compute_hash(anchor.anchor_text)
            assert len(anchor.anchor_text) <= 200

    def test_path_to_nonexistent_file_skips_anchor(
        self, project_root: Path, distiller_factory
    ) -> None:
        """AC: path resolves to a missing file → no anchor produced."""
        candidate = _candidate(
            content="See ghost/missing.rs for details",
            assistant_message="```rust\nfn ghost() {}\n```",
        )
        bullet = distiller_factory().distill(candidate)
        assert bullet.anchors == []

    def test_path_outside_project_root_skips_anchor(
        self, project_root: Path, tmp_path: Path, distiller_factory
    ) -> None:
        """AC: path lives outside project_root → no anchor."""
        outside = tmp_path / "other_project"
        outside.mkdir()
        (outside / "leak.py").write_text("def leak(): pass\n", encoding="utf-8")

        # Reference the absolute path that lies outside project_root
        leak_abs = (outside / "leak.py").as_posix()
        candidate = _candidate(
            content=f"check {leak_abs}",
            assistant_message=f"```python\ndef leak(): pass\n```\nfile: {leak_abs}",
        )
        bullet = distiller_factory().distill(candidate)
        assert bullet.anchors == []

    def test_multi_anchor_per_bullet(
        self, project_root: Path, distiller_factory
    ) -> None:
        """AC: a single bullet can carry multiple anchors across files."""
        _write(
            project_root,
            "src/alpha.py",
            "def alpha(x):\n    return x + 1\n",
        )
        _write(
            project_root,
            "src/beta.py",
            "def beta(y):\n    return y * 2\n",
        )

        candidate = _candidate(
            content=(
                "alpha() in src/alpha.py and beta() in src/beta.py "
                "form the pipeline"
            ),
            assistant_message=(
                "```python\ndef alpha(x):\n    return x + 1\n```\n"
                "and:\n"
                "```python\ndef beta(y):\n    return y * 2\n```"
            ),
        )

        bullet = distiller_factory().distill(candidate)
        paths = {a.file_path for a in bullet.anchors}
        assert "src/alpha.py" in paths
        assert "src/beta.py" in paths

    def test_empty_input_produces_empty_list_not_none(
        self, distiller_factory
    ) -> None:
        """AC: with nothing extractable, ``anchors`` is ``[]`` (not None)."""
        candidate = _candidate(content="hello world, no code here")
        bullet = distiller_factory().distill(candidate)
        assert bullet.anchors == []
        assert isinstance(bullet.anchors, list)

    def test_dedupe_per_bullet(
        self, project_root: Path, distiller_factory
    ) -> None:
        """AC: duplicates by ``(file_path, anchor_hash)`` are merged."""
        _write(
            project_root,
            "src/dup.py",
            "def shared():\n    return 'hi'\n",
        )
        # Mention the same file + symbol twice in the assistant turn so the
        # extractor would otherwise emit two identical anchors.
        candidate = _candidate(
            content="shared() in src/dup.py is the helper",
            assistant_message=(
                "src/dup.py shows:\n"
                "```python\ndef shared():\n    return 'hi'\n```\n"
                "Again in src/dup.py:\n"
                "```python\ndef shared():\n    return 'hi'\n```"
            ),
        )
        bullet = distiller_factory().distill(candidate)
        # Hash sets must be unique
        keys = {(a.file_path, a.anchor_hash) for a in bullet.anchors}
        assert len(keys) == len(bullet.anchors)


# ---------------------------------------------------------------------------
# Cross-cutting / Edge cases
# ---------------------------------------------------------------------------


class TestAnchorExtractionEdge:
    def test_best_effort_on_unreadable_file(
        self, project_root: Path, distiller_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File reads that raise OSError must NOT bubble out of distill()."""
        _write(project_root, "src/bad.py", "def bad():\n    pass\n")

        original = Path.open

        def boom(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            if self.name == "bad.py":
                raise OSError("simulated read failure")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", boom)

        candidate = _candidate(
            content="bad() in src/bad.py crashes",
            assistant_message="```python\ndef bad():\n    pass\n```",
        )
        bullet = distiller_factory().distill(candidate)  # must not raise
        assert bullet.anchors == []

    def test_oversized_file_skipped(
        self, project_root: Path, distiller_factory, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files larger than 5 MB are not read at all."""
        from memorus.core.engines.reflector import distiller as dmod

        monkeypatch.setattr(dmod, "_ANCHOR_FILE_MAX_BYTES", 50)
        # Write a file larger than the patched cap
        _write(project_root, "src/big.py", "x = 1\n" * 100)
        candidate = _candidate(
            content="see src/big.py",
            assistant_message="```python\nx = 1\n```",
        )
        bullet = distiller_factory().distill(candidate)
        assert bullet.anchors == []

    def test_binary_file_skipped(
        self, project_root: Path, distiller_factory
    ) -> None:
        """Files containing NUL bytes are treated as binary and skipped."""
        target = project_root / "src" / "blob.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"prefix\x00suffix")
        candidate = _candidate(
            content="blob in src/blob.json",
            assistant_message="```\nbinary stuff\n```",
        )
        bullet = distiller_factory().distill(candidate)
        assert bullet.anchors == []

    def test_anchor_text_capped_at_200(
        self, project_root: Path, distiller_factory
    ) -> None:
        """``anchor_text`` must never exceed the 200-char Pydantic cap."""
        long_body = "// tail\n" + ("x" * 600) + "\n"
        _write(project_root, "src/long.rs", "fn long() {\n" + long_body + "}\n")
        candidate = _candidate(
            content="fn long in src/long.rs",
            assistant_message=(
                "```rust\nfn long() {\n" + long_body + "}\n```"
            ),
        )
        bullet = distiller_factory().distill(candidate)
        for anchor in bullet.anchors:
            assert len(anchor.anchor_text) <= 200

    def test_path_normalized_forward_slash(
        self, project_root: Path, distiller_factory
    ) -> None:
        """``file_path`` is always forward-slash, regardless of OS separator."""
        _write(
            project_root,
            "src/nested/path.py",
            "def nested():\n    return 1\n",
        )
        # Author the reference using backslashes (Windows-style)
        candidate = _candidate(
            content=r"check src\nested\path.py for nested()",
            assistant_message=(
                "```python\ndef nested():\n    return 1\n```"
            ),
        )
        bullet = distiller_factory().distill(candidate)
        for anchor in bullet.anchors:
            assert "/" in anchor.file_path
            assert "\\" not in anchor.file_path


# ---------------------------------------------------------------------------
# Cross-language fixture parity
# ---------------------------------------------------------------------------


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures" / "anchor_extract_fixture.json"
)


class TestAnchorCrossLanguageFixture:
    """Both Python and Rust extractors must agree on the anchor file_path set.

    The fixture defines:
    * ``files``: relative_path → file contents (written to a fresh tmp root)
    * ``input``: the assistant turn / bullet content fed to the extractor
    * ``expected_file_paths``: the set of anchor.file_path values both sides
      must produce
    """

    def test_fixture_exists(self) -> None:
        assert FIXTURE_PATH.is_file(), (
            f"missing cross-language fixture {FIXTURE_PATH}"
        )

    def test_python_matches_expected_file_paths(
        self, tmp_path: Path
    ) -> None:
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        # Stand up the project root from the fixture
        (tmp_path / ".git").mkdir()
        for rel, body in data["files"].items():
            target = tmp_path / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")

        verification = VerificationConfig(project_root=str(tmp_path))
        distiller = BulletDistiller(ReflectorConfig(), verification=verification)

        candidate = _candidate(
            content=data["input"]["bullet_content"],
            assistant_message=data["input"]["assistant_message"],
        )
        bullet = distiller.distill(candidate)
        produced = {a.file_path for a in bullet.anchors}
        assert produced == set(data["expected_file_paths"]), (
            f"expected={data['expected_file_paths']}, got={sorted(produced)}"
        )
