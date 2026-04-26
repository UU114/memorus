"""Unit tests for `memorus verify` and `memorus conflicts --type` (STORY-R105)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from memorus.core.cli.main import cli
from memorus.core.engines.curator.conflict import Conflict, ConflictType
from memorus.core.engines.verifier.engine import VerificationOutcome
from memorus.core.types import Anchor, VerifiedStatus


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _patch_create_memory(mock_memory: MagicMock) -> Any:
    return patch("memorus.core.cli.main._create_memory", return_value=mock_memory)


def _make_memory(records: list[dict[str, Any]]) -> MagicMock:
    """Build a mocked Memory whose ``get_all`` returns *records*.

    The mock also exposes ``_config`` (real :class:`MemorusConfig`) and a
    spy ``_build_metadata_update_fn`` so the verify command can call it.
    """
    from memorus.core.config import MemorusConfig

    m = MagicMock()
    m.get_all.return_value = {"memories": records}
    m._config = MemorusConfig()

    # Track update_fn invocations so tests can assert on writes.
    update_calls: list[tuple[str, dict[str, Any]]] = []

    def _fake_update(bullet_id: str, patch: dict[str, Any]) -> None:
        update_calls.append((bullet_id, patch))

    m._update_calls = update_calls
    m._build_metadata_update_fn.return_value = _fake_update
    return m


def _bullet_record(
    bullet_id: str,
    *,
    content: str = "stub",
    scope: str = "global",
    anchors: list[dict[str, Any]] | None = None,
    verified_status: str | None = None,
) -> dict[str, Any]:
    """Build a mem0 payload dict matching ``BulletFactory.from_mem0_payload``."""
    metadata: dict[str, Any] = {"memorus_scope": scope}
    if anchors is not None:
        # BulletMetadata.anchors round-trips through model_dump(mode='json')
        # — list of dicts is the canonical shape stored in mem0 payload.
        metadata["memorus_anchors"] = anchors
    if verified_status is not None:
        metadata["memorus_verified_status"] = verified_status
    return {"id": bullet_id, "memory": content, "metadata": metadata}


# ---------------------------------------------------------------------------
# verify --dry-run: no DB writes
# ---------------------------------------------------------------------------


class TestVerifyDryRun:
    def test_dry_run_writes_nothing(
        self, runner: CliRunner, tmp_path: Any
    ) -> None:
        """`verify --dry-run` must not call update_fn at all."""
        # Provide a real anchor + matching file so the bullet would be
        # VERIFIED on a real run; that ensures we'd issue a write if the
        # dry-run guard were broken.
        target = tmp_path / "src" / "alpha.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "def alpha():\n    return 'kept'\n",
            encoding="utf-8",
        )
        anchor_text = "def alpha():\n    return 'kept'"
        anchor_dict = {
            "file_path": "src/alpha.py",
            "anchor_text": anchor_text,
            "anchor_hash": Anchor.compute_hash(anchor_text),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        mock_memory = _make_memory(
            [_bullet_record("b1", anchors=[anchor_dict])]
        )
        # Pin project_root to the tempdir so the verifier resolves anchors.
        from memorus.core.config import MemorusConfig

        mock_memory._config = MemorusConfig(
            verification={"project_root": str(tmp_path)}
        )

        with _patch_create_memory(mock_memory):
            result = runner.invoke(
                cli, ["verify", "--dry-run", "--json"]
            )
        assert result.exit_code == 0, result.output
        # No writes — neither the update-fn factory call nor any real patch.
        assert mock_memory._update_calls == []
        # Engine still produced a verified outcome in the report.
        report = json.loads(result.output)
        assert report["verified"] == 1
        assert report["anchors_added"] == 0


# ---------------------------------------------------------------------------
# verify --rehydrate-anchors: anchor backfill
# ---------------------------------------------------------------------------


class TestVerifyRehydrateAnchors:
    def test_rehydrate_adds_anchors_on_anchorable_bullets_only(
        self, runner: CliRunner, tmp_path: Any
    ) -> None:
        """Three-bullet fixture: 1 anchorable, 2 not. Counter == 1."""
        # Anchorable: file exists and the bullet content references it.
        target = tmp_path / "src" / "service.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        body = "def serve():\n    return 'ok'\n"
        target.write_text(body, encoding="utf-8")
        anchorable_content = (
            "Use src/service.py — see this code:\n"
            "```python\n"
            f"{body}"
            "```"
        )

        records = [
            # 1) Anchorable, currently empty anchors → should add anchor.
            _bullet_record("anchorable", content=anchorable_content, anchors=[]),
            # 2) No path mention — extraction yields nothing.
            _bullet_record(
                "preference",
                content="Prefer kebab-case for branch names.",
                anchors=[],
            ),
            # 3) Path-like text but file does not exist on disk → nothing.
            _bullet_record(
                "ghost",
                content="See ghost/missing_file.py for details.",
                anchors=[],
            ),
        ]
        mock_memory = _make_memory(records)
        from memorus.core.config import MemorusConfig

        mock_memory._config = MemorusConfig(
            verification={"project_root": str(tmp_path)}
        )

        with _patch_create_memory(mock_memory):
            result = runner.invoke(
                cli, ["verify", "--rehydrate-anchors", "--json"]
            )
        assert result.exit_code == 0, result.output
        report = json.loads(result.output)
        assert report["anchors_added"] == 1, report
        # Exactly one anchor write went out (per anchorable bullet) plus
        # three status writes (one per bullet).
        anchor_writes = [
            (bid, p) for bid, p in mock_memory._update_calls
            if "memorus_anchors" in p
        ]
        assert len(anchor_writes) == 1
        assert anchor_writes[0][0] == "anchorable"


# ---------------------------------------------------------------------------
# verify --stale-only: skip already-verified bullets
# ---------------------------------------------------------------------------


class TestVerifyStaleOnly:
    def test_stale_only_skips_verified_bullets(
        self, runner: CliRunner, tmp_path: Any
    ) -> None:
        """`--stale-only` must skip rows whose verified_status is 'verified'."""
        # Build one verified-status bullet + one stale + one null.
        records = [
            _bullet_record("v", verified_status="verified", anchors=[]),
            _bullet_record("s", verified_status="stale", anchors=[]),
            _bullet_record("u", verified_status=None, anchors=[]),
        ]
        mock_memory = _make_memory(records)

        # Patch VerificationEngine.verify so we can count how many times it
        # was invoked. With --stale-only we expect 2 calls (s + u), not 3.
        with _patch_create_memory(mock_memory), patch(
            "memorus.core.engines.verifier.engine.VerificationEngine.verify"
        ) as verify_method:
            verify_method.return_value = VerificationOutcome(
                verified_status=VerifiedStatus.NOT_APPLICABLE,
                trust_score=None,
                verified_at=datetime.now(timezone.utc),
            )

            result = runner.invoke(
                cli, ["verify", "--stale-only", "--json"]
            )
            assert result.exit_code == 0, result.output
            assert verify_method.call_count == 2


# ---------------------------------------------------------------------------
# conflicts --type filter
# ---------------------------------------------------------------------------


class TestConflictsTypeFilter:
    def test_type_filter_returns_only_matching(
        self, runner: CliRunner
    ) -> None:
        """`conflicts --type anchor_mismatch` filters to that type only."""
        sample = [
            Conflict(
                memory_a_id="aaaaaaaa-1111",
                memory_b_id="bbbbbbbb-2222",
                memory_a_content="alpha",
                memory_b_content="beta",
                similarity=0.7,
                reason="anchor_mismatch",
                conflict_type=ConflictType.ANCHOR_MISMATCH,
            ),
            Conflict(
                memory_a_id="cccccccc-3333",
                memory_b_id="dddddddd-4444",
                memory_a_content="x",
                memory_b_content="y",
                similarity=0.6,
                reason="opposing_pair",
                conflict_type=ConflictType.OPPOSING_PAIR,
            ),
        ]
        mock_memory = MagicMock()
        mock_memory.detect_conflicts.return_value = sample

        with _patch_create_memory(mock_memory):
            result = runner.invoke(
                cli, ["conflicts", "--type", "anchor_mismatch", "--json"]
            )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload) == 1
        assert payload[0]["reason"] == "anchor_mismatch"
        # Make sure omitting the flag still returns both.
        with _patch_create_memory(mock_memory):
            full = runner.invoke(cli, ["conflicts", "--json"])
        assert full.exit_code == 0
        assert len(json.loads(full.output)) == 2
