"""STORY-R102 — RetrievalPipeline verifier stage integration tests.

Three scenarios per the Story DoD:
1. Verifier enabled, one anchor mutated on disk -> one result has
   ``verified_status="stale"``, others verified or not_applicable.
2. Verifier enabled, all anchors intact -> every result is "verified".
3. Verifier disabled (``VerificationEngine=None``) -> output shape is
   identical to pre-R102 state (no ``verified_status`` attribute written).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from memorus.core.config import VerificationConfig
from memorus.core.engines.generator.engine import GeneratorEngine
from memorus.core.engines.generator.score_merger import ScoredBullet
from memorus.core.engines.verifier.engine import VerificationEngine
from memorus.core.pipeline.retrieval import RetrievalPipeline
from memorus.core.types import Anchor, BulletMetadata
from memorus.core.utils.token_counter import TokenBudgetTrimmer


@pytest.fixture
def tmp_project_root(tmp_path: Path) -> Path:
    """Create a tiny fake project tree with three files for anchor lookups."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "auth.py").write_text(
        "def verify_token(tok):\n    return tok == 'ok'\n", encoding="utf-8"
    )
    (root / "db.py").write_text(
        "def connect(url):\n    return url\n", encoding="utf-8"
    )
    (root / "util.py").write_text(
        "def slugify(s):\n    return s.replace(' ', '-')\n", encoding="utf-8"
    )
    return root


def _anchor(root: Path, file_name: str, text: str) -> Anchor:
    return Anchor(
        file_path=file_name,
        anchor_text=text,
        anchor_hash=Anchor.compute_hash(text),
        created_at=datetime.now(timezone.utc),
    )


def _bullet_with_anchor(bid: str, anchor: Anchor) -> BulletMetadata:
    # BulletMetadata is a Pydantic model without a ``content`` field —
    # verifier only needs anchors / verified_at / verified_status to run,
    # bullet_id alignment is done by caller via the loader mapping.
    return BulletMetadata(anchors=[anchor])


def _scored(bid: str) -> ScoredBullet:
    return ScoredBullet(
        bullet_id=bid,
        content=f"bullet {bid}",
        final_score=0.9,
        keyword_score=5.0,
        semantic_score=0.8,
        decay_weight=1.0,
        recency_boost=1.0,
    )


def _build_pipeline(
    scored: list[ScoredBullet],
    verifier: VerificationEngine | None,
    loader,
    update_fn=None,
) -> RetrievalPipeline:
    """Build a pipeline with a mock generator that returns ``scored``."""
    generator = MagicMock(spec=GeneratorEngine)
    generator.search.return_value = scored
    generator.mode = "full"
    trimmer = MagicMock(spec=TokenBudgetTrimmer)
    trimmer.trim.side_effect = lambda x: x  # pass-through
    trimmer.token_budget = 1000

    return RetrievalPipeline(
        generator=generator,
        trimmer=trimmer,
        decay_engine=None,
        verification_engine=verifier,
        bullet_loader=loader,
        update_fn=update_fn,
    )


# ---------------------------------------------------------------------------
# AC1: Verifier ON, one file mutated -> one stale
# ---------------------------------------------------------------------------


def test_verifier_flags_single_stale_bullet(tmp_project_root: Path) -> None:
    # Anchors reference live text in each file.
    a1 = _anchor(tmp_project_root, "auth.py", "def verify_token(tok):")
    a2 = _anchor(tmp_project_root, "db.py", "def connect(url):")
    a3 = _anchor(tmp_project_root, "util.py", "def slugify(s):")

    bid1 = "11111111-1111-1111-1111-111111111111"
    bid2 = "22222222-2222-2222-2222-222222222222"
    bid3 = "33333333-3333-3333-3333-333333333333"

    b1 = _bullet_with_anchor(bid1, a1)
    b2 = _bullet_with_anchor(bid2, a2)
    b3 = _bullet_with_anchor(bid3, a3)

    # Mutate auth.py so b1 becomes stale.
    (tmp_project_root / "auth.py").write_text(
        "def check_token(tok):\n    return tok == 'ok'\n", encoding="utf-8"
    )

    verifier = VerificationEngine(
        VerificationConfig(
            enabled=True,
            ttl_seconds=0,  # disable TTL so we always re-check
            project_root=str(tmp_project_root),
        )
    )

    loaded = {bid1: b1, bid2: b2, bid3: b3}

    def loader(ids: list[str]) -> list[BulletMetadata | None]:
        return [loaded.get(i) for i in ids]

    scored = [_scored(bid1), _scored(bid2), _scored(bid3)]
    pipeline = _build_pipeline(scored, verifier, loader)

    result = pipeline.search(
        query="anything", bullets=[], limit=10
    )

    assert len(result.results) == 3
    by_id = {r.bullet_id: r for r in result.results}
    assert by_id[bid1].verified_status == "stale"
    assert by_id[bid1].trust_score == pytest.approx(0.3)
    assert by_id[bid2].verified_status == "verified"
    assert by_id[bid2].trust_score == pytest.approx(1.0)
    assert by_id[bid3].verified_status == "verified"


# ---------------------------------------------------------------------------
# AC2: Verifier ON, all anchors intact -> all verified
# ---------------------------------------------------------------------------


def test_verifier_all_verified(tmp_project_root: Path) -> None:
    a1 = _anchor(tmp_project_root, "auth.py", "def verify_token(tok):")
    a2 = _anchor(tmp_project_root, "db.py", "def connect(url):")

    bid1 = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    bid2 = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

    b1 = _bullet_with_anchor(bid1, a1)
    b2 = _bullet_with_anchor(bid2, a2)

    verifier = VerificationEngine(
        VerificationConfig(
            enabled=True,
            ttl_seconds=0,
            project_root=str(tmp_project_root),
        )
    )

    loaded = {bid1: b1, bid2: b2}

    def loader(ids: list[str]) -> list[BulletMetadata | None]:
        return [loaded.get(i) for i in ids]

    # Write-back should be invoked once per bullet.
    writes: list[tuple[str, dict]] = []

    def update_fn(bid: str, patch: dict) -> None:
        writes.append((bid, dict(patch)))

    scored = [_scored(bid1), _scored(bid2)]
    pipeline = _build_pipeline(scored, verifier, loader, update_fn=update_fn)

    result = pipeline.search(query="q", bullets=[], limit=10)

    assert len(result.results) == 2
    for r in result.results:
        assert r.verified_status == "verified"
        assert r.trust_score == pytest.approx(1.0)

    # verified_at + verified_status persisted for every bullet.
    persisted_ids = {bid for bid, _ in writes}
    assert persisted_ids == {bid1, bid2}
    for _, patch in writes:
        assert "memorus_verified_at" in patch
        assert patch["memorus_verified_status"] == "verified"


# ---------------------------------------------------------------------------
# AC3: Verifier OFF -> zero-cost, fields stay None
# ---------------------------------------------------------------------------


def test_verifier_disabled_no_fields_written() -> None:
    bid1 = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    bid2 = "dddddddd-dddd-dddd-dddd-dddddddddddd"

    # Loader must NOT be consulted when verifier is None.
    loader_calls: list[Any] = []

    def loader(ids: list[str]) -> list[BulletMetadata | None]:
        loader_calls.append(ids)
        return [None] * len(ids)

    scored = [_scored(bid1), _scored(bid2)]
    pipeline = _build_pipeline(scored, verifier=None, loader=loader)

    result = pipeline.search(query="q", bullets=[], limit=10)

    assert len(result.results) == 2
    for r in result.results:
        assert r.verified_status is None
        assert r.trust_score is None
    assert loader_calls == []  # zero-cost: loader never called


# ---------------------------------------------------------------------------
# Safety: loader exception is logged + swallowed, search still returns
# ---------------------------------------------------------------------------


def test_verifier_loader_raises_is_swallowed(tmp_project_root: Path) -> None:
    verifier = VerificationEngine(
        VerificationConfig(
            enabled=True, ttl_seconds=0, project_root=str(tmp_project_root)
        )
    )

    def boom(ids: list[str]) -> list[BulletMetadata]:
        raise RuntimeError("loader dead")

    scored = [_scored("zz")]
    pipeline = _build_pipeline(scored, verifier, boom)

    result = pipeline.search(query="q", bullets=[], limit=10)

    # Search result still returned — fields stay None.
    assert len(result.results) == 1
    assert result.results[0].verified_status is None
    assert result.results[0].trust_score is None
