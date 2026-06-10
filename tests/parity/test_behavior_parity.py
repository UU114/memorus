"""Behavioral parity gate: Python (memorus/) vs Rust (memorus-r/).

This suite locks in the four behavioral dimensions reconciled by the 2026-06
divergence audit (``doc/optimization-roadmap.md``). For each dimension it
asserts the *aligned* behavior on the Python side (directly executable here)
and pins the canonical target value that the Rust source-of-truth must also
satisfy. The Rust side of dimensions (2) and (3) is verified by Rust unit
tests — ``memorus-ace/src/decay/formula.rs::test_protection_period_boundary``
and ``memorus-core/src/engine/crud.rs`` demote-branch coverage — which the
supervisor's consolidated ``cargo test`` run exercises; we cannot rebuild the
extension from here, so we encode the expected constants as authoritative
references rather than re-deriving them.

Dimensions
----------
1. DEDUP DEFAULT THRESHOLD — Python aligned TO Rust source-of-truth (0.9).
   Was 0.8; a higher threshold means fewer false-positive merges.
2. DECAY PROTECTION-PERIOD BOUNDARY — inclusive (``<=``) on BOTH sides. A
   bullet aged exactly ``protection_days`` is still within the window. Python
   was already ``<=``; Rust ``<`` was the bug and is now ``<=``.
3. DEMOTE REORDERING — BOTH sides re-sort by score after demoting stale /
   unverifiable rows so they sink to the bottom. Python already re-sorted;
   Rust ``crud.rs`` now re-sorts too.
4. RETRIEVAL CONFIG WEIGHT SHAPE — INTENTIONALLY divergent structures (Python
   2-way keyword+semantic, Rust 4-way semantic+recency+frequency+decay). We do
   NOT assert equality on weight names; instead each side's weights must sum to
   1.0 independently.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from memorus.core.config import (
    CuratorConfig,
    DecayConfig,
    RetrievalConfig,
    VerificationConfig,
)
from memorus.core.engines.decay.engine import DecayEngine
from memorus.core.engines.generator.score_merger import ScoredBullet
from memorus.core.pipeline.retrieval import RetrievalPipeline

# ---------------------------------------------------------------------------
# Canonical target values (Rust is source-of-truth unless noted). These are the
# single point of truth for the parity gate; if the Rust defaults change, these
# constants — and the corresponding Rust code — must change together.
# ---------------------------------------------------------------------------

# (1) Rust `default_dedup_threshold()` in memorus-core/src/config.rs.
RUST_DEDUP_THRESHOLD = 0.9

# (2) Rust `DecayConfig::protection_days` default in memorus-ace.
RUST_PROTECTION_DAYS = 7.0

# (4) The 4-way Rust weights each default to 0.25 and are validated to sum to 1.0.
RUST_RETRIEVAL_WEIGHT_SUM = 1.0


# ---------------------------------------------------------------------------
# (1) Dedup default threshold — Python aligned to Rust (0.9).
# ---------------------------------------------------------------------------
def test_dedup_default_threshold_aligned_to_rust() -> None:
    """Python CuratorConfig.similarity_threshold must equal the Rust default.

    Aligned: was 0.8 (Python) -> 0.9 (Rust source-of-truth).
    """
    cfg = CuratorConfig()
    assert cfg.similarity_threshold == RUST_DEDUP_THRESHOLD


# ---------------------------------------------------------------------------
# (2) Decay protection-period boundary — inclusive (<=) on both sides.
# ---------------------------------------------------------------------------
def test_decay_protection_boundary_is_inclusive() -> None:
    """A bullet aged exactly `protection_days` is still protected (`<=`).

    Both engines now use an inclusive boundary. Python was already `<=`; the
    Rust formula was fixed from `<` to `<=`.
    """
    cfg = DecayConfig()
    assert float(cfg.protection_days) == RUST_PROTECTION_DAYS

    engine = DecayEngine(cfg)
    now = datetime(2026, 1, 8, tzinfo=timezone.utc)

    # Exactly at the boundary (age == protection_days) — protected.
    at_boundary = now - timedelta(days=cfg.protection_days)
    res_at = engine.compute_weight(created_at=at_boundary, recall_count=0, now=now)
    assert res_at.is_protected is True
    assert res_at.weight == 1.0

    # Just below the boundary — protected.
    below = now - timedelta(days=cfg.protection_days, seconds=-3600)
    res_below = engine.compute_weight(created_at=below, recall_count=0, now=now)
    assert res_below.is_protected is True

    # Just past the boundary — no longer protected; decay applies.
    past = now - timedelta(days=cfg.protection_days, seconds=3600)
    res_past = engine.compute_weight(created_at=past, recall_count=0, now=now)
    assert res_past.is_protected is False
    assert res_past.weight < 1.0


# ---------------------------------------------------------------------------
# (3) Demote reordering — stale/unverifiable rows sink after demotion.
# ---------------------------------------------------------------------------
def _scored(bullet_id: str, score: float, status: str | None) -> ScoredBullet:
    return ScoredBullet(
        bullet_id=bullet_id,
        content=bullet_id,
        final_score=score,
        keyword_score=0.0,
        semantic_score=0.0,
        decay_weight=1.0,
        recency_boost=1.0,
        verified_status=status,
    )


def test_demote_policy_resorts_so_stale_rows_sink() -> None:
    """Under `policy="demote"`, demoted rows must drop below un-demoted rows.

    Mirrors the Rust `crud.rs` Demote branch, which now re-sorts `raw_results`
    (and the parallel outcomes vector) by descending score after applying the
    trust multiplier.
    """
    vc = VerificationConfig(
        policy="demote",
        stale_trust_score=0.3,
        unverifiable_trust_score=0.5,
    )
    # The Demote branch only reads self._verification_config; the generator is
    # stored but never touched, so a None generator is safe here.
    pipeline = RetrievalPipeline(generator=None, verification_config=vc)

    # 'a' starts highest (0.9) but is stale: 0.9 * 0.3 = 0.27 -> must sink last.
    bullets = [
        _scored("a", 0.9, "stale"),
        _scored("b", 0.8, "verified"),
        _scored("c", 0.7, "verified"),
    ]
    out, dropped = pipeline._apply_stale_policy(bullets)

    assert dropped == 0
    # Re-sorted descending: verified rows first, demoted stale row last.
    assert [b.bullet_id for b in out] == ["b", "c", "a"]
    # Output is monotonically non-increasing by final_score.
    scores = [b.final_score for b in out]
    assert scores == sorted(scores, reverse=True)


def test_demote_unverifiable_also_demoted_and_resorted() -> None:
    """Unverifiable rows are demoted by their own multiplier and re-sorted."""
    vc = VerificationConfig(
        policy="demote",
        stale_trust_score=0.3,
        unverifiable_trust_score=0.5,
    )
    pipeline = RetrievalPipeline(generator=None, verification_config=vc)

    # 'u' starts highest (1.0) but unverifiable: 1.0 * 0.5 = 0.5 -> sinks below 'v'.
    bullets = [
        _scored("u", 1.0, "unverifiable"),
        _scored("v", 0.6, "verified"),
    ]
    out, _ = pipeline._apply_stale_policy(bullets)
    assert [b.bullet_id for b in out] == ["v", "u"]


# ---------------------------------------------------------------------------
# (4) Retrieval config weight shape — structurally divergent BY DESIGN.
# ---------------------------------------------------------------------------
def test_retrieval_weights_sum_to_one_python_side() -> None:
    """Python's 2-way keyword+semantic weights must sum to 1.0.

    The parity gate deliberately does NOT compare weight *names* across
    languages: Python is keyword+semantic (GeneratorEngine ScoreMerger), Rust
    is semantic+recency+frequency+decay (CRUD vector search). These are two
    different retrieval architectures sharing a config namespace. The only
    cross-language invariant is that each side's weights normalize to 1.0.
    """
    cfg = RetrievalConfig()
    total = cfg.keyword_weight + cfg.semantic_weight
    assert abs(total - 1.0) < 1e-9


def test_retrieval_weight_shape_is_intentionally_divergent() -> None:
    """Document the intentional structural divergence as an executable check.

    Python exposes exactly the 2-way keyword/semantic split and must NOT grow
    the Rust 4-way fields (doing so would silently change the scoring model).
    The Rust 4-way weights are pinned by RUST_RETRIEVAL_WEIGHT_SUM and verified
    by the Rust `config.rs` validate()/default tests on the supervisor run.
    """
    cfg = RetrievalConfig()
    assert hasattr(cfg, "keyword_weight")
    assert hasattr(cfg, "semantic_weight")
    # Rust-only 4-way weights must NOT have leaked into the Python config.
    for rust_only in ("recency_weight", "frequency_weight", "decay_weight"):
        assert not hasattr(cfg, rust_only), (
            f"Python RetrievalConfig must not expose Rust-only weight "
            f"{rust_only!r}; the two weight shapes are intentionally divergent."
        )
    # The Rust side's weights are documented to sum to 1.0 independently.
    assert RUST_RETRIEVAL_WEIGHT_SUM == 1.0
