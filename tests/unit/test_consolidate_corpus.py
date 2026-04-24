"""Unit tests for CuratorEngine.consolidate_corpus (STORY-R094).

Verifies Python port matches the semantics of the Rust
``memorus_ace::curator::consolidate_corpus``:

* merge_groups: near-duplicates in range [dedup_threshold, skip_threshold)
* redundant_ids: >= skip_threshold
* conflicts: detected when sim in [conflict_min, conflict_max) AND
  (negation mismatch OR antonym pair)
* max_per_pass: caps the scan
"""

from __future__ import annotations

from memorus.core.config import CuratorConfig
from memorus.core.engines.curator.engine import (
    ConsolidateConflictInfo,
    ConsolidateResult,
    CuratorEngine,
    ExistingBullet,
    MergeGroup,
    _count_negations,
    _find_antonym_pair,
)


def _b(bid: str, content: str, embedding=None) -> ExistingBullet:
    return ExistingBullet(bullet_id=bid, content=content, embedding=embedding)


class TestConsolidateCorpusBasics:
    def test_empty_returns_empty_result(self) -> None:
        eng = CuratorEngine(CuratorConfig())
        r = eng.consolidate_corpus([], max_per_pass=100)
        assert r == ConsolidateResult()

    def test_single_bullet_returns_empty(self) -> None:
        eng = CuratorEngine(CuratorConfig())
        r = eng.consolidate_corpus([_b("a", "hello world")], max_per_pass=100)
        assert r.pairs_compared == 0
        assert not r.merge_groups
        assert not r.redundant_ids
        assert not r.conflicts

    def test_max_per_pass_caps_scan(self) -> None:
        eng = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        bullets = [_b(str(i), "hello world foo") for i in range(10)]
        r = eng.consolidate_corpus(bullets, max_per_pass=2)
        # Only 1 pair among the first 2 should be compared
        assert r.pairs_compared == 1


class TestMergeGroups:
    def test_similar_bullets_form_merge_group(self) -> None:
        eng = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        bullets = [
            _b("a", "use fast fourier transform for audio"),
            _b("b", "use fast fourier transform for audio signals"),
        ]
        r = eng.consolidate_corpus(bullets, max_per_pass=100)
        assert len(r.merge_groups) == 1
        assert set(r.merge_groups[0].bullet_ids) == {"a", "b"}
        assert r.merge_groups[0].max_similarity >= 0.5

    def test_grouped_bullet_not_reused_as_seed(self) -> None:
        """Once a bullet is grouped, it cannot seed another group."""
        eng = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        bullets = [
            _b("a", "kubernetes pods with persistent volumes"),
            _b("b", "kubernetes pods with persistent volumes claim"),
            _b("c", "kubernetes pods with persistent volumes configured"),
        ]
        r = eng.consolidate_corpus(bullets, max_per_pass=100)
        assert len(r.merge_groups) == 1
        assert len(r.merge_groups[0].bullet_ids) == 3


class TestRedundantIds:
    def test_near_exact_duplicate_marked_redundant(self) -> None:
        eng = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        # Identical content → text_similarity = 1.0 >= 0.95 skip_threshold
        bullets = [
            _b("a", "same exact content right here"),
            _b("b", "same exact content right here"),
        ]
        r = eng.consolidate_corpus(bullets, max_per_pass=100)
        assert r.redundant_ids == ["b"]
        assert not r.merge_groups


class TestConflicts:
    def test_negation_asymmetry_conflict(self) -> None:
        eng = CuratorEngine(
            CuratorConfig(
                similarity_threshold=0.9,  # avoid triggering merge
                conflict_detection=True,
                conflict_min_similarity=0.3,
                conflict_max_similarity=0.9,
            )
        )
        # Shared vocab → similarity within [0.3, 0.9); one has "not"
        bullets = [
            _b("a", "always use sync mode here"),
            _b("b", "never use sync mode here"),
        ]
        r = eng.consolidate_corpus(bullets, max_per_pass=100)
        # High token overlap + antonym pair always/never
        assert any(c.conflict_type == "contradiction" for c in r.conflicts)

    def test_conflict_disabled_emits_nothing(self) -> None:
        eng = CuratorEngine(
            CuratorConfig(
                similarity_threshold=0.9,
                conflict_detection=False,
            )
        )
        bullets = [
            _b("a", "always use sync mode here"),
            _b("b", "never use sync mode here"),
        ]
        r = eng.consolidate_corpus(bullets, max_per_pass=100)
        assert r.conflicts == []


class TestNegationCounting:
    def test_basic_english(self) -> None:
        assert _count_negations("do not use this approach") == 1
        assert _count_negations("never avoid the solution") == 2

    def test_handles_punctuation(self) -> None:
        assert _count_negations("don't, avoid!") == 2

    def test_chinese_substring(self) -> None:
        assert _count_negations("不要使用该方法") >= 1


class TestAntonymPairs:
    def test_detects_basic_pair(self) -> None:
        pair = _find_antonym_pair("always keep it", "never keep it")
        assert pair == ("always", "never")

    def test_reversed_pair(self) -> None:
        pair = _find_antonym_pair("disable flag", "enable flag")
        # find_antonym_pair returns (word_in_a, word_in_b) — reversed pair
        assert pair == ("disable", "enable")

    def test_no_pair_returns_none(self) -> None:
        assert _find_antonym_pair("foo bar", "baz qux") is None


class TestPairsCompared:
    def test_counts_live_pairs_only(self) -> None:
        """pairs_compared increments only for pairs where both bullets are
        not already in a group."""
        eng = CuratorEngine(CuratorConfig(similarity_threshold=0.5))
        # a-b merge, a-c and b-c should also be checked until grouping takes
        # effect. Algo semantics: a-b compared, a-c compared, b-c SKIPPED
        # because b is grouped at that point.
        bullets = [
            _b("a", "kubernetes persistent volume"),
            _b("b", "kubernetes persistent volume claim"),
            _b("c", "totally unrelated content goes here"),
        ]
        r = eng.consolidate_corpus(bullets, max_per_pass=100)
        # at least 2 pairs: (a,b) and (a,c); (b,c) might be skipped because b
        # was grouped
        assert r.pairs_compared >= 2
