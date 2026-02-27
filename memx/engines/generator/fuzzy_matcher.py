"""L2 FuzzyMatcher — fuzzy matching layer for the Generator search engine.

Uses 2-gram tokenization for Chinese text and simplified suffix-stripping
stemmer for English.  Scores are computed as the ratio of matched grams to
total query grams, scaled to the range [0, max_score] (default 10).

Usage::

    matcher = FuzzyMatcher(max_score=10.0)
    result = matcher.match("数据库管理", "数据分析和数据库操作")
    # result.score > 0 because "数据" bigram is shared
"""

from __future__ import annotations

from memx.engines.generator.exact_matcher import MatchResult
from memx.utils.text_processing import extract_tokens


class FuzzyMatcher:
    """L2 fuzzy matcher based on token overlap ratio.

    Chinese text is split into 2-grams; English words are stemmed.  The score
    is ``min(matched_grams / total_query_grams * max_score, max_score)``.
    """

    def __init__(self, max_score: float = 10.0) -> None:
        self._max_score = max_score

    @property
    def max_score(self) -> float:
        """Maximum possible fuzzy match score."""
        return self._max_score

    # -- public API ---------------------------------------------------------

    def match(self, query: str, content: str) -> MatchResult:
        """Score *content* against *query* using fuzzy token overlap.

        Returns a :class:`MatchResult` with score in [0, max_score] and a list
        of matched tokens in ``matched_terms``.
        """
        query_tokens = extract_tokens(query)
        if not query_tokens:
            return MatchResult()

        content_tokens_set = set(extract_tokens(content))
        if not content_tokens_set:
            return MatchResult()

        matched: list[str] = []
        for token in query_tokens:
            if token in content_tokens_set:
                matched.append(token)

        hit_ratio = len(matched) / len(query_tokens)
        score = min(hit_ratio * self._max_score, self._max_score)

        return MatchResult(
            score=score,
            matched_terms=matched,
            details={
                "query_tokens": query_tokens,
                "hit_count": len(matched),
                "total_grams": len(query_tokens),
                "hit_ratio": hit_ratio,
            },
        )

    def match_batch(
        self,
        query: str,
        contents: list[str],
    ) -> list[MatchResult]:
        """Score each item in *contents* against *query*.

        Pre-tokenizes the query once and reuses it for all content items.
        Returns a list of :class:`MatchResult` in the same order as *contents*.
        """
        query_tokens = extract_tokens(query)
        if not query_tokens:
            return [MatchResult() for _ in contents]

        total_grams = len(query_tokens)
        results: list[MatchResult] = []

        for content in contents:
            content_tokens_set = set(extract_tokens(content))
            if not content_tokens_set:
                results.append(MatchResult())
                continue

            matched: list[str] = []
            for token in query_tokens:
                if token in content_tokens_set:
                    matched.append(token)

            hit_ratio = len(matched) / total_grams
            score = min(hit_ratio * self._max_score, self._max_score)

            results.append(MatchResult(
                score=score,
                matched_terms=matched,
                details={
                    "query_tokens": query_tokens,
                    "hit_count": len(matched),
                    "total_grams": total_grams,
                    "hit_ratio": hit_ratio,
                },
            ))

        return results
