"""TokenBudgetTrimmer — trim search results to fit within a token budget.

Fills results from highest FinalScore downward until the cumulative
estimated token count reaches the budget or the maximum result count
is hit.  At least one result is always returned (the highest-scoring
bullet), even if it alone exceeds the budget.

Token estimation uses a simple character-count heuristic:
    tokens ≈ len(text) / chars_per_token

CJK characters are denser (~1.5 chars/token on average), so the
estimator detects CJK runs and applies a lower ratio automatically.
"""

from __future__ import annotations

import re

from memx.engines.generator.score_merger import ScoredBullet

# Regex matching contiguous CJK Unified Ideograph runs
_CJK_RE: re.Pattern[str] = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df"
    r"\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]+"
)

# Default chars-per-token ratio for CJK text
_CJK_CHARS_PER_TOKEN: float = 1.5


class TokenBudgetTrimmer:
    """Trim a ranked list of ScoredBullet to fit within a token budget.

    The trimmer iterates over results sorted by ``final_score`` (descending)
    and accumulates tokens until *token_budget* or *max_results* is reached.

    Guarantee: at least one result is always returned for non-empty input,
    even if that single result exceeds the token budget on its own.

    Usage::

        trimmer = TokenBudgetTrimmer(token_budget=2000, max_results=5)
        trimmed = trimmer.trim(scored_bullets)
    """

    def __init__(
        self,
        token_budget: int = 2000,
        max_results: int = 5,
        chars_per_token: float = 4.0,
    ) -> None:
        self._token_budget = token_budget
        self._max_results = max_results
        self._chars_per_token = chars_per_token

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trim(self, results: list[ScoredBullet]) -> list[ScoredBullet]:
        """Trim *results* to fit within the configured token budget.

        Results are assumed to be pre-sorted by ``final_score`` descending
        (as returned by :class:`ScoreMerger`).  The method iterates from
        the highest score downward, accumulating token estimates, and stops
        when the budget or max-results cap is reached.

        Args:
            results: Scored bullets sorted by final_score descending.

        Returns:
            A (possibly shorter) list preserving the original order.
            Empty input yields an empty list.
        """
        if not results:
            return []

        # Sort by final_score descending (defensive — should already be sorted)
        sorted_results = sorted(results, key=lambda r: r.final_score, reverse=True)

        trimmed: list[ScoredBullet] = []
        tokens_used: int = 0

        for bullet in sorted_results:
            if len(trimmed) >= self._max_results:
                break

            estimated = self.estimate_tokens(bullet.content)

            # Guarantee: always include at least 1 result
            if trimmed and tokens_used + estimated > self._token_budget:
                break

            trimmed.append(bullet)
            tokens_used += estimated

        return trimmed

    def estimate_tokens(self, text: str) -> int:
        """Estimate the token count of *text*.

        Uses a simple heuristic: ``len(text) / chars_per_token``.
        CJK character runs are counted separately with a lower
        chars-per-token ratio (~1.5) to better reflect real tokenizer
        behaviour for ideographic scripts.

        Args:
            text: The text to estimate.

        Returns:
            Estimated token count (always >= 0).
        """
        if not text:
            return 0

        # Count CJK characters
        cjk_chars = sum(len(m.group()) for m in _CJK_RE.finditer(text))
        non_cjk_chars = len(text) - cjk_chars

        cjk_tokens = cjk_chars / _CJK_CHARS_PER_TOKEN if cjk_chars > 0 else 0.0
        non_cjk_tokens = (
            non_cjk_chars / self._chars_per_token if non_cjk_chars > 0 else 0.0
        )

        return int(cjk_tokens + non_cjk_tokens)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def token_budget(self) -> int:
        """Configured token budget."""
        return self._token_budget

    @property
    def max_results(self) -> int:
        """Maximum number of results to return."""
        return self._max_results

    @property
    def chars_per_token(self) -> float:
        """Characters-per-token ratio for non-CJK text."""
        return self._chars_per_token
