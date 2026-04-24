# mypy: disable-error-code="untyped-decorator"
"""Benchmark: token cost of batch path vs single-turn path (STORY-R095).

Acceptance: batch path <= 60% of single-turn tokens for equivalent info.

The benchmark stubs out the network LLM call but keeps the *prompt sizes*
real so the comparison is meaningful. Token counts use the same heuristic
as :func:`memorus.core.engines.reflector.batch_analyzer.estimate_tokens`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from memorus.core.config import ReflectorConfig
from memorus.core.engines.reflector.batch_analyzer import (
    BatchAnalyzer,
    estimate_tokens,
)
from memorus.core.engines.reflector.inbox import Inbox, make_entry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_TURNS = [
    (
        "How do I rebase my branch onto main without losing my commits?",
        "Use `git rebase main` while on your feature branch. Use `-i` for interactive, "
        "allowing you to reword, squash, or drop commits individually.",
    ),
    (
        "What's the safest way to force-push after a rebase?",
        "Prefer `git push --force-with-lease`. It aborts if the remote moved since your "
        "last fetch, preventing you from clobbering teammate commits.",
    ),
    (
        "Can I configure rebase to auto-stash before it runs?",
        "Yes: `git config --global rebase.autoStash true`. Git will stash unstaged "
        "changes before the rebase and reapply them afterwards.",
    ),
    (
        "How do I abort a rebase that went wrong?",
        "`git rebase --abort` returns the branch to its original state. If you already "
        "finished, use `git reflog` to find the original HEAD and reset to it.",
    ),
    (
        "Is it okay to rebase a branch that's been pushed to a shared remote?",
        "Generally no — rebasing rewrites history and forces collaborators to reset "
        "their local copies. Use merge on shared branches; keep rebase for local cleanup.",
    ),
    (
        "How do I squash commits while rebasing?",
        "During `git rebase -i`, mark commits with `squash` (or `s`) to fold them "
        "into the previous commit. Edit the combined message when prompted.",
    ),
    (
        "Does rebase preserve commit authors?",
        "Yes. Rebase re-applies commits as the original authors, though dates change. "
        "Use `--committer-date-is-author-date` to keep committer dates aligned too.",
    ),
    (
        "Can I rebase across branches that don't share history?",
        "Not without `--allow-unrelated-histories`. Prefer a merge for unrelated "
        "histories; rebase expects a linear ancestry.",
    ),
    (
        "What is an interactive rebase marker like `pick`?",
        "`pick` keeps the commit as-is. Other markers (`reword`, `edit`, `squash`, "
        "`fixup`, `drop`) describe how that commit should be handled when applied.",
    ),
    (
        "How do I finish a rebase that paused on a conflict?",
        "Resolve the conflicts, `git add` the fixed files, then `git rebase --continue`. "
        "Use `--skip` to drop the current commit or `--abort` to bail out.",
    ),
]


# ---------------------------------------------------------------------------
# Benchmark helper
# ---------------------------------------------------------------------------


def _build_inbox(tmp_path: Path) -> Inbox:
    box = Inbox(tmp_path / "inbox.jsonl")
    for i, (user, asst) in enumerate(SAMPLE_TURNS):
        box.append(
            make_entry(
                user_message=user,
                assistant_message=asst,
                conversation_id="bench",
                turn_offset=i,
            )
        )
    return box


def _single_turn_tokens() -> int:
    """Simulate the realtime hybrid path prompt cost across 10 turns."""
    # A realistic single-turn hybrid prompt repeats the system prompt for each
    # LLM call. The existing LLMEvaluator uses a system prompt of ~500 tokens
    # and the LLMDistiller another ~400. Total per turn: ~900 tokens of
    # overhead + ~200 tokens of turn content.
    from memorus.core.engines.reflector.llm_evaluator import (
        _EVAL_SYSTEM_PROMPT,  # type: ignore[attr-defined]
        _EVAL_USER_TEMPLATE,  # type: ignore[attr-defined]
    )
    from memorus.core.engines.reflector.llm_distiller import (
        _DISTILL_SYSTEM_PROMPT,  # type: ignore[attr-defined]
        _DISTILL_USER_TEMPLATE,  # type: ignore[attr-defined]
    )

    total = 0
    for user, asst in SAMPLE_TURNS:
        eval_user = _EVAL_USER_TEMPLATE.format(
            user_message=user, assistant_message=asst, tool_context="",
        )
        distill_user = _DISTILL_USER_TEMPLATE.format(
            knowledge_type="knowledge",
            section="general",
            score=60,
            content=user + "\n" + asst,
        )
        total += estimate_tokens(_EVAL_SYSTEM_PROMPT + "\n" + eval_user)
        total += estimate_tokens(_DISTILL_SYSTEM_PROMPT + "\n" + distill_user)
    return total


@pytest.mark.benchmark(group="batch-tokens")
def test_batch_path_uses_fewer_tokens(tmp_path: Path) -> None:
    """Batch path should use <= 60% of single-turn tokens for equivalent info."""
    single_total = _single_turn_tokens()

    # Stub LLM: record analyse / generate prompt tokens, return empty payloads.
    counts = {"analyze": 0, "generate": 0}

    def stub_llm(prompt: str, *, max_tokens: int, stage: str) -> str:
        counts[stage] += estimate_tokens(prompt)
        if stage == "analyze":
            return '{"themes": []}'
        return '{"bullets": []}'

    cfg = ReflectorConfig(mode="hybrid")
    cfg.batch.batch_enabled = True
    cfg.batch.batch_min_turns = 1
    cfg.batch.max_batch_size = len(SAMPLE_TURNS)
    inbox = _build_inbox(tmp_path)
    analyzer = BatchAnalyzer(cfg, inbox, llm_caller=stub_llm)
    analyzer.run_once(force=True)

    batch_total = counts["analyze"] + counts["generate"]

    amortized_per_turn = batch_total / len(SAMPLE_TURNS)
    single_per_turn = single_total / len(SAMPLE_TURNS)
    ratio = batch_total / single_total

    # Write a terse one-liner to stderr so `pytest -s` users can see it.
    print(
        f"\n[R095 benchmark] single_total={single_total} batch_total={batch_total} "
        f"single/turn={single_per_turn:.1f} batch/turn={amortized_per_turn:.1f} "
        f"ratio={ratio:.3f}"
    )

    assert ratio <= 0.60, (
        f"batch path must use <= 60% of single-turn tokens, got ratio={ratio:.3f} "
        f"(single_total={single_total}, batch_total={batch_total})"
    )
