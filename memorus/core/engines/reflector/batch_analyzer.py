"""BatchAnalyzer — idle-time batch distillation (STORY-R095).

Drains the inbox produced by :class:`ReflectorRouter` into bullets using a
two-step LLM pipeline:

  **Stage 1 — Analyze**: cluster the batch into themes / entities / timeline.
  **Stage 2 — Generate**: emit one bullet per theme with ``sources`` pointing
  back to the original inbox turn IDs.

Design goals:

* Plug into :class:`memorus.core.daemon.orchestrator.IdleOrchestrator` as a
  new stage executed **before** ``consolidate_corpus``. No parallel
  orchestrator.
* Bound each LLM prompt by ``prompt_token_budget`` (default 8000 tokens).
* On failure, retry up to ``max_batch_retries`` times. After the final
  failure, fall back to single-turn realtime distillation using the
  existing reflector components — the inbox entries are still marked
  ``consumed`` so the turns are not lost.
* LLM calls are abstracted behind an ``LlmCaller`` protocol so tests can
  inject deterministic fakes (and the token benchmark can measure prompt
  sizes without real network traffic).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol

from memorus.core.config import ReflectorBatchConfig, ReflectorConfig
from memorus.core.engines.reflector.inbox import Inbox, InboxEntry
from memorus.core.types import (
    BulletSection,
    CandidateBullet,
    KnowledgeType,
    SourceRef,
    SourceType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM abstraction
# ---------------------------------------------------------------------------


class LlmCaller(Protocol):
    """Callable abstraction over ``litellm.completion``.

    Implementations receive a full prompt string (system + user concatenated
    with ``\n---\n``) and a ``max_tokens`` hint. Returns the raw response
    text. Raise any Exception on failure — BatchAnalyzer will retry.
    """

    def __call__(self, prompt: str, *, max_tokens: int, stage: str) -> str: ...


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ANALYZE_SYSTEM = """\
You are analyzing a batch of conversation turns to extract learnable knowledge.
Return a JSON object (no prose, no markdown fences) with this shape:
{
  "themes": [
    {"id": "t1", "title": "short title", "turn_ids": ["<turn_id>", ...], "entities": ["..."], "tools": ["..."]}
  ]
}
Group related turns into the same theme. Only include a theme if it teaches
something actionable (a command, preference, pitfall, or workflow). If the
batch contains nothing learnable, return {"themes": []}.
"""

_GENERATE_SYSTEM = """\
You are generating reusable knowledge bullets from analysed themes.
Each bullet must follow "When [condition], [action], because [reason]." Return
JSON (no prose, no markdown) shaped:
{
  "bullets": [
    {
      "theme_id": "t1",
      "distilled_rule": "...",
      "content": "one-paragraph expansion",
      "section": "commands|debugging|architecture|workflow|tools|patterns|preferences|general",
      "knowledge_type": "method|trick|pitfall|preference|knowledge",
      "related_tools": ["..."],
      "key_entities": ["..."],
      "tags": ["..."],
      "turn_ids": ["<turn_id>", ...]
    }
  ]
}
"""

# Rough char-per-token heuristic used when a real tokenizer is unavailable.
_CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """Conservative token estimate (len / 4). Fine for prompt-size budgeting."""
    if not text:
        return 0
    return max(1, int(len(text) / _CHARS_PER_TOKEN) + 1)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class BatchReport:
    """Summary of one BatchAnalyzer pass."""

    batches: int = 0
    turns_processed: int = 0
    bullets_produced: int = 0
    bullets_fallback: int = 0
    batches_retried: int = 0
    batches_failed: int = 0
    consumed_ids: list[str] = field(default_factory=list)
    skipped_ids: list[str] = field(default_factory=list)
    analyze_tokens: int = 0  # sum of approximate prompt tokens sent to Analyze
    generate_tokens: int = 0


# ---------------------------------------------------------------------------
# BatchAnalyzer
# ---------------------------------------------------------------------------


class BatchAnalyzer:
    """Convert queued inbox turns into CandidateBullets."""

    def __init__(
        self,
        reflector_config: ReflectorConfig,
        inbox: Inbox,
        llm_caller: Optional[LlmCaller] = None,
        fallback_distill: Optional[Callable[[InboxEntry], list[CandidateBullet]]] = None,
    ) -> None:
        self._reflector_config = reflector_config
        self._batch_cfg: ReflectorBatchConfig = reflector_config.batch
        self._inbox = inbox
        self._llm = llm_caller or _DefaultLlmCaller(reflector_config)
        self._fallback = fallback_distill or _default_fallback_distill

    # ------------------------------------------------------------------
    # Triggers
    # ------------------------------------------------------------------

    def should_run(self, now: Optional[datetime] = None) -> bool:
        """Return True iff there is enough work to justify a batch."""
        pending = self._inbox.list_pending()
        if not pending:
            return False
        if len(pending) >= self._batch_cfg.batch_min_turns:
            return True
        age = self._inbox.oldest_pending_age_seconds(now=now)
        if age is not None and age >= self._batch_cfg.batch_max_age_seconds:
            return True
        return False

    # ------------------------------------------------------------------
    # Main entry point (called by IdleOrchestrator, or `memorus inbox flush`)
    # ------------------------------------------------------------------

    def run_once(self, *, force: bool = False) -> BatchReport:
        """Drain as many batches as the configured budget allows in one pass."""
        self._inbox.recover_crashed()

        report = BatchReport()
        while True:
            if not force and not self.should_run():
                break

            pending = self._inbox.list_pending()
            if not pending:
                break

            batches = self._split_into_batches(pending)
            if not batches:
                break

            for batch in batches:
                self._process_batch(batch, report)

            # Only drain once when forced to avoid infinite loops on
            # permanent failures.
            if force:
                break

            # When not forced, if the first iteration didn't produce any
            # state change, break to avoid spinning.
            if not report.consumed_ids and not report.skipped_ids:
                break

        return report

    # ------------------------------------------------------------------
    # Batching
    # ------------------------------------------------------------------

    def _split_into_batches(self, pending: list[InboxEntry]) -> list[list[InboxEntry]]:
        """Greedily pack pending entries into batches bounded by size & tokens."""
        max_size = self._batch_cfg.max_batch_size
        token_budget = self._batch_cfg.prompt_token_budget
        # Reserve ~15% headroom for the Analyze system prompt.
        content_budget = int(token_budget * 0.85)

        batches: list[list[InboxEntry]] = []
        current: list[InboxEntry] = []
        current_tokens = 0
        for entry in pending:
            est = estimate_tokens(
                self._render_turn_for_analyze(entry),
            )
            # Always include a turn even if it alone exceeds the budget —
            # splitting a single turn makes no sense; the LLM will truncate.
            if current and (
                len(current) >= max_size or current_tokens + est > content_budget
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(entry)
            current_tokens += est
        if current:
            batches.append(current)
        return batches

    # ------------------------------------------------------------------
    # Per-batch pipeline
    # ------------------------------------------------------------------

    def _process_batch(self, batch: list[InboxEntry], report: BatchReport) -> None:
        if not batch:
            return
        # Claim exactly these entries so they transition to in_progress.
        # ``inbox.claim`` takes batch_size — to be safe we claim then
        # filter to the ids we want; any extras are requeued.
        claimed = self._inbox.claim(len(batch))
        wanted_ids = {e.id for e in batch}
        extras = [c for c in claimed if c.id not in wanted_ids]
        if extras:
            self._inbox.requeue(c.id for c in extras)
        active = [c for c in claimed if c.id in wanted_ids]
        if not active:
            return
        report.batches += 1
        report.turns_processed += len(active)

        last_error: Optional[Exception] = None
        for attempt in range(1, self._batch_cfg.max_batch_retries + 1):
            try:
                bullets, tok_a, tok_g = self._run_two_step(active)
            except Exception as e:
                last_error = e
                logger.warning(
                    "BatchAnalyzer: batch attempt %d/%d failed: %s",
                    attempt, self._batch_cfg.max_batch_retries, e,
                )
                report.batches_retried += 1
                continue

            report.analyze_tokens += tok_a
            report.generate_tokens += tok_g
            report.bullets_produced += len(bullets)
            self._inbox.mark_consumed(e.id for e in active)
            report.consumed_ids.extend(e.id for e in active)
            logger.info(
                "BatchAnalyzer: batch of %d turns -> %d bullets (attempt %d)",
                len(active), len(bullets), attempt,
            )
            # Hand the bullets off through the callback hook.
            self._emit_bullets(bullets)
            return

        # All retries exhausted -> fall back to per-turn realtime distillation.
        logger.error(
            "BatchAnalyzer: batch of %d turns failed after %d attempts (%s); "
            "falling back to single-turn distillation",
            len(active), self._batch_cfg.max_batch_retries, last_error,
        )
        report.batches_failed += 1
        fallback_bullets: list[CandidateBullet] = []
        for entry in active:
            try:
                fallback_bullets.extend(self._fallback(entry))
            except Exception as e:
                logger.warning(
                    "BatchAnalyzer: fallback distill failed for %s: %s", entry.id, e,
                )
        report.bullets_fallback += len(fallback_bullets)
        report.bullets_produced += len(fallback_bullets)
        self._inbox.mark_consumed(e.id for e in active)
        report.consumed_ids.extend(e.id for e in active)
        self._emit_bullets(fallback_bullets)

    # ------------------------------------------------------------------
    # Two-step LLM prompts
    # ------------------------------------------------------------------

    def _run_two_step(
        self, batch: list[InboxEntry],
    ) -> tuple[list[CandidateBullet], int, int]:
        """Run Analyze and Generate LLM calls. Returns (bullets, a_tokens, g_tokens)."""
        # --- Stage 1: Analyze ---
        analyze_user = self._build_analyze_user(batch)
        analyze_prompt = _ANALYZE_SYSTEM + "\n---\n" + analyze_user
        analyze_tokens = estimate_tokens(analyze_prompt)
        analyze_raw = self._llm(
            analyze_prompt,
            max_tokens=self._reflector_config.max_eval_tokens,
            stage="analyze",
        )
        themes = _parse_themes(analyze_raw)
        if not themes:
            # Nothing learnable; return empty bullets but count the prompt.
            return [], analyze_tokens, 0

        # --- Stage 2: Generate ---
        generate_user = self._build_generate_user(batch, themes)
        generate_prompt = _GENERATE_SYSTEM + "\n---\n" + generate_user
        generate_tokens = estimate_tokens(generate_prompt)
        generate_raw = self._llm(
            generate_prompt,
            max_tokens=self._reflector_config.max_distill_tokens * max(1, len(themes)),
            stage="generate",
        )
        raw_bullets = _parse_bullets(generate_raw)

        entries_by_id = {e.id: e for e in batch}
        bullets: list[CandidateBullet] = []
        for rb in raw_bullets:
            bullet = _raw_bullet_to_candidate(rb, entries_by_id)
            if bullet is not None:
                bullets.append(bullet)
        return bullets, analyze_tokens, generate_tokens

    def _build_analyze_user(self, batch: list[InboxEntry]) -> str:
        parts = []
        for e in batch:
            parts.append(self._render_turn_for_analyze(e))
        return "\n\n".join(parts)

    def _render_turn_for_analyze(self, e: InboxEntry) -> str:
        user = (e.content or "").strip()
        assistant = (e.assistant_message or "").strip()
        header = f"<turn id=\"{e.id}\" conv=\"{e.conversation_id}\" offset=\"{e.turn_offset}\">"
        footer = "</turn>"
        return f"{header}\nUSER: {user}\nASSISTANT: {assistant}\n{footer}"

    def _build_generate_user(
        self, batch: list[InboxEntry], themes: list[dict[str, Any]]
    ) -> str:
        # Provide both the themes JSON and a compact id->snippet index so the
        # LLM can cite the right turn_ids back at us.
        index_lines = []
        for e in batch:
            snippet = (e.content or "").replace("\n", " ")
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            index_lines.append(f"- {e.id}: {snippet}")
        return (
            "Themes:\n"
            + json.dumps({"themes": themes}, ensure_ascii=False)
            + "\n\nTurn index:\n"
            + "\n".join(index_lines)
        )

    # ------------------------------------------------------------------
    # Bullet emission hook (overridable by callers)
    # ------------------------------------------------------------------

    _emit_callback: Optional[Callable[[list[CandidateBullet]], None]] = None

    def set_emit_callback(
        self, cb: Callable[[list[CandidateBullet]], None]
    ) -> None:
        """Register a callback that receives bullets produced by the pipeline."""
        self._emit_callback = cb

    def _emit_bullets(self, bullets: list[CandidateBullet]) -> None:
        if not bullets:
            return
        if self._emit_callback is None:
            logger.debug(
                "BatchAnalyzer: no emit callback configured; %d bullets dropped",
                len(bullets),
            )
            return
        try:
            self._emit_callback(list(bullets))
        except Exception as e:
            logger.error("BatchAnalyzer: emit callback raised: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def _strip_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        # drop opening fence
        parts = text.split("\n", 1)
        text = parts[1] if len(parts) == 2 else ""
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_themes(raw: str) -> list[dict[str, Any]]:
    text = _strip_fences(raw)
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("BatchAnalyzer: analyze response not JSON: %s", e)
        return []
    themes = data.get("themes") if isinstance(data, dict) else None
    if not isinstance(themes, list):
        return []
    clean: list[dict[str, Any]] = []
    for t in themes:
        if isinstance(t, dict):
            clean.append(t)
    return clean


def _parse_bullets(raw: str) -> list[dict[str, Any]]:
    text = _strip_fences(raw)
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("BatchAnalyzer: generate response not JSON: %s", e)
        return []
    bullets = data.get("bullets") if isinstance(data, dict) else None
    if not isinstance(bullets, list):
        return []
    return [b for b in bullets if isinstance(b, dict)]


def _raw_bullet_to_candidate(
    raw: dict[str, Any], entries_by_id: dict[str, InboxEntry],
) -> Optional[CandidateBullet]:
    content = str(raw.get("content") or raw.get("distilled_rule") or "").strip()
    distilled = raw.get("distilled_rule")
    distilled_str = str(distilled).strip() if isinstance(distilled, str) else None
    if not content and distilled_str:
        content = distilled_str
    if not content:
        return None

    try:
        section = BulletSection(raw.get("section", "general"))
    except ValueError:
        section = BulletSection.GENERAL
    try:
        knowledge_type = KnowledgeType(raw.get("knowledge_type", "knowledge"))
    except ValueError:
        knowledge_type = KnowledgeType.KNOWLEDGE

    turn_ids = raw.get("turn_ids") or []
    if not isinstance(turn_ids, list):
        turn_ids = []
    sources: list[SourceRef] = []
    for tid in turn_ids:
        entry = entries_by_id.get(str(tid))
        if entry is None:
            continue
        try:
            ts = _parse_iso(entry.timestamp)
            sources.append(
                SourceRef(
                    conversation_id=entry.conversation_id,
                    turn_hash=entry.turn_hash,
                    turn_offset=entry.turn_offset,
                    timestamp=ts,
                    role=entry.role,
                )
            )
        except Exception as e:
            logger.debug("BatchAnalyzer: dropping bad source turn %s: %s", tid, e)

    related_tools = _str_list(raw.get("related_tools"))
    key_entities = _str_list(raw.get("key_entities"))
    tags = _str_list(raw.get("tags"))

    return CandidateBullet(
        content=content[:500],
        distilled_rule=(distilled_str[:200] if distilled_str else None),
        section=section,
        knowledge_type=knowledge_type,
        source_type=SourceType.INTERACTION,
        instructivity_score=60.0,
        related_tools=related_tools,
        key_entities=key_entities,
        tags=tags,
        sources=sources,
    )


def _str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v[:10]:
        if isinstance(item, str):
            out.append(item)
        else:
            out.append(str(item))
    return out


def _parse_iso(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------


def _default_fallback_distill(entry: InboxEntry) -> list[CandidateBullet]:
    """Last-resort fallback: run the rules reflector on the single turn."""
    try:
        from memorus.core.engines.reflector.engine import ReflectorEngine
        from memorus.core.config import ReflectorConfig
        from memorus.core.types import InteractionEvent

        cfg = ReflectorConfig(mode="rules")
        # Disable batch path when falling back to avoid re-queueing.
        cfg.batch.batch_enabled = False
        engine = ReflectorEngine(cfg)
        event = InteractionEvent(
            user_message=entry.content or "",
            assistant_message=entry.assistant_message or "",
            session_id=entry.conversation_id,
            metadata={
                "conversation_id": entry.conversation_id,
                "turn_offset": entry.turn_offset,
                "role": entry.role,
            },
        )
        return engine.reflect(event)
    except Exception as e:
        logger.warning("fallback distill (rules) failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Default LLM caller
# ---------------------------------------------------------------------------


class _DefaultLlmCaller:
    """Thin wrapper around ``litellm.completion`` matching :class:`LlmCaller`."""

    def __init__(self, cfg: ReflectorConfig) -> None:
        self._cfg = cfg

    def __call__(self, prompt: str, *, max_tokens: int, stage: str) -> str:
        try:
            from litellm import completion
        except ImportError as e:
            raise RuntimeError("litellm not installed") from e

        # Split prompt into (system, user) using "\n---\n" delimiter inserted
        # by the caller.
        if "\n---\n" in prompt:
            system, user = prompt.split("\n---\n", 1)
        else:
            system, user = "", prompt

        kwargs: dict[str, Any] = {
            "model": self._cfg.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": self._cfg.llm_temperature,
        }
        if self._cfg.llm_api_base:
            kwargs["api_base"] = self._cfg.llm_api_base
        if self._cfg.llm_api_key:
            kwargs["api_key"] = self._cfg.llm_api_key

        response = completion(**kwargs)
        return response.choices[0].message.content.strip()
