"""LLMEvaluator — LLM-based knowledge evaluation for the Reflector pipeline.

Replaces Stage 1 (PatternDetector) + Stage 2 (KnowledgeScorer) in LLM mode.
In hybrid mode, acts as a refinement layer after rule-based pre-screening.

Single LLM call per interaction: should_record + classification + scoring.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from memx.core.config import ReflectorConfig
from memx.core.types import (
    BulletSection,
    DetectedPattern,
    InteractionEvent,
    KnowledgeType,
    ScoredCandidate,
)

logger = logging.getLogger(__name__)

_EVAL_SYSTEM_PROMPT = """\
You are a knowledge extraction evaluator for an AI memory system.
Your job: decide whether a user-assistant interaction contains knowledge worth remembering long-term.

Criteria for recording:
- Error fixes, debugging insights, configuration tips → HIGH value
- Reusable methods, workflows, best practices → HIGH value
- User preferences, coding style conventions → MEDIUM value
- New tool discoveries, library usage patterns → MEDIUM value
- Trivial chit-chat, one-off questions, purely code output → LOW value (reject)

Respond ONLY with a JSON object (no markdown, no explanation):
{
  "should_record": true/false,
  "knowledge_type": "method"|"trick"|"pitfall"|"preference"|"knowledge",
  "section": "commands"|"debugging"|"architecture"|"workflow"|"tools"|"patterns"|"preferences"|"general",
  "instructivity_score": 0-100,
  "summary": "one-line summary of the knowledge"
}
"""

_EVAL_USER_TEMPLATE = """\
=== User Message ===
{user_message}

=== Assistant Response ===
{assistant_message}
"""

# Valid enum values for validation
_VALID_TYPES = frozenset(e.value for e in KnowledgeType)
_VALID_SECTIONS = frozenset(e.value for e in BulletSection)


class LLMEvaluator:
    """Evaluate interactions via LLM to decide what to remember."""

    def __init__(self, config: ReflectorConfig) -> None:
        self._config = config
        self._model = config.llm_model
        self._api_base = config.llm_api_base
        self._api_key = config.llm_api_key
        self._max_tokens = config.max_eval_tokens
        self._temperature = config.llm_temperature

    def evaluate(self, event: InteractionEvent) -> Optional[ScoredCandidate]:
        """Evaluate an interaction via LLM. Returns None if not worth recording.

        Gracefully returns None on any LLM/parse failure.
        """
        try:
            from litellm import completion
        except ImportError:
            logger.warning("litellm not installed; LLM evaluation unavailable")
            return None

        # Truncate to avoid excessive token usage
        user_msg = event.user_message[:1500]
        asst_msg = event.assistant_message[:2000]

        user_content = _EVAL_USER_TEMPLATE.format(
            user_message=user_msg,
            assistant_message=asst_msg,
        )

        try:
            kwargs: dict = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "max_tokens": self._max_tokens,
                "temperature": self._temperature,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            if self._api_key:
                kwargs["api_key"] = self._api_key

            response = completion(**kwargs)
            raw_text = response.choices[0].message.content.strip()
            logger.debug("LLMEvaluator raw response: %s", raw_text[:200])
        except Exception as e:
            logger.warning("LLM evaluation call failed: %s", e)
            return None

        return self._parse_response(raw_text, event)

    def _parse_response(
        self, raw: str, event: InteractionEvent
    ) -> Optional[ScoredCandidate]:
        """Parse LLM JSON response into ScoredCandidate."""
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("LLMEvaluator: failed to parse JSON: %s | raw=%s", e, raw[:100])
            return None

        if not data.get("should_record", False):
            logger.debug("LLMEvaluator: should_record=False, skipping")
            return None

        # Validate and map fields
        kt_str = str(data.get("knowledge_type", "knowledge")).lower()
        if kt_str not in _VALID_TYPES:
            kt_str = "knowledge"
        knowledge_type = KnowledgeType(kt_str)

        sec_str = str(data.get("section", "general")).lower()
        if sec_str not in _VALID_SECTIONS:
            sec_str = "general"
        section = BulletSection(sec_str)

        try:
            score = float(data.get("instructivity_score", 50))
            score = max(0.0, min(100.0, score))
        except (TypeError, ValueError):
            score = 50.0

        summary = str(data.get("summary", ""))

        # Build content from summary + original interaction
        content = summary if summary else event.assistant_message[:400]

        pattern = DetectedPattern(
            pattern_type="llm_evaluated",
            content=content,
            confidence=score / 100.0,
            source_event=event,
            metadata={"detection_rule": "llm_evaluator"},
        )

        return ScoredCandidate(
            pattern=pattern,
            section=section,
            knowledge_type=knowledge_type,
            instructivity_score=score,
        )

    def refine(self, candidate: ScoredCandidate) -> ScoredCandidate:
        """Refine a rule-detected candidate with LLM re-scoring (hybrid mode).

        If LLM call fails, returns the original candidate unchanged.
        """
        event = candidate.pattern.source_event
        if event is None:
            return candidate

        llm_result = self.evaluate(event)
        if llm_result is None:
            # LLM says not worth recording or call failed — keep rule result
            return candidate

        # Blend: use LLM classification but average the scores
        blended_score = (candidate.instructivity_score + llm_result.instructivity_score) / 2.0

        return ScoredCandidate(
            pattern=candidate.pattern,
            section=llm_result.section,
            knowledge_type=llm_result.knowledge_type,
            instructivity_score=max(0.0, min(100.0, blended_score)),
        )
