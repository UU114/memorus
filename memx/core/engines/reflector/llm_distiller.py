"""LLMDistiller — LLM-based knowledge distillation for the Reflector pipeline.

Replaces Stage 4 (BulletDistiller) in LLM/hybrid mode.
Produces structured "When [condition], [action], because [reason]" rules.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from memx.core.config import ReflectorConfig
from memx.core.types import (
    CandidateBullet,
    ScoredCandidate,
    SourceType,
)

logger = logging.getLogger(__name__)

_DISTILL_SYSTEM_PROMPT = """\
You are a knowledge distiller for an AI memory system.
Your job: convert a raw interaction into a concise, reusable knowledge rule.

Output format — respond ONLY with a JSON object (no markdown, no explanation):
{
  "distilled_rule": "When [condition/trigger], [action/recommendation], because [reason/evidence].",
  "content": "Expanded explanation in 1-3 sentences if needed.",
  "related_tools": ["tool1", "tool2"],
  "key_entities": ["entity1", "entity2"],
  "tags": ["tag1", "tag2"]
}

Guidelines:
- distilled_rule MUST follow the "When..., ..., because..." pattern
- Keep distilled_rule under 150 chars
- content provides supporting detail (under 300 chars)
- related_tools: CLI tools, libraries, frameworks mentioned
- key_entities: specific config keys, file names, class names, API endpoints
- tags: 2-4 topic tags for retrieval
"""

_DISTILL_USER_TEMPLATE = """\
Knowledge type: {knowledge_type}
Section: {section}
Score: {score}

=== Raw Content ===
{content}
"""


class LLMDistiller:
    """Distill scored candidates into structured Bullets via LLM."""

    def __init__(self, config: ReflectorConfig) -> None:
        self._config = config
        self._model = config.llm_model
        self._api_base = config.llm_api_base
        self._api_key = config.llm_api_key
        self._max_tokens = config.max_distill_tokens
        self._temperature = config.llm_temperature

    def distill(self, candidate: ScoredCandidate) -> CandidateBullet:
        """Distill a ScoredCandidate via LLM.

        Falls back to basic truncation if LLM call fails.
        """
        try:
            result = self._call_llm(candidate)
            if result is not None:
                return result
        except Exception as e:
            logger.warning("LLMDistiller call failed, using fallback: %s", e)

        return self._fallback_distill(candidate)

    def _call_llm(self, candidate: ScoredCandidate) -> Optional[CandidateBullet]:
        """Make the LLM distillation call."""
        try:
            from litellm import completion
        except ImportError:
            logger.warning("litellm not installed; LLM distillation unavailable")
            return None

        user_content = _DISTILL_USER_TEMPLATE.format(
            knowledge_type=candidate.knowledge_type.value,
            section=candidate.section.value,
            score=candidate.instructivity_score,
            content=candidate.pattern.content[:2000],
        )

        try:
            kwargs: dict = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": _DISTILL_SYSTEM_PROMPT},
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
            logger.debug("LLMDistiller raw response: %s", raw_text[:200])
        except Exception as e:
            logger.warning("LLM distillation call failed: %s", e)
            return None

        return self._parse_response(raw_text, candidate)

    def _parse_response(
        self, raw: str, candidate: ScoredCandidate
    ) -> Optional[CandidateBullet]:
        """Parse LLM JSON response into CandidateBullet."""
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning("LLMDistiller: failed to parse JSON: %s | raw=%s", e, raw[:100])
            return None

        distilled_rule = str(data.get("distilled_rule", ""))[:200]
        content = str(data.get("content", ""))[:500]
        if not content and not distilled_rule:
            return None

        # Use distilled_rule as primary content, with expanded content as fallback
        primary_content = content if content else distilled_rule

        related_tools = data.get("related_tools", [])
        if not isinstance(related_tools, list):
            related_tools = []
        related_tools = [str(t).lower() for t in related_tools[:10]]

        key_entities = data.get("key_entities", [])
        if not isinstance(key_entities, list):
            key_entities = []
        key_entities = [str(e) for e in key_entities[:10]]

        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).lower() for t in tags[:10]]

        return CandidateBullet(
            content=primary_content,
            distilled_rule=distilled_rule if distilled_rule else None,
            section=candidate.section,
            knowledge_type=candidate.knowledge_type,
            source_type=SourceType.INTERACTION,
            instructivity_score=candidate.instructivity_score,
            related_tools=related_tools,
            key_entities=key_entities,
            tags=tags,
        )

    @staticmethod
    def _fallback_distill(candidate: ScoredCandidate) -> CandidateBullet:
        """Fallback: create bullet without LLM, same as rule-based distiller."""
        return CandidateBullet(
            content=candidate.pattern.content[:500],
            section=candidate.section,
            knowledge_type=candidate.knowledge_type,
            source_type=SourceType.INTERACTION,
            instructivity_score=candidate.instructivity_score,
        )
