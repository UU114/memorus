"""BulletDistiller — Stage 4 of the Reflector pipeline. Distill scored candidates into Bullets."""

from __future__ import annotations

import re
from typing import Optional

import logging

from memorus.core.config import ReflectorConfig
from memorus.core.types import (
    BulletSection,
    CandidateBullet,
    KnowledgeType,
    ScoredCandidate,
    SourceRef,
    SourceType,
)

logger = logging.getLogger(__name__)

# Common tool/command names for extraction
_KNOWN_TOOLS = frozenset({
    "git", "docker", "npm", "pip", "cargo", "brew", "apt",
    "kubectl", "terraform", "ansible", "make", "cmake",
    "pytest", "ruff", "mypy", "black", "flake8",
    "curl", "wget", "ssh", "scp", "rsync",
    "python", "node", "java", "go", "rustc", "ruby",
    "yarn", "pnpm", "bun", "deno", "poetry", "uv",
    "postgres", "redis", "nginx", "systemctl",
})


class BulletDistiller:
    """Distill ScoredCandidates into compact CandidateBullets."""

    def __init__(self, config: Optional[ReflectorConfig] = None):
        cfg = config or ReflectorConfig()
        self._max_content = cfg.max_content_length
        self._max_code_lines = cfg.max_code_lines

    def distill(self, candidate: ScoredCandidate) -> CandidateBullet:
        """Convert a ScoredCandidate into a CandidateBullet."""
        raw_content = candidate.pattern.content
        content = self._truncate_content(raw_content)
        was_truncated = len(content) < len(raw_content)
        tools = self._extract_tools(raw_content, candidate.pattern.metadata)
        entities = self._extract_entities(raw_content)
        sources = self._build_sources(candidate)

        logger.debug(
            "BulletDistiller.distill: content_len=%d->%d truncated=%s tools=%s entities=%s sources=%d",
            len(raw_content), len(content), was_truncated, tools, entities[:5],
            len(sources),
        )

        return CandidateBullet(
            content=content,
            section=candidate.section,
            knowledge_type=candidate.knowledge_type,
            source_type=SourceType.INTERACTION,
            instructivity_score=candidate.instructivity_score,
            related_tools=tools,
            key_entities=entities,
            sources=sources,
        )

    @staticmethod
    def _build_sources(candidate: ScoredCandidate) -> list[SourceRef]:
        """Build SourceRefs from the originating InteractionEvent (if any).

        We record the assistant turn because assistant messages carry the
        actionable knowledge; the user prompt is contextual. ``turn_offset``
        comes from ``event.metadata["turn_offset"]`` when the caller tracks
        conversation position, else falls back to 0.
        """
        event = candidate.pattern.source_event
        if event is None:
            return []

        conversation_id = event.session_id or event.user_id
        if not conversation_id:
            return []

        meta = event.metadata or {}
        try:
            turn_offset = int(meta.get("turn_offset", 0))
        except (TypeError, ValueError):
            turn_offset = 0
        turn_offset = max(turn_offset, 0)

        turn_text = event.assistant_message or event.user_message
        turn_hash = SourceRef.compute_turn_hash(turn_text)

        return [
            SourceRef(
                conversation_id=conversation_id,
                turn_hash=turn_hash,
                turn_offset=turn_offset,
                timestamp=event.timestamp,
                role="assistant" if event.assistant_message else "user",
            )
        ]

    def _truncate_content(self, content: str) -> str:
        """Truncate content at sentence boundary, respecting max length."""
        if len(content) <= self._max_content:
            return content

        truncated = content[:self._max_content]
        # Try to cut at sentence boundary
        for sep in [". ", "\u3002", "\n", "; ", ", "]:
            idx = truncated.rfind(sep)
            if idx > self._max_content * 0.5:
                return truncated[:idx + len(sep)].rstrip()
        return truncated.rstrip() + "..."

    def _truncate_code(self, code: str) -> str:
        """Truncate code to max_code_lines."""
        lines = code.splitlines()
        if len(lines) <= self._max_code_lines:
            return code
        return "\n".join(lines[:self._max_code_lines]) + "\n..."

    @staticmethod
    def _extract_tools(content: str, context: dict) -> list[str]:
        """Extract tool names from content and context."""
        tools: set[str] = set()

        # From context metadata
        if tool := context.get("tool"):
            tools.add(str(tool).lower())
        if tool_list := context.get("tools"):
            if isinstance(tool_list, list):
                tools.update(str(t).lower() for t in tool_list)

        # From content — match known tools
        words = set(re.findall(r'\b\w+\b', content.lower()))
        tools.update(words & _KNOWN_TOOLS)

        return sorted(tools)

    @staticmethod
    def _extract_entities(content: str) -> list[str]:
        """Extract key entities: quoted terms, CamelCase, dotted names."""
        entities: set[str] = set()

        # Quoted strings (backtick, double-quote)
        entities.update(re.findall(r'`([^`]+)`', content))
        entities.update(re.findall(r'"([^"]{2,30})"', content))

        # CamelCase / PascalCase identifiers
        entities.update(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', content))

        # File-like dotted names (e.g., "config.yaml", "main.py")
        entities.update(
            m for m in re.findall(r'\b[\w-]+\.[\w.]+\b', content)
            if not m[0].isdigit()  # Exclude version numbers like "2.0"
        )

        # Limit to 10 most relevant
        return sorted(entities)[:10]
