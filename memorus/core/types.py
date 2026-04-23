"""Memorus core data types: BulletMetadata, enums, and auxiliary models."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


class BulletSection(str, Enum):
    """Knowledge section categories for Bullet classification."""

    COMMANDS = "commands"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    WORKFLOW = "workflow"
    TOOLS = "tools"
    PATTERNS = "patterns"
    PREFERENCES = "preferences"
    GENERAL = "general"


class KnowledgeType(str, Enum):
    """Type of knowledge encoded in a Bullet."""

    METHOD = "method"
    TRICK = "trick"
    PITFALL = "pitfall"
    PREFERENCE = "preference"
    KNOWLEDGE = "knowledge"


class SourceType(str, Enum):
    """How the Bullet was created."""

    INTERACTION = "interaction"
    MANUAL = "manual"
    IMPORT = "import"


# ---------------------------------------------------------------------------
# SourceRef — immutable reference to a conversation turn that produced a Bullet
# ---------------------------------------------------------------------------


# Maximum number of SourceRef items retained per Bullet. Beyond this, merge
# keeps the earliest SOURCES_CAP // 2 + latest SOURCES_CAP // 2.
SOURCES_CAP = 50


class SourceRef(BaseModel):
    """Immutable pointer to the original conversation turn that yielded a Bullet.

    The combination ``(conversation_id, turn_offset)`` uniquely identifies a
    turn within a conversation; ``turn_hash`` is a content-address allowing
    later verification and cross-workspace lookups.
    """

    conversation_id: str
    turn_hash: str  # SHA256 hex digest, first 16 chars
    turn_offset: int = Field(ge=0)
    timestamp: datetime
    role: str  # "user" | "assistant"

    model_config = {"frozen": True}

    @staticmethod
    def compute_turn_hash(text: str) -> str:
        """Compute the canonical 16-char SHA256 prefix used for ``turn_hash``."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return digest[:16]


def merge_sources(
    a: list["SourceRef"],
    b: list["SourceRef"],
    cap: int = SOURCES_CAP,
) -> list["SourceRef"]:
    """Merge two SourceRef lists with semantics identical to the Rust side.

    Semantics (MUST match ``memorus-core::models::merge_sources``):
    1. Dedup by ``(conversation_id, turn_offset)`` — first occurrence wins.
    2. Sort ascending by ``timestamp``.
    3. If ``len > cap``: keep earliest ``cap // 2`` + latest ``cap // 2``.
    """
    seen: set[tuple[str, int]] = set()
    merged: list[SourceRef] = []
    for src in [*a, *b]:
        key = (src.conversation_id, src.turn_offset)
        if key in seen:
            continue
        seen.add(key)
        merged.append(src)

    merged.sort(key=lambda s: s.timestamp)

    if len(merged) > cap:
        half = cap // 2
        merged = merged[:half] + merged[-half:]
    return merged


# ---------------------------------------------------------------------------
# Auxiliary types used by Reflector pipeline
# ---------------------------------------------------------------------------


class InteractionEvent(BaseModel):
    """A single user-assistant interaction fed to the Reflector."""

    user_message: str
    assistant_message: str
    user_id: str = ""
    session_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)  # type: ignore[type-arg]


class DetectedPattern(BaseModel):
    """Output of PatternDetector — a knowledge pattern found in an interaction."""

    pattern_type: str = ""
    content: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source_event: InteractionEvent | None = None
    metadata: dict = Field(default_factory=dict)  # type: ignore[type-arg]


class ScoredCandidate(BaseModel):
    """Output of KnowledgeScorer — a DetectedPattern with scoring applied."""

    pattern: DetectedPattern = Field(default_factory=DetectedPattern)
    section: BulletSection = BulletSection.GENERAL
    knowledge_type: KnowledgeType = KnowledgeType.KNOWLEDGE
    instructivity_score: float = Field(default=50.0, ge=0.0, le=100.0)
    key_entities: list[str] = Field(default_factory=list)
    related_tools: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)


class CandidateBullet(BaseModel):
    """Output of BulletDistiller — ready to be curated and stored."""

    content: str = ""
    distilled_rule: str | None = None
    section: BulletSection = BulletSection.GENERAL
    knowledge_type: KnowledgeType = KnowledgeType.KNOWLEDGE
    source_type: SourceType = SourceType.INTERACTION
    instructivity_score: float = Field(default=50.0, ge=0.0, le=100.0)
    key_entities: list[str] = Field(default_factory=list)
    related_tools: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    scope: str = "global"
    sources: list[SourceRef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Bullet metadata model
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BulletMetadata(BaseModel):
    """ACE Bullet metadata attached to every Memorus memory record.

    All fields carry sensible defaults so ``BulletMetadata()`` always succeeds.
    Fields are stored in mem0 vector-store payloads under the ``memorus_`` prefix.
    """

    section: BulletSection = BulletSection.GENERAL
    knowledge_type: KnowledgeType = KnowledgeType.KNOWLEDGE
    instructivity_score: float = Field(default=50.0, ge=0.0, le=100.0)
    recall_count: int = Field(default=0, ge=0)
    last_recall: datetime | None = None
    decay_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    related_tools: list[str] = Field(default_factory=list)
    related_files: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    distilled_rule: str | None = None
    source_type: SourceType = SourceType.INTERACTION
    scope: str = "global"
    schema_version: int = 1
    incompatible_tags: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    sources: list[SourceRef] = Field(default_factory=list)
