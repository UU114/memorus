"""MemX core data types: BulletMetadata, enums, and auxiliary models."""

from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Core Bullet metadata model
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BulletMetadata(BaseModel):
    """ACE Bullet metadata attached to every MemX memory record.

    All fields carry sensible defaults so ``BulletMetadata()`` always succeeds.
    Fields are stored in mem0 vector-store payloads under the ``memx_`` prefix.
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
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
