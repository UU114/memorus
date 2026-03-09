# mypy: disable-error-code="untyped-decorator"
"""Shared fixtures for performance benchmark tests."""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from memx.core.engines.curator.engine import ExistingBullet
from memx.core.engines.decay.engine import BulletDecayInfo
from memx.core.engines.generator.engine import BulletForSearch
from memx.core.engines.generator.metadata_matcher import MetadataInfo
from memx.core.types import CandidateBullet, InteractionEvent

# Seed for reproducibility across benchmark runs
_RNG_SEED = 42

_TOOL_CHOICES = ["pytest", "git", "docker", "vim", "cargo"]
_ENTITY_PREFIX = "entity_"
_TAG_CHOICES = ["testing", "devops", "editor", "lang"]
_TOPIC_CHOICES = ["pytest", "git", "docker", "vim", "rust"]


@pytest.fixture()
def generate_bullets() -> Any:
    """Factory fixture: generate N mock BulletForSearch for benchmarking."""

    def _generate(n: int = 5000) -> list[BulletForSearch]:
        rng = random.Random(_RNG_SEED)
        now = datetime.now(timezone.utc)
        bullets: list[BulletForSearch] = []
        for i in range(n):
            bullets.append(
                BulletForSearch(
                    bullet_id=f"bench-{i:06d}",
                    content=(
                        f"This is benchmark bullet {i} about "
                        f"{rng.choice(_TOPIC_CHOICES)} with details on "
                        f"usage patterns and common pitfalls"
                    ),
                    metadata=MetadataInfo(
                        related_tools=[rng.choice(_TOOL_CHOICES)],
                        key_entities=[f"{_ENTITY_PREFIX}{i % 100}"],
                        tags=[rng.choice(_TAG_CHOICES)],
                    ),
                    created_at=now - timedelta(days=rng.randint(0, 90)),
                    decay_weight=rng.uniform(0.1, 1.0),
                    extra={},
                )
            )
        return bullets

    return _generate


@pytest.fixture()
def generate_decay_infos() -> Any:
    """Factory fixture: generate N mock BulletDecayInfo for benchmarking."""

    def _generate(n: int = 5000) -> list[BulletDecayInfo]:
        rng = random.Random(_RNG_SEED)
        now = datetime.now(timezone.utc)
        infos: list[BulletDecayInfo] = []
        for i in range(n):
            created = now - timedelta(days=rng.randint(0, 180))
            recall_count = rng.randint(0, 20)
            last_recall = (
                now - timedelta(days=rng.randint(0, 30))
                if recall_count > 0
                else None
            )
            infos.append(
                BulletDecayInfo(
                    bullet_id=f"decay-{i:06d}",
                    created_at=created,
                    recall_count=recall_count,
                    last_recall=last_recall,
                    current_weight=rng.uniform(0.0, 1.0),
                )
            )
        return infos

    return _generate


@pytest.fixture()
def generate_existing_bullets() -> Any:
    """Factory fixture: generate N mock ExistingBullet for curator benchmarking."""

    def _generate(n: int = 100) -> list[ExistingBullet]:
        rng = random.Random(_RNG_SEED)
        bullets: list[ExistingBullet] = []
        for i in range(n):
            bullets.append(
                ExistingBullet(
                    bullet_id=f"existing-{i:04d}",
                    content=(
                        f"Existing bullet {i} about "
                        f"{rng.choice(_TOPIC_CHOICES)} covering "
                        f"best practices and troubleshooting"
                    ),
                )
            )
        return bullets

    return _generate


@pytest.fixture()
def sample_interaction_event() -> InteractionEvent:
    """A realistic interaction event for reflector benchmarking."""
    return InteractionEvent(
        user_message=(
            "How do I use pytest fixtures with parametrize? "
            "I keep getting errors when combining them. "
            "I prefer using conftest.py for shared fixtures."
        ),
        assistant_message=(
            "You can use @pytest.fixture with @pytest.mark.parametrize together. "
            "Put shared fixtures in conftest.py and use indirect=True for "
            "parametrized fixture arguments. Here's an example pattern..."
        ),
        user_id="bench-user",
        session_id="bench-session",
    )


@pytest.fixture()
def generate_candidate_bullets() -> Any:
    """Factory fixture: generate N mock CandidateBullet for curator benchmarking."""

    def _generate(n: int = 1) -> list[CandidateBullet]:
        rng = random.Random(_RNG_SEED + 1)
        candidates: list[CandidateBullet] = []
        for i in range(n):
            candidates.append(
                CandidateBullet(
                    content=(
                        f"Candidate bullet {i}: use pytest fixtures with "
                        f"parametrize by setting indirect=True for "
                        f"{rng.choice(_TOPIC_CHOICES)} workflows"
                    ),
                    related_tools=[rng.choice(_TOOL_CHOICES)],
                    key_entities=[f"{_ENTITY_PREFIX}{i}"],
                    tags=[rng.choice(_TAG_CHOICES)],
                )
            )
        return candidates

    return _generate
