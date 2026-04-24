"""Memory Trust / Verification Layer engine (STORY-R101).

Exposes :class:`VerificationEngine` — the read-path verifier that checks a
batch of bullets' anchors against the current filesystem and returns per-bullet
:class:`VerificationOutcome` tuples. The engine is purely functional: it never
mutates bullets, never writes to the vector store.

Opt-in via ``config.verification.enabled`` (default ``True``). When disabled,
the caller is expected to short-circuit and never construct an engine.
"""

from __future__ import annotations

from memorus.core.engines.verifier.engine import (
    VerificationEngine,
    VerificationOutcome,
)

__all__ = [
    "VerificationEngine",
    "VerificationOutcome",
]
