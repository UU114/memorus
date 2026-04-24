"""VerificationEngine — checks Bullet anchors against the live filesystem.

The engine implements the read-path half of the Memory Trust layer. For each
bullet it inspects ``bullet.anchors`` and returns a :class:`VerificationOutcome`
describing the current ``VerifiedStatus`` plus the config-derived
``trust_score``.

Protocol (must stay byte-identical with the Rust ``FileSystemVerifier``):

* Empty ``anchors`` → ``VerifiedStatus.NOT_APPLICABLE`` + ``trust_score = None``.
* TTL hit — ``now - bullet.verified_at < ttl_seconds`` AND
  ``bullet.verified_status is not None`` → reuse the stored state, zero I/O.
* For every anchor:

  * resolve ``project_root / anchor.file_path``; missing or outside-root
    paths, or any ``OSError`` during reading → ``VerifiedStatus.UNVERIFIABLE``.
  * read the file once per batch; look for ``anchor.anchor_text`` as a plain
    substring (single linear ``str.find``).
  * re-compute ``sha256(anchor_text)`` and compare to ``anchor.anchor_hash``
    — guards against a tampered anchor itself.

* All anchors pass → ``VerifiedStatus.VERIFIED``. Any anchor fails the content
  check → ``VerifiedStatus.STALE``.

The engine NEVER persists ``verified_at`` or ``verified_status`` back into the
bullet. Callers (retrieval pipeline / CLI verify command) decide whether to
write outcomes back to the store.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from memorus.core.config import VerificationConfig
from memorus.core.types import Anchor, BulletMetadata, VerifiedStatus

logger = logging.getLogger(__name__)


# Walk-up markers used to locate the project root when config.project_root is
# not set. Order mirrors the Rust implementation.
_PROJECT_ROOT_MARKERS: tuple[str, ...] = (".git", "Cargo.toml", "pyproject.toml")


# ---------------------------------------------------------------------------
# Outcome type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationOutcome:
    """Result of verifying a single bullet's anchors against the filesystem.

    Deliberately decoupled from :class:`BulletMetadata` — the Verifier returns
    outcomes so downstream stages (retrieval / CLI) can decide whether to
    persist them back onto the bullet.
    """

    verified_status: VerifiedStatus
    # ``None`` for ``NOT_APPLICABLE`` bullets (no anchors → no meaningful score).
    trust_score: float | None
    verified_at: datetime


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VerificationEngine:
    """Pure-function verifier for a batch of bullets.

    The engine is stateless between calls: all TTL caching lives on the bullet
    itself via ``verified_at`` / ``verified_status``. Construct once per
    verification session with the active :class:`VerificationConfig`.
    """

    def __init__(self, config: VerificationConfig) -> None:
        self._config = config

    # ----- public API --------------------------------------------------

    def verify(self, bullet: BulletMetadata) -> VerificationOutcome:
        """Verify one bullet. Equivalent to ``verify_many([bullet])[0]``."""
        return self.verify_many([bullet])[0]

    def verify_many(
        self, bullets: list[BulletMetadata]
    ) -> list[VerificationOutcome]:
        """Verify a batch, reading each referenced file at most once.

        TTL-hits return a synthetic outcome reusing ``bullet.verified_at`` and
        ``bullet.verified_status`` — zero filesystem I/O is performed.
        """
        now = _utcnow()
        outcomes: list[VerificationOutcome | None] = [None] * len(bullets)

        # Resolve project root lazily — only needed if any bullet requires I/O.
        project_root: Path | None = None
        # ``file_cache`` is populated on demand and scoped to this single call.
        # Keys are absolute canonical paths; value ``None`` means read-failed
        # (propagated as ``UNVERIFIABLE``).
        file_cache: dict[Path, str | None] = {}

        for idx, bullet in enumerate(bullets):
            # 1) No anchors → NotApplicable, trust_score = None.
            if not bullet.anchors:
                outcomes[idx] = VerificationOutcome(
                    verified_status=VerifiedStatus.NOT_APPLICABLE,
                    trust_score=None,
                    verified_at=now,
                )
                continue

            # 2) TTL hit — reuse stored state, zero I/O.
            if self._is_ttl_hit(bullet, now):
                # verified_status stored as str; coerce safely back to enum for
                # the outcome, falling back to the stored string if it was an
                # unknown value (should not happen — BulletMetadata validator
                # rejects unknown strings).
                stored = bullet.verified_status
                assert stored is not None  # _is_ttl_hit guarantees this
                reused_status = VerifiedStatus(stored)
                outcomes[idx] = VerificationOutcome(
                    verified_status=reused_status,
                    trust_score=bullet.trust_score,
                    verified_at=bullet.verified_at or now,
                )
                continue

            # 3) Real verification — needs project_root resolved.
            if project_root is None:
                project_root = self._resolve_project_root()

            status = self._check_bullet(bullet, project_root, file_cache)
            trust = self._trust_score_for(status)
            outcomes[idx] = VerificationOutcome(
                verified_status=status,
                trust_score=trust,
                verified_at=now,
            )

        # Sanity: every slot is populated.
        assert all(o is not None for o in outcomes)
        return [o for o in outcomes if o is not None]

    # ----- project_root resolution -------------------------------------

    def _resolve_project_root(self) -> Path:
        """Resolve the project root.

        Priority: ``config.project_root`` (must exist on disk) → cwd walk-up
        looking for ``.git`` / ``Cargo.toml`` / ``pyproject.toml`` markers.
        When no marker is found, fall back to the current working directory
        — the in-root check then guards the actual anchor paths.
        """
        configured = self._config.project_root
        if configured:
            root = Path(configured).expanduser()
            # resolve(strict=False) canonicalizes while allowing non-existent
            # intermediate segments so the Unverifiable path hits later.
            return root.resolve(strict=False)

        # cwd walk-up
        cwd = Path.cwd().resolve()
        for candidate in (cwd, *cwd.parents):
            for marker in _PROJECT_ROOT_MARKERS:
                if (candidate / marker).exists():
                    return candidate
        return cwd

    # ----- internal per-bullet verification ----------------------------

    def _check_bullet(
        self,
        bullet: BulletMetadata,
        project_root: Path,
        file_cache: dict[Path, str | None],
    ) -> VerifiedStatus:
        """Verify a single bullet's anchors, mutating ``file_cache`` on reads.

        Returns the coarsest status across all anchors, using the first
        ``UNVERIFIABLE`` or ``STALE`` to short-circuit.
        """
        for anchor in bullet.anchors:
            abs_path = (project_root / anchor.file_path).resolve(strict=False)

            if not _in_root(abs_path, project_root):
                return VerifiedStatus.UNVERIFIABLE

            # Read (or fetch cached) file content. ``None`` means read failed.
            content = self._read_cached(abs_path, file_cache)
            if content is None:
                return VerifiedStatus.UNVERIFIABLE

            if content.find(anchor.anchor_text) < 0:
                return VerifiedStatus.STALE

            # Defensive: re-hash anchor_text. Mismatch means the anchor row
            # itself has been tampered with — treat as stale rather than
            # silently trusting the stale hash.
            if _sha256(anchor.anchor_text) != anchor.anchor_hash:
                return VerifiedStatus.STALE

        return VerifiedStatus.VERIFIED

    def _read_cached(
        self,
        abs_path: Path,
        file_cache: dict[Path, str | None],
    ) -> str | None:
        """Return file content, reading at most once per ``file_cache``.

        Missing files or I/O errors cache ``None`` so subsequent bullets that
        reference the same file also get ``UNVERIFIABLE`` without re-raising.
        """
        if abs_path in file_cache:
            return file_cache[abs_path]

        try:
            # ``str`` open so the substring ``find`` works on unicode content.
            # errors="replace" keeps a handful of non-UTF8 bytes (e.g. binary
            # stubs in tests) from crashing the whole batch — substring
            # matches still succeed on the textual prefix.
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except (OSError, ValueError) as exc:
            logger.debug("verifier: read failed for %s: %s", abs_path, exc)
            file_cache[abs_path] = None
            return None

        file_cache[abs_path] = content
        return content

    # ----- TTL + trust-score helpers -----------------------------------

    def _is_ttl_hit(self, bullet: BulletMetadata, now: datetime) -> bool:
        """True when the stored verification state is still within TTL."""
        ttl = self._config.ttl_seconds
        if ttl <= 0:
            return False
        if bullet.verified_at is None or bullet.verified_status is None:
            return False
        # Both datetimes expected to be tz-aware; be defensive in case a
        # caller loaded a naive datetime.
        prev = bullet.verified_at
        if prev.tzinfo is None:
            prev = prev.replace(tzinfo=timezone.utc)
        delta = (now - prev).total_seconds()
        return delta < float(ttl)

    def _trust_score_for(self, status: VerifiedStatus) -> float | None:
        """Map a status onto the config-driven trust score."""
        if status is VerifiedStatus.VERIFIED:
            return self._config.verified_trust_score
        if status is VerifiedStatus.STALE:
            return self._config.stale_trust_score
        if status is VerifiedStatus.UNVERIFIABLE:
            return self._config.unverifiable_trust_score
        # NOT_APPLICABLE — no meaningful score.
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _in_root(abs_path: Path, root: Path) -> bool:
    """True when ``abs_path`` is the same as, or lives beneath, ``root``.

    Uses ``resolve`` + ``relative_to`` rather than string prefix to correctly
    reject ``..`` traversals and differing casing on case-sensitive platforms.
    """
    try:
        abs_path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


__all__ = [
    "VerificationEngine",
    "VerificationOutcome",
]

# The import is intentionally referenced so ``ruff`` / ``pyflakes`` do not
# complain — it keeps ``Anchor`` and ``Iterable`` available for type-checking
# while staying unused at runtime.
_ = (Anchor, Iterable, os)
