"""Regression tests for metadata-preserving mem0 write-backs (P0).

mem0 1.0.x ``Memory.update(memory_id, data)`` accepts content only and routes
into ``_update_memory(memory_id, data, existing_embeddings, metadata=None)``.
With ``metadata=None`` the stored payload metadata is rebuilt from scratch
(``new_metadata = {}``) — every ``memorus_*`` field (decay weight, verification
TTL / trust_score, recall_count, anchors, scope, curator-merged fields) is
silently wiped.

These tests use a FAKE mem0 that faithfully emulates that 1.0.x semantics
(``update`` -> content-only -> metadata reset to ``{}``) and assert that the
memorus write-back paths preserve ``memorus_*`` metadata by driving the
metadata-preserving ``_update_memory`` API instead.

NOTE: the existing MagicMock-based tests in ``tests/unit/test_memory.py``
(``test_run_decay_sweep_with_memories``) cannot catch this regression: a
MagicMock ``update`` swallows any signature and never mutates a real store, so
the data loss is masked. A faithful fake is required.
"""

from __future__ import annotations

import copy
import hashlib

import pytest

from memorus.core.config import MemorusConfig
from memorus.core.memory import Memory


# ---------------------------------------------------------------------------
# Fake mem0 faithfully emulating 1.0.x update semantics
# ---------------------------------------------------------------------------

# mem0's get() splits the stored payload into "core"/"promoted" keys and the
# remaining "metadata". We mirror that split so memorus code sees the same
# shape it sees against the real SDK.
_CORE_KEYS = {"data", "hash", "created_at", "updated_at", "id"}
_PROMOTED_KEYS = {"user_id", "agent_id", "run_id", "actor_id", "role"}


class FakeMem0:
    """Minimal but faithful emulation of mem0 1.0.x payload/update behaviour.

    Each row is a flat payload dict, exactly like mem0's vector store. The
    ``memorus_*`` fields live as additional payload keys (outside the core /
    promoted sets), which is how the real SDK stores them.
    """

    def __init__(self) -> None:
        # memory_id -> flat payload dict
        self._rows: dict[str, dict] = {}

    # -- seeding helper (test-only) -----------------------------------------
    def seed(self, memory_id: str, data: str, metadata: dict) -> None:
        payload = {
            "data": data,
            "hash": hashlib.md5(data.encode()).hexdigest(),
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        }
        payload.update(copy.deepcopy(metadata))
        self._rows[memory_id] = payload

    # -- read API ------------------------------------------------------------
    def get(self, memory_id: str):
        payload = self._rows.get(memory_id)
        if payload is None:
            return None
        result = {
            "id": memory_id,
            "memory": payload.get("data", ""),
            "hash": payload.get("hash"),
            "created_at": payload.get("created_at"),
            "updated_at": payload.get("updated_at"),
        }
        for key in _PROMOTED_KEYS:
            if key in payload:
                result[key] = payload[key]
        extra = {
            k: v
            for k, v in payload.items()
            if k not in _CORE_KEYS and k not in _PROMOTED_KEYS
        }
        if extra:
            result["metadata"] = copy.deepcopy(extra)
        return result

    def get_all(self, **kwargs):
        memories = [self.get(mid) for mid in self._rows]
        return {"memories": memories}

    # -- low-level update emulating mem0 1.0.x _update_memory ----------------
    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        existing = self._rows.get(memory_id)
        if existing is None:
            raise ValueError(f"invalid memory_id {memory_id}")

        # This is the crux of the 1.0.x bug: metadata=None -> {}. ANY existing
        # memorus_* metadata not re-supplied here is dropped.
        new_payload = copy.deepcopy(metadata) if metadata is not None else {}
        new_payload["data"] = data
        new_payload["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_payload["created_at"] = existing.get("created_at")
        new_payload["updated_at"] = "2026-06-01T00:00:00+00:00"
        # Promoted session ids are carried over by the real SDK.
        for key in _PROMOTED_KEYS:
            if key not in new_payload and key in existing:
                new_payload[key] = existing[key]
        self._rows[memory_id] = new_payload
        return memory_id

    # -- public content-only update (the trap) ------------------------------
    def update(self, memory_id, data):
        # Faithful to mem0 1.0.x: no metadata is threaded through, so the
        # underlying call wipes the payload metadata to {}.
        self._update_memory(memory_id, data, {})
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        self._rows.pop(memory_id, None)
        return {"message": "Memory deleted successfully!"}


# ---------------------------------------------------------------------------
# Sanity: the fake actually reproduces the data-loss bug
# ---------------------------------------------------------------------------


def test_fake_mem0_reproduces_metadata_wipe() -> None:
    """The fake must wipe metadata on a content-only update (1.0.x trap)."""
    fake = FakeMem0()
    fake.seed(
        "b1",
        "old content",
        {"memorus_decay_weight": 0.9, "memorus_scope": "project:x"},
    )

    fake.update("b1", "new content")

    got = fake.get("b1")
    assert got["memory"] == "new content"
    # Metadata is gone — this is exactly the silent data loss we guard against.
    assert "metadata" not in got


def _make_memory(fake: FakeMem0) -> Memory:
    m = Memory.__new__(Memory)
    m._config = MemorusConfig(ace_enabled=True)
    m._mem0 = fake
    m._mem0_init_error = None
    return m


# ---------------------------------------------------------------------------
# Decay-weight write-back must preserve memorus_* metadata
# ---------------------------------------------------------------------------


def test_decay_sweep_preserves_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_decay_sweep weight write-back keeps all other memorus_* fields."""
    fake = FakeMem0()
    fake.seed(
        "b1",
        "a durable fact",
        {
            "memorus_created_at": "2026-01-01T00:00:00+00:00",
            "memorus_recall_count": 3,
            "memorus_decay_weight": 1.0,
            "memorus_scope": "project:alpha",
            "memorus_trust_score": 0.8,
            "memorus_verified_at": "2026-05-01T00:00:00+00:00",
            "memorus_anchors": ["anchor-1", "anchor-2"],
        },
    )

    m = _make_memory(fake)

    # Force the decay engine to report a changed (but non-archiving) weight so
    # the write-back branch runs deterministically regardless of decay tuning.
    # run_decay_sweep imports DecayEngine locally from its source module, so we
    # must patch it there (not on memorus.core.memory).
    import memorus.core.engines.decay.engine as decay_mod

    class _Result:
        def __init__(self, weight: float) -> None:
            self.weight = weight
            self.should_archive = False

    class _Sweep:
        def __init__(self) -> None:
            self.errors: list[str] = []
            self.permanent = 0
            self.unchanged = 0
            self.updated = 1
            self.archived = 0
            self.details = {"b1": _Result(0.5)}

    class _Engine:
        def __init__(self, *a, **k) -> None:
            pass

        def sweep(self, bullets):
            return _Sweep()

    monkeypatch.setattr(decay_mod, "DecayEngine", _Engine)

    result = m.run_decay_sweep(archive=False)
    assert result["updated"] == 1

    got = fake.get("b1")
    meta = got.get("metadata", {})

    # The decay weight was written...
    assert meta.get("memorus_decay_weight") == pytest.approx(0.5)
    # ...and NONE of the other memorus_* fields were lost.
    assert meta.get("memorus_recall_count") == 3
    assert meta.get("memorus_scope") == "project:alpha"
    assert meta.get("memorus_trust_score") == pytest.approx(0.8)
    assert meta.get("memorus_verified_at") == "2026-05-01T00:00:00+00:00"
    assert meta.get("memorus_anchors") == ["anchor-1", "anchor-2"]
    # Content is unchanged.
    assert got["memory"] == "a durable fact"


# ---------------------------------------------------------------------------
# Verification write-back must preserve memorus_* metadata
# ---------------------------------------------------------------------------


def test_verification_write_back_preserves_metadata() -> None:
    """The retrieval verification update_fn keeps unrelated memorus_* fields."""
    fake = FakeMem0()
    fake.seed(
        "v1",
        "a verifiable claim",
        {
            "memorus_decay_weight": 0.77,
            "memorus_recall_count": 5,
            "memorus_scope": "global",
            "memorus_anchors": ["src://doc#1"],
        },
    )

    m = _make_memory(fake)

    update_fn = m._build_metadata_update_fn()
    assert update_fn is not None

    update_fn(
        "v1",
        {
            "memorus_verified_at": "2026-06-01T00:00:00+00:00",
            "memorus_verified_status": "verified",
            "memorus_trust_score": 0.95,
        },
    )

    meta = fake.get("v1").get("metadata", {})
    # New verification fields landed...
    assert meta.get("memorus_verified_at") == "2026-06-01T00:00:00+00:00"
    assert meta.get("memorus_verified_status") == "verified"
    assert meta.get("memorus_trust_score") == pytest.approx(0.95)
    # ...and the pre-existing fields survived.
    assert meta.get("memorus_decay_weight") == pytest.approx(0.77)
    assert meta.get("memorus_recall_count") == 5
    assert meta.get("memorus_scope") == "global"
    assert meta.get("memorus_anchors") == ["src://doc#1"]


# ---------------------------------------------------------------------------
# Curator-merge content update must preserve memorus_* metadata
# ---------------------------------------------------------------------------


def test_merge_update_preserves_metadata() -> None:
    """_merge_update rewrites content but keeps the bullet's metadata."""
    fake = FakeMem0()
    fake.seed(
        "m1",
        "original content",
        {
            "memorus_decay_weight": 0.6,
            "memorus_scope": "project:beta",
            "memorus_trust_score": 0.5,
        },
    )

    m = _make_memory(fake)

    m._merge_update("m1", "merged improved content")

    got = fake.get("m1")
    assert got["memory"] == "merged improved content"
    meta = got.get("metadata", {})
    assert meta.get("memorus_decay_weight") == pytest.approx(0.6)
    assert meta.get("memorus_scope") == "project:beta"
    assert meta.get("memorus_trust_score") == pytest.approx(0.5)
