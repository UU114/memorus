"""Read-only storage backend for .ace/playbook.jsonl (Git Fallback).

Loads team knowledge bullets from a JSONL file distributed via git.
Strictly read-only: no write code paths exist in this module.

Includes read-time deduplication with playbook.cache (STORY-057):
  - First load: deduplicate bullets by content similarity, cache result
  - Subsequent loads: use cache directly (zero overhead)
  - Cache invalidation: rebuild when playbook.jsonl mtime changes

Vector cache for semantic search (STORY-055):
  - First search: auto-generate .ace/playbook.vec (gitignored)
  - Cache staleness: rebuild when playbook.jsonl mtime changes
  - ONNXEmbedder unavailable: graceful fallback to keyword search
  - After load: vectors resident in memory, zero disk I/O for search
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default embedding model used by the core ONNX embedder
_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_DIM = 384


@dataclass
class TeamBulletRecord:
    """Lightweight read-only representation of a team bullet from JSONL.

    Uses a plain dataclass (not Pydantic) so this module has zero
    dependency on STORY-050 TeamBullet which may not exist yet.
    Once TeamBullet lands, records can be converted via ``to_team_bullet()``.
    """

    content: str = ""
    section: str = "general"
    knowledge_type: str = "Knowledge"
    instructivity_score: float = 50.0
    schema_version: int = 2
    author_id: str = ""
    enforcement: str = "suggestion"
    tags: list[str] = field(default_factory=list)
    incompatible_tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Return True if this bullet should be included in search results."""
        return bool(self.content)


class GitFallbackStorage:
    """Read-only storage backend for .ace/playbook.jsonl.

    Implements a ``search()`` interface compatible with StorageBackend Protocol.
    All data is loaded lazily on first access.  If the playbook file does not
    exist the storage simply returns empty results without raising errors.
    """

    def __init__(
        self,
        playbook_path: Path | str | None = None,
        *,
        expected_model: str = _DEFAULT_MODEL,
        expected_dim: int = _DEFAULT_DIM,
    ) -> None:
        if playbook_path is not None:
            self._path: Path | None = Path(playbook_path)
        else:
            self._path = self._find_playbook()

        self._expected_model = expected_model
        self._expected_dim = expected_dim

        self._bullets: list[TeamBulletRecord] = []
        self._header: dict[str, Any] | None = None
        self._loaded: bool = False
        self._model_mismatch: bool = False

        # Vector cache state (STORY-055)
        self._vectors: Any | None = None  # np.ndarray (N, dim) or None
        self._embedder: Any | None = None  # ONNXEmbedder instance or None
        self._vectors_initialized: bool = False

    # -- StorageBackend Protocol -----------------------------------------------

    def search(self, query: str, *, limit: int = 10, **kwargs: Any) -> list[dict[str, Any]]:
        """Search with vector similarity if available, else keyword fallback.

        Returns a list of dicts (compatible with StorageBackend Protocol).
        Vector search is attempted first; if embedder is unavailable or
        model fingerprint mismatches, falls back to keyword search.
        """
        self._ensure_loaded()
        self._ensure_vectors()

        if self._vectors is not None and not self._model_mismatch:
            return self._vector_search(query, limit)
        return self._keyword_search(query, limit)

    def _keyword_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Keyword-based search against loaded bullets."""
        query_lower = query.lower()
        scored: list[tuple[TeamBulletRecord, float]] = []

        for bullet in self._bullets:
            content_lower = bullet.content.lower()
            if query_lower in content_lower:
                scored.append((bullet, 1.0))
            else:
                # Partial tag match
                if any(query_lower in t.lower() for t in bullet.tags):
                    scored.append((bullet, 0.5))

        # Sort by relevance then instructivity_score descending
        scored.sort(key=lambda x: (-x[1], -x[0].instructivity_score))

        results: list[dict[str, Any]] = []
        for bullet, score in scored[:limit]:
            results.append({
                "content": bullet.content,
                "section": bullet.section,
                "knowledge_type": bullet.knowledge_type,
                "instructivity_score": bullet.instructivity_score,
                "tags": bullet.tags,
                "enforcement": bullet.enforcement,
                "score": score,
                "source": "git_fallback",
            })
        return results

    # -- Public helpers --------------------------------------------------------

    @property
    def header(self) -> dict[str, Any] | None:
        """Return parsed header dict, or None if no header was found."""
        self._ensure_loaded()
        return self._header

    @property
    def model_mismatch(self) -> bool:
        """True when the playbook embedding model does not match local model."""
        self._ensure_loaded()
        return self._model_mismatch

    @property
    def bullet_count(self) -> int:
        """Number of active bullets loaded."""
        self._ensure_loaded()
        return len(self._bullets)

    # -- Internal --------------------------------------------------------------

    @staticmethod
    def _find_playbook() -> Path | None:
        """Walk up from cwd to git root looking for .ace/playbook.jsonl."""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            candidate = parent / ".ace" / "playbook.jsonl"
            if candidate.exists():
                return candidate
            if (parent / ".git").exists():
                break
        return None

    def _ensure_loaded(self) -> None:
        """Lazy load on first access."""
        if self._loaded:
            return
        self._loaded = True

        if not self._path or not self._path.exists():
            return

        try:
            with self._path.open("r", encoding="utf-8") as f:
                for lineno, raw_line in enumerate(f, 1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Invalid JSON at %s:%d, skipping", self._path, lineno
                        )
                        continue

                    if data.get("_header"):
                        self._parse_header(data)
                        continue

                    self._parse_bullet(data, lineno)
        except UnicodeDecodeError:
            logger.warning(
                "Non-UTF-8 encoding in %s, cannot load playbook", self._path
            )
            return

        # Deduplication with cache (STORY-057)
        if self._bullets:
            if not self._load_dedup_cache():
                original_count = len(self._bullets)
                self._bullets = self._deduplicate(self._bullets)
                self._save_dedup_cache(original_count)
                logger.info(
                    "Deduplicated: %d → %d bullets",
                    original_count, len(self._bullets),
                )

        logger.info(
            "Loaded %d team bullets from %s", len(self._bullets), self._path
        )

    def _parse_header(self, data: dict[str, Any]) -> None:
        """Parse the header line and check model fingerprint."""
        self._header = data
        model = data.get("model", "")
        dim = data.get("dim", 0)

        if model != self._expected_model or dim != self._expected_dim:
            self._model_mismatch = True
            logger.warning(
                "Embedding model mismatch: playbook uses %s (dim=%s), "
                "local model is %s (dim=%s). Falling back to keyword search.",
                model,
                dim,
                self._expected_model,
                self._expected_dim,
            )

    def _parse_bullet(self, data: dict[str, Any], lineno: int) -> None:
        """Parse a single bullet line into a TeamBulletRecord."""
        try:
            known_keys = {
                "content", "section", "knowledge_type", "instructivity_score",
                "schema_version", "author_id", "enforcement", "tags",
                "incompatible_tags",
            }
            known = {k: v for k, v in data.items() if k in known_keys}
            extra = {k: v for k, v in data.items() if k not in known_keys}
            bullet = TeamBulletRecord(**known, extra=extra)
            if bullet.is_active:
                self._bullets.append(bullet)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid TeamBullet at %s:%d, skipping", self._path, lineno
            )

    # -- Vector cache (STORY-055) ----------------------------------------------

    @property
    def _vec_cache_path(self) -> Path | None:
        """Return path to .ace/playbook.vec, or None if no playbook."""
        if self._path is None:
            return None
        return self._path.parent / "playbook.vec"

    @property
    def vectors_available(self) -> bool:
        """True when vector search is active (embeddings loaded in memory)."""
        return self._vectors is not None

    def _ensure_vectors(self) -> None:
        """Lazily initialize vector cache on first search call."""
        if self._vectors_initialized:
            return
        self._vectors_initialized = True

        if not self._bullets or self._path is None:
            return

        # Try loading cached vectors from disk
        if self._load_vector_cache():
            return

        # Build fresh vectors using ONNXEmbedder
        self._build_vector_cache()

    def _get_embedder(self) -> Any | None:
        """Try to instantiate an ONNXEmbedder. Returns None on failure."""
        if self._embedder is not None:
            return self._embedder
        try:
            from memx.core.embeddings.onnx import ONNXEmbedder
            self._embedder = ONNXEmbedder()
            return self._embedder
        except (ImportError, Exception) as exc:
            logger.info("ONNX embedder not available, vector search disabled: %s", exc)
            return None

    def _build_vector_cache(self) -> None:
        """Generate vector embeddings for all loaded bullets."""
        import numpy as np

        embedder = self._get_embedder()
        if embedder is None:
            return

        texts = [b.content for b in self._bullets]
        if not texts:
            return

        try:
            raw_vectors = embedder.embed_batch(texts)
            self._vectors = np.array(raw_vectors, dtype=np.float32)
        except Exception as exc:
            logger.warning("Failed to generate embeddings: %s", exc)
            self._vectors = None
            return

        # Persist cache to disk (best-effort)
        self._save_vector_cache()

    def _save_vector_cache(self) -> None:
        """Persist vector cache to .ace/playbook.vec (numpy npz format)."""
        import numpy as np

        vec_path = self._vec_cache_path
        if vec_path is None or self._vectors is None or self._path is None:
            return

        try:
            source_mtime = os.path.getmtime(self._path)
        except OSError:
            return

        try:
            np.savez_compressed(
                str(vec_path),
                vectors=self._vectors,
                source_mtime=np.array([source_mtime], dtype=np.float64),
                model=np.array([self._expected_model]),
                dim=np.array([self._expected_dim], dtype=np.int32),
            )
            self._ensure_vec_gitignored()
            logger.info(
                "Saved vector cache: %d vectors to %s",
                len(self._vectors), vec_path,
            )
        except OSError as exc:
            logger.warning("Failed to write vector cache to %s: %s", vec_path, exc)

    def _load_vector_cache(self) -> bool:
        """Try loading existing vector cache. Returns True if valid and loaded."""
        import numpy as np

        vec_path = self._vec_cache_path
        if vec_path is None:
            return False

        # numpy savez adds .npz extension
        npz_path = vec_path.with_suffix(".vec.npz") if not str(vec_path).endswith(".npz") else vec_path
        # Try both with and without .npz
        candidates = [vec_path, vec_path.parent / (vec_path.name + ".npz")]
        actual_path: Path | None = None
        for c in candidates:
            if c.exists():
                actual_path = c
                break

        if actual_path is None:
            return False

        try:
            data = np.load(str(actual_path), allow_pickle=False)
        except Exception as exc:
            logger.warning("Corrupt vector cache at %s: %s", actual_path, exc)
            try:
                actual_path.unlink()
            except OSError:
                pass
            return False

        # Validate source freshness
        if self._path is None:
            return False
        try:
            current_mtime = os.path.getmtime(self._path)
        except OSError:
            return False

        cached_mtime = float(data["source_mtime"][0])
        if abs(cached_mtime - current_mtime) > 0.01:
            logger.info("Vector cache expired (mtime changed), rebuilding")
            return False

        # Validate model fingerprint
        cached_model = str(data["model"][0])
        cached_dim = int(data["dim"][0])
        if cached_model != self._expected_model or cached_dim != self._expected_dim:
            logger.info("Vector cache model mismatch, rebuilding")
            return False

        vectors = data["vectors"]
        if len(vectors) != len(self._bullets):
            logger.info(
                "Vector cache count mismatch (%d vs %d bullets), rebuilding",
                len(vectors), len(self._bullets),
            )
            return False

        self._vectors = vectors
        logger.info("Loaded vector cache: %d vectors from %s", len(vectors), actual_path)
        return True

    def _vector_search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Cosine similarity search against cached vectors."""
        import numpy as np

        embedder = self._get_embedder()
        if embedder is None or self._vectors is None:
            return self._keyword_search(query, limit)

        try:
            query_vec = np.array(embedder.embed(query), dtype=np.float32)
        except Exception as exc:
            logger.warning("Failed to embed query, falling back to keyword: %s", exc)
            return self._keyword_search(query, limit)

        # Cosine similarity (vectors are L2-normalized by ONNXEmbedder)
        norms = np.linalg.norm(self._vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        # Guard against zero-norm vectors
        safe_norms = np.where(norms > 1e-9, norms, 1.0)
        safe_query_norm = max(float(query_norm), 1e-9)

        sims = np.dot(self._vectors, query_vec) / (safe_norms * safe_query_norm)

        # Get top-k indices with similarity > 0.3
        top_indices = np.argsort(sims)[::-1][:limit]

        results: list[dict[str, Any]] = []
        for idx in top_indices:
            sim = float(sims[idx])
            if sim <= 0.3:
                break
            bullet = self._bullets[idx]
            results.append({
                "content": bullet.content,
                "section": bullet.section,
                "knowledge_type": bullet.knowledge_type,
                "instructivity_score": bullet.instructivity_score,
                "tags": bullet.tags,
                "enforcement": bullet.enforcement,
                "score": sim,
                "source": "git_fallback",
            })
        return results

    def _ensure_vec_gitignored(self) -> None:
        """Ensure .ace/.gitignore includes playbook.vec and playbook.cache."""
        if self._path is None:
            return
        gitignore_path = self._path.parent / ".gitignore"
        entries = {"playbook.vec", "playbook.vec.npz", "playbook.cache"}
        try:
            existing_content = ""
            if gitignore_path.exists():
                existing_content = gitignore_path.read_text(encoding="utf-8")
            existing_lines = set(existing_content.splitlines())
            missing = entries - existing_lines
            if missing:
                content = existing_content
                if content and not content.endswith("\n"):
                    content += "\n"
                for entry in sorted(missing):
                    content += entry + "\n"
                gitignore_path.write_text(content, encoding="utf-8")
        except OSError:
            logger.warning("Failed to update .gitignore at %s", gitignore_path)

    # -- Deduplication (STORY-057) ---------------------------------------------

    @property
    def _cache_path(self) -> Path | None:
        """Return path to .ace/playbook.cache, or None if no playbook."""
        if self._path is None:
            return None
        return self._path.parent / "playbook.cache"

    def _deduplicate(self, bullets: list[TeamBulletRecord]) -> list[TeamBulletRecord]:
        """Remove near-duplicate bullets, keeping highest instructivity_score."""
        if len(bullets) <= 1:
            return list(bullets)

        kept: list[TeamBulletRecord] = []
        for bullet in sorted(bullets, key=lambda b: -b.instructivity_score):
            is_dup = False
            for existing in kept:
                sim = self._text_similarity(bullet.content, existing.content)
                if sim >= 0.90:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(bullet)
        return kept

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Word-level Jaccard similarity."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def _load_dedup_cache(self) -> bool:
        """Try to load dedup cache. Return True if cache is valid and applied."""
        cache_path = self._cache_path
        if cache_path is None or not cache_path.exists():
            return False

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt dedup cache at %s, will rebuild", cache_path)
            try:
                cache_path.unlink()
            except OSError:
                pass
            return False

        # Validate cache freshness against source mtime
        assert self._path is not None  # guaranteed by caller
        try:
            source_mtime = os.path.getmtime(self._path)
        except OSError:
            return False

        cached_mtime = cache_data.get("source_mtime")
        if cached_mtime is None or abs(cached_mtime - source_mtime) > 0.01:
            logger.info("Dedup cache expired (mtime changed), rebuilding")
            return False

        # Validate index count matches current bullets
        original_count = cache_data.get("original_count", -1)
        if original_count != len(self._bullets):
            logger.info(
                "Dedup cache count mismatch (%d vs %d), rebuilding",
                original_count, len(self._bullets),
            )
            return False

        deduped_indices: list[int] = cache_data.get("deduped_indices", [])
        # Apply cached indices to filter bullets
        try:
            self._bullets = [self._bullets[i] for i in deduped_indices]
        except IndexError:
            logger.warning("Invalid indices in dedup cache, rebuilding")
            return False

        logger.info(
            "Loaded dedup cache: %d → %d bullets",
            original_count, len(self._bullets),
        )
        return True

    def _save_dedup_cache(self, original_count: int) -> None:
        """Persist dedup result as a cache file alongside the playbook."""
        cache_path = self._cache_path
        if cache_path is None:
            return

        assert self._path is not None  # guaranteed by caller
        try:
            source_mtime = os.path.getmtime(self._path)
        except OSError:
            return

        # Build index mapping: find the original indices of kept bullets.
        # We need to re-read original bullets to map indices correctly.
        # Instead, store the deduped content for index reconstruction on load.
        # Since _deduplicate re-sorts by score, we store indices into
        # the *original* bullet list that survived dedup.
        # Re-read original bullets to find matching indices.
        original_bullets = self._read_raw_bullets()
        deduped_contents = {b.content for b in self._bullets}
        deduped_indices: list[int] = []
        seen_contents: set[str] = set()
        for i, bullet in enumerate(original_bullets):
            if bullet.content in deduped_contents and bullet.content not in seen_contents:
                deduped_indices.append(i)
                seen_contents.add(bullet.content)

        cache_data = {
            "source_mtime": source_mtime,
            "original_count": original_count,
            "deduped_count": len(self._bullets),
            "deduped_indices": deduped_indices,
        }

        try:
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
            # Ensure .ace/.gitignore includes playbook.cache
            self._ensure_cache_gitignored()
        except OSError:
            logger.warning("Failed to write dedup cache to %s", cache_path)

    def _read_raw_bullets(self) -> list[TeamBulletRecord]:
        """Re-read all bullets from JSONL without dedup (for index mapping)."""
        if self._path is None or not self._path.exists():
            return []
        bullets: list[TeamBulletRecord] = []
        try:
            with self._path.open("r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if data.get("_header"):
                        continue
                    known_keys = {
                        "content", "section", "knowledge_type",
                        "instructivity_score", "schema_version", "author_id",
                        "enforcement", "tags", "incompatible_tags",
                    }
                    known = {k: v for k, v in data.items() if k in known_keys}
                    extra = {k: v for k, v in data.items() if k not in known_keys}
                    try:
                        bullet = TeamBulletRecord(**known, extra=extra)
                        if bullet.is_active:
                            bullets.append(bullet)
                    except (TypeError, ValueError):
                        continue
        except (OSError, UnicodeDecodeError):
            pass
        return bullets

    def _ensure_cache_gitignored(self) -> None:
        """Ensure .ace/.gitignore includes playbook.cache."""
        if self._path is None:
            return
        gitignore_path = self._path.parent / ".gitignore"
        entry = "playbook.cache"
        try:
            if gitignore_path.exists():
                content = gitignore_path.read_text(encoding="utf-8")
                if entry in content:
                    return
                if not content.endswith("\n"):
                    content += "\n"
                content += entry + "\n"
            else:
                content = entry + "\n"
            gitignore_path.write_text(content, encoding="utf-8")
        except OSError:
            logger.warning("Failed to update .gitignore at %s", gitignore_path)
