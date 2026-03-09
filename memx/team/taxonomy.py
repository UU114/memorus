"""TagTaxonomy — tag normalization and alignment for team knowledge.

Provides a canonical tag vocabulary that ensures consistent naming across team
members.  Tags are normalized through a three-tier cascade:

  1. Exact alias match (O(1) dict lookup)
  2. Case-insensitive match against all canonical tags
  3. Vector similarity fallback (cosine >= 0.9 threshold)

Taxonomy can be loaded from:
  - ACE Sync Server via ``pull_taxonomy()``
  - Local cache ``~/.ace/team_cache/{team_id}/taxonomy.json``
  - Git Fallback project-level ``.ace/taxonomy.json``
  - Built-in preset templates (always available)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset taxonomy templates
# ---------------------------------------------------------------------------

PRESET_CATEGORIES: dict[str, list[str]] = {
    "languages": [
        "python", "rust", "typescript", "javascript", "go", "java",
        "c", "cpp", "csharp", "ruby", "swift", "kotlin", "php",
        "scala", "haskell", "elixir", "dart", "lua", "r", "sql",
    ],
    "frameworks": [
        "react", "vue", "angular", "django", "fastapi", "flask",
        "express", "nextjs", "nuxt", "svelte", "actix", "axum",
        "spring", "rails", "laravel", "gin", "echo",
    ],
    "domains": [
        "security", "architecture", "testing", "devops", "database",
        "networking", "concurrency", "api-design", "ci-cd", "monitoring",
        "observability", "cloud", "containerization", "microservices",
    ],
    "practices": [
        "error-handling", "performance", "debugging", "logging",
        "code-review", "refactoring", "documentation", "dependency-management",
        "version-control", "configuration", "deployment",
    ],
}

PRESET_ALIASES: dict[str, str] = {
    # Languages
    "py": "python",
    "Python3": "python",
    "python3": "python",
    "rs": "rust",
    "ts": "typescript",
    "TS": "typescript",
    "js": "javascript",
    "JS": "javascript",
    "golang": "go",
    "c++": "cpp",
    "C++": "cpp",
    "c#": "csharp",
    "C#": "csharp",
    # Frameworks
    "reactjs": "react",
    "React.js": "react",
    "react.js": "react",
    "vuejs": "vue",
    "Vue.js": "vue",
    "vue.js": "vue",
    "angularjs": "angular",
    "next.js": "nextjs",
    "Next.js": "nextjs",
    "nuxt.js": "nuxt",
    "Nuxt.js": "nuxt",
    # Domains
    "k8s": "kubernetes",
    "K8S": "kubernetes",
    "db": "database",
    "DB": "database",
    "ci/cd": "ci-cd",
    "CI/CD": "ci-cd",
    # Practices
    "perf": "performance",
    "docs": "documentation",
    "doc": "documentation",
}


# ---------------------------------------------------------------------------
# TagTaxonomy model
# ---------------------------------------------------------------------------


class TagTaxonomy(BaseModel):
    """Canonical tag vocabulary with alias resolution.

    Attributes:
        version: Schema / revision number.
        updated_at: When this taxonomy was last updated.
        categories: Mapping of category name -> list of canonical tags.
        aliases: Mapping of alias string -> canonical tag.
    """

    version: int = 1
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    categories: dict[str, list[str]] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)

    # -- derived caches (not serialized) ------------------------------------

    _all_tags_set: set[str] | None = None
    _lower_map: dict[str, str] | None = None

    model_config = {"extra": "allow"}

    def all_tags(self) -> set[str]:
        """Return all canonical tags across all categories."""
        if self._all_tags_set is None:
            tags: set[str] = set()
            for tag_list in self.categories.values():
                tags.update(tag_list)
            object.__setattr__(self, "_all_tags_set", tags)
        return self._all_tags_set  # type: ignore[return-value]

    def _build_lower_map(self) -> dict[str, str]:
        """Build lowercase -> canonical tag mapping for case-insensitive lookup."""
        if self._lower_map is None:
            mapping: dict[str, str] = {}
            for tag in self.all_tags():
                mapping[tag.lower()] = tag
            object.__setattr__(self, "_lower_map", mapping)
        return self._lower_map  # type: ignore[return-value]

    def invalidate_cache(self) -> None:
        """Clear derived caches after mutation."""
        object.__setattr__(self, "_all_tags_set", None)
        object.__setattr__(self, "_lower_map", None)

    def normalize(self, tag: str) -> str:
        """Normalize a tag using the taxonomy.

        Resolution order:
          1. Exact alias match
          2. Case-insensitive match against canonical tags
          3. Return original tag unchanged
        """
        # 1. Exact alias match
        if tag in self.aliases:
            return self.aliases[tag]

        # 2. Case-insensitive match
        lower_map = self._build_lower_map()
        lower = tag.lower()
        if lower in lower_map:
            return lower_map[lower]

        # 3. No match
        return tag

    def normalize_tags(self, tags: list[str]) -> list[str]:
        """Normalize a list of tags, preserving order and removing duplicates."""
        seen: set[str] = set()
        result: list[str] = []
        for tag in tags:
            normalized = self.normalize(tag)
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    def merge(self, other: TagTaxonomy) -> TagTaxonomy:
        """Merge another taxonomy into this one.

        ``other`` takes priority for aliases (project-level overrides team-level).
        Categories are union-merged.
        """
        merged_categories: dict[str, list[str]] = {}
        # Start with self categories
        for cat, tags in self.categories.items():
            merged_categories[cat] = list(tags)
        # Merge other categories
        for cat, tags in other.categories.items():
            existing = set(merged_categories.get(cat, []))
            existing.update(tags)
            merged_categories[cat] = sorted(existing)

        merged_aliases = {**self.aliases, **other.aliases}

        return TagTaxonomy(
            version=max(self.version, other.version),
            updated_at=max(self.updated_at, other.updated_at),
            categories=merged_categories,
            aliases=merged_aliases,
        )


# ---------------------------------------------------------------------------
# Preset taxonomy factory
# ---------------------------------------------------------------------------


def build_preset_taxonomy() -> TagTaxonomy:
    """Create a TagTaxonomy from built-in preset templates."""
    return TagTaxonomy(
        version=1,
        updated_at=datetime.now(timezone.utc),
        categories=dict(PRESET_CATEGORIES),
        aliases=dict(PRESET_ALIASES),
    )


# ---------------------------------------------------------------------------
# Taxonomy loading from various sources
# ---------------------------------------------------------------------------


def _sanitize_team_id(team_id: str) -> str:
    """Remove unsafe characters from team_id for use as directory name."""
    return re.sub(r"[^\w\-.]", "_", team_id)


def _cache_path(team_id: str) -> Path:
    """Return the local cache path for taxonomy: ~/.ace/team_cache/{team_id}/taxonomy.json."""
    safe_id = _sanitize_team_id(team_id)
    return Path.home() / ".ace" / "team_cache" / safe_id / "taxonomy.json"


def load_taxonomy_from_cache(team_id: str) -> TagTaxonomy | None:
    """Load taxonomy from local cache file.

    Returns None if cache does not exist or is corrupt.
    """
    path = _cache_path(team_id)
    if not path.exists():
        return None

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Corrupt taxonomy cache at %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Invalid taxonomy cache format at %s", path)
        return None

    try:
        return TagTaxonomy.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to parse taxonomy cache at %s: %s", path, exc)
        return None


def save_taxonomy_to_cache(taxonomy: TagTaxonomy, team_id: str) -> bool:
    """Persist taxonomy to local cache.

    Returns True on success, False on failure.
    """
    path = _cache_path(team_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(
                taxonomy.model_dump(mode="json"),
                f,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        logger.info("Saved taxonomy cache to %s", path)
        return True
    except OSError as exc:
        logger.warning("Failed to save taxonomy cache to %s: %s", path, exc)
        return False


def load_taxonomy_from_git_fallback(
    project_root: Path | None = None,
) -> TagTaxonomy | None:
    """Load taxonomy from project-level .ace/taxonomy.json (Git Fallback).

    Walks up from cwd (or given root) to git root looking for .ace/taxonomy.json.
    Returns None if not found.
    """
    if project_root is not None:
        candidate = project_root / ".ace" / "taxonomy.json"
        if candidate.exists():
            return _load_taxonomy_file(candidate)
        return None

    # Walk up to git root
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / ".ace" / "taxonomy.json"
        if candidate.exists():
            return _load_taxonomy_file(candidate)
        if (parent / ".git").exists():
            break
    return None


def _load_taxonomy_file(path: Path) -> TagTaxonomy | None:
    """Load and validate a taxonomy JSON file."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load taxonomy from %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Invalid taxonomy format at %s", path)
        return None

    try:
        return TagTaxonomy.model_validate(data)
    except Exception as exc:
        logger.warning("Failed to parse taxonomy at %s: %s", path, exc)
        return None


def _convert_sync_response(response: Any) -> TagTaxonomy:
    """Convert a TaxonomyResponse from AceSyncClient to TagTaxonomy.

    The server response contains a flat list of TaxonomyTag objects, each with
    ``name``, ``aliases``, and optional ``parent``.  We convert this into
    TagTaxonomy's categories + aliases structure.
    """
    categories: dict[str, list[str]] = {}
    aliases: dict[str, str] = {}

    for tag_entry in response.tags:
        # Group by parent if available, otherwise use "uncategorized"
        parent = tag_entry.parent or "uncategorized"
        if parent not in categories:
            categories[parent] = []
        if tag_entry.name not in categories[parent]:
            categories[parent].append(tag_entry.name)

        # Register aliases
        for alias in tag_entry.aliases:
            aliases[alias] = tag_entry.name

    return TagTaxonomy(
        version=1,
        updated_at=datetime.now(timezone.utc),
        categories=categories,
        aliases=aliases,
    )


# ---------------------------------------------------------------------------
# Vector similarity fallback
# ---------------------------------------------------------------------------


def normalize_with_vector_fallback(
    tag: str,
    taxonomy: TagTaxonomy,
    embedder: Any | None = None,
    *,
    threshold: float = 0.9,
) -> str:
    """Normalize a tag with vector similarity as final fallback.

    Resolution order:
      1. Taxonomy.normalize() (exact alias + case-insensitive)
      2. Vector cosine similarity >= threshold
      3. Return original tag unchanged

    Args:
        tag: Raw tag string to normalize.
        taxonomy: The active TagTaxonomy.
        embedder: An object with ``embed(text) -> list[float]`` method, or None.
        threshold: Minimum cosine similarity to consider a match (default 0.9).
    """
    # Try taxonomy first (fast path)
    normalized = taxonomy.normalize(tag)
    if normalized != tag:
        return normalized

    # Vector fallback
    if embedder is None:
        return tag

    try:
        tag_vec = embedder.embed(tag)
    except Exception as exc:
        logger.debug("Failed to embed tag %r: %s", tag, exc)
        return tag

    best_match: str | None = None
    best_score: float = 0.0

    for canonical in taxonomy.all_tags():
        try:
            canonical_vec = embedder.embed(canonical)
        except Exception:
            continue
        score = _cosine_similarity(tag_vec, canonical_vec)
        if score > best_score:
            best_match = canonical
            best_score = score

    if best_score >= threshold and best_match is not None:
        return best_match

    return tag


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0

    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# High-level taxonomy resolver
# ---------------------------------------------------------------------------


class TaxonomyResolver:
    """Resolves and caches the active taxonomy for a team.

    Loads taxonomy from multiple sources with priority:
      1. Server (via AceSyncClient) — freshest, saved to cache
      2. Local cache (~/.ace/team_cache/{team_id}/taxonomy.json)
      3. Git Fallback (.ace/taxonomy.json in project)
      4. Built-in preset templates (always available)

    Git Fallback taxonomy is merged on top (project-level overrides).
    """

    def __init__(
        self,
        team_id: str = "default",
        *,
        project_root: Path | None = None,
        embedder: Any | None = None,
        vector_threshold: float = 0.9,
    ) -> None:
        self._team_id = team_id
        self._project_root = project_root
        self._embedder = embedder
        self._vector_threshold = vector_threshold
        self._taxonomy: TagTaxonomy | None = None

    @property
    def taxonomy(self) -> TagTaxonomy:
        """Return the active taxonomy, loading lazily if needed."""
        if self._taxonomy is None:
            self._taxonomy = self._load()
        return self._taxonomy

    def reload(self) -> TagTaxonomy:
        """Force reload taxonomy from all sources."""
        self._taxonomy = self._load()
        return self._taxonomy

    def normalize(self, tag: str) -> str:
        """Normalize a single tag using the active taxonomy + vector fallback."""
        return normalize_with_vector_fallback(
            tag,
            self.taxonomy,
            self._embedder,
            threshold=self._vector_threshold,
        )

    def normalize_tags(self, tags: list[str]) -> list[str]:
        """Normalize a list of tags, removing duplicates."""
        seen: set[str] = set()
        result: list[str] = []
        for tag in tags:
            normalized = self.normalize(tag)
            if normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    def update_from_server(self, sync_response: Any) -> None:
        """Update taxonomy from a TaxonomyResponse pulled from the server.

        Converts the server response, saves to cache, and merges with
        Git Fallback if available.
        """
        server_taxonomy = _convert_sync_response(sync_response)
        save_taxonomy_to_cache(server_taxonomy, self._team_id)

        # Merge with Git Fallback (project-level overrides)
        git_taxonomy = load_taxonomy_from_git_fallback(self._project_root)
        if git_taxonomy is not None:
            self._taxonomy = server_taxonomy.merge(git_taxonomy)
        else:
            self._taxonomy = server_taxonomy

        # Also merge presets as baseline
        preset = build_preset_taxonomy()
        self._taxonomy = preset.merge(self._taxonomy)

    def _load(self) -> TagTaxonomy:
        """Load taxonomy from available sources with fallback chain."""
        # Try local cache first
        cached = load_taxonomy_from_cache(self._team_id)

        # Try Git Fallback
        git_taxonomy = load_taxonomy_from_git_fallback(self._project_root)

        # Start with preset as base
        base = build_preset_taxonomy()

        if cached is not None:
            base = base.merge(cached)
            logger.info("Loaded taxonomy from cache for team %s", self._team_id)

        if git_taxonomy is not None:
            base = base.merge(git_taxonomy)
            logger.info("Merged Git Fallback taxonomy")

        return base
