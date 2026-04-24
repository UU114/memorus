"""TopicPage persistence — SQLite table + Obsidian-compatible MD files.

Mirrors the Rust ``memorus-r/memorus-ace/src/topic/store.rs`` so the two
can share a single SQLite file:

.. code:: sql

    CREATE TABLE topic_pages (
        id TEXT PRIMARY KEY,          -- "topic_<8-char hex>"
        slug TEXT NOT NULL UNIQUE,    -- kebab-case
        title TEXT NOT NULL,
        summary TEXT NOT NULL,
        bullet_ids TEXT NOT NULL,     -- JSON array
        source_hash TEXT NOT NULL,    -- hex digest
        model_hash TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS topic_pages (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    bullet_ids TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    model_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_topic_pages_source_hash
    ON topic_pages(source_hash);
"""


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TopicPage:
    """First-class retrieval target: a narrative summary over N bullets."""

    id: str                 # "topic_<8-char hex>"
    slug: str               # kebab-case, globally unique
    title: str
    summary: str
    bullet_ids: list[str] = field(default_factory=list)
    source_hash: str = ""
    model_hash: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_row(self) -> tuple:
        return (
            self.id,
            self.slug,
            self.title,
            self.summary,
            json.dumps(self.bullet_ids, ensure_ascii=False),
            self.source_hash,
            self.model_hash,
            _to_rfc3339(self.created_at),
            _to_rfc3339(self.updated_at),
        )

    @classmethod
    def from_row(cls, row: Iterable) -> "TopicPage":
        (
            id_, slug, title, summary, bullets_json,
            source_hash, model_hash, created_at, updated_at,
        ) = row
        try:
            ids = json.loads(bullets_json) if bullets_json else []
            if not isinstance(ids, list):
                ids = []
        except (json.JSONDecodeError, TypeError):
            ids = []
        return cls(
            id=id_,
            slug=slug,
            title=title,
            summary=summary,
            bullet_ids=[str(b) for b in ids],
            source_hash=source_hash or "",
            model_hash=model_hash or "",
            created_at=_from_rfc3339(created_at),
            updated_at=_from_rfc3339(updated_at),
        )


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def compute_source_hash(bullet_ids: Iterable[str]) -> str:
    """SHA256 over sorted(bullet_ids); first 16 hex chars.

    Bullet IDs are deduped and compared as strings so Rust + Python always
    produce the same digest for the same cluster. Empty input yields the
    sha256 of an empty sorted list (i.e. ``sha256("[]")``).
    """
    unique = sorted({str(b) for b in bullet_ids if b is not None})
    payload = json.dumps(unique, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def generate_topic_id(source_hash: str) -> str:
    """Produce a ``topic_<8-char hex>`` identifier derived from *source_hash*."""
    if not source_hash:
        source_hash = "0000000000000000"
    return f"topic_{source_hash[:8]}"


_SLUG_ALNUM = re.compile(r"[^\w一-鿿]+", re.UNICODE)


def slugify(title: str, *, max_len: int = 60) -> str:
    """Kebab-case slug derived from *title*.

    Non-alphanumeric Unicode characters collapse into a single ``-``. CJK
    ranges are preserved so Chinese titles still produce useful slugs.
    """
    base = (title or "").strip().lower()
    if not base:
        return "untitled"
    # Replace underscores with hyphens so "foo_bar" → "foo-bar".
    base = base.replace("_", "-")
    cleaned = _SLUG_ALNUM.sub("-", base)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    if not cleaned:
        return "untitled"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-") or "untitled"
    return cleaned


def compute_drift_fraction(
    old_ids: Iterable[str],
    new_ids: Iterable[str],
) -> float:
    """Jaccard-distance style drift: ``1 - |A∩B| / |A∪B|``.

    Matches the Rust implementation — used to decide whether a page
    needs regenerating (default threshold ``0.20``). Identical sets
    yield ``0.0``; disjoint sets yield ``1.0``.
    """
    a = {str(x) for x in old_ids}
    b = {str(x) for x in new_ids}
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    inter = a & b
    return 1.0 - (len(inter) / len(union))


# ---------------------------------------------------------------------------
# SQLite + Markdown store
# ---------------------------------------------------------------------------


class SqliteTopicStore:
    """SQLite-backed TopicPage store with on-disk Markdown mirror.

    The Markdown mirror directory (``.ace/pages`` by default) gets one
    ``<slug>.md`` per page. The SQLite file is the source of truth; the
    MD files are a derived representation for Obsidian users.
    """

    def __init__(
        self,
        sqlite_path: str = "",
        pages_dir: str = ".ace/pages",
    ) -> None:
        self._path = sqlite_path if sqlite_path else ":memory:"
        self._is_memory = self._path == ":memory:"
        self._pages_dir = Path(pages_dir)
        self._write_lock = threading.Lock()
        self._local = threading.local()

        if self._is_memory:
            self._memory_conn: Optional[sqlite3.Connection] = sqlite3.connect(
                self._path, detect_types=0, check_same_thread=False,
            )
        else:
            self._memory_conn = None

        conn = self._conn()
        conn.executescript(CREATE_TABLE_SQL)
        conn.commit()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        if self._memory_conn is not None:
            try:
                self._memory_conn.close()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to close topic-store memory conn", exc_info=True)
            self._memory_conn = None
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                logger.debug("Failed to close topic-store conn", exc_info=True)
            self._local.conn = None

    def __enter__(self) -> "SqliteTopicStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert(self, page: TopicPage) -> TopicPage:
        """Insert or replace *page* by primary key + mirror to disk."""
        now = datetime.now(timezone.utc)
        page.updated_at = now
        # Ensure slug uniqueness by disambiguating with the short id suffix
        # if the slug is already taken by a different id.
        with self._write_lock:
            page.slug = self._ensure_unique_slug(page.slug, page.id)
            conn = self._conn()
            conn.execute(
                "INSERT OR REPLACE INTO topic_pages "
                "(id, slug, title, summary, bullet_ids, source_hash, "
                " model_hash, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                page.to_row(),
            )
            conn.commit()
        return page

    def delete(self, page_id: str) -> bool:
        """Remove a page by id. Also deletes its on-disk MD file."""
        page = self.get_by_id(page_id)
        if page is None:
            return False
        with self._write_lock:
            conn = self._conn()
            conn.execute("DELETE FROM topic_pages WHERE id = ?", (page_id,))
            conn.commit()
        try:
            md = self._pages_dir / f"{page.slug}.md"
            if md.exists():
                md.unlink()
        except OSError as e:
            logger.warning("Failed to delete MD file for %s: %s", page.slug, e)
        return True

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_by_id(self, page_id: str) -> Optional[TopicPage]:
        conn = self._conn()
        row = conn.execute(
            "SELECT id, slug, title, summary, bullet_ids, source_hash, "
            "model_hash, created_at, updated_at "
            "FROM topic_pages WHERE id = ?",
            (page_id,),
        ).fetchone()
        return TopicPage.from_row(row) if row else None

    def get_by_slug(self, slug: str) -> Optional[TopicPage]:
        conn = self._conn()
        row = conn.execute(
            "SELECT id, slug, title, summary, bullet_ids, source_hash, "
            "model_hash, created_at, updated_at "
            "FROM topic_pages WHERE slug = ?",
            (slug,),
        ).fetchone()
        return TopicPage.from_row(row) if row else None

    def get_by_source_hash(self, source_hash: str) -> Optional[TopicPage]:
        """Return the page whose source_hash matches, if any (drift check)."""
        if not source_hash:
            return None
        conn = self._conn()
        row = conn.execute(
            "SELECT id, slug, title, summary, bullet_ids, source_hash, "
            "model_hash, created_at, updated_at "
            "FROM topic_pages WHERE source_hash = ? LIMIT 1",
            (source_hash,),
        ).fetchone()
        return TopicPage.from_row(row) if row else None

    def list_all(self) -> list[TopicPage]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT id, slug, title, summary, bullet_ids, source_hash, "
            "model_hash, created_at, updated_at "
            "FROM topic_pages ORDER BY updated_at DESC",
        ).fetchall()
        return [TopicPage.from_row(r) for r in rows]

    # ------------------------------------------------------------------
    # MD file mirror
    # ------------------------------------------------------------------

    def write_md(
        self,
        page: TopicPage,
        bullet_lookup: dict[str, str] | None = None,
        related_titles: list[str] | None = None,
    ) -> Path:
        """Write *page* as Obsidian-compatible markdown under *pages_dir*."""
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        body = render_markdown(page, bullet_lookup or {}, related_titles or [])
        target = self._pages_dir / f"{page.slug}.md"
        target.write_text(body, encoding="utf-8")
        return target

    def md_path(self, slug: str) -> Path:
        return self._pages_dir / f"{slug}.md"

    # ------------------------------------------------------------------
    # Slug uniqueness
    # ------------------------------------------------------------------

    def _ensure_unique_slug(self, slug: str, own_id: str) -> str:
        conn = self._conn()
        row = conn.execute(
            "SELECT id FROM topic_pages WHERE slug = ?",
            (slug,),
        ).fetchone()
        if row is None or row[0] == own_id:
            return slug
        # Disambiguate with the trailing 8-hex id portion.
        suffix = own_id.replace("topic_", "")[:8] or "x"
        disambiguated = f"{slug}-{suffix}"
        return disambiguated[:80]


# ---------------------------------------------------------------------------
# Markdown renderer (Obsidian-compatible)
# ---------------------------------------------------------------------------


def render_markdown(
    page: TopicPage,
    bullet_lookup: dict[str, str],
    related_titles: list[str],
) -> str:
    """Render a :class:`TopicPage` as Obsidian-compatible markdown.

    YAML front matter includes the page id, title, bullet_ids, model_hash
    and updated_at so callers can round-trip the file back into SQLite if
    needed.
    """
    front = [
        "---",
        f"id: {page.id}",
        f"slug: {page.slug}",
        f"title: {_yaml_escape(page.title)}",
        "bullet_ids:",
    ]
    for bid in page.bullet_ids:
        front.append(f"  - {bid}")
    front.append(f"source_hash: {page.source_hash}")
    front.append(f"model_hash: {page.model_hash}")
    front.append(f"updated_at: {_to_rfc3339(page.updated_at)}")
    front.append("---")
    front.append("")

    body = [f"# {page.title}", "", page.summary.strip(), ""]
    body.append("## Backing Bullets")
    for bid in page.bullet_ids:
        snippet = (bullet_lookup.get(bid, "") or "").strip()
        snippet_one_line = snippet.replace("\n", " ")
        if len(snippet_one_line) > 120:
            snippet_one_line = snippet_one_line[:117] + "..."
        if snippet_one_line:
            body.append(f"- {bid}: {snippet_one_line}")
        else:
            body.append(f"- {bid}")
    body.append("")

    if related_titles:
        body.append("## Related Topics")
        for t in related_titles:
            body.append(f"- [[{t}]]")
        body.append("")

    return "\n".join(front + body)


def write_page_file(
    page: TopicPage,
    pages_dir: Path | str,
    bullet_lookup: dict[str, str] | None = None,
    related_titles: list[str] | None = None,
) -> Path:
    """Convenience wrapper: write *page* markdown to *pages_dir*/<slug>.md."""
    pages_path = Path(pages_dir)
    pages_path.mkdir(parents=True, exist_ok=True)
    target = pages_path / f"{page.slug}.md"
    target.write_text(
        render_markdown(page, bullet_lookup or {}, related_titles or []),
        encoding="utf-8",
    )
    return target


# ---------------------------------------------------------------------------
# Timestamp + YAML helpers
# ---------------------------------------------------------------------------


def _to_rfc3339(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _from_rfc3339(value: str) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now(timezone.utc)


def _yaml_escape(value: str) -> str:
    """Quote a YAML scalar if it contains characters that trip plain scalar rules."""
    if value == "":
        return '""'
    if any(ch in value for ch in (":", "#", "\n", "\"", "'", "[", "]", "{", "}", ",")):
        escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return f'"{escaped}"'
    return value


__all__ = [
    "CREATE_TABLE_SQL",
    "SqliteTopicStore",
    "TopicPage",
    "compute_drift_fraction",
    "compute_source_hash",
    "generate_topic_id",
    "render_markdown",
    "slugify",
    "write_page_file",
]
