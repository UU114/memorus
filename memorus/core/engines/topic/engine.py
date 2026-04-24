"""TopicEngine — runs the TopicPage pipeline inside IdleOrchestrator.

Pipeline:

1. Load active bullets (caller supplies list of :class:`BulletNode`).
2. Cluster via :class:`TopicClusterer` (R096 edge graph → Louvain).
3. For each cluster:
   a. compute ``source_hash`` from sorted bullet ids
   b. if an existing page has the same hash → skip (no drift).
   c. else if hash differs but the bullet sets overlap ≥ (1 - drift) →
      regenerate: LLM summarise + upsert.
   d. else → new page: LLM summarise + insert.
4. For orphaned pages (their bullet_ids no longer produce any matching
   cluster), mark for review but **do not** delete on first pass — per
   the story's explicit "DO NOT physically delete orphans" rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from memorus.core.config import TopicsConfig
from memorus.core.engines.topic.clusterer import (
    BulletNode,
    Cluster,
    TopicClusterer,
)
from memorus.core.engines.topic.store import (
    SqliteTopicStore,
    TopicPage,
    compute_drift_fraction,
    compute_source_hash,
    generate_topic_id,
    slugify,
)
from memorus.core.engines.topic.summarizer import (
    FallbackSummarizer,
    Summarizer,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Report types
# ---------------------------------------------------------------------------


@dataclass
class TopicRunReport:
    """Outcome of one :meth:`TopicEngine.run` pass."""

    clusters_found: int = 0
    pages_created: int = 0
    pages_updated: int = 0
    pages_unchanged: int = 0
    orphan_candidates: list[str] = field(default_factory=list)
    llm_calls: int = 0
    fallback_calls: int = 0
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# TopicEngine
# ---------------------------------------------------------------------------


LoadBulletsFn = Callable[[], list[BulletNode]]


class TopicEngine:
    """Entry point invoked by :class:`IdleOrchestrator` after consolidate."""

    def __init__(
        self,
        config: TopicsConfig,
        clusterer: TopicClusterer,
        summarizer: Summarizer,
        store: SqliteTopicStore,
        load_bullets: LoadBulletsFn,
    ) -> None:
        self._config = config
        self._clusterer = clusterer
        self._summarizer = summarizer
        self._store = store
        self._load_bullets = load_bullets

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, *, force: bool = False) -> TopicRunReport:
        """Run one full cluster → summarize → persist pass.

        When ``force`` is ``True`` every cluster is regenerated — used by
        ``memorus topics regen --force``.
        """
        import time

        start = time.monotonic()
        report = TopicRunReport()

        if not self._config.enabled and not force:
            logger.debug("TopicEngine disabled (topics.enabled=False)")
            return report

        bullets = self._load_bullets()
        if not bullets:
            report.duration_seconds = time.monotonic() - start
            return report

        clusters = self._clusterer.cluster(bullets)
        report.clusters_found = len(clusters)
        if not clusters:
            report.duration_seconds = time.monotonic() - start
            return report

        bullet_lookup = {b.bullet_id: b.content for b in bullets}
        seen_source_hashes: set[str] = set()
        active_page_ids: set[str] = set()

        for cluster in clusters:
            sh = compute_source_hash(cluster.bullet_ids)
            seen_source_hashes.add(sh)

            existing = self._store.get_by_source_hash(sh)
            if existing is not None and not force:
                # No drift — skip.
                report.pages_unchanged += 1
                active_page_ids.add(existing.id)
                continue

            # Drift check against any page overlapping on ids.
            drift_target = self._find_drift_target(cluster)
            if drift_target is not None and not force:
                drift = compute_drift_fraction(
                    drift_target.bullet_ids, cluster.bullet_ids,
                )
                if drift < self._config.page_regen_drift_threshold:
                    report.pages_unchanged += 1
                    active_page_ids.add(drift_target.id)
                    continue

            # Otherwise summarise + upsert.
            page = self._summarise_cluster(cluster, bullet_lookup, sh, report)
            page_id = page.id

            if drift_target is not None:
                # Regenerate: keep the old id so we don't leak an orphan.
                page.id = drift_target.id
                page.created_at = drift_target.created_at
                report.pages_updated += 1
                page_id = drift_target.id
            else:
                report.pages_created += 1

            saved = self._store.upsert(page)
            active_page_ids.add(saved.id)
            self._write_md_safely(saved, bullet_lookup)
            _ = page_id  # keep linter quiet

        # Orphan detection — any persisted page whose id is not active
        # this pass. Do NOT physically delete; surface for review.
        try:
            for existing in self._store.list_all():
                if existing.id in active_page_ids:
                    continue
                if existing.source_hash in seen_source_hashes:
                    continue
                report.orphan_candidates.append(existing.id)
        except Exception as e:  # defensive — diagnostics must not fail run
            logger.warning("TopicEngine orphan scan failed: %s", e)

        report.duration_seconds = time.monotonic() - start
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summarise_cluster(
        self,
        cluster: Cluster,
        bullet_lookup: dict[str, str],
        source_hash: str,
        report: TopicRunReport,
    ) -> TopicPage:
        contents = [bullet_lookup.get(bid, "") for bid in cluster.bullet_ids]
        try:
            sr = self._summarizer.summarize(contents)
            if sr.model_hash.startswith("fallback:"):
                report.fallback_calls += 1
            else:
                report.llm_calls += 1
        except Exception as e:
            logger.warning("Summarizer failed (%s) — falling back", e)
            report.fallback_calls += 1
            sr = FallbackSummarizer().summarize(contents)

        topic_id = generate_topic_id(source_hash)
        slug = slugify(sr.title) or topic_id
        page = TopicPage(
            id=topic_id,
            slug=slug,
            title=sr.title,
            summary=sr.summary,
            bullet_ids=list(cluster.bullet_ids),
            source_hash=source_hash,
            model_hash=sr.model_hash,
        )
        return page

    def _find_drift_target(self, cluster: Cluster) -> Optional[TopicPage]:
        """Find the best-matching existing page by id overlap."""
        best: Optional[TopicPage] = None
        best_overlap = 0
        cluster_ids = set(cluster.bullet_ids)
        if not cluster_ids:
            return None
        try:
            candidates = self._store.list_all()
        except Exception as e:  # defensive
            logger.warning("TopicEngine list_all failed: %s", e)
            return None
        for page in candidates:
            overlap = len(cluster_ids.intersection(page.bullet_ids))
            if overlap > best_overlap:
                best_overlap = overlap
                best = page
        # Require at least half-overlap to consider it a drift target —
        # prevents two unrelated clusters from stomping each other.
        if best is None:
            return None
        needed = max(1, len(cluster_ids) // 2)
        if best_overlap < needed:
            return None
        return best

    def _write_md_safely(
        self,
        page: TopicPage,
        bullet_lookup: dict[str, str],
    ) -> None:
        try:
            self._store.write_md(page, bullet_lookup=bullet_lookup)
        except OSError as e:
            logger.warning("TopicEngine MD write failed for %s: %s", page.slug, e)


__all__ = ["LoadBulletsFn", "TopicEngine", "TopicRunReport"]
