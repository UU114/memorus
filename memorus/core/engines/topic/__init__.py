"""TopicPage aggregation layer (STORY-R097).

Adds a layer ABOVE bullets: cluster related bullets via Louvain community
detection on the R096 edge graph, summarize each cluster into a narrative
TopicPage, expose pages as a first-class retrieval target.

Opt-in via ``config.topics.enabled = True`` — when disabled, every entry
point is a no-op and Generator behavior is byte-identical to R096.
"""

from __future__ import annotations

from memorus.core.engines.topic.clusterer import (
    BulletNode,
    Cluster,
    TopicClusterer,
    louvain_communities,
)
from memorus.core.engines.topic.engine import TopicEngine, TopicRunReport
from memorus.core.engines.topic.store import (
    SqliteTopicStore,
    TopicPage,
    compute_source_hash,
    render_markdown,
    write_page_file,
)
from memorus.core.engines.topic.summarizer import (
    FallbackSummarizer,
    LLMSummarizer,
    Summarizer,
    SummaryResult,
)

__all__ = [
    "BulletNode",
    "Cluster",
    "FallbackSummarizer",
    "LLMSummarizer",
    "SqliteTopicStore",
    "Summarizer",
    "SummaryResult",
    "TopicClusterer",
    "TopicEngine",
    "TopicPage",
    "TopicRunReport",
    "compute_source_hash",
    "louvain_communities",
    "render_markdown",
    "write_page_file",
]
