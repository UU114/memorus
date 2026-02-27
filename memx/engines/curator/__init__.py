"""Curator engine — semantic deduplication for MemX memories."""

from memx.engines.curator.engine import (
    CurateResult,
    CuratorEngine,
    ExistingBullet,
    MergeCandidate,
)
from memx.engines.curator.merger import (
    KeepBestStrategy,
    MergeContentStrategy,
    MergeResult,
    MergeStrategy,
    get_merge_strategy,
)

__all__ = [
    "CurateResult",
    "CuratorEngine",
    "ExistingBullet",
    "KeepBestStrategy",
    "MergeCandidate",
    "MergeContentStrategy",
    "MergeResult",
    "MergeStrategy",
    "get_merge_strategy",
]
