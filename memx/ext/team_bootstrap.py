"""Conditional Team Layer bootstrap for MemX.

This module is the ONLY place that imports from ``memx.team``.
Core code (``memx.core``) never directly depends on Team — all
Team functionality is injected at startup via ``try_bootstrap_team()``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def try_bootstrap_team(
    memory: Any,
    config_path: Optional[str] = None,
) -> bool:
    """Conditionally inject Team Layer into Memory.

    Returns True if Team Layer was successfully bootstrapped.
    This is the ONLY module that imports from memx.team.

    Args:
        memory:      A Memory instance to inject Team components into.
        config_path: Explicit path to team_config.yaml.  If None, auto-discovers.

    Behavior:
        - Team package not installed  -> DEBUG log, return False
        - Team not configured/enabled -> DEBUG log, return False
        - Team enabled                -> inject MultiPoolRetriever, return True
        - Any exception               -> WARNING log, return False (Core unaffected)
    """
    try:
        from memx.team.config import TeamConfig, load_team_config  # noqa: F401
    except ImportError:
        logger.debug("Team Layer not installed, skipping bootstrap")
        return False

    try:
        # Load team config
        path_arg = Path(config_path) if config_path else None
        team_config: TeamConfig = load_team_config(path_arg)

        # Check if explicitly enabled or Git Fallback available
        git_fallback_available = _detect_git_fallback()

        if not team_config.enabled and not git_fallback_available:
            logger.debug("Team Layer not enabled and no Git Fallback, skipping")
            return False

        # Build and inject MultiPoolRetriever
        retriever = _build_multi_pool_retriever(
            memory, team_config, git_fallback_available
        )

        if retriever is not None and memory._retrieval_pipeline is not None:
            memory._retrieval_pipeline._team_retriever = retriever
            logger.info(
                "Team Layer bootstrapped (mode=%s)",
                "federation" if team_config.server_url else "git-fallback",
            )
        else:
            logger.debug(
                "Team Layer configured but retrieval pipeline not ready, "
                "storing retriever for deferred injection"
            )
            memory._team_retriever = retriever

        return True

    except Exception:
        logger.warning(
            "Team Layer bootstrap failed, falling back to Core only",
            exc_info=True,
        )
        return False


def _detect_git_fallback() -> bool:
    """Check if .ace/playbook.jsonl exists in current directory or parents.

    Walks up the directory tree until a .git directory is found (git root)
    or the filesystem root is reached.
    """
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".ace" / "playbook.jsonl").exists():
            return True
        if (parent / ".git").exists():
            break  # stop at git root
    return False


def _build_multi_pool_retriever(
    memory: Any,
    team_config: Any,
    git_fallback: bool,
) -> Any:
    """Build MultiPoolRetriever with appropriate Team storage backends.

    Returns the retriever instance, or None if no pools could be created.
    If MultiPoolRetriever is not yet implemented (STORY-056), falls back
    to returning the first available storage backend directly.
    """
    pools: list[tuple[str, Any]] = []

    # Git Fallback pool
    if git_fallback:
        try:
            from memx.team.git_storage import GitFallbackStorage

            git_storage = GitFallbackStorage()
            pools.append(("git_fallback", git_storage))
            logger.debug("Added Git Fallback storage pool")
        except ImportError:
            logger.warning("GitFallbackStorage not available")

    # Federation pool (remote server)
    if team_config.server_url:
        try:
            from memx.team.cache_storage import TeamCacheStorage  # type: ignore[import-not-found]

            cache_storage = TeamCacheStorage(team_config)
            pools.append(("federation", cache_storage))
            logger.debug("Added Federation cache storage pool")
        except ImportError:
            logger.debug(
                "TeamCacheStorage not available (STORY not yet implemented)"
            )

    if not pools:
        logger.debug("No Team pools available, skipping retriever build")
        return None

    # Try to use MultiPoolRetriever if available (STORY-056)
    try:
        from memx.team.merger import MultiPoolRetriever  # type: ignore[import-not-found]

        return MultiPoolRetriever(
            local_backend=getattr(memory, "_storage_backend", None),
            team_pools=pools,
            boost_config=team_config.layer_boost,
        )
    except ImportError:
        # MultiPoolRetriever not yet implemented — return pools directly
        # so that downstream code can iterate them manually
        logger.debug(
            "MultiPoolRetriever not available (STORY-056), "
            "storing raw pools for future use"
        )
        return pools
