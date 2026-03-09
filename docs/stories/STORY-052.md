# STORY-052: 实现 ext/team_bootstrap.py 条件注入

**Epic:** EPIC-009 (Core/Team 解耦重构)
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a MemX developer
I want Team Layer to be injected conditionally at startup
So that Core code never directly depends on Team

---

## Description

### Background
Core/Team 解耦的关键胶水层。`team_bootstrap.py` 是唯一知道 Team Layer 存在的模块，它在 `Memory` 初始化时条件检测 Team 包是否安装、是否配置启用，然后通过组合模式将 Team 的 `MultiPoolRetriever` 注入到 `RetrievalPipeline` 中。

Core 代码永远不会 `import memx.team`，所有 Team 依赖通过此 bootstrap 模块条件注入。

### Scope
**In scope:**
- `try_bootstrap_team()` 函数实现
- Team 包安装检测（ImportError 捕获）
- TeamConfig 加载和检测
- Git Fallback 自动检测（`.ace/playbook.jsonl`）
- MultiPoolRetriever 注入到 RetrievalPipeline
- Memory 初始化流程集成

**Out of scope:**
- Team Layer 内部实现
- Federation 同步逻辑

### User Flow
1. 用户调用 `Memory()` 初始化
2. Memory 内部调用 `try_bootstrap_team()`
3. Team 未安装 → 静默跳过，纯 Core 模式
4. Team 已安装但未配置 → 静默跳过
5. Team 已安装且配置启用 → 注入 MultiPoolRetriever
6. 检测到 `.ace/playbook.jsonl` → 自动启用 Git Fallback

---

## Acceptance Criteria

- [ ] `try_bootstrap_team(memory, config_path)` 实现条件导入
- [ ] Team 包未安装时静默跳过（ImportError 捕获，DEBUG 日志）
- [ ] Team 未配置时静默跳过
- [ ] Team 启用时正确注入 MultiPoolRetriever 到 RetrievalPipeline
- [ ] Git Fallback 自动检测 `.ace/playbook.jsonl` 文件存在
- [ ] Memory 初始化流程调用 `try_bootstrap_team`
- [ ] 注入失败时不影响 Core 功能（异常捕获 + WARNING 日志）
- [ ] mypy --strict 通过

---

## Technical Notes

### Components
- `memx/ext/team_bootstrap.py` — 唯一知道 Team 存在的胶水层
- `memx/core/memory.py` — Memory.__init__ 调用 bootstrap

### Implementation

```python
# memx/ext/team_bootstrap.py
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def try_bootstrap_team(memory, config_path: str | None = None) -> bool:
    """Conditionally inject Team Layer into Memory.

    Returns True if Team Layer was successfully bootstrapped.
    This is the ONLY module that imports from memx.team.
    """
    try:
        from memx.team.config import TeamConfig, load_team_config
        from memx.team.merger import MultiPoolRetriever
    except ImportError:
        logger.debug("Team Layer not installed, skipping bootstrap")
        return False

    try:
        # Load team config
        team_config = load_team_config(config_path)

        # Check if explicitly enabled or Git Fallback available
        git_fallback_available = _detect_git_fallback()

        if not team_config.enabled and not git_fallback_available:
            logger.debug("Team Layer not enabled and no Git Fallback, skipping")
            return False

        # Build and inject MultiPoolRetriever
        retriever = _build_multi_pool_retriever(
            memory, team_config, git_fallback_available
        )
        memory._retrieval_pipeline.set_retriever(retriever)

        logger.info(
            "Team Layer bootstrapped (mode=%s)",
            "federation" if team_config.server_url else "git-fallback",
        )
        return True

    except Exception:
        logger.warning("Team Layer bootstrap failed, falling back to Core only", exc_info=True)
        return False


def _detect_git_fallback() -> bool:
    """Check if .ace/playbook.jsonl exists in current directory or parents."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".ace" / "playbook.jsonl").exists():
            return True
        if (parent / ".git").exists():
            break  # stop at git root
    return False


def _build_multi_pool_retriever(memory, team_config, git_fallback: bool):
    """Build MultiPoolRetriever with appropriate Team storage."""
    from memx.team.merger import MultiPoolRetriever

    pools = []

    if git_fallback:
        from memx.team.git_storage import GitFallbackStorage
        git_storage = GitFallbackStorage()
        pools.append(("git_fallback", git_storage))

    if team_config.server_url:
        from memx.team.cache_storage import TeamCacheStorage
        cache_storage = TeamCacheStorage(team_config)
        pools.append(("federation", cache_storage))

    return MultiPoolRetriever(
        local_backend=memory._storage_backend,
        team_pools=pools,
        boost_config=team_config.layer_boost,
    )
```

### Memory Integration

```python
# In Memory.__init__
from memx.ext.team_bootstrap import try_bootstrap_team

class Memory:
    def __init__(self, ...):
        # ... existing init ...
        # Try to bootstrap Team Layer (always safe to call)
        self._team_enabled = try_bootstrap_team(self)
```

### Edge Cases
- `.ace/playbook.jsonl` 存在但为空文件 → GitFallbackStorage 返回空结果
- TeamConfig 文件存在但格式错误 → WARNING 日志，跳过 Team
- 多个 `.ace/playbook.jsonl` 在不同父目录 → 使用最近的一个

---

## Dependencies

**Prerequisite Stories:**
- STORY-048: 重构 memx/ → memx/core/
- STORY-049: TeamConfig 独立配置

**Blocked Stories:**
- STORY-053: 解耦验证测试套件

---

## Definition of Done

- [ ] `try_bootstrap_team()` 函数实现
- [ ] Team 未安装时静默跳过（验证 ImportError 路径）
- [ ] Team 启用时正确注入 MultiPoolRetriever
- [ ] Git Fallback 自动检测逻辑正确
- [ ] Memory.__init__ 集成调用
- [ ] 单元测试覆盖所有分支（未安装、未配置、已启用、注入失败）
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **条件导入 + 检测逻辑:** 2 points
- **MultiPoolRetriever 构建 + 注入:** 2 points
- **测试覆盖:** 1 point
- **Total:** 5 points

**Rationale:** 胶水层代码量不大，但需要处理多种条件分支和异常场景，且是 Core/Team 解耦的关键节点。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
