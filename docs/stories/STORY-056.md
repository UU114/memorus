# STORY-056: 实现 MultiPoolRetriever + Shadow Merge

**Epic:** EPIC-010 (Git Fallback 团队记忆)
**Priority:** Should Have
**Story Points:** 8
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 5

---

## User Story

As a developer
I want Local and Team search results merged intelligently
So that I get the best knowledge from both pools

---

## Description

### Background
MultiPoolRetriever 是 Team Memory 的检索核心。它实现 `StorageBackend` Protocol，组合 Local Pool 和一个或多个 Team Pool（Git Fallback / Federation Cache），并通过 Shadow Merge 算法将结果智能合并。

Shadow Merge 的核心逻辑：
- Local 结果加权 ×1.5（优先本地经验）
- Team 结果加权 ×1.0
- `enforcement: "mandatory"` 的 TeamBullet 跳过加权直接优先
- incompatible_tags 冲突检测：互斥标签保留高分，无互斥+高相似度互补保留
- 合并延迟目标 < 5ms（纯内存计算）

### Scope
**In scope:**
- MultiPoolRetriever 实现 StorageBackend Protocol
- 并行查询 Local + Team Pool
- Shadow Merge 算法
- incompatible_tags 冲突检测
- mandatory enforcement 优先处理
- Team Pool 查询失败时降级

**Out of scope:**
- Team Pool 的存储实现（STORY-054/059）
- 去重逻辑（STORY-057）

### User Flow
1. 用户调用 `memory.search("how to handle errors in Rust")`
2. MultiPoolRetriever 并行查询 Local Pool 和 Team Pool
3. Shadow Merge 合并两组结果
4. 用户获得来自两个知识池的最佳结果

---

## Acceptance Criteria

- [ ] `MultiPoolRetriever` 实现 `StorageBackend` Protocol
- [ ] 并行查询 Local + Team Pool（使用 concurrent.futures 或 asyncio）
- [ ] Shadow Merge: Local boost ×1.5, Team boost ×1.0（可通过 LayerBoostConfig 配置）
- [ ] `enforcement: "mandatory"` 的 TeamBullet 跳过加权直接优先
- [ ] Incompatible Tags 冲突判定：
  - 标签互斥（A.tags ∩ B.incompatible_tags ≠ ∅）→ 保留高分
  - 无互斥 + 相似度 ≥ 0.8 → 互补保留两条
- [ ] 兜底：无 incompatible_tags 的旧数据用相似度 ≥ 0.95 判定冲突（近似重复）
- [ ] Shadow Merge 延迟 < 5ms（纯内存计算）
- [ ] Team Pool 查询失败时静默降级，仅返回 Local 结果
- [ ] 合并结果按 boosted_score 降序排列
- [ ] mypy --strict 通过

---

## Technical Notes

### Components
- `memx/team/merger.py` — MultiPoolRetriever + ShadowMerger

### Data Structures

```python
@dataclass
class ScoredResult:
    """A search result with source and boosted score."""
    bullet: BulletMetadata | TeamBullet
    raw_score: float
    boosted_score: float
    source: str  # "local" | "team_git" | "team_cache"
    is_mandatory: bool = False
```

### Implementation Sketch

```python
# memx/team/merger.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from memx.team.types import TeamBullet

class MultiPoolRetriever:
    """Combines Local and Team pools with Shadow Merge."""

    def __init__(self, local_backend, team_pools, boost_config):
        self._local = local_backend
        self._team_pools = team_pools  # list of (name, StorageBackend)
        self._boost = boost_config

    def search(self, query: str, limit: int = 10, **kwargs):
        """Query all pools in parallel, then Shadow Merge."""
        results = []

        # Parallel query
        with ThreadPoolExecutor(max_workers=len(self._team_pools) + 1) as executor:
            futures = {}
            futures[executor.submit(self._local.search, query, limit * 2)] = "local"
            for name, pool in self._team_pools:
                futures[executor.submit(pool.search, query, limit * 2)] = name

            for future in as_completed(futures):
                source = futures[future]
                try:
                    pool_results = future.result(timeout=0.5)
                    results.extend(
                        self._score_results(pool_results, source)
                    )
                except Exception:
                    logger.warning("Pool %s query failed, skipping", source)

        # Shadow Merge
        merged = self._shadow_merge(results)
        return merged[:limit]

    def _score_results(self, results, source):
        """Apply layer boost to results."""
        scored = []
        for r in results:
            raw_score = getattr(r, "score", r.instructivity_score / 100)
            is_mandatory = (
                isinstance(r, TeamBullet) and r.enforcement == "mandatory"
            )
            boost = self._boost.local_boost if source == "local" else self._boost.team_boost
            boosted = raw_score * boost if not is_mandatory else 999.0

            scored.append(ScoredResult(
                bullet=r,
                raw_score=raw_score,
                boosted_score=boosted,
                source=source,
                is_mandatory=is_mandatory,
            ))
        return scored

    def _shadow_merge(self, results: list[ScoredResult]) -> list:
        """Merge results with conflict detection."""
        # Sort by boosted_score descending
        results.sort(key=lambda r: -r.boosted_score)

        merged = []
        seen_tags = {}  # tag -> best result

        for r in results:
            # Check incompatible_tags conflict
            tags = getattr(r.bullet, "tags", [])
            incomp = getattr(r.bullet, "incompatible_tags", [])

            conflict = False
            for existing in merged:
                existing_tags = getattr(existing.bullet, "tags", [])
                existing_incomp = getattr(existing.bullet, "incompatible_tags", [])

                # Mutual incompatibility check
                if self._tags_conflict(tags, existing_incomp) or \
                   self._tags_conflict(existing_tags, incomp):
                    # Keep higher score (already sorted)
                    conflict = True
                    break

                # Fallback: near-duplicate check (no incompatible_tags)
                if not incomp and not existing_incomp:
                    sim = self._content_similarity(r.bullet.content, existing.bullet.content)
                    if sim >= 0.95:
                        conflict = True  # near duplicate, keep first
                        break

            if not conflict or r.is_mandatory:
                merged.append(r)

        return [r.bullet for r in merged]

    @staticmethod
    def _tags_conflict(tags_a, incompatible_b) -> bool:
        return bool(set(tags_a) & set(incompatible_b))

    @staticmethod
    def _content_similarity(a: str, b: str) -> float:
        """Quick text similarity for dedup fallback."""
        # Simple Jaccard similarity on word sets
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
```

### Performance Considerations
- Shadow Merge 是纯内存计算，目标 < 5ms
- 并行查询使用 ThreadPoolExecutor，每个 Pool 超时 500ms
- 冲突检测是 O(n²)，但 n 通常 < 20（limit * 2 per pool）

### Edge Cases
- 所有 Team Pool 查询失败 → 返回纯 Local 结果
- Local Pool 为空 → 返回 Team 结果
- 所有 Pool 都为空 → 返回空列表
- mandatory Bullet 与 Local 冲突 → mandatory 优先
- 多个 mandatory Bullet 互相冲突 → 按 score 排序

---

## Dependencies

**Prerequisite Stories:**
- STORY-048: 重构 memx/ → memx/core/
- STORY-050: TeamBullet 数据模型

**Blocked Stories:**
- STORY-058: Git Fallback 集成测试
- STORY-072: Mandatory 逃生舱（Sprint 7）

---

## Definition of Done

- [ ] `MultiPoolRetriever` 实现 StorageBackend Protocol
- [ ] 并行查询 + 超时处理
- [ ] Shadow Merge 算法完整实现
- [ ] incompatible_tags 冲突检测
- [ ] mandatory enforcement 优先处理
- [ ] 降级逻辑（Team 失败时返回 Local）
- [ ] 性能验证：Shadow Merge < 5ms
- [ ] 单元测试覆盖：正常合并、冲突处理、降级、mandatory
- [ ] mypy --strict 通过
- [ ] ruff check 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **MultiPoolRetriever 框架:** 2 points
- **Shadow Merge 算法:** 3 points
- **incompatible_tags 冲突检测:** 1.5 points
- **降级逻辑 + 边界处理:** 0.5 points
- **测试:** 1 point
- **Total:** 8 points

**Rationale:** Sprint 5 中最大的单体 Story。Shadow Merge 的冲突判定逻辑（incompatible_tags + 相似度兜底 + mandatory 优先）有较高复杂度，需要充分的测试覆盖。

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
