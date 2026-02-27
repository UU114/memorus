# STORY-028: 组装 GeneratorEngine + 降级模式

**Epic:** EPIC-005 (Generator 混合检索引擎)
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a MemX system
I want a complete Generator engine with automatic degradation
So that search always works even without Embedding

---

## Description

### Background
GeneratorEngine 是混合检索的顶层编排器，协调 L1-L4 四个 Matcher 和 ScoreMerger 完成搜索。自动检测 Embedding 是否可用，支持 "full" 和 "degraded" 两种运行模式。

### Scope
**In scope:**
- 编排 L1→L2→L3→L4→ScoreMerger 完整搜索流程
- mode 属性：full / degraded 自动切换
- 降级时跳过 L4
- Embedding 恢复后自动切回 full
- 每个 Matcher 独立 try-catch

**Out of scope:**
- Reranker
- Token budget trimming（STORY-029）

---

## Acceptance Criteria

- [ ] `search()` 编排 L1→L2→L3→L4→ScoreMerger 完整流程
- [ ] `mode` 属性：Embedding 可用 → "full"，否则 → "degraded"
- [ ] 降级时跳过 L4，仅 L1-L3 + ScoreMerger（semantic_weight 自动归零）
- [ ] 降级事件记录 WARNING 日志（仅首次）
- [ ] Embedding 恢复后自动切回 "full"（每次 search 检查可用性）
- [ ] 单个 Matcher 异常不影响其他 Matcher 执行
- [ ] 返回按 FinalScore 降序排列的结果列表

---

## Technical Notes

### API Design

```python
class GeneratorEngine:
    def __init__(
        self,
        config: RetrievalConfig = None,
        vector_searcher: VectorSearcher | None = None,
    ): ...

    def search(
        self,
        query: str,
        bullets: list[BulletForSearch],  # content + metadata for matching
        limit: int = 20,
        filters: dict | None = None,
    ) -> list[ScoredBullet]: ...

    @property
    def mode(self) -> str:
        """'full' or 'degraded'"""
```

### Components
- `memx/engines/generator/engine.py`

### Architecture
```
GeneratorEngine.search(query, bullets)
  ├── L1: ExactMatcher.match_batch(query, contents)
  ├── L2: FuzzyMatcher.match_batch(query, contents)
  ├── L3: MetadataMatcher.match(query, metadatas)
  ├── L4: VectorSearcher.search(query)  [skip if degraded]
  └── ScoreMerger.merge(keyword_results, semantic_results, bullet_infos)
      → list[ScoredBullet] sorted by final_score desc
```

### Error Handling Pattern
每个 Matcher 独立 try-catch，失败返回空结果：
```python
try:
    l1_results = self._exact_matcher.match_batch(query, contents)
except Exception as e:
    logger.warning("L1 ExactMatcher failed: %s", e)
    l1_results = [MatchResult() for _ in contents]
```

---

## Dependencies

**Prerequisite Stories:**
- STORY-023~026: 四个 Matcher
- STORY-027: ScoreMerger

**Blocked Stories:**
- STORY-029: TokenBudgetTrimmer
- STORY-030: RetrievalPipeline

---

## Definition of Done

- [ ] `memx/engines/generator/engine.py` 实现 GeneratorEngine
- [ ] full 模式 + degraded 模式测试
- [ ] Matcher 故障隔离测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **编排逻辑:** 2 points
- **降级模式 + 自动恢复:** 1.5 points
- **错误隔离:** 1 point
- **测试:** 0.5 points
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
