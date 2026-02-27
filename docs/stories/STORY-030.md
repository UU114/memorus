# STORY-030: 实现 RetrievalPipeline + RecallReinforcer

**Epic:** EPIC-005 (Generator 混合检索引擎)
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As MemXMemory.search()
I want a retrieval pipeline that automatically reinforces recalled memories
So that frequently used knowledge persists

---

## Description

### Background
RetrievalPipeline 是 search() 操作的顶层编排器，串联 GeneratorEngine → TokenBudgetTrimmer → RecallReinforcer。搜索结果返回后，异步更新被命中记忆的 recall_count，使其衰退速度减慢。

### Scope
**In scope:**
- RetrievalPipeline.search() 编排 Generator → Trimmer → Reinforce
- RecallReinforcer 异步更新 recall_count
- SearchResult 数据结构
- 对接 MemXMemory.search()（ace_enabled=True 时使用此管线）
- 集成 IngestPipeline 的 Curator 对接

**Out of scope:**
- Reranker
- 缓存层

---

## Acceptance Criteria

- [ ] `RetrievalPipeline.search()` 编排 Generator → Trimmer → Reinforce
- [ ] RecallReinforcer 异步更新 recall_count（不阻塞返回）
- [ ] 返回 `SearchResult(results, mode, total_candidates)`
- [ ] ace_enabled=True 时 `MemXMemory.search()` 调用此管线
- [ ] Generator 异常 → 降级到 mem0 原生 search
- [ ] Trimmer 异常 → 返回未裁剪结果
- [ ] Reinforcer 异常 → 仅记录 WARNING，不影响搜索结果返回

---

## Technical Notes

### API Design

```python
@dataclass
class SearchResult:
    results: list[ScoredBullet]
    mode: str                    # "full" | "degraded" | "fallback"
    total_candidates: int = 0

class RetrievalPipeline:
    def __init__(
        self,
        generator: GeneratorEngine,
        trimmer: TokenBudgetTrimmer | None = None,
        decay_engine: DecayEngine | None = None,
        mem0_search_fn: Callable | None = None,  # fallback
    ): ...

    def search(
        self,
        query: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 5,
        filters: dict | None = None,
    ) -> SearchResult: ...
```

### Components
- `memx/pipeline/retrieval.py`

### Architecture
```
RetrievalPipeline.search(query)
  ├── Load bullets from mem0 (get_all or filtered)
  ├── GeneratorEngine.search(query, bullets)
  ├── TokenBudgetTrimmer.trim(results)
  ├── [async] DecayEngine.reinforce(hit_ids)
  └── SearchResult
```

### Integration with Memory
需要在 `memx/memory.py` 中对接：
```python
# Memory.__init__() — 初始化 RetrievalPipeline
# Memory.search() — ace_enabled 时使用 RetrievalPipeline
```

### Also: IngestPipeline Curator 对接
在 `memx/pipeline/ingest.py` 中完善 Curator 集成（Step 3 的 placeholder）。

---

## Dependencies

**Prerequisite Stories:**
- STORY-028: GeneratorEngine
- STORY-029: TokenBudgetTrimmer
- STORY-021: Decay reinforce
- STORY-004: MemXMemory ✓

**Blocked Stories:**
- STORY-031: Generator 测试全覆盖（Sprint 3）

---

## Definition of Done

- [ ] `memx/pipeline/retrieval.py` 实现 RetrievalPipeline
- [ ] `memx/memory.py` 对接 RetrievalPipeline
- [ ] `memx/pipeline/ingest.py` 对接 CuratorEngine
- [ ] 搜索 + 降级 + 异常处理测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **RetrievalPipeline 编排:** 2 points
- **Memory 对接:** 1 point
- **IngestPipeline Curator 对接:** 1 point
- **测试:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
