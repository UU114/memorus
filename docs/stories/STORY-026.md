# STORY-026: 实现 VectorSearcher (L4) 适配器

**Epic:** EPIC-005 (Generator 混合检索引擎)
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a search engine
I want semantic vector search from mem0
So that meaning-based retrieval complements keyword search

---

## Description

### Background
L4 VectorSearcher 是 Generator 的语义检索层，包装 mem0 的 VectorStore.search() 调用。当 Embedding 可用时提供语义相似度分数；不可用时静默返回空结果以支持降级模式。

### Scope
**In scope:**
- 调用 mem0 VectorStore.search()
- 返回归一化相似度分数 0-1
- Embedding 异常时返回空结果
- filters 透传

**Out of scope:**
- 自行实现 embedding（STORY-036 ONNXEmbedder）
- 向量索引管理

---

## Acceptance Criteria

- [ ] 调用 mem0 的 VectorStore.search() 获取语义搜索结果
- [ ] 返回归一化相似度分数 [0.0, 1.0]
- [ ] Embedding 不可用时返回空结果列表（不报错）
- [ ] 支持 filters 参数透传到 mem0
- [ ] 支持 limit 参数控制返回条数
- [ ] VectorStore 异常被捕获并记录 WARNING 日志

---

## Technical Notes

### API Design

```python
@dataclass
class VectorMatch:
    """A semantic search result."""
    bullet_id: str
    score: float           # [0.0, 1.0] normalized
    content: str = ""
    metadata: dict = field(default_factory=dict)

class VectorSearcher:
    def __init__(self, search_fn: Callable | None = None): ...

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: dict | None = None,
    ) -> list[VectorMatch]: ...

    @property
    def available(self) -> bool:
        """Whether the vector search backend is available."""
```

### Components
- `memx/engines/generator/vector_searcher.py`

### Implementation Notes
- search_fn 是一个回调，由 Memory 层注入（封装 mem0 的 search）
- 分数归一化：mem0 返回的 distance/similarity 转换为 [0, 1]
- 当 search_fn 为 None → available=False → search 返回空列表

---

## Dependencies

**Prerequisite Stories:**
- STORY-004: MemXMemory Decorator ✓

**Blocked Stories:**
- STORY-027: ScoreMerger

---

## Definition of Done

- [ ] `memx/engines/generator/vector_searcher.py` 实现
- [ ] 正常路径 + 降级路径测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **VectorSearcher 逻辑:** 1.5 points
- **降级 + 异常处理:** 1 point
- **测试:** 0.5 points
- **Total:** 3 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
