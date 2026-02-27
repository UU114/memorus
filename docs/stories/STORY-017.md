# STORY-017: 实现 CuratorEngine 核心去重逻辑

**Epic:** EPIC-003 (Curator 语义去重引擎)
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a MemX system
I want duplicate memories automatically detected
So that the Playbook stays clean and compact

---

## Description

### Background
当 Reflector 蒸馏出新的 CandidateBullet 后，Curator 需要将其与现有记忆对比。如果语义相似度超过阈值，标记为 Merge（交给 MergeStrategy 处理）；否则标记为 Insert（新增）。Curator 不直接操作存储层，只做决策。

### Scope
**In scope:**
- CuratorEngine 核心去重判断逻辑
- CurateResult 结果数据结构
- 相似度计算（cosine similarity）
- 阈值配置（CuratorConfig.similarity_threshold）
- 与 IngestPipeline 的集成接口

**Out of scope:**
- MergeStrategy 实现（STORY-018）
- 冲突检测（STORY-047）
- 实际 embedding 向量计算（使用传入的向量或文本相似度）

---

## Acceptance Criteria

- [ ] 候选 Bullet 与现有记忆列表计算 cosine similarity
- [ ] similarity >= 阈值（默认 0.8）→ 标记为 Merge，关联最相似的现有记忆
- [ ] similarity < 阈值 → 标记为 Insert
- [ ] 阈值通过 `CuratorConfig.similarity_threshold` 配置
- [ ] 返回 `CurateResult(to_add, to_merge, to_skip)`
- [ ] 空的现有记忆列表 → 所有候选都是 Insert
- [ ] 多个候选与同一记忆匹配 → 各自独立判断

---

## Technical Notes

### API Design

```python
@dataclass
class ExistingBullet:
    """Representation of an existing memory for comparison."""
    bullet_id: str
    content: str
    embedding: list[float] | None = None
    metadata: dict = field(default_factory=dict)

@dataclass
class MergeCandidate:
    """A candidate that should be merged with an existing bullet."""
    candidate: CandidateBullet
    existing: ExistingBullet
    similarity: float

@dataclass
class CurateResult:
    to_add: list[CandidateBullet]      # new memories to insert
    to_merge: list[MergeCandidate]     # candidates to merge with existing
    to_skip: list[CandidateBullet]     # explicitly skipped (e.g., exact duplicate)

class CuratorEngine:
    def __init__(self, config: CuratorConfig = None): ...

    def curate(
        self,
        candidates: list[CandidateBullet],
        existing: list[ExistingBullet],
    ) -> CurateResult: ...

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float: ...

    @staticmethod
    def text_similarity(a: str, b: str) -> float:
        """Fallback: simple text-based similarity when embeddings unavailable."""
```

### Components
- `memx/engines/curator/engine.py`

### Similarity Strategies
1. **Primary**: cosine similarity on embedding vectors（当向量可用时）
2. **Fallback**: 基于文本的简易相似度（token overlap ratio）— 用于无 embedding 的降级模式

### Edge Cases
- 候选内容为空 → skip
- 现有记忆 embedding 为 None → 使用 text_similarity 降级
- 相似度恰好等于阈值 → 视为 Merge
- 大量现有记忆（1000+）→ 需保证线性扫描性能可接受

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata 模型 ✓
- STORY-002: BulletFactory ✓

**Blocked Stories:**
- STORY-018: MergeStrategy
- STORY-019: Curator 测试

---

## Definition of Done

- [ ] `memx/engines/curator/engine.py` 实现 CuratorEngine
- [ ] cosine_similarity 和 text_similarity 函数实现
- [ ] CurateResult 数据结构完整
- [ ] 基础单元测试通过
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **CuratorEngine 主逻辑:** 2 points
- **相似度计算:** 1.5 points
- **数据结构 + 降级处理:** 1.5 points
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
