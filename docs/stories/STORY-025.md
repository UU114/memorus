# STORY-025: 实现 MetadataMatcher (L3)

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
I want metadata-based matching
So that tool names and entity references boost relevance

---

## Description

### Background
L3 MetadataMatcher 利用 BulletMetadata 中的结构化字段（related_tools, key_entities, tags）进行匹配。当查询中出现工具名、实体名或标签时，给予额外加分。

### Scope
**In scope:**
- related_tools 前缀匹配（"git" 匹配 "git-rebase"）
- key_entities 前缀匹配
- tags 精确匹配
- 元数据匹配分数 0-10

**Out of scope:**
- 语义理解（不做 "版本控制" → "git" 的映射）

---

## Acceptance Criteria

- [ ] related_tools 前缀匹配："git" 匹配 tools 中含 "git", "git-rebase", "git-stash" 的记忆
- [ ] key_entities 前缀匹配："React" 匹配 entities 中含 "React", "ReactDOM" 的记忆
- [ ] tags 精确匹配："python" 匹配 tags 中含 "python" 的记忆
- [ ] 元数据匹配分数 0-10（每种匹配类型各贡献一部分分数）
- [ ] 大小写不敏感
- [ ] 空 metadata 字段 → 0 分

---

## Technical Notes

### API Design

```python
@dataclass
class MetadataInfo:
    """Metadata fields for a bullet, used in matching."""
    related_tools: list[str] = field(default_factory=list)
    key_entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

class MetadataMatcher:
    def __init__(
        self,
        tools_score: float = 4.0,
        entities_score: float = 3.0,
        tags_score: float = 3.0,
    ): ...

    def match(
        self,
        query: str,
        metadata: MetadataInfo,
    ) -> MatchResult: ...
```

### Components
- `memx/engines/generator/metadata_matcher.py`

### Scoring
- tools 命中: +4.0（最多 4.0）
- entities 命中: +3.0（最多 3.0）
- tags 命中: +3.0（最多 3.0）
- 总计最多 10.0

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata ✓

**Blocked Stories:**
- STORY-027: ScoreMerger

---

## Definition of Done

- [ ] `memx/engines/generator/metadata_matcher.py` 实现
- [ ] 单元测试覆盖三种匹配类型
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **三种匹配逻辑:** 2 points
- **测试:** 1 point
- **Total:** 3 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
