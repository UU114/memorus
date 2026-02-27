# STORY-023: 实现 ExactMatcher (L1)

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
I want exact keyword matching
So that precise term hits get highest scores

---

## Description

### Background
Generator 检索引擎采用 4 层匹配架构。L1 ExactMatcher 是最基础的精确匹配层，对用户查询中的关键词在记忆内容中做全词匹配。命中精确关键词给予最高加分。

### Scope
**In scope:**
- 全词匹配检测（word boundary aware）
- 中英文双语支持
- 命中 +15 分（可配置）
- 返回匹配详情（命中词、位置）

**Out of scope:**
- 模糊匹配（STORY-024）
- 元数据匹配（STORY-025）
- 向量检索（STORY-026）

---

## Acceptance Criteria

- [ ] 英文全词匹配：word boundary 感知（"git" 不匹配 "digital"）
- [ ] 中文匹配：字符级子串匹配
- [ ] 命中 +15 分（默认值，可通过参数配置）
- [ ] 多关键词命中时累加分数
- [ ] 大小写不敏感匹配
- [ ] 支持中英文混合查询
- [ ] 性能：5000 条记忆 < 3ms

---

## Technical Notes

### API Design

```python
@dataclass
class MatchResult:
    """Result from a single matcher layer."""
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

class ExactMatcher:
    def __init__(self, hit_score: float = 15.0): ...

    def match(
        self,
        query: str,
        content: str,
    ) -> MatchResult: ...

    def match_batch(
        self,
        query: str,
        contents: list[str],
    ) -> list[MatchResult]: ...
```

### Components
- `memx/engines/generator/exact_matcher.py`

### Implementation Notes
- 英文分词：用空格 + 标点分割，过滤 stopwords
- 中文分词：字符级 n-gram（2字以上关键词做子串匹配）
- 使用 `re.compile` 预编译 word boundary pattern 提升性能

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata ✓

**Blocked Stories:**
- STORY-027: ScoreMerger

---

## Definition of Done

- [ ] `memx/engines/generator/exact_matcher.py` 实现
- [ ] 中英文匹配单元测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **匹配逻辑:** 1.5 points
- **中文支持:** 1 point
- **测试:** 0.5 points
- **Total:** 3 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
