# STORY-024: 实现 FuzzyMatcher (L2)

**Epic:** EPIC-005 (Generator 混合检索引擎)
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a search engine
I want fuzzy matching for approximate queries
So that typos and variants still find relevant memories

---

## Description

### Background
L2 FuzzyMatcher 处理用户查询中的近似匹配场景，如拼写变体、词干形式差异等。中文使用 2-gram 分词匹配，英文使用简化词干化匹配。

### Scope
**In scope:**
- 中文 2-gram 分词匹配
- 英文词干化匹配（简化 Porter stemmer 或后缀剥离）
- 模糊匹配分数 0-10
- 文本预处理工具函数

**Out of scope:**
- jieba 等重量级分词库（可未来替换）
- 编辑距离匹配（复杂度过高）

---

## Acceptance Criteria

- [ ] 中文 2-gram 分词匹配（"数据库" → ["数据", "据库"]，匹配含"数据"的记忆）
- [ ] 英文词干化："running"/"runs"/"ran" 可匹配 "run"
- [ ] 模糊匹配分数 0-10（按命中率 = 命中 gram 数 / 查询 gram 总数 × 10）
- [ ] 性能：5000 条 < 5ms
- [ ] 空查询返回 0 分
- [ ] 纯标点/特殊字符查询返回 0 分

---

## Technical Notes

### API Design

```python
class FuzzyMatcher:
    def __init__(self, max_score: float = 10.0): ...

    def match(self, query: str, content: str) -> MatchResult: ...
    def match_batch(self, query: str, contents: list[str]) -> list[MatchResult]: ...

# memx/utils/text_processing.py
def tokenize_chinese(text: str) -> list[str]:
    """Split Chinese text into 2-grams."""

def stem_english(word: str) -> str:
    """Simple English stemming (suffix stripping)."""

def extract_tokens(text: str) -> list[str]:
    """Extract mixed Chinese/English tokens from text."""
```

### Components
- `memx/engines/generator/fuzzy_matcher.py`
- `memx/utils/text_processing.py` — 文本处理工具函数

### Implementation Notes
- 英文词干化：简化版后缀剥离（-ing, -ed, -s, -tion→-te, -ly）
- 中文 2-gram：滑动窗口生成 bigrams
- 分数公式：`min(matched_grams / total_query_grams × max_score, max_score)`

---

## Dependencies

**Prerequisite Stories:**
- STORY-001: BulletMetadata ✓

**Blocked Stories:**
- STORY-027: ScoreMerger

---

## Definition of Done

- [ ] `memx/engines/generator/fuzzy_matcher.py` 实现
- [ ] `memx/utils/text_processing.py` 文本处理工具实现
- [ ] 中英文模糊匹配测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **中文 2-gram:** 1.5 points
- **英文词干化:** 1.5 points
- **text_processing 工具:** 1 point
- **测试:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
