# STORY-019: Curator 单元测试

**Epic:** EPIC-003 (Curator 语义去重引擎)
**Priority:** Must Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As a QA engineer
I want Curator fully tested
So that deduplication is reliable

---

## Description

### Background
Curator 的去重判断和合并策略直接影响记忆质量。需要全面测试确保阈值边界、合并行为、降级模式正确。

### Scope
**In scope:**
- CuratorEngine.curate() 去重判断测试
- cosine_similarity / text_similarity 数值测试
- MergeStrategy 合并结果测试
- 空输入 / 边界条件测试
- 降级模式（无 embedding）测试

**Out of scope:**
- 性能基准测试
- 与 IngestPipeline 的集成测试

---

## Acceptance Criteria

- [ ] 去重阈值边界测试：similarity=0.79 → Insert, similarity=0.80 → Merge
- [ ] cosine_similarity 精度测试（正交向量=0, 相同向量=1, 反向向量=-1→0）
- [ ] text_similarity 基础测试（完全相同=1.0, 无重叠=0.0）
- [ ] `keep_best` 策略结果验证
- [ ] `merge_content` 策略结果验证（content 合并、字段并集）
- [ ] 空 existing 列表 → 全部 Insert
- [ ] 空 candidates 列表 → 空结果
- [ ] embedding 为 None → 自动降级到 text_similarity
- [ ] 覆盖率 > 85%

---

## Technical Notes

### Test File
- `tests/unit/test_curator.py`

### Test Structure

```python
class TestCosineSimilarity:
    # identical vectors → 1.0
    # orthogonal vectors → 0.0
    # empty vectors → handle gracefully

class TestTextSimilarity:
    # identical strings → 1.0
    # no overlap → 0.0
    # partial overlap → 0.x

class TestCuratorEngine:
    # all insert (no existing)
    # all merge (all above threshold)
    # mixed insert/merge
    # threshold boundary (0.79 vs 0.80)
    # fallback to text_similarity

class TestKeepBestStrategy:
    # higher score wins
    # same score → longer content wins
    # metadata union

class TestMergeContentStrategy:
    # content concatenation
    # field union
    # updated_at refresh
```

---

## Dependencies

**Prerequisite Stories:**
- STORY-017: CuratorEngine
- STORY-018: MergeStrategy

---

## Definition of Done

- [ ] `tests/unit/test_curator.py` 全部通过
- [ ] 覆盖率 > 85%
- [ ] 所有 acceptance criteria 对应至少 1 个测试用例
- [ ] ruff check 通过

---

## Story Points Breakdown

- **CuratorEngine 测试:** 2 points
- **MergeStrategy 测试:** 1.5 points
- **Similarity 函数测试:** 0.5 points
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
