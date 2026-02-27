# STORY-029: 实现 TokenBudgetTrimmer

**Epic:** EPIC-005 (Generator 混合检索引擎)
**Priority:** Should Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 2

---

## User Story

As an API consumer
I want search results within token budget
So that context injection doesn't exceed LLM limits

---

## Description

### Background
搜索结果注入 LLM 上下文时需控制总 token 数，避免超出模型上下文窗口。TokenBudgetTrimmer 按 FinalScore 从高到低填充，直到达到 token 预算或条数上限。

### Scope
**In scope:**
- 按 FinalScore 排序后从高到低填充
- token 预算限制
- 条数限制
- 简单 token 估算（4 chars ≈ 1 token）

**Out of scope:**
- 精确 tokenizer（tiktoken 等）
- 截断单条内容

---

## Acceptance Criteria

- [ ] 按 FinalScore 从高到低填充，累计 token 不超过 budget（默认 2000）
- [ ] 条数不超过 max_results（默认 5）
- [ ] token 估算：`len(content) / 4`（中文可能需调整系数）
- [ ] 返回裁剪后的结果列表（保持排序）
- [ ] 空输入返回空列表
- [ ] 所有结果都超出预算 → 至少返回 1 条（最高分）

---

## Technical Notes

### API Design

```python
class TokenBudgetTrimmer:
    def __init__(
        self,
        token_budget: int = 2000,
        max_results: int = 5,
        chars_per_token: float = 4.0,
    ): ...

    def trim(
        self,
        results: list[ScoredBullet],
    ) -> list[ScoredBullet]: ...

    def estimate_tokens(self, text: str) -> int: ...
```

### Components
- `memx/utils/token_counter.py`

### Implementation Notes
- 中文字符 token 比更高（约 1.5 chars/token），可做简单检测
- 保底策略：至少返回 1 条结果

---

## Dependencies

**Prerequisite Stories:**
- STORY-028: GeneratorEngine（提供 ScoredBullet 输入）

**Blocked Stories:**
- STORY-030: RetrievalPipeline

---

## Definition of Done

- [ ] `memx/utils/token_counter.py` 实现 TokenBudgetTrimmer
- [ ] 预算裁剪测试
- [ ] 边界条件测试
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **Trimmer 逻辑:** 1.5 points
- **Token 估算:** 0.5 points
- **测试:** 1 point
- **Total:** 3 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
