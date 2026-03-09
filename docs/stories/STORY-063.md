# STORY-063: 实现 Redactor 团队脱敏引擎

**Epic:** EPIC-011 (Federation Mode MVP)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a privacy-conscious team member
I want my knowledge sanitized before sharing with the team
So that sensitive data never reaches the team pool

---

## Description

### Background
当 Local 知识被提名到 Team Pool 时，必须经过脱敏处理。Redactor 提供三层脱敏：L1 确定性规则脱敏、L2 用户审核确认、L3 可选 LLM 泛化。这确保敏感信息（路径、密钥、个人信息）不会泄露到团队层面。

### Scope
**In scope:**
- L1 确定性脱敏：复用 Core PrivacySanitizer + Team 扩展规则
- L2 用户审核：展示脱敏后内容给用户确认（不可跳过）
- L3 LLM 泛化（可选配置）
- 脱敏结果附加 `context_summary`

**Out of scope:**
- 提名流水线编排（STORY-064）
- Server 端脱敏
- LLM Provider 集成（仅定义接口，由 ext/ 注入）

### User Flow
1. Nominator 选中一个高质量 Local Bullet 准备提名
2. L1：Redactor 使用 PrivacySanitizer + custom_patterns 自动脱敏
3. L2：展示脱敏结果给用户，用户确认或编辑
4. L3（可选）：如配置 `llm_generalize=true`，LLM 进一步泛化
5. 输出：脱敏后的 TeamBullet + context_summary

---

## Acceptance Criteria

- [ ] L1 确定性脱敏：复用 Core PrivacySanitizer + Team 扩展规则（`custom_patterns`）
- [ ] L2 用户审核：展示脱敏后内容给用户确认（**不可跳过**）
- [ ] L3 LLM 泛化（可选）：`redactor.llm_generalize = true` 时启用
- [ ] 脱敏结果支持附加 `context_summary`
- [ ] 单元测试覆盖各种敏感信息格式
- [ ] custom_patterns 支持 regex 格式
- [ ] 脱敏前后内容 diff 可供用户审查

---

## Technical Notes

### Components
- **File:** `memx/team/redactor.py`
- **Dependencies:** `memx/core/privacy/sanitizer.py`（复用）

### RedactorConfig (from TeamConfig)
```python
class RedactorConfig(BaseModel):
    custom_patterns: list[str] = []  # Additional regex patterns
    llm_generalize: bool = False
    llm_provider: str | None = None  # e.g., "openai", "anthropic"
    show_diff: bool = True  # Show before/after diff to user
```

### Redactor Interface
```python
class Redactor:
    def __init__(self, config: RedactorConfig, sanitizer: PrivacySanitizer):
        ...

    def redact_l1(self, bullet: Bullet) -> RedactedResult:
        """L1: Deterministic sanitization"""
        ...

    async def redact_l3(self, result: RedactedResult) -> RedactedResult:
        """L3: Optional LLM generalization"""
        ...

    def prepare_for_review(self, result: RedactedResult) -> ReviewPayload:
        """L2: Format for user review"""
        ...

    def finalize(self, result: RedactedResult, user_edits: str | None) -> TeamBullet:
        """Apply user edits and produce final TeamBullet"""
        ...
```

### Team-Specific Patterns (examples)
- 项目路径: `/home/user/projects/secret-project/` → `[PROJECT_PATH]`
- 内部 URL: `https://internal.company.com/...` → `[INTERNAL_URL]`
- IP 地址: `192.168.x.x` → `[INTERNAL_IP]`
- 数据库连接串

### Edge Cases
- Bullet 内容全部被脱敏 → 警告用户，建议放弃提名
- 用户编辑后引入新的敏感信息 → L1 再次扫描
- LLM 泛化失败 → 回退到 L1 结果

---

## Dependencies

**Prerequisite Stories:**
- STORY-011: PrivacySanitizer（Core，已完成）

**Blocked Stories:**
- STORY-064: Nominator 提名流水线

**External Dependencies:**
- LLM Provider（L3 可选，通过 ext/ 注入）

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] L1 确定性脱敏测试（各种敏感信息格式）
  - [ ] custom_patterns 测试
  - [ ] L2 审核接口测试
  - [ ] L3 LLM 泛化测试（mock LLM）
  - [ ] context_summary 生成测试
  - [ ] 全部脱敏后警告测试
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **L1 脱敏逻辑:** 2 points
- **L2 审核接口:** 1 point
- **L3 LLM 泛化:** 1 point
- **测试:** 1 point
- **Total:** 5 points

**Rationale:** 三层脱敏逻辑 + 用户交互接口，需要仔细处理各种敏感信息格式。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
