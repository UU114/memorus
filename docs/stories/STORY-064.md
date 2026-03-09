# STORY-064: 实现 Nominator 提名流水线

**Epic:** EPIC-011 (Federation Mode MVP)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a developer
I want my high-quality local knowledge automatically suggested for team sharing
So that the team benefits from my discoveries

---

## Description

### Background
Nominator 是从 Local Pool 到 Team Pool 的知识提名管道。它自动检测高质量候选 Bullet，控制提名频率，编排 Redactor 脱敏流程，并最终通过 AceSyncClient 上传到 Team Pool。

### Scope
**In scope:**
- 自动检测候选 Bullet（基于 recall_count 和 instructivity_score）
- 频率控制（每会话最多 N 次提名提示）
- 静默模式（不弹窗，手动查看）
- Session 结束批量汇总
- Redactor → 用户确认 → 上传编排
- 永久忽略标记

**Out of scope:**
- Redactor 脱敏逻辑（STORY-063）
- AceSyncClient 推送接口（STORY-068）
- Server 端审核流程

### User Flow
1. Session 期间，Nominator 后台扫描 Local Pool 中的高质量 Bullet
2. 检测到候选：`recall_count > 3` 且 `instructivity_score > 0.7`
3. 弹窗提示用户（或静默记录）
4. 用户同意 → Redactor L1 脱敏 → 展示给用户审核（L2）
5. 用户确认 → 通过 AceSyncClient 上传
6. Session 结束时显示待提名汇总

---

## Acceptance Criteria

- [ ] 自动检测候选：`recall_count > min_recall_count` 且 `instructivity_score > min_score`（可配置）
- [ ] 频率控制：每会话最多 `max_prompts_per_session` 次弹窗（默认 1）
- [ ] 静默模式：`silent=true` 时不弹窗，`ace nominate list` 主动查看
- [ ] Session 结束时批量汇总待提名列表
- [ ] 编排 Redactor → 用户确认 → AceSyncClient.nominate_bullet 上传
- [ ] 用户可标记永久忽略特定 Bullet（`ignored_bullet_ids` 持久化）
- [ ] 已提名的 Bullet 不会重复提名

---

## Technical Notes

### Components
- **File:** `memx/team/nominator.py`
- **Config:** `AutoNominateConfig`（from TeamConfig）

### AutoNominateConfig
```python
class AutoNominateConfig(BaseModel):
    enabled: bool = True
    min_recall_count: int = 3
    min_score: float = 0.7
    max_prompts_per_session: int = 1
    silent: bool = False
```

### Nominator Interface
```python
class Nominator:
    def __init__(self, config, redactor, sync_client, storage):
        ...

    def scan_candidates(self, bullets: list[Bullet]) -> list[Bullet]:
        """Find nomination candidates from local pool"""
        ...

    async def nominate(self, bullet: Bullet) -> NominationResult:
        """Full pipeline: redact → review → upload"""
        ...

    def get_pending_nominations(self) -> list[Bullet]:
        """List candidates waiting for user action"""
        ...

    def ignore_bullet(self, bullet_id: str):
        """Permanently ignore a bullet for nomination"""
        ...

    def session_summary(self) -> NominationSummary:
        """End-of-session nomination summary"""
        ...
```

### Persistence
- `~/.ace/team_cache/{team_id}/nomination_state.json`
  - `ignored_bullet_ids: list[str]`
  - `nominated_bullet_ids: list[str]`
  - `session_prompt_count: int`

### Edge Cases
- Bullet 在提名过程中被 Curator 删除 → 跳过，清理候选列表
- 用户拒绝 L2 审核 → 标记为本次 Session 跳过（不永久忽略）
- AceSyncClient 上传失败 → 保留在待提名列表，下次重试
- 无 AceSyncClient（Git Fallback 模式）→ 提名功能不可用，静默跳过

---

## Dependencies

**Prerequisite Stories:**
- STORY-063: Redactor 脱敏引擎
- STORY-060: AceSyncClient 拉取接口

**Blocked Stories:**
- STORY-066: Team CLI 命令

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] 候选检测逻辑测试
  - [ ] 频率控制测试
  - [ ] 静默模式测试
  - [ ] 提名编排流程测试（mock Redactor + SyncClient）
  - [ ] 永久忽略测试
  - [ ] Session 汇总测试
  - [ ] 无 SyncClient 时跳过测试
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **候选检测 + 频率控制:** 2 points
- **提名编排流水线:** 2 points
- **测试:** 1 point
- **Total:** 5 points

**Rationale:** 多组件编排 + 状态管理 + 用户交互，中等复杂度。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
