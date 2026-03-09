# STORY-068: 实现 AceSyncClient — 推送接口

**Epic:** EPIC-012 (Team 治理与高级功能)
**Priority:** Could Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a team member
I want to vote on and correct team knowledge
So that the team knowledge base improves over time

---

## Description

### Background
AceSyncClient 的拉取接口（STORY-060）已实现只读通信。本 Story 扩展推送（write）接口，支持知识提名上传、投票和纠正提交。这是 Federation Mode 双向通信的关键组件。

### Scope
**In scope:**
- `nominate_bullet(sanitized_bullet)` 上传到 Staging
- `cast_vote(bullet_id, "up"|"down")` 投票
- `propose_supersede(origin_id, new_bullet)` 纠正
- `priority: "urgent"` 字段支持
- 网络失败可重试错误

**Out of scope:**
- Server 端审核逻辑
- 三层治理客户端逻辑（STORY-069）
- Supersede 检测逻辑（STORY-070）

### User Flow
1. Nominator 完成脱敏后调用 `nominate_bullet(bullet)` 上传
2. 用户通过 CLI 调用 `cast_vote(id, "up")` 投票
3. Reflector 检测到纠正后调用 `propose_supersede(origin, new)` 提交

---

## Acceptance Criteria

- [ ] `nominate_bullet(sanitized_bullet)` 上传到 Staging
- [ ] `cast_vote(bullet_id, "up"|"down")` 投票
- [ ] `propose_supersede(origin_id, new_bullet)` 提交纠正
- [ ] 支持 `priority: "urgent"` 字段
- [ ] 错误处理：网络失败时返回可重试错误（`RetryableError`）
- [ ] 幂等性：重复提名同一 Bullet 返回已存在提示

---

## Technical Notes

### Components
- **File:** `memx/team/sync_client.py` — 扩展现有 AceSyncClient

### API Endpoints (expected Server contract)
- `POST /api/v1/bullets/nominate`
  - Request: `{ bullet: TeamBullet, priority?: "normal"|"urgent" }`
  - Response: `{ id, status: "staging" }`
- `POST /api/v1/bullets/{id}/vote`
  - Request: `{ vote: "up"|"down" }`
  - Response: `{ id, upvotes, downvotes }`
- `POST /api/v1/bullets/supersede`
  - Request: `{ origin_id, new_bullet: TeamBullet, priority?: "urgent" }`
  - Response: `{ id, status: "pending_review" }`

### Error Handling
```python
class RetryableError(Exception):
    """Network failure, can retry"""
    def __init__(self, message, retry_after=None):
        ...

class PermanentError(Exception):
    """Auth failure, bad request, etc."""
    ...
```

### Edge Cases
- 上传内容超过 Server 大小限制 → PermanentError
- 重复提名 → Server 返回 409 Conflict → 返回已存在提示
- 投票的 Bullet 不存在 → 404 → PermanentError
- 网络超时 → RetryableError + 建议重试时间

---

## Dependencies

**Prerequisite Stories:**
- STORY-060: AceSyncClient 拉取接口

**Blocked Stories:**
- STORY-069: 三层审核治理
- STORY-070: Supersede 知识纠正

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] nominate_bullet 测试
  - [ ] cast_vote 测试
  - [ ] propose_supersede 测试
  - [ ] RetryableError 测试
  - [ ] PermanentError 测试
  - [ ] 幂等性测试
- [ ] Mock Server fixture 更新支持推送接口
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **三个推送接口:** 3 points
- **错误处理 + 重试:** 1 point
- **测试:** 1 point
- **Total:** 5 points

**Rationale:** 三个写接口 + 幂等性 + 错误分类，与拉取接口复杂度相当。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
