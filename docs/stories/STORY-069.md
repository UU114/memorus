# STORY-069: 实现三层审核治理逻辑（客户端）

**Epic:** EPIC-012 (Team 治理与高级功能)
**Priority:** Could Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 7

---

## User Story

As a team curator
I want different approval rules for different knowledge
So that sensitive knowledge gets proper review

---

## Description

### Background
Federation Mode MVP（Sprint 6）已完成 Nominator 提名和 AceSyncClient 推送。但目前所有提名一视同仁，缺乏差异化审核策略。本 Story 在客户端实现三层审核的分类标记和投票影响逻辑，使高质量知识快速入池、敏感知识得到人工把关。

### Scope
**In scope:**
- 客户端自动审批建议标记（score ≥ 90 + 非敏感标签 → `auto_approve`）
- 敏感标签识别与强制 Curator 审核标记（`curator_required`）
- CLI 投票命令 `ace upvote/downvote <id>`
- 投票结果影响本地缓存中 TeamBullet 的 `effective_score`
- 防积压通知逻辑（Staging 超 50 条或最早 Pending 超 7 天）
- 超 30 天未审核自动拒绝（客户端缓存清理）

**Out of scope:**
- Server 端审核逻辑（独立项目）
- 贡献者信誉系统（Server 端维护）
- 投票结果的 Server 端持久化同步（由 STORY-068 cast_vote 覆盖）

### User Flow
1. Nominator 检测到候选 Bullet 后，GovernanceClassifier 评估审核级别
2. score ≥ 90 且标签非敏感 → 标记 `review_level: "auto_approve"`
3. 标签含 `security`/`architecture`/`mandatory` → 标记 `review_level: "curator_required"`
4. 其他情况 → 标记 `review_level: "p2p_review"`
5. 用户通过 `ace upvote <id>` 或 `ace downvote <id>` 投票
6. 投票调整本地缓存中的 `effective_score`（upvote: +5, downvote: -10）
7. Staging 积压超阈值时，CLI 输出提醒 Curator

---

## Acceptance Criteria

- [ ] 客户端标记：score ≥ 90 + 非敏感标签 → `review_level: "auto_approve"` 建议
- [ ] 敏感标签（`security`, `architecture`, `mandatory`）→ 标记为 `curator_required`
- [ ] `ace upvote <id>` CLI 命令正确调用 `AceSyncClient.cast_vote(id, "up")`
- [ ] `ace downvote <id>` CLI 命令正确调用 `AceSyncClient.cast_vote(id, "down")`
- [ ] 投票结果影响 TeamBullet 的 `effective_score`（本地缓存中 upvote: +5, downvote: -10）
- [ ] 不采纳 AI 执行结果作为投票信号
- [ ] 防积压：Staging 超 50 条或最早 Pending 超 7 天 → 输出通知信息
- [ ] 超 30 天未审核的 Staging Bullet → 本地缓存标记为 `rejected`
- [ ] 自动审批入池后初始低权重（`effective_score` 乘以 0.5 衰减因子）

---

## Technical Notes

### Components
- **File:** `memx/team/nominator.py` — 扩展 GovernanceClassifier 逻辑
- **File:** `memx/team/cli.py` — 添加 `ace upvote/downvote` 命令
- **File:** `memx/team/cache_storage.py` — effective_score 调整逻辑

### Governance Classification Logic
```python
SENSITIVE_TAGS = {"security", "architecture", "mandatory"}

def classify_review_level(bullet: TeamBullet) -> str:
    """Classify the review level for a nominated bullet."""
    tags = set(bullet.metadata.tags)
    if tags & SENSITIVE_TAGS:
        return "curator_required"
    if bullet.metadata.instructivity_score >= 90:
        return "auto_approve"
    return "p2p_review"
```

### CLI Commands
```python
# ace upvote <bullet_id>
# ace downvote <bullet_id>
# Both call AceSyncClient.cast_vote() and update local cache
```

### Vote Impact on effective_score
```python
VOTE_WEIGHTS = {"up": +5, "down": -10}
AUTO_APPROVE_DECAY = 0.5  # initial low weight for auto-approved bullets

def apply_vote(bullet: TeamBullet, vote: str) -> None:
    if vote == "up":
        bullet.upvotes += 1
    else:
        bullet.downvotes += 1
    # Recalculate effective_score locally
    bullet.effective_score += VOTE_WEIGHTS[vote]
```

### Edge Cases
- 用户对同一 Bullet 重复投票 → 忽略（幂等）
- 投票的 Bullet 不在本地缓存 → 提示 `ace team sync` 后重试
- Staging 积压统计依赖 Server 返回的 Staging 列表 → 降级为不通知

---

## Dependencies

**Prerequisite Stories:**
- STORY-068: AceSyncClient 推送接口（cast_vote 方法）
- STORY-064: Nominator 提名流水线
- STORY-066: Team CLI 命令（CLI 框架）

**Blocked Stories:**
- STORY-073: 治理集成测试

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] GovernanceClassifier 分类逻辑测试（auto_approve / curator_required / p2p_review）
  - [ ] 敏感标签识别测试
  - [ ] 投票 effective_score 调整测试
  - [ ] 防积压通知阈值测试
  - [ ] 超 30 天自动拒绝测试
  - [ ] 重复投票幂等性测试
- [ ] CLI 命令集成测试（upvote/downvote）
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **GovernanceClassifier 逻辑:** 2 points
- **CLI upvote/downvote:** 1 point
- **effective_score 调整 + 积压逻辑:** 1 point
- **测试:** 1 point
- **Total:** 5 points

**Rationale:** 分类逻辑是核心，投票和积压管理相对简单，但涉及多个文件修改。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
