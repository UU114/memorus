# STORY-070: 实现 Supersede 知识纠正流程

**Epic:** EPIC-012 (Team 治理与高级功能)
**Priority:** Could Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 7

---

## User Story

As a developer who found wrong team knowledge
I want to propose a correction
So that the entire team gets the updated version

---

## Description

### Background
当团队知识库中存在错误或过时的知识时，开发者在本地修正后，这个修正只存在于 Local Pool。本 Story 实现 Supersede 流程：Reflector 检测到 Local 纠正了来自 Team Pool 的知识后，提示用户是否提交 Supersede Proposal，经 Redactor 脱敏后上传到 Team Server。全员下次同步即可获得更新版本。

### Scope
**In scope:**
- Reflector 扩展接口：检测 Local Bullet 是否纠正了 Team Pool 来源的知识
- Supersede Proposal 生成（包含 `origin_id` + 新内容）
- 用户确认/拒绝 Supersede 提交
- 拒绝时仅 Local Pool 保留（Shadow Merge 覆盖）
- `priority: "urgent"` 级别支持
- Team Bullet 更新后通知用户重新评估本地覆盖

**Out of scope:**
- Server 端 Supersede 审核逻辑
- 自动合并冲突解决
- urgent Supersede 的即时推送（WebSocket，未来增强）

### User Flow
1. 用户在 Local 中修正了一条 Team 来源的知识
2. Reflector 检测到修正模式（语义相似度 ≥ 0.8 且内容不同）
3. 系统提示："检测到你修正了团队知识 [原内容摘要]，是否提交纠正提案？"
4. 用户选择"是" → 经 Redactor 脱敏 → 调用 `AceSyncClient.propose_supersede(origin_id, new_bullet)`
5. 用户选择"否" → 仅 Local 保留，Shadow Merge 自动覆盖 Team 版本
6. Team Bullet 更新后同步到本地，检测到本地存在旧版覆盖 → 通知用户重新评估

---

## Acceptance Criteria

- [ ] Reflector 检测到 Local 纠正了 Team Pool 来源的知识（语义相似度 ≥ 0.8 且内容差异显著）
- [ ] 提示用户是否提交 Supersede Proposal
- [ ] 拒绝提交 → 仅 Local Pool 保留（Shadow Merge 覆盖团队版本）
- [ ] 同意提交 → 经 Redactor 脱敏 → `propose_supersede(origin_id, new_bullet)` 上传
- [ ] Supersede Proposal 包含 `origin_id`（原 TeamBullet ID）和脱敏后新内容
- [ ] 支持 `priority: "urgent"` 级别
- [ ] Team Bullet 更新后，检测到本地存在旧版覆盖 → 通知用户重新评估
- [ ] 用户可标记永久忽略某个 Supersede 检测

---

## Technical Notes

### Components
- **File:** `memx/team/nominator.py` — 添加 Supersede 检测和提交逻辑
- **File:** `memx/team/merger.py` — Supersede 版本冲突检测
- **File:** `memx/team/cache_storage.py` — 更新后通知逻辑

### Supersede Detection Logic
```python
class SupersedeDetector:
    """Detect when a Local bullet supersedes a Team bullet."""

    def detect(self, local_bullet: Bullet, team_bullets: list[TeamBullet]) -> Optional[SupersedeCandidate]:
        """
        Check if local_bullet is a correction of any team bullet.

        Criteria:
        - Semantic similarity >= 0.8 (same topic)
        - Content differs significantly (not a duplicate)
        - Local bullet has higher instructivity_score or more recent
        """
        for team_bullet in team_bullets:
            similarity = compute_similarity(local_bullet, team_bullet)
            if similarity >= 0.8 and not is_content_identical(local_bullet, team_bullet):
                return SupersedeCandidate(
                    origin_id=team_bullet.id,
                    local_bullet=local_bullet,
                    team_bullet=team_bullet,
                    similarity=similarity,
                )
        return None
```

### Supersede Proposal Model
```python
@dataclass
class SupersedeProposal:
    origin_id: str          # original TeamBullet ID
    new_content: str        # redacted new content
    priority: str = "normal"  # "normal" | "urgent"
    reason: str = ""        # optional reason from user
```

### Post-Sync Update Detection
```python
def check_stale_overrides(local_bullets, updated_team_bullets):
    """After sync, check if any local overrides are now stale."""
    stale = []
    for team_bullet in updated_team_bullets:
        # Find local bullets that were overriding the old version
        local_override = find_local_override(local_bullets, team_bullet.origin_id)
        if local_override:
            stale.append((local_override, team_bullet))
    return stale
```

### Edge Cases
- 同一 Team Bullet 多人同时提交 Supersede → Server 处理冲突（客户端只提交）
- Supersede 提交后 Server 拒绝 → 显示拒绝原因，保留 Local 覆盖
- urgent Supersede 提交 → 标记 `priority: "urgent"`，Server 决定是否跳过 Staging
- 用户忽略某个 Supersede 检测 → 持久化到 `~/.ace/ignored_supersedes.json`

---

## Dependencies

**Prerequisite Stories:**
- STORY-068: AceSyncClient 推送接口（propose_supersede 方法）
- STORY-056: MultiPoolRetriever + Shadow Merge
- STORY-063: Redactor 脱敏引擎

**Blocked Stories:**
- STORY-073: 治理集成测试

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] Supersede 检测逻辑测试（相似度阈值、内容差异判定）
  - [ ] 用户确认/拒绝流程测试
  - [ ] Redactor 脱敏集成测试
  - [ ] propose_supersede 调用测试
  - [ ] urgent 优先级测试
  - [ ] 更新后通知逻辑测试
  - [ ] 永久忽略功能测试
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **Supersede 检测逻辑:** 2 points
- **提交流程（Redactor → Upload）:** 1 point
- **更新后通知 + 忽略列表:** 1 point
- **测试:** 1 point
- **Total:** 5 points

**Rationale:** 检测逻辑是核心复杂度，提交流程复用现有 Redactor 和 SyncClient 接口。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
