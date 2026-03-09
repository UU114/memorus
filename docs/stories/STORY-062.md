# STORY-062: 实现墓碑机制 + Full Sync Check

**Epic:** EPIC-011 (Federation Mode MVP)
**Priority:** Should Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a developer
I want deleted team knowledge to be properly cleaned up
So that my cache doesn't contain stale entries

---

## Description

### Background
当 Team 知识在 Server 端被删除时，客户端缓存需要同步清理。墓碑机制通过 `status: tombstone` 标记实现软删除，保留 90 天后清理。当客户端长时间未同步（超过墓碑保留期）时，需要执行 Full Sync Check 来校验本地缓存完整性。

### Scope
**In scope:**
- 同步时处理 `status: tombstone` 记录
- 墓碑记录 90 天保留期后清理
- Full Sync Check（全量 ID 校验）
- 本地缓存中删除多余 Bullet

**Out of scope:**
- Server 端墓碑生成（Server 职责）
- 客户端删除上报（STORY-068 推送接口）

### User Flow
1. 增量同步时收到 `status: tombstone` 的 Bullet
2. 从缓存中标记该 Bullet 为已删除（软删除）
3. 90 天后彻底从缓存清理
4. 如果 `last_sync_timestamp` 早于 Server 的墓碑清理时间 → 触发 Full Sync Check
5. Full Sync Check：拉取 Server 全量 ID 列表，删除本地多余条目

---

## Acceptance Criteria

- [ ] 服务端删除 → 同步时接收 `status: tombstone` 记录
- [ ] 墓碑记录保留 90 天后清理
- [ ] `last_sync_timestamp` 早于墓碑清理时间 → 强制全量 ID 校验
- [ ] Full Sync Check 删除本地多余 Bullet
- [ ] 墓碑清理不影响缓存容量计算
- [ ] Full Sync Check 完成后重置 sync_state

---

## Technical Notes

### Components
- **File:** `memx/team/cache_storage.py` — tombstone 处理扩展

### Tombstone Flow
```python
def process_tombstones(self, index_entries):
    for entry in index_entries:
        if entry.status == "tombstone":
            self.mark_deleted(entry.id, deleted_at=entry.updated_at)
    self.cleanup_expired_tombstones(retention_days=90)

def full_sync_check(self, server_ids: set[str]):
    local_ids = set(self.bullets.keys())
    stale_ids = local_ids - server_ids
    for sid in stale_ids:
        self.remove(sid)
```

### Trigger Conditions for Full Sync Check
- `last_sync_timestamp` < `server_tombstone_cutoff`（Server 在响应中返回）
- 手动触发：`ace team sync --full`

### Edge Cases
- 墓碑记录本身在清理期间同步 → 忽略已清理的墓碑
- Full Sync Check 期间网络断开 → 中止，下次重试
- 大量删除（>500 条）→ 分批处理

---

## Dependencies

**Prerequisite Stories:**
- STORY-061: Team Cache 同步流程

**Blocked Stories:** None（被 STORY-067 集成测试覆盖）

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] 墓碑处理测试
  - [ ] 90 天清理测试（mock 时间）
  - [ ] Full Sync Check 测试
  - [ ] 容量计算不含墓碑测试
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **墓碑处理逻辑:** 1 point
- **Full Sync Check:** 1 point
- **测试:** 1 point
- **Total:** 3 points

**Rationale:** 逻辑清晰，主要是状态管理和 ID 集合运算，复杂度适中。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
