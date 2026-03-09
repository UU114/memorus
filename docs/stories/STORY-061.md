# STORY-061: 实现 Team Cache 同步流程

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
I want team cache to sync automatically at session start
So that I always have recent team knowledge

---

## Description

### Background
Team Cache 需要与 ACE Sync Server 保持同步。本 Story 实现同步编排逻辑：Session 启动时后台异步增量拉取，定时刷新，同步状态持久化。同步过程不阻塞用户操作。

### Scope
**In scope:**
- Session Start 后台异步增量同步
- 定时刷新（默认每 1 小时）
- 同步状态持久化（`sync_state.json`）
- 首次同步全量拉取
- Server 不可达时降级使用缓存

**Out of scope:**
- 墓碑机制（STORY-062）
- 订阅过滤（STORY-065）
- 推送/提名（STORY-064, STORY-068）

### User Flow
1. Memory 初始化 → team_bootstrap 创建 TeamCacheStorage
2. 检查 `sync_state.json` 获取 `last_sync_timestamp`
3. 后台线程启动增量同步（`pull_index(since=last_sync_timestamp)`）
4. 增量数据写入 TeamCacheStorage
5. 更新 `sync_state.json`
6. 定时器每 1 小时触发下一次增量同步
7. 用户操作期间同步在后台静默进行

---

## Acceptance Criteria

- [ ] Session Start 时后台异步拉取增量（`updated_at` 差分）
- [ ] 同步不阻塞用户操作（后台线程/asyncio.create_task）
- [ ] 定时刷新（默认每 1 小时，可配置 `cache_ttl_minutes`）
- [ ] 同步状态持久化到 `sync_state.json`（last_sync_timestamp）
- [ ] 首次同步为全量拉取（since=None）
- [ ] Server 不可达时使用上次缓存快照（WARNING 日志）
- [ ] 同步期间检索仍可用（读旧缓存）
- [ ] 同步完成后新数据立即可检索

---

## Technical Notes

### Components
- **File:** `memx/team/cache_storage.py` — sync 方法扩展
- **State File:** `~/.ace/team_cache/{team_id}/sync_state.json`

### sync_state.json Format
```json
{
  "last_sync_timestamp": "2026-03-08T12:00:00Z",
  "last_sync_status": "success",
  "total_bullets": 1234,
  "sync_count": 42
}
```

### Sync Flow (Pseudocode)
```python
async def sync(self):
    state = load_sync_state()
    try:
        index = await client.pull_index(since=state.last_sync_timestamp)
        new_ids = [b.id for b in index.bullets if b.status != "tombstone"]
        if new_ids:
            bullets = await client.fetch_bullets(new_ids)
            self.upsert_bullets(bullets)
        self.enforce_capacity_limit()
        save_sync_state(timestamp=now())
    except SyncConnectionError:
        logger.warning("Server unreachable, using cached data")
```

### Threading Model
- 使用 `threading.Thread(daemon=True)` 或 `asyncio.create_task`
- 同步写入使用 Lock 保护
- 定时器使用 `threading.Timer` 或 `asyncio.sleep` loop

### Edge Cases
- 同步中途网络断开 → 部分数据已写入，下次增量从断点继续
- 多个 Memory 实例共享同一 team_id → 文件锁或 last-write-wins
- sync_state.json 损坏 → 重建（全量同步）

---

## Dependencies

**Prerequisite Stories:**
- STORY-059: TeamCacheStorage（缓存存储）
- STORY-060: AceSyncClient 拉取接口（网络层）

**Blocked Stories:**
- STORY-062: 墓碑机制
- STORY-065: subscribed_tags 过滤

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] 增量同步流程测试
  - [ ] 全量同步（首次）测试
  - [ ] sync_state.json 持久化测试
  - [ ] Server 不可达降级测试
  - [ ] 定时刷新触发测试
  - [ ] 同步期间读操作不阻塞测试
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **同步编排逻辑:** 2 points
- **后台线程 + 定时器:** 2 points
- **测试:** 1 point
- **Total:** 5 points

**Rationale:** 异步后台同步 + 状态持久化 + 竞态条件处理，中等偏高复杂度。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
