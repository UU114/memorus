# STORY-065: 实现 subscribed_tags 订阅过滤

**Epic:** EPIC-011 (Federation Mode MVP)
**Priority:** Should Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a frontend developer
I want to subscribe to #frontend #react tags
So that I only get relevant team knowledge

---

## Description

### Background
团队知识库可能包含大量不同领域的知识。订阅过滤允许开发者按 tags 选择性同步，只缓存与自己工作相关的知识，减少噪音和缓存开销。

### Scope
**In scope:**
- `subscribed_tags` 配置项
- 同步时按 tags 过滤
- 未订阅标签时拉取全量
- 修改订阅后自动调整缓存
- team_id 隔离

**Out of scope:**
- Tag Taxonomy 归一化（STORY-071）
- Server 端 tag 管理

### User Flow
1. 用户在 TeamConfig 中配置 `subscribed_tags: ["frontend", "react", "typescript"]`
2. 同步时 AceSyncClient 将 tags 传给 Server 作为过滤条件
3. Server 只返回匹配 tags 的 Bullet
4. 未配置 subscribed_tags 时，拉取全量（受 cache_max_bullets 限制）
5. 用户修改 subscribed_tags 后，下次同步自动重新调整缓存

---

## Acceptance Criteria

- [ ] `subscribed_tags` 配置项支持标签列表
- [ ] 同步时按 tags 过滤请求（传给 AceSyncClient.pull_index）
- [ ] 未订阅任何标签时拉取全量（受 cache_max_bullets 限制）
- [ ] 修改订阅后下次同步自动调整缓存（清除不匹配的旧条目）
- [ ] 不同团队 (`team_id`) 的缓存路径隔离
- [ ] subscribed_tags 变更检测（与上次同步比较）

---

## Technical Notes

### Components
- **File:** `memx/team/cache_storage.py`（订阅变更检测）
- **File:** `memx/team/sync_client.py`（tags 参数传递）

### Implementation
```python
# In sync flow
def sync(self):
    current_tags = self.config.subscribed_tags
    last_tags = self.sync_state.get("subscribed_tags", [])

    if set(current_tags) != set(last_tags):
        # Tags changed - need to resync
        self.clear_cache()
        self.sync_state["last_sync_timestamp"] = None  # Force full sync

    index = self.client.pull_index(
        since=self.sync_state.get("last_sync_timestamp"),
        tags=current_tags or None  # None = all
    )
    ...
```

### sync_state.json Extension
```json
{
  "last_sync_timestamp": "...",
  "subscribed_tags": ["frontend", "react"],
  "...": "..."
}
```

### Edge Cases
- 订阅标签从 ["a", "b"] 变为 ["b", "c"] → 清除 "a" 相关缓存，补充 "c"
- 空 subscribed_tags 列表 vs None → 都表示全量
- 标签大小写不敏感（统一 lowercase）

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
  - [ ] tags 过滤同步测试
  - [ ] 全量拉取（无 tags）测试
  - [ ] 订阅变更检测测试
  - [ ] 缓存清理测试
  - [ ] team_id 隔离测试
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **过滤逻辑:** 1 point
- **变更检测 + 缓存调整:** 1 point
- **测试:** 1 point
- **Total:** 3 points

**Rationale:** 逻辑较简单，主要是参数传递和变更检测。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
