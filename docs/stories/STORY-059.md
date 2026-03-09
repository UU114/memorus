# STORY-059: 实现 TeamCacheStorage

**Epic:** EPIC-011 (Federation Mode MVP)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a team member with Federation Mode
I want team knowledge cached locally
So that search is fast without remote calls

---

## Description

### Background
Federation Mode 需要一个本地缓存层来存储从 ACE Sync Server 同步的 Team 知识。TeamCacheStorage 是 Federation 的核心存储组件，实现 `StorageBackend` Protocol，提供向量检索和关键词检索能力。

### Scope
**In scope:**
- 实现 `StorageBackend` Protocol 的完整接口
- 本地缓存目录 `~/.ace/team_cache/{team_id}/`
- 向量检索（内存索引）+ 关键词检索
- 缓存容量上限管理（`cache_max_bullets`，默认 2000）
- 按 `effective_score` 保留 Top-N

**Out of scope:**
- 同步逻辑（STORY-061）
- 墓碑机制（STORY-062）
- 订阅过滤（STORY-065）

### User Flow
1. Memory 初始化时，team_bootstrap 检测到 Federation 配置
2. 创建 TeamCacheStorage 实例，指定 team_id
3. 加载缓存目录中的 TeamBullet 数据到内存
4. 构建内存向量索引
5. 检索时同时支持向量相似度和关键词匹配
6. 缓存为空时返回空结果（不报错）

---

## Acceptance Criteria

- [ ] `TeamCacheStorage` 实现 `StorageBackend` Protocol
- [ ] 缓存存储在 `~/.ace/team_cache/{team_id}/`
- [ ] 支持向量检索（内存索引）+ 关键词检索
- [ ] 缓存上限 `cache_max_bullets`（默认 2000），按 effective_score 保留 Top-N
- [ ] 缓存为空时返回空结果（不报错）
- [ ] 缓存持久化到磁盘（JSON 格式）
- [ ] 加载时自动重建内存索引

---

## Technical Notes

### Components
- **File:** `memx/team/cache_storage.py`
- **Protocol:** 实现 `StorageBackend` 接口（search, add, update, delete）
- **Storage:** `~/.ace/team_cache/{team_id}/bullets.json` + `vectors.npy`

### Key Implementation Details
- 内存中维护 `dict[str, TeamBullet]` 和 numpy 向量矩阵
- search 方法支持 `query_vector` 和 `query_text` 两种模式
- 容量超限时按 `effective_score` 排序，淘汰低分条目
- 线程安全：使用 `threading.Lock` 保护写操作

### Edge Cases
- team_id 包含特殊字符时路径安全处理
- 缓存文件损坏时重建（WARNING 日志）
- 磁盘空间不足时的错误处理

---

## Dependencies

**Prerequisite Stories:**
- STORY-048: memx/core/ 包结构（已完成）
- STORY-050: TeamBullet 数据模型（已完成）

**Blocked Stories:**
- STORY-061: Team Cache 同步流程
- STORY-062: 墓碑机制
- STORY-065: subscribed_tags 过滤
- STORY-066: Team CLI 命令

**External Dependencies:**
- numpy（向量索引）

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] StorageBackend Protocol 合规测试
  - [ ] 向量检索准确性测试
  - [ ] 关键词检索测试
  - [ ] 容量上限淘汰测试
  - [ ] 空缓存行为测试
  - [ ] 缓存持久化/加载测试
- [ ] Integration tests passing
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **Storage 实现:** 3 points
- **索引构建:** 1 point
- **测试:** 1 point
- **Total:** 5 points

**Rationale:** 需要实现完整的 StorageBackend Protocol，包含向量索引和持久化，但无网络逻辑。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
