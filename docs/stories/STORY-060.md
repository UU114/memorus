# STORY-060: 实现 AceSyncClient — 拉取接口

**Epic:** EPIC-011 (Federation Mode MVP)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a team member
I want to sync team knowledge from the server
So that I have up-to-date team knowledge locally

---

## Description

### Background
AceSyncClient 是 Federation Mode 的网络层，负责与 ACE Sync Server 通信。本 Story 实现拉取（read-only）接口，包括增量索引拉取、Bullet 批量获取和 Tag Taxonomy 下载。

### Scope
**In scope:**
- `pull_index(since, tags)` 增量拉取 Bullet 索引
- `fetch_bullets(ids)` 获取完整 Bullet 数据（含向量）
- `pull_taxonomy()` 拉取 Tag Taxonomy
- HTTP 客户端（httpx，支持 async）
- API Key / Bearer Token 认证
- 超时和重试配置

**Out of scope:**
- 推送接口（STORY-068）
- 同步编排逻辑（STORY-061）
- 具体的 Server 实现

### User Flow
1. AceSyncClient 使用 TeamConfig 中的 server_url 和认证信息初始化
2. 调用 `pull_index(since=last_sync_time)` 获取变更列表
3. 根据变更列表调用 `fetch_bullets(ids)` 获取完整数据
4. 可选调用 `pull_taxonomy()` 更新标签体系

---

## Acceptance Criteria

- [ ] `pull_index(since, tags)` 增量拉取 Bullet 索引
- [ ] `fetch_bullets(ids)` 获取完整 Bullet 数据（含向量）
- [ ] `pull_taxonomy()` 拉取 Tag Taxonomy
- [ ] HTTP 客户端使用 httpx（支持 async）
- [ ] 支持 API Key 和 Bearer Token 认证
- [ ] 网络超时和重试配置（默认 timeout=30s，retries=3）
- [ ] Server 不可达时抛出可捕获异常（由调用方决定降级）
- [ ] 响应数据验证（Pydantic 模型）

---

## Technical Notes

### Components
- **File:** `memx/team/sync_client.py`
- **HTTP Client:** httpx.AsyncClient
- **Auth:** Header-based (API-Key or Bearer Token)

### API Endpoints (expected Server contract)
- `GET /api/v1/bullets/index?since={timestamp}&tags={tags}` — 增量索引
  - Response: `{ bullets: [{ id, updated_at, status }], cursor }`
- `POST /api/v1/bullets/fetch` — 批量获取
  - Request: `{ ids: [str] }`
  - Response: `{ bullets: [TeamBullet] }`
- `GET /api/v1/taxonomy` — Tag Taxonomy
  - Response: `{ tags: [{ name, aliases, parent }] }`

### Configuration
```python
class SyncClientConfig:
    server_url: str
    auth_token: str
    auth_type: Literal["api_key", "bearer"] = "bearer"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff: float = 1.0
```

### Edge Cases
- Server 返回 429 (Rate Limit) → 指数退避重试
- Server 返回 401 (Unauthorized) → 立即失败，清晰错误信息
- 大批量 fetch_bullets → 自动分批（每批 100 条）
- 网络中断 → 超时后抛出 `SyncConnectionError`

---

## Dependencies

**Prerequisite Stories:**
- STORY-049: TeamConfig 独立配置模型（已完成）

**Blocked Stories:**
- STORY-061: Team Cache 同步流程
- STORY-064: Nominator 提名流水线
- STORY-068: AceSyncClient 推送接口

**External Dependencies:**
- httpx PyPI 包

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Unit tests written and passing (≥80% coverage)
  - [ ] pull_index 正常响应测试
  - [ ] fetch_bullets 批量获取测试
  - [ ] pull_taxonomy 测试
  - [ ] 认证头正确设置测试
  - [ ] 超时和重试行为测试
  - [ ] Server 不可达异常测试
  - [ ] 429 Rate Limit 退避测试
- [ ] Mock Server fixture 可复用
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **HTTP Client 封装:** 2 points
- **API 接口实现:** 2 points
- **测试 + Mock:** 1 point
- **Total:** 5 points

**Rationale:** 三个 API 接口 + httpx 异步封装 + 认证/重试机制，中等复杂度。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
