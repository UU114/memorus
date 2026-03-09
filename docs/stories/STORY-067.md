# STORY-067: Federation MVP 端到端集成测试

**Epic:** EPIC-011 (Federation Mode MVP)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-03-08
**Sprint:** 6

---

## User Story

As a QA engineer
I want comprehensive tests for Federation Mode
So that sync, search, and nomination work reliably

---

## Description

### Background
Federation Mode 涉及多个组件的协作：AceSyncClient、TeamCacheStorage、Nominator、Redactor。本 Story 创建端到端集成测试，验证完整的同步、检索、提名流程，确保各组件正确协作。

### Scope
**In scope:**
- Mock AceSyncServer pytest fixture
- 增量同步测试
- 墓碑机制测试
- Local + Team Cache 合并检索测试
- 提名流程端到端测试
- Server 不可达降级测试
- 性能测试

**Out of scope:**
- 真实 Server 测试（无 Server 实现）
- 治理功能测试（STORY-073）

### Test Scenarios
1. **Sync Flow:** Client → pull_index → fetch_bullets → cache → search
2. **Tombstone:** Delete on server → sync → removed from cache
3. **Merged Search:** Local + Team Cache → MultiPoolRetriever → ranked results
4. **Nomination:** detect candidate → redact → review → upload
5. **Degradation:** Server down → local-only results
6. **Performance:** Team Cache search < 100ms

---

## Acceptance Criteria

- [ ] Mock AceSyncServer 实现（pytest fixture）
- [ ] 测试：增量同步正确拉取新 Bullet
- [ ] 测试：墓碑机制正确清理删除的 Bullet
- [ ] 测试：search 返回 Local + Team Cache 合并结果
- [ ] 测试：提名流程端到端（detect → redact → upload）
- [ ] 测试：Server 不可达时降级为纯 Local
- [ ] 性能测试：含 Team Cache 检索 < 100ms
- [ ] 所有测试可在 CI 中无外部依赖运行

---

## Technical Notes

### Components
- **Files:**
  - `tests/integration/test_team_retrieval.py`（检索集成测试）
  - `tests/unit/team/test_federation_e2e.py`（Federation 端到端）
  - `tests/performance/test_bench_team_search.py`（性能基准）
  - `tests/conftest.py`（Mock Server fixture）

### Mock AceSyncServer
```python
@pytest.fixture
def mock_ace_server():
    """In-memory mock ACE Sync Server"""
    server = MockAceServer()
    # Pre-populate with test data
    server.add_bullets([
        TeamBullet(id="team-1", content="Use connection pooling for DB", tags=["backend"]),
        TeamBullet(id="team-2", content="React.memo prevents re-renders", tags=["frontend"]),
    ])
    yield server

@pytest.fixture
def sync_client(mock_ace_server):
    """AceSyncClient connected to mock server"""
    return AceSyncClient(config=TeamConfig(server_url=mock_ace_server.url))
```

### Performance Test
```python
@pytest.mark.benchmark
def test_team_cache_search_latency(benchmark, team_cache_with_1000_bullets):
    result = benchmark(team_cache_with_1000_bullets.search, query="database optimization")
    assert benchmark.stats["mean"] < 0.1  # < 100ms
```

### Edge Cases to Test
- 空 Team Cache + 有 Local 结果 → 只返回 Local
- 有 Team Cache + 空 Local → 只返回 Team
- 同时有重复内容 → Shadow Merge 正确去重
- 提名上传失败 → 保留在待提名列表

---

## Dependencies

**Prerequisite Stories:**
- STORY-059~066: 所有 Sprint 6 Federation 组件

**Blocked Stories:** None

**External Dependencies:** None

---

## Definition of Done

- [ ] Code implemented and committed to feature branch
- [ ] Mock Server fixture 可被其他测试复用
- [ ] 所有集成测试通过
- [ ] 性能基准测试通过（< 100ms）
- [ ] CI 中可无外部依赖运行
- [ ] Code reviewed and approved
- [ ] Acceptance criteria validated (all ✓)

---

## Story Points Breakdown

- **Mock Server fixture:** 2 points
- **集成测试场景:** 2 points
- **性能测试:** 1 point
- **Total:** 5 points

**Rationale:** Mock Server 搭建 + 多场景覆盖 + 性能基准，测试工作量较大。

---

## Progress Tracking

**Status History:**
- 2026-03-08: Created

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
