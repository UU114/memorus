# STORY-007: mem0 兼容性测试套件

**Epic:** EPIC-001 — Bullet 数据模型与配置基础
**Priority:** Must Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a QA engineer
I want automated tests verifying mem0 API compatibility
So that we catch breaking changes early

---

## Description

### Background
MemX 的核心承诺是 mem0 API 100% 兼容。这个 Story 创建一个完整的兼容性测试套件，验证 `memx.Memory` 在 `ace_enabled=False` 模式下与 `mem0.Memory` 行为完全一致。这些测试将作为 CI 门禁，任何破坏兼容性的改动都会被拦截。

### Scope

**In scope:**
- 提取/编写 mem0 核心 API 测试用例
- 使用 `memx.Memory` 替代 `mem0.Memory` 运行所有测试
- ace_enabled=False 模式下 100% 通过
- 测试 add, search, get_all, get, update, delete, delete_all, history, reset
- config dict 兼容性测试

**Out of scope:**
- ace_enabled=True 模式测试（各引擎自带测试）
- mem0 内部实现细节测试（仅测试公开 API）
- 性能测试（STORY-045）

---

## Acceptance Criteria

- [ ] `tests/integration/test_mem0_compat.py` 包含 ≥ 15 个测试用例
- [ ] 覆盖所有 mem0 公开 API 方法：
  - [ ] `add()` 基础添加
  - [ ] `add()` 带 user_id, agent_id, metadata
  - [ ] `search()` 基础搜索
  - [ ] `search()` 带 user_id, limit, filters
  - [ ] `get_all()` 获取所有记忆
  - [ ] `get()` 获取单条
  - [ ] `update()` 更新记忆
  - [ ] `delete()` 删除单条
  - [ ] `delete_all()` 删除全部
  - [ ] `history()` 变更历史
  - [ ] `reset()` 重置
- [ ] 所有测试使用 `memx.Memory` 而非 `mem0.Memory`
- [ ] ace_enabled=False 模式下 100% 通过
- [ ] config dict 兼容测试：mem0 格式 config 传入 memx.Memory 正常工作
- [ ] 返回值格式与 mem0 一致（dict 结构、字段名称）

---

## Technical Notes

### File Location
`tests/integration/test_mem0_compat.py`

### Testing Strategy

```python
import pytest
from memx import Memory

@pytest.fixture
def memory():
    """Create MemX Memory with ace_enabled=False (default)."""
    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {"collection_name": "test", "host": "localhost", "port": 6333}
        }
    }
    m = Memory(config=config)
    yield m
    m.reset()

# Alternative: use in-memory vector store for faster tests
@pytest.fixture
def memory_inmem():
    """Create MemX Memory with in-memory store."""
    m = Memory()  # defaults to in-memory
    yield m


class TestMem0ApiCompat:
    def test_add_basic(self, memory):
        result = memory.add("Python is great", user_id="test_user")
        assert "results" in result or "id" in result

    def test_add_with_metadata(self, memory):
        result = memory.add(
            "Use pytest for testing",
            user_id="test_user",
            metadata={"category": "tools"}
        )
        assert result is not None

    def test_search_basic(self, memory):
        memory.add("Redis is a fast cache", user_id="u1")
        results = memory.search("fast cache", user_id="u1")
        assert "results" in results
        assert len(results["results"]) > 0

    def test_get_all(self, memory):
        memory.add("Test memory", user_id="u1")
        all_memories = memory.get_all(user_id="u1")
        assert "results" in all_memories

    # ... more tests
```

### Key Considerations
- 需要一个可用的 VectorStore 后端（推荐 in-memory 或本地 Qdrant）
- 可使用 `pytest.mark.integration` 标记，允许单独运行
- 返回值结构验证需参考 mem0 实际返回格式
- 某些 mem0 功能需要 LLM（如 add 时的记忆提取）→ 可 mock LLM 或使用简单配置

### Edge Cases
- 空搜索查询 → 应返回空结果而非错误
- 删除不存在的 ID → mem0 行为是什么？MemX 应该一致
- 重复 add 相同内容 → 行为应与 mem0 一致

---

## Dependencies

**Prerequisite Stories:**
- STORY-004: MemXMemory
- STORY-006: 项目骨架

**Blocked Stories:** None directly（但作为 CI 门禁影响所有后续 Story）

**External Dependencies:**
- 可用的 VectorStore 后端（测试用）
- 可能需要 mock LLM provider

---

## Definition of Done

- [ ] `tests/integration/test_mem0_compat.py` 实现
- [ ] ≥ 15 个测试用例
- [ ] ace_enabled=False 模式下 100% 通过
- [ ] CI 中可作为必过门禁运行
- [ ] `ruff check` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Test case design:** 1 point
- **Test implementation:** 2 points
- **Fixture + CI config:** 1 point
- **Total:** 4 points

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
