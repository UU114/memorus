# STORY-005: 实现 AsyncMemXMemory 异步包装类

**Epic:** EPIC-001 — Bullet 数据模型与配置基础
**Priority:** Must Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a developer using async patterns
I want async version of MemX Memory
So that I can integrate with async frameworks

---

## Description

### Background
mem0 提供了 `AsyncMemory` 类，用于 async/await 模式下的集成。MemX 需要对应的 `AsyncMemory` 包装类，行为与同步版本 `Memory` 完全一致——ace_enabled 开关逻辑相同，API 签名相同（但所有方法为 async）。

### Scope

**In scope:**
- `memx.AsyncMemory` 类（包装 `mem0.AsyncMemory`）
- 所有方法的 async 版本
- ace_enabled 开关逻辑（与同步版一致）
- 异步测试

**Out of scope:**
- 同步 Memory 实现（STORY-004）
- Pipeline 的异步版本（Pipeline 本身已设计为 async）

---

## Acceptance Criteria

- [ ] `memx.AsyncMemory` 包装 `mem0.AsyncMemory`
- [ ] 所有方法为 async：add, search, get_all, get, update, delete, delete_all, history, reset
- [ ] ace_enabled=False 时直接代理到 mem0 AsyncMemory
- [ ] ace_enabled=True 时调用 Pipeline（或 fallback 到代理）
- [ ] 与同步版 Memory 共享相同的 MemXConfig 解析逻辑
- [ ] 异步测试覆盖核心路径（add, search 的代理模式）
- [ ] `from memx import AsyncMemory` 工作

---

## Technical Notes

### File Location
`memx/async_memory.py`

### Implementation Sketch

```python
from mem0 import AsyncMemory as Mem0AsyncMemory
from memx.config import MemXConfig

class AsyncMemory:
    def __init__(self, config=None):
        self._config = MemXConfig.from_dict(config or {})
        self._mem0 = Mem0AsyncMemory(config=self._config.to_mem0_config())
        self._ingest_pipeline = None
        self._retrieval_pipeline = None

    async def add(self, messages, user_id=None, agent_id=None,
                  run_id=None, metadata=None, filters=None,
                  prompt=None, **kwargs):
        if not self._config.ace_enabled or self._ingest_pipeline is None:
            return await self._mem0.add(messages, user_id=user_id, ...)
        # ACE async path
        ...

    async def search(self, query, user_id=None, agent_id=None,
                     run_id=None, limit=100, filters=None, **kwargs):
        if not self._config.ace_enabled or self._retrieval_pipeline is None:
            return await self._mem0.search(query, user_id=user_id, ...)
        ...

    # async proxy methods
    async def get_all(self, **kwargs): return await self._mem0.get_all(**kwargs)
    async def get(self, memory_id): return await self._mem0.get(memory_id)
    # ... etc
```

### Key Design
- 内部逻辑与 `Memory` 完全对称，仅加 `async/await`
- 共享 `MemXConfig` 和 Pipeline 初始化逻辑
- 可考虑提取公共基类或 mixin 减少重复

### Edge Cases
- asyncio event loop 未运行时构造 AsyncMemory → 应在 __init__ 中不做 async 操作
- Pipeline 的 async process() 异常 → 降级到 mem0 async 代理

---

## Dependencies

**Prerequisite Stories:**
- STORY-004: MemXMemory（参考实现）
- STORY-006: 项目骨架

**Blocked Stories:** None directly

**External Dependencies:** mem0ai 的 AsyncMemory 可用

---

## Definition of Done

- [ ] Code implemented in `memx/async_memory.py`
- [ ] `from memx import AsyncMemory` 工作
- [ ] Async tests in `tests/unit/test_async_memory.py`
- [ ] All tests passing（pytest-asyncio）
- [ ] `ruff check` 通过
- [ ] `mypy` 通过
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **AsyncMemory class:** 1.5 points
- **Async tests:** 1 point
- **Config reuse:** 0.5 points
- **Total:** 3 points

**Rationale:** 逻辑与同步版对称，主要工作是 async/await 适配和异步测试编写。

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
