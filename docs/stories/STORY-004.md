# STORY-004: 实现 MemXMemory Decorator 包装类

**Epic:** EPIC-001 — Bullet 数据模型与配置基础
**Priority:** Must Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 1

---

## User Story

As a mem0 user migrating to MemX
I want to use the same API without code changes
So that migration is zero-effort

---

## Description

### Background
MemXMemory 是 MemX 的核心公开 API，它包装 mem0 的 Memory 类。设计采用 Decorator 模式：`ace_enabled=False` 时所有调用直接代理到 mem0（零开销透传）；`ace_enabled=True` 时通过 IngestPipeline 和 RetrievalPipeline 处理。这确保了 100% mem0 API 兼容性——用户只需将 `from mem0 import Memory` 改为 `from memx import Memory`。

### Scope

**In scope:**
- `memx.Memory` 类（包装 `mem0.Memory`）
- 所有 mem0 公开方法的代理：add, search, get_all, get, update, delete, delete_all, history, reset
- `ace_enabled=False` 模式的完整透传
- `ace_enabled=True` 模式的管线调用接口（管线实现由 STORY-014, STORY-030 完成）
- `from_config()` 类方法兼容 mem0 config dict
- ACE 新增方法占位：status(), export(), import_data(), run_decay_sweep()

**Out of scope:**
- IngestPipeline 实现（STORY-014）
- RetrievalPipeline 实现（STORY-030）
- AsyncMemXMemory（STORY-005）

### Architecture

```
User Code
    │
    ▼
memx.Memory(config)
    │
    ├── ace_enabled=False ──→ self._mem0.add/search/... (直接代理)
    │
    └── ace_enabled=True
        ├── .add() ──→ IngestPipeline.process() ──→ self._mem0.add()
        ├── .search() ──→ RetrievalPipeline.search()
        └── .get/update/delete/... ──→ self._mem0.xxx() (代理 + metadata 处理)
```

---

## Acceptance Criteria

- [ ] `memx.Memory` 可通过与 `mem0.Memory` 相同的方式构造
- [ ] `Memory(config_dict)` 兼容 mem0 config dict 格式
- [ ] `ace_enabled=False`（默认）时所有方法直接代理到 mem0
- [ ] 代理方法签名完全一致：add(messages, user_id=None, agent_id=None, run_id=None, metadata=None, filters=None, prompt=None)
- [ ] search 签名：search(query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None)
- [ ] get_all, get, update, delete, delete_all, history, reset 签名不变
- [ ] `ace_enabled=True` 时 add() 调用 IngestPipeline（如果管线未初始化则 fallback 到直接代理）
- [ ] `ace_enabled=True` 时 search() 调用 RetrievalPipeline（如果管线未初始化则 fallback 到直接代理）
- [ ] `from_config(config_dict)` 类方法正常工作
- [ ] 新增方法 `status()`, `export()`, `import_data()`, `run_decay_sweep()` 存在（可抛 NotImplementedError 占位）
- [ ] 运行 mem0 核心测试用例通过（ace_enabled=False 模式）

---

## Technical Notes

### File Location
`memx/memory.py`

### Implementation Sketch

```python
import logging
from typing import Any, Optional
from mem0 import Memory as Mem0Memory
from memx.config import MemXConfig

logger = logging.getLogger(__name__)

class Memory:
    """MemX Memory - drop-in replacement for mem0.Memory.

    ACE OFF: direct proxy to mem0.Memory (zero overhead)
    ACE ON: pipeline processing with graceful degradation
    """

    def __init__(self, config: Optional[dict] = None):
        self._config = MemXConfig.from_dict(config or {})
        self._mem0 = Mem0Memory(config=self._config.to_mem0_config())

        # Pipeline initialization (lazy)
        self._ingest_pipeline = None
        self._retrieval_pipeline = None

        if self._config.ace_enabled:
            self._init_ace_engines()

    def _init_ace_engines(self) -> None:
        """Initialize ACE engines. Failures degrade gracefully."""
        try:
            # Will be implemented when pipeline stories are done
            pass
        except Exception as e:
            logger.warning(f"ACE initialization failed, running in proxy mode: {e}")

    def add(self, messages, user_id=None, agent_id=None, run_id=None,
            metadata=None, filters=None, prompt=None, **kwargs) -> dict:
        if not self._config.ace_enabled or self._ingest_pipeline is None:
            return self._mem0.add(messages, user_id=user_id,
                                  agent_id=agent_id, run_id=run_id,
                                  metadata=metadata, filters=filters,
                                  prompt=prompt)
        # ACE path - delegate to IngestPipeline
        ...

    def search(self, query, user_id=None, agent_id=None, run_id=None,
               limit=100, filters=None, **kwargs) -> dict:
        if not self._config.ace_enabled or self._retrieval_pipeline is None:
            return self._mem0.search(query, user_id=user_id,
                                     agent_id=agent_id, run_id=run_id,
                                     limit=limit, filters=filters)
        # ACE path - delegate to RetrievalPipeline
        ...

    # Proxy methods
    def get_all(self, **kwargs): return self._mem0.get_all(**kwargs)
    def get(self, memory_id): return self._mem0.get(memory_id)
    def update(self, memory_id, data): return self._mem0.update(memory_id, data)
    def delete(self, memory_id): return self._mem0.delete(memory_id)
    def delete_all(self, **kwargs): return self._mem0.delete_all(**kwargs)
    def history(self, memory_id): return self._mem0.history(memory_id)
    def reset(self): return self._mem0.reset()
```

### Key Design Decisions
1. **Lazy Pipeline Init**: Pipeline 只在 `ace_enabled=True` 时初始化，且失败时降级到代理模式
2. **kwargs 透传**: 所有方法接受 `**kwargs` 确保 mem0 未来新参数不会破坏兼容
3. **内部 _mem0 属性**: 直接暴露底层 mem0 实例，方便测试和高级用户

### Edge Cases
- `config=None` → 使用全默认配置（ace_enabled=False）
- mem0 初始化失败（如缺少 API key）→ 应该抛出原始错误，不吞掉
- Pipeline 初始化失败 → 降级为代理模式，不抛异常
- 用户传入 mem0 不认识的 kwarg → 代理模式下直接传给 mem0，由 mem0 处理

---

## Dependencies

**Prerequisite Stories:**
- STORY-003: MemXConfig
- STORY-006: 项目骨架

**Blocked Stories:**
- STORY-005: AsyncMemXMemory（需要 Memory 类作为参考）
- STORY-007: mem0 兼容测试（需要 Memory 类存在）
- STORY-014: IngestPipeline（需要 Memory 类来集成）
- STORY-030: RetrievalPipeline（需要 Memory 类来集成）
- STORY-026: VectorSearcher（需要通过 Memory 访问 mem0 VectorStore）
- STORY-032: IntegrationManager（需要 Memory 类实例）

**External Dependencies:**
- mem0ai 包必须可导入

---

## Definition of Done

- [ ] Code implemented in `memx/memory.py`
- [ ] `from memx import Memory` 工作
- [ ] ace_enabled=False 下所有 mem0 方法正确代理
- [ ] Unit tests in `tests/unit/test_memory.py`
- [ ] All tests passing
- [ ] `ruff check` 通过
- [ ] `mypy` 通过
- [ ] Acceptance criteria all validated
- [ ] Code reviewed and approved

---

## Story Points Breakdown

- **Memory class + proxy methods:** 2 points
- **Config integration + from_config:** 1 point
- **ACE mode scaffolding:** 1 point
- **Unit tests:** 1 point
- **Total:** 5 points

**Rationale:** 方法代理本身简单，但 config 解析、ACE 初始化逻辑和 mem0 兼容性需要仔细处理。

---

## Progress Tracking

**Status History:**
- 2026-02-27: Created by Scrum Master

**Actual Effort:** TBD

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
