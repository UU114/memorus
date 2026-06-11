# STORY-032: 实现 IntegrationManager + BaseHook 抽象

**Epic:** EPIC-006 (Integration Layer)
**Priority:** Should Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As an AI product builder
I want a hook registration system
So that Memorus automatically integrates with my product's lifecycle

---

## Description

### Background
Memorus 通过三个 Hook 点与宿主 AI 产品集成：PreInferenceHook（用户输入前自动召回记忆）、PostActionHook（工具调用后自动蒸馏）、SessionEndHook（会话结束时兜底蒸馏 + Decay sweep）。IntegrationManager 是这些 Hook 的注册中心和生命周期管理器。本 Story 定义抽象层，具体 CLI 实现在 STORY-033/034。

### Scope
**In scope:**
- BaseHook 抽象基类
- PreInferenceHook / PostActionHook / SessionEndHook 接口定义
- IntegrationManager 注册/注销/查询 Hook
- IntegrationConfig 控制各 Hook 启停
- Hook 执行的错误隔离

**Out of scope:**
- CLI 具体 Hook 实现（STORY-033/034）
- MCP Server 集成
- API Middleware 集成

---

## Acceptance Criteria

- [ ] `BaseHook` ABC 定义 `name` 属性和 `enabled` 属性
- [ ] `PreInferenceHook` 定义 `async on_user_input(input: str) -> ContextInjection`
- [ ] `PostActionHook` 定义 `async on_tool_result(event: ToolEvent) -> None`
- [ ] `SessionEndHook` 定义 `async on_session_end(session_id: str) -> None`
- [ ] `ContextInjection` 数据类包含 `memories: list[dict]`, `format: str`, `rendered: str`
- [ ] `ToolEvent` 数据类包含 `tool_name: str`, `input: dict`, `output: str`, `session_id: str`
- [ ] `IntegrationManager.register_hooks(hooks)` 注册 Hook 列表
- [ ] `IntegrationManager.unregister_all()` 注销所有 Hook
- [ ] `IntegrationManager` 根据 `IntegrationConfig` 自动跳过禁用的 Hook
- [ ] 单个 Hook 执行异常不影响其他 Hook，记录 WARNING 日志

---

## Technical Notes

### Components
- `memorus/integration/__init__.py` — 包入口，导出公共 API
- `memorus/integration/hooks.py` — BaseHook, PreInferenceHook, PostActionHook, SessionEndHook, ContextInjection, ToolEvent
- `memorus/integration/manager.py` — IntegrationManager

### API Design

```python
# hooks.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class ContextInjection:
    memories: list[dict]
    format: str  # "xml" | "markdown" | "plain"
    rendered: str  # formatted context string

@dataclass(frozen=True)
class ToolEvent:
    tool_name: str
    input: dict
    output: str
    session_id: str

class BaseHook(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def enabled(self) -> bool:
        return True

class PreInferenceHook(BaseHook):
    @abstractmethod
    async def on_user_input(self, input: str) -> ContextInjection: ...

class PostActionHook(BaseHook):
    @abstractmethod
    async def on_tool_result(self, event: ToolEvent) -> None: ...

class SessionEndHook(BaseHook):
    @abstractmethod
    async def on_session_end(self, session_id: str) -> None: ...

# manager.py
class IntegrationManager:
    def __init__(self, memory: Memory, config: IntegrationConfig): ...
    def register_hooks(self, hooks: list[BaseHook]) -> None: ...
    def unregister_all(self) -> None: ...
    def get_hooks(self, hook_type: type) -> list[BaseHook]: ...
    async def fire_pre_inference(self, input: str) -> ContextInjection | None: ...
    async def fire_post_action(self, event: ToolEvent) -> None: ...
    async def fire_session_end(self, session_id: str) -> None: ...
```

### Dependencies on Existing Code
- `memorus/config.py:IntegrationConfig` — 已定义（auto_recall, auto_reflect, sweep_on_exit）
- `memorus/memory.py:Memory` — search() 和 add() 方法

### Edge Cases
- 注册空 Hook 列表 → 无操作
- 同一类型注册多个 Hook → 按注册顺序执行
- Hook 执行超时 → 不设硬超时（由调用方决定）
- fire_pre_inference 在 auto_recall=False 时 → 返回 None

---

## Dependencies

**Prerequisite Stories:**
- STORY-004: MemorusMemory Decorator ✓（已完成）

**Blocked Stories:**
- STORY-033: CLI PreInferenceHook
- STORY-034: CLI PostActionHook + SessionEndHook
- STORY-035: Integration 测试

---

## Definition of Done

- [ ] `memorus/integration/hooks.py` 定义全部抽象类和数据类
- [ ] `memorus/integration/manager.py` 实现 IntegrationManager
- [ ] 单元测试覆盖注册/注销/fire/错误隔离
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **数据类定义:** 1 point
- **BaseHook + 三个子类:** 1 point
- **IntegrationManager:** 1.5 points
- **测试:** 0.5 points
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
