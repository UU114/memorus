# STORY-035: Integration 单元测试

**Epic:** EPIC-006 (Integration Layer)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a QA engineer
I want Integration hooks fully tested
So that auto-learn and auto-recall are reliable in production

---

## Description

### Background
Integration Layer 是 Memorus 与宿主 AI 产品的核心连接点。Hook 的可靠性直接影响用户体验——PreInference 失败意味着 AI 丢失记忆，PostAction 失败意味着知识不被学习，SessionEnd 失败意味着衰退不执行。本 Story 对全部 Integration 组件进行系统测试。

### Scope
**In scope:**
- IntegrationManager 注册/注销/fire 测试
- CLIPreInferenceHook 召回 + 格式化测试
- CLIPostActionHook 异步蒸馏测试
- CLISessionEndHook 信号处理测试
- Hook 错误隔离测试
- 配置启停测试

**Out of scope:**
- 与真实 Claude Code 的集成测试
- 性能测试

---

## Acceptance Criteria

- [ ] IntegrationManager 测试：注册/注销 Hook、fire 各事件类型、空注册列表
- [ ] PreInferenceHook 测试：正常召回、空结果、XML/Markdown/Plain 三种格式输出
- [ ] PostActionHook 测试：异步蒸馏触发验证、ToolEvent 格式化
- [ ] SessionEndHook 测试：sweep 执行验证、兜底蒸馏触发
- [ ] 信号处理测试：SIGINT 触发 session_end（mock signal）
- [ ] 错误隔离测试：单 Hook 异常不影响其他 Hook
- [ ] 配置测试：auto_recall=False 时 PreInference 不执行、auto_reflect=False 时 PostAction 不执行
- [ ] 覆盖率 > 90%（`memorus/integration/`）

---

## Technical Notes

### Test Files
- `tests/unit/test_integration_manager.py` — IntegrationManager 测试
- `tests/unit/test_cli_hooks.py` — CLI Hook 具体实现测试

### Testing Strategy

```python
# Mock Memory for isolation
class MockMemory:
    def search(self, query, **kwargs):
        return [{"id": "1", "memory": "test", "score": 0.9}]

    def add(self, messages, user_id, **kwargs):
        return {"results": [{"id": "1"}]}

    def get_all(self, **kwargs):
        return [{"id": "1", "memory": "test"}]

# Async test pattern
@pytest.mark.asyncio
async def test_pre_inference_hook():
    hook = CLIPreInferenceHook(MockMemory(), IntegrationConfig())
    result = await hook.on_user_input("test query")
    assert result.memories
    assert "<memorus-context>" in result.rendered

# Signal test (mock signal module)
def test_session_end_on_sigint(monkeypatch):
    fired = []
    monkeypatch.setattr(signal, "signal", lambda *args: None)
    # Verify handler registration and execution
```

### Dependencies on Existing Code
- `memorus/integration/` — 全部 STORY-032/033/034 实现
- `memorus/engines/decay/engine.py` — DecayEngine（用于 SessionEnd 测试）

---

## Dependencies

**Prerequisite Stories:**
- STORY-034: CLI PostActionHook + SessionEndHook
- STORY-033: CLI PreInferenceHook
- STORY-032: IntegrationManager + BaseHook 抽象

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] `tests/unit/test_integration_manager.py` 实现
- [ ] `tests/unit/test_cli_hooks.py` 实现
- [ ] 所有 async 测试通过（pytest-asyncio）
- [ ] 覆盖率 > 90%
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **IntegrationManager 测试:** 1.5 points
- **PreInferenceHook 测试:** 1 point
- **PostAction + SessionEnd 测试:** 1.5 points
- **信号处理 + 错误隔离测试:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
