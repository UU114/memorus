# STORY-034: 实现 CLI PostActionHook + SessionEndHook

**Epic:** EPIC-006 (Integration Layer)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a CLI AI tool
I want knowledge captured after tool calls and at session end
So that learning happens automatically without user intervention

---

## Description

### Background
PostActionHook 在 AI 工具调用（如代码执行、文件编辑）完成后触发，将工具上下文发送给 Reflector 进行知识蒸馏。SessionEndHook 在会话结束时触发兜底蒸馏和 Decay sweep。两者配合实现"无感学习"。首要集成目标是 Claude Code 的 `PostToolUse` 和 `Stop`/`SessionEnd` Hook。

### Scope
**In scope:**
- `CLIPostActionHook`：接收 ToolEvent → 异步触发 IngestPipeline
- `CLISessionEndHook`：兜底蒸馏 + DecayEngine.sweep()
- 信号处理：SIGTERM/SIGINT 触发 SessionEnd
- 异步执行：蒸馏不阻塞主流程

**Out of scope:**
- 会话级别的蒸馏去重（Curator 在 IngestPipeline 中处理）
- 多 Session 并行管理（Daemon 负责）

---

## Acceptance Criteria

- [ ] `CLIPostActionHook` 继承 `PostActionHook`
- [ ] `on_tool_result(event)` 将 ToolEvent 格式化为 messages 并调用 `memory.add()` (异步)
- [ ] 蒸馏在后台线程执行（`ThreadPoolExecutor`），不阻塞调用方
- [ ] `CLISessionEndHook` 继承 `SessionEndHook`
- [ ] `on_session_end(session_id)` 执行：①收集未蒸馏的会话上下文 ②调用 memory.add() ③调用 DecayEngine.sweep()
- [ ] 注册 `signal.signal(SIGTERM, handler)` 和 `signal.signal(SIGINT, handler)` 触发 session_end
- [ ] 信号处理器中确保 sweep 完成后才退出（最多等待 5s）
- [ ] PostAction 执行异常不传播，仅 WARNING 日志
- [ ] SessionEnd 执行异常不传播，仅 WARNING 日志

---

## Technical Notes

### Components
- `memorus/integration/cli_hooks.py` — CLIPostActionHook, CLISessionEndHook

### API Design

```python
class CLIPostActionHook(PostActionHook):
    def __init__(self, memory: Memory, config: IntegrationConfig):
        self._memory = memory
        self._config = config
        self._executor = ThreadPoolExecutor(max_workers=1)

    @property
    def name(self) -> str:
        return "cli_post_action"

    @property
    def enabled(self) -> bool:
        return self._config.auto_reflect

    async def on_tool_result(self, event: ToolEvent) -> None:
        # Fire-and-forget async distillation
        messages = self._format_tool_event(event)
        self._executor.submit(self._memory.add, messages, event.session_id)

    def _format_tool_event(self, event: ToolEvent) -> list[dict]:
        return [
            {"role": "assistant", "content": f"Used tool: {event.tool_name}"},
            {"role": "tool", "content": event.output},
        ]


class CLISessionEndHook(SessionEndHook):
    def __init__(
        self,
        memory: Memory,
        decay_engine: DecayEngine,
        config: IntegrationConfig,
    ):
        self._memory = memory
        self._decay = decay_engine
        self._config = config

    @property
    def name(self) -> str:
        return "cli_session_end"

    @property
    def enabled(self) -> bool:
        return self._config.sweep_on_exit

    async def on_session_end(self, session_id: str) -> None:
        # 1. Flush any pending distillation
        # 2. Run decay sweep
        bullets = self._memory.get_all(user_id=session_id)
        self._decay.sweep(bullets)
```

### Signal Handling

```python
import signal
import asyncio

def setup_signal_handlers(manager: IntegrationManager, session_id: str):
    def handler(signum, frame):
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(
                asyncio.wait_for(
                    manager.fire_session_end(session_id),
                    timeout=5.0,
                )
            )
        finally:
            loop.close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
```

### Dependencies on Existing Code
- `memorus/integration/hooks.py` — PostActionHook, SessionEndHook（STORY-032）
- `memorus/pipeline/ingest.py:IngestPipeline` — add 路径
- `memorus/engines/decay/engine.py:DecayEngine.sweep()` — 衰退扫描
- `memorus/config.py:IntegrationConfig` — auto_reflect, sweep_on_exit

### Edge Cases
- ToolEvent.output 超长（>10000 字符）→ 截断后传入
- 信号在 sweep 执行中再次触发 → 忽略重复信号
- Windows 不支持 SIGTERM → 仅注册 SIGINT（Ctrl+C）
- ThreadPoolExecutor 中的任务未完成就退出 → SessionEnd 中 executor.shutdown(wait=True, timeout=5)

---

## Dependencies

**Prerequisite Stories:**
- STORY-032: IntegrationManager + BaseHook 抽象
- STORY-014: IngestPipeline ✓（已完成）
- STORY-021: Decay sweep + reinforce ✓（已完成）

**Blocked Stories:**
- STORY-035: Integration 测试

---

## Definition of Done

- [ ] `memorus/integration/cli_hooks.py` 实现 CLIPostActionHook + CLISessionEndHook
- [ ] 信号处理逻辑实现
- [ ] 异步蒸馏不阻塞主流程（测试验证）
- [ ] sweep 在 session_end 中正确执行
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **CLIPostActionHook + 异步蒸馏:** 1.5 points
- **CLISessionEndHook + sweep:** 1.5 points
- **信号处理:** 1 point
- **测试:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
