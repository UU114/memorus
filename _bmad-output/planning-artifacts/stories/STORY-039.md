# STORY-039: Daemon 降级逻辑

**Epic:** EPIC-007 (本地 Embedding + Daemon)
**Priority:** Should Have
**Story Points:** 3
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a Memorus system
I want graceful fallback when Daemon is unavailable
So that Memorus always works regardless of Daemon status

---

## Description

### Background
Daemon 是性能优化组件，不是必要依赖。当 Daemon 未启动、崩溃或不响应时，Memorus 必须透明降级到直接调用 Memory 库模式（冷启动但功能完整）。降级和恢复对用户完全透明，仅通过日志通知。

### Scope
**In scope:**
- DaemonClient.ping() 失败 → 回退到直接调用 Memory
- 降级/恢复自动检测
- 降级 WARNING 日志（仅首次）
- 恢复 INFO 日志
- 集成到 Memory 层

**Out of scope:**
- 自动启动 Daemon
- Daemon 自动重启（watchdog）

---

## Acceptance Criteria

- [ ] `Memory` 初始化时若 `DaemonConfig.enabled=True`，尝试 ping Daemon
- [ ] ping 成功 → 使用 DaemonClient 路径（recall/curate 通过 IPC）
- [ ] ping 失败 → 回退到直接调用本地 Memory（冷启动）
- [ ] 降级时记录 `WARNING: Daemon unavailable, falling back to direct mode`（仅首次）
- [ ] 每次 search/add 操作前检测 Daemon 可用性（lazy check，非每次 ping）
- [ ] Daemon 恢复（ping 成功）→ 自动切回 IPC 模式，记录 `INFO: Daemon reconnected`
- [ ] 降级过程对调用方完全透明（返回值格式一致）
- [ ] 降级模式下功能完整（仅性能可能较慢）

---

## Technical Notes

### Implementation

降级逻辑集成到 `memorus/memory.py` 的 `MemorusMemory` 类：

```python
class MemorusMemory:
    def __init__(self, config):
        self._daemon_client: DaemonClient | None = None
        self._daemon_available: bool = False
        self._degraded_logged: bool = False

        if config.daemon.enabled:
            self._daemon_client = DaemonClient(config.daemon)
            self._daemon_available = asyncio.run(self._daemon_client.ping())
            if not self._daemon_available:
                if not self._degraded_logged:
                    logger.warning("Daemon unavailable, falling back to direct mode")
                    self._degraded_logged = True

    def search(self, query, **kwargs):
        if self._daemon_available:
            try:
                return asyncio.run(self._daemon_client.recall(query, **kwargs))
            except DaemonUnavailableError:
                self._daemon_available = False
                logger.warning("Daemon lost, falling back to direct mode")
        # Direct mode
        return self._direct_search(query, **kwargs)

    def add(self, messages, user_id, **kwargs):
        if self._daemon_available:
            try:
                return asyncio.run(self._daemon_client.curate(messages, user_id))
            except DaemonUnavailableError:
                self._daemon_available = False
                logger.warning("Daemon lost, falling back to direct mode")
        # Direct mode
        return self._direct_add(messages, user_id, **kwargs)

    def _check_daemon_recovery(self):
        """Periodic check if daemon came back (called every N operations)."""
        if self._daemon_client and not self._daemon_available:
            if asyncio.run(self._daemon_client.ping()):
                self._daemon_available = True
                self._degraded_logged = False
                logger.info("Daemon reconnected, switching to IPC mode")
```

### Recovery Strategy
- 不在每次操作时 ping（性能开销）
- 使用计数器：每 10 次操作尝试一次 ping
- 或使用时间间隔：距上次失败 30 秒后再尝试

### Dependencies on Existing Code
- `memorus/memory.py:MemorusMemory` — 主 Memory 类
- `memorus/daemon/client.py:DaemonClient` — IPC 客户端（STORY-038）
- `memorus/config.py:DaemonConfig` — enabled 配置

### Edge Cases
- Daemon 在请求中间崩溃 → DaemonUnavailableError → 降级
- Daemon 频繁崩溃重启 → 避免 log spam（首次降级/恢复才记录）
- DaemonConfig.enabled=False → 完全跳过 Daemon 逻辑
- 异步事件循环冲突 → 在同步上下文中使用 asyncio.run()，注意嵌套循环

---

## Dependencies

**Prerequisite Stories:**
- STORY-038: DaemonClient + IPC Transport

**Blocked Stories:**
- STORY-040: Daemon 测试

---

## Definition of Done

- [ ] `memorus/memory.py` 集成 Daemon 降级逻辑
- [ ] 降级/恢复自动切换测试
- [ ] 日志输出符合预期（仅关键时刻记录）
- [ ] 降级模式下 search/add 功能完整
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **降级逻辑集成:** 1.5 points
- **恢复检测策略:** 0.5 points
- **测试:** 1 point
- **Total:** 3 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
