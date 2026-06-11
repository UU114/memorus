# STORY-040: Daemon 测试

**Epic:** EPIC-007 (本地 Embedding + Daemon)
**Priority:** Should Have
**Story Points:** 4
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a QA engineer
I want Daemon fully tested
So that multi-session lifecycle and IPC communication are reliable

---

## Description

### Background
MemorusDaemon 是 Memorus 中最复杂的单体组件（8pt），涉及进程管理、IPC 通信、Session 生命周期、空闲超时等多个子系统。充分测试对于保证 Daemon 可靠性至关重要，尤其是跨平台（Windows Named Pipe / Unix Socket）差异。

### Scope
**In scope:**
- MemorusDaemon 启动/关闭生命周期测试
- IPC 协议全部 6 个命令测试
- Session 注册/注销管理测试
- 空闲超时自动退出测试
- DaemonClient 全部方法测试
- 降级逻辑测试
- PID 文件管理测试

**Out of scope:**
- 性能压测
- 真实多进程 E2E 测试（CI 环境不稳定）

---

## Acceptance Criteria

- [ ] 启动测试：Daemon 正常启动 → PID 文件创建 → IPC 监听就绪
- [ ] 重复启动测试：PID 文件存在且进程存活 → 启动失败并抛出明确错误
- [ ] Stale PID 测试：PID 文件存在但进程已死 → 清理 PID 文件 → 正常启动
- [ ] 关闭测试：shutdown 命令 → PID 文件删除 → Socket/Pipe 清理
- [ ] IPC 命令测试：6 个命令全覆盖（ping, recall, curate, session_register, session_unregister, shutdown）
- [ ] Session 管理测试：注册 3 个 Session → 注销 2 个 → 只剩 1 个活跃
- [ ] 空闲超时测试：最后 Session 注销后 → 等待超时 → Daemon 自动退出
- [ ] Client 测试：DaemonClient 6 个方法 mock transport 测试
- [ ] 降级测试：Daemon 不可用 → DaemonUnavailableError → Memory 降级到直接模式
- [ ] 恢复测试：Daemon 恢复 → Memory 切回 IPC 模式
- [ ] 覆盖率 > 85%（`memorus/daemon/`）

---

## Technical Notes

### Test Files
- `tests/unit/test_daemon_server.py` — MemorusDaemon 服务端测试
- `tests/unit/test_daemon_client.py` — DaemonClient 测试
- `tests/unit/test_ipc_transport.py` — IPCTransport 测试
- `tests/unit/test_daemon_degradation.py` — 降级逻辑测试

### Testing Strategy

```python
# Mock IPC transport for unit testing
class MockTransport(IPCTransport):
    def __init__(self, responses: list[bytes]):
        self._responses = responses
        self._sent: list[bytes] = []

    async def connect(self): pass
    async def send(self, data: bytes): self._sent.append(data)
    async def recv(self) -> bytes: return self._responses.pop(0)
    async def close(self): pass

# Daemon lifecycle test (in-process, no real IPC)
@pytest.mark.asyncio
async def test_daemon_lifecycle():
    config = DaemonConfig(idle_timeout_seconds=1)
    daemon = MemorusDaemon(config)
    # Start in-process (skip real IPC binding)
    assert daemon._sessions == {}

    # Register session
    resp = await daemon.handle_request(DaemonRequest(
        cmd="session_register", data={"session_id": "s1"}
    ))
    assert resp.status == "ok"
    assert len(daemon._sessions) == 1

    # Unregister → idle timer starts
    resp = await daemon.handle_request(DaemonRequest(
        cmd="session_unregister", data={"session_id": "s1"}
    ))
    assert len(daemon._sessions) == 0

# PID file test
def test_stale_pid_cleanup(tmp_path):
    pid_file = tmp_path / "daemon.pid"
    pid_file.write_text("99999999")  # nonexistent PID
    # Daemon should clean up stale PID and start normally
```

### Platform-Specific Testing
- 使用 `sys.platform` 条件跳过不支持的平台测试
- `@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")`
- Mock Named Pipe 操作在 Linux CI 上

### Dependencies on Existing Code
- `memorus/daemon/` — 全部 STORY-037/038/039 实现
- `memorus/memory.py` — 降级逻辑集成

---

## Dependencies

**Prerequisite Stories:**
- STORY-039: Daemon 降级逻辑
- STORY-038: DaemonClient + IPC Transport
- STORY-037: MemorusDaemon 服务端

**Blocked Stories:**
- None

---

## Definition of Done

- [ ] 全部 4 个测试文件实现
- [ ] 所有 async 测试通过（pytest-asyncio）
- [ ] 跨平台 skip 标记正确
- [ ] 覆盖率 > 85%
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **Daemon 服务端测试:** 1.5 points
- **Client + IPC 测试:** 1 point
- **降级/恢复测试:** 1 point
- **PID + 平台适配测试:** 0.5 points
- **Total:** 4 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
