# STORY-038: 实现 DaemonClient + IPC Transport

**Epic:** EPIC-007 (本地 Embedding + Daemon)
**Priority:** Should Have
**Story Points:** 5
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a Hook implementation
I want a client to communicate with Daemon
So that I can recall/curate through IPC without cold start

---

## Description

### Background
DaemonClient 是 MemorusDaemon 的客户端封装，供 CLI Hook 和 CLI 命令使用。它封装了 IPC 连接细节（Named Pipe / Unix Socket），提供类型安全的 API，并自动探测 Daemon 是否运行。当 Daemon 不可用时抛出可捕获的 `DaemonUnavailableError`，供上层进行降级处理。

### Scope
**In scope:**
- DaemonClient 封装全部 6 个 IPC 命令
- IPC transport 抽象层（跨平台）
- 自动探测 Daemon 可用性（ping）
- 连接超时和重试策略
- `DaemonUnavailableError` 异常

**Out of scope:**
- 降级逻辑（STORY-039）
- Daemon 服务端（STORY-037）
- 自动启动 Daemon

---

## Acceptance Criteria

- [ ] `DaemonClient` 封装 6 个命令：`ping()`, `recall()`, `curate()`, `register_session()`, `unregister_session()`, `shutdown()`
- [ ] `ping()` 返回 `bool`，True=Daemon 可用
- [ ] `recall(query, user_id)` 返回搜索结果列表
- [ ] `curate(messages, user_id)` 返回 add 结果
- [ ] `register_session(session_id)` / `unregister_session(session_id)` 管理 Session
- [ ] `shutdown()` 优雅关闭 Daemon
- [ ] 支持 Named Pipe (Windows) + Unix Socket (Linux/Mac)
- [ ] 连接超时默认 2 秒，可配置
- [ ] Daemon 不可用时抛 `DaemonUnavailableError`（继承 `ConnectionError`）
- [ ] IPC transport 抽象为独立模块，便于未来扩展（如 TCP）

---

## Technical Notes

### Components
- `memorus/daemon/client.py` — DaemonClient
- `memorus/daemon/ipc.py` — IPCTransport 抽象层

### API Design

```python
# ipc.py
class IPCTransport(ABC):
    @abstractmethod
    async def connect(self) -> None: ...
    @abstractmethod
    async def send(self, data: bytes) -> None: ...
    @abstractmethod
    async def recv(self) -> bytes: ...
    @abstractmethod
    async def close(self) -> None: ...

class NamedPipeTransport(IPCTransport):
    """Windows Named Pipe transport."""
    def __init__(self, pipe_name: str = r"\\.\pipe\memorus-daemon"): ...

class UnixSocketTransport(IPCTransport):
    """Unix Socket transport."""
    def __init__(self, socket_path: str = "~/.memorus/daemon.sock"): ...

def get_transport(config: DaemonConfig) -> IPCTransport:
    """Factory: auto-select transport based on platform."""
    if sys.platform == "win32":
        return NamedPipeTransport(config.socket_path or DEFAULT_PIPE)
    return UnixSocketTransport(config.socket_path or DEFAULT_SOCKET)


# client.py
class DaemonUnavailableError(ConnectionError):
    """Raised when Daemon is not running or unreachable."""

class DaemonClient:
    def __init__(self, config: DaemonConfig | None = None):
        self._config = config or DaemonConfig()
        self._transport: IPCTransport | None = None

    async def ping(self) -> bool:
        try:
            resp = await self._request(DaemonRequest(cmd="ping"))
            return resp.status == "ok"
        except (ConnectionError, TimeoutError):
            return False

    async def recall(self, query: str, user_id: str = "default") -> list[dict]:
        resp = await self._request(DaemonRequest(
            cmd="recall",
            data={"query": query, "user_id": user_id},
        ))
        return resp.data.get("results", [])

    async def curate(self, messages: list[dict], user_id: str) -> dict:
        resp = await self._request(DaemonRequest(
            cmd="curate",
            data={"messages": messages, "user_id": user_id},
        ))
        return resp.data

    async def register_session(self, session_id: str) -> None:
        await self._request(DaemonRequest(
            cmd="session_register",
            data={"session_id": session_id},
        ))

    async def unregister_session(self, session_id: str) -> None:
        await self._request(DaemonRequest(
            cmd="session_unregister",
            data={"session_id": session_id},
        ))

    async def shutdown(self) -> None:
        await self._request(DaemonRequest(cmd="shutdown"))

    async def _request(self, req: DaemonRequest) -> DaemonResponse:
        transport = get_transport(self._config)
        try:
            await asyncio.wait_for(transport.connect(), timeout=self._config.connect_timeout)
            await transport.send(json.dumps(asdict(req)).encode())
            raw = await asyncio.wait_for(transport.recv(), timeout=self._config.request_timeout)
            return DaemonResponse(**json.loads(raw))
        except (ConnectionError, TimeoutError, OSError) as e:
            raise DaemonUnavailableError(f"Daemon unavailable: {e}") from e
        finally:
            await transport.close()
```

### Dependencies on Existing Code
- `memorus/daemon/server.py:DaemonRequest, DaemonResponse` — 共享数据类
- `memorus/config.py:DaemonConfig` — socket_path, enabled 等配置

### Edge Cases
- Daemon 进程在请求途中崩溃 → recv 超时 → 抛 DaemonUnavailableError
- Socket 文件存在但进程已死 → connect 失败 → DaemonUnavailableError
- Named Pipe 已被其他进程占用 → connect 失败
- 并发多个 Client 调用 → 每次请求创建新连接（无连接池，简化实现）
- JSON 响应格式错误 → 解析失败，抛 DaemonUnavailableError

---

## Dependencies

**Prerequisite Stories:**
- STORY-037: MemorusDaemon 服务端

**Blocked Stories:**
- STORY-039: Daemon 降级逻辑
- STORY-040: Daemon 测试

---

## Definition of Done

- [ ] `memorus/daemon/ipc.py` 实现 IPCTransport + NamedPipeTransport + UnixSocketTransport
- [ ] `memorus/daemon/client.py` 实现 DaemonClient
- [ ] 单元测试覆盖全部 6 个命令（mock transport）
- [ ] DaemonUnavailableError 在各失败场景正确抛出
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **IPCTransport 抽象 + 双平台实现:** 2 points
- **DaemonClient 6 个命令:** 1.5 points
- **错误处理 + 超时:** 0.5 points
- **测试:** 1 point
- **Total:** 5 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
