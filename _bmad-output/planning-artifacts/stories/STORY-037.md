# STORY-037: 实现 MemorusDaemon 服务端

**Epic:** EPIC-007 (本地 Embedding + Daemon)
**Priority:** Should Have
**Story Points:** 8
**Status:** Not Started
**Assigned To:** Unassigned
**Created:** 2026-02-27
**Sprint:** 3

---

## User Story

As a power user
I want a daemon process to avoid cold starts
So that Hook calls are fast and ONNX models stay loaded in memory

---

## Description

### Background
每次 CLI Hook 调用都需要初始化 Memorus（加载配置、ONNX 模型、向量索引等），冷启动可能需要 1-3 秒。MemorusDaemon 是一个常驻后台进程，预加载所有重量级资源，通过 IPC 提供服务。多个 CLI Session 共享同一个 Daemon 实例，实现近乎零延迟的 Hook 调用。

### Scope
**In scope:**
- Daemon 进程启动/关闭生命周期管理
- PID 文件管理（防重复启动）
- IPC 协议：6 个命令（ping, recall, curate, session_register, session_unregister, shutdown）
- 多 Session 注册管理
- 空闲超时自动退出（默认 5 分钟无活跃 Session）
- 请求/响应 JSON 协议
- 健康检查

**Out of scope:**
- TCP 网络监听（仅本地 IPC）
- DaemonClient 实现（STORY-038）
- 降级逻辑（STORY-039）

---

## Acceptance Criteria

- [ ] `MemorusDaemon` 可通过 `daemon.start()` 启动为后台进程
- [ ] 启动时创建 PID 文件（`~/.memorus/daemon.pid`），退出时删除
- [ ] PID 文件已存在且进程存活 → 拒绝重复启动，抛出明确错误
- [ ] IPC 监听：Windows 使用 Named Pipe (`\\.\pipe\memorus-daemon`)，Linux/Mac 使用 Unix Socket (`~/.memorus/daemon.sock`)
- [ ] 支持 6 个 IPC 命令：`ping`, `recall`, `curate`, `session_register`, `session_unregister`, `shutdown`
- [ ] `ping` → 返回 `{"status": "ok", "version": "1.0.0", "sessions": N}`
- [ ] `recall` → 调用 Memory.search() 返回结果
- [ ] `curate` → 调用 IngestPipeline 处理 bullets
- [ ] `session_register/unregister` → 管理活跃 Session 列表
- [ ] `shutdown` → 优雅关闭（flush pending → close IPC → delete PID → exit）
- [ ] 空闲超时：最后一个 Session unregister 后启动倒计时（默认 5 分钟），期间无新 Session → 自动关闭
- [ ] 所有请求处理异常不导致 Daemon 崩溃，返回 error response

---

## Technical Notes

### Components
- `memorus/daemon/__init__.py` — 包入口
- `memorus/daemon/server.py` — MemorusDaemon

### API Design

```python
class MemorusDaemon:
    def __init__(self, config: DaemonConfig):
        self._config = config
        self._memory: Memory | None = None
        self._sessions: dict[str, datetime] = {}
        self._idle_timer: asyncio.Task | None = None

    async def start(self) -> None:
        self._check_pid()
        self._write_pid()
        self._memory = Memory(config={"ace_enabled": True})
        await self._start_ipc_server()

    async def stop(self) -> None:
        await self._flush_pending()
        self._remove_pid()

    async def handle_request(self, request: DaemonRequest) -> DaemonResponse:
        match request.cmd:
            case "ping":
                return DaemonResponse(status="ok", data={
                    "version": __version__,
                    "sessions": len(self._sessions),
                })
            case "recall":
                results = self._memory.search(request.data["query"])
                return DaemonResponse(status="ok", data={"results": results})
            case "curate":
                result = self._memory.add(request.data["messages"], request.data["user_id"])
                return DaemonResponse(status="ok", data=result)
            case "session_register":
                self._sessions[request.data["session_id"]] = datetime.now()
                self._cancel_idle_timer()
                return DaemonResponse(status="ok")
            case "session_unregister":
                self._sessions.pop(request.data["session_id"], None)
                if not self._sessions:
                    self._start_idle_timer()
                return DaemonResponse(status="ok")
            case "shutdown":
                await self.stop()
                return DaemonResponse(status="ok")
            case _:
                return DaemonResponse(status="error", error=f"Unknown command: {request.cmd}")
```

### IPC Protocol

```
REQUEST:  {"cmd": "ping"}
RESPONSE: {"status": "ok", "version": "1.0.0", "sessions": 2}

REQUEST:  {"cmd": "recall", "data": {"query": "async error handling", "user_id": "user1"}}
RESPONSE: {"status": "ok", "data": {"results": [...]}}

REQUEST:  {"cmd": "session_register", "data": {"session_id": "sess-abc"}}
RESPONSE: {"status": "ok"}

REQUEST:  {"cmd": "shutdown"}
RESPONSE: {"status": "ok"}
```

### Data Classes

```python
@dataclass(frozen=True)
class DaemonRequest:
    cmd: str
    data: dict = field(default_factory=dict)

@dataclass
class DaemonResponse:
    status: str  # "ok" | "error"
    data: dict = field(default_factory=dict)
    error: str | None = None
```

### IPC Transport (Platform-Specific)

```python
# Windows: Named Pipe
PIPE_NAME = r"\\.\pipe\memorus-daemon"

# Linux/Mac: Unix Socket
SOCKET_PATH = Path("~/.memorus/daemon.sock").expanduser()

async def _start_ipc_server(self):
    if sys.platform == "win32":
        # asyncio.start_server on Named Pipe (Windows)
        # or use win32pipe for lower-level control
        server = await asyncio.start_server(
            self._handle_connection, path=PIPE_NAME
        )
    else:
        server = await asyncio.start_unix_server(
            self._handle_connection, path=str(SOCKET_PATH)
        )
```

### Dependencies on Existing Code
- `memorus/config.py:DaemonConfig` — 已定义（enabled, idle_timeout_seconds, socket_path）
- `memorus/memory.py:Memory` — search(), add() API
- `memorus/embeddings/onnx.py:ONNXEmbedder`（STORY-036）

### Edge Cases
- Daemon 进程被 kill -9 → PID 文件残留 → 启动时检测进程是否存活，清理 stale PID
- Socket 文件残留 → 启动时删除旧 socket 文件
- 并发请求 → asyncio 单线程事件循环天然序列化
- 超大 request payload → 设置 max_request_size（默认 1MB）
- Windows Named Pipe 权限 → 仅当前用户可访问
- Daemon 启动期间 ONNX 下载失败 → 退出并清理 PID 文件

---

## Dependencies

**Prerequisite Stories:**
- STORY-036: ONNXEmbedder Provider
- STORY-004: MemorusMemory Decorator ✓（已完成）

**Blocked Stories:**
- STORY-038: DaemonClient + IPC Transport
- STORY-039: Daemon 降级逻辑
- STORY-040: Daemon 测试

---

## Definition of Done

- [ ] `memorus/daemon/server.py` 实现 MemorusDaemon
- [ ] PID 文件管理正确（创建/清理/stale 检测）
- [ ] IPC 6 个命令全部实现
- [ ] 空闲超时自动退出
- [ ] Windows Named Pipe + Unix Socket 双平台支持
- [ ] mypy --strict 通过
- [ ] ruff check 通过

---

## Story Points Breakdown

- **Daemon 生命周期 (start/stop/PID):** 2 points
- **IPC 协议 + 请求路由:** 2 points
- **Session 管理 + 空闲超时:** 1.5 points
- **跨平台 IPC transport:** 1.5 points
- **测试:** 1 point
- **Total:** 8 points

---

**This story was created using BMAD Method v6 - Phase 4 (Implementation Planning)**
