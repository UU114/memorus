# lurus-memorus (2b-svc-memorus)

AI Agent 记忆引擎 — mem0 fork + ACE v2.0 自适应上下文引擎。自动蒸馏、语义去重、时间衰退、团队联邦。Platform 产品组 (P0)。

> **双引擎**：本仓库是 Python 引擎（默认发行，依赖 mem0/ONNX 生态）；Rust core 是
> 单一事实源，位于 github.com/LurusTech/Lurus-memorus-r（12 crates，含生产级
> memorus-server REST/MCP/daemon + 自带 deploy/）。`pip install memorus[rust]`
> 后 `MEMORUS_BACKEND=rust` 可切换到 Rust 后端（同一 Python API）。

- Namespace / Port: `lurus-system` / 8880
- Transport: REST API + MCP (stdio)
- Storage: SQLite (local) / Qdrant (K8s optional) — PVC-backed
- Repo: github.com/LurusTech/lurus-memorus (private)
- Deployment: **直接 kubectl apply，ArgoCD 显示 Unknown**（不影响运行）

## Tech Stack

| Layer | Choice |
|-------|--------|
| Runtime | Python 3.12 |
| Core | mem0ai, pydantic v2, click（可选 Rust 后端经 `memorus_r` binding） |
| Embeddings | ONNX (offline) / OpenAI (cloud) |
| API | FastAPI + uvicorn |
| MCP | `mcp[fastmcp]` |
| Test | pytest, pytest-asyncio, pytest-benchmark（含 tests/parity 跨语言闸门） |
| Lint | ruff, mypy |

## Behavior (since 2026-06, see CHANGELOG.md)

- ACE 管线**默认开启**（`ace_enabled=True`）；reflector 默认 `rules` 模式，完全离线，
  LLM 蒸馏（hybrid/llm）是显式 opt-in。
- Sanitizer fail-closed：脱敏失败时丢弃整次 ingest，绝不落盘原文。
- `project:`/`team:` scope 检索 = 该 scope ∪ global；search 分数归一化 [0,1]。
- 配置 canonical schema 为 Rust 嵌套形式；Python 扁平键经别名兼容。

## Directory (one level)

- `memorus/core/` — ACE engine (reflector/curator/decay/generator)
- `memorus/team/` — federation layer
- `memorus/ext/` — REST API, MCP server, agent tools
- `tests/`, `examples/`, `deploy/`, `Dockerfile`, `pyproject.toml`

## Commands

```bash
# Install
pip install -e ".[all]"

# Run
MEMORUS_API_KEY=dev-secret memorus-api --host 0.0.0.0 --port 8880
memorus-mcp                         # MCP stdio

# CLI
memorus status
memorus search "trading strategy"
memorus learn "Always check RSI before entry"

# Test
pytest tests/ -v

# Docker
docker build -t ghcr.io/lurustech/lurus-memorus:main .
```

## REST API (OpenAPI: `api/openapi.yaml`)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Liveness probe (no auth) |
| POST | `/memories` | Add memory |
| GET | `/memories/search` | Semantic search |
| GET | `/memories` | List all |
| GET/DELETE | `/memories/{id}` | Get / delete |
| GET | `/status` | KB statistics |

> Rust `memorus-server`（Lurus-memorus-r）提供独立的 `/api/v1/*` REST 面
> （多 PUT update / history / reset，常数时间鉴权），两者不互换。

Auth: `X-API-Key` header (= `MEMORUS_API_KEY` env)

## Consumers

| Consumer | Method | user_id mapping |
|----------|--------|----------------|
| lurus-lucrum | REST | Zitadel sub |
| lurus-creator | REST (scope: `project:{id}`) | local project ID |
| lurus-switch | embedded lib (SQLite) | device user |
| kova-memory | `MemoryProvider` trait (Rust) | agent id |

## Env Vars

| Var | Required | Description |
|-----|----------|-------------|
| `MEMORUS_API_KEY` | Yes | REST auth key |
| `OPENAI_API_KEY` | env-config mode | Enables env-driven config (embedder+LLM via lurus-newapi) |
| `OPENAI_BASE_URL` | No | OpenAI-compatible endpoint override |
| `MEMORUS_DATA_DIR` | No | Data dir (default `/data`) |
| `MEMORUS_ACE_ENABLED` | No | ACE pipeline (**default true**; set `false` to disable) |
| `MEMORUS_BACKEND` | No | `python` (default) / `rust` (requires `memorus[rust]`) |

## BMAD

| Resource | Path |
|----------|------|
| PRD | `./_bmad-output/planning-artifacts/prd.md` |
| Architecture | `./_bmad-output/planning-artifacts/architecture.md` |
| Sprint Plan | `./_bmad-output/planning-artifacts/sprint-plan.md` |
| Sprint Status | `./_bmad-output/planning-artifacts/sprint-status.yaml` |
| Stories | `./_bmad-output/planning-artifacts/stories/` |

---
_BMAD artifacts last review: 2026-05-18 — governance: `lurus/doc/audit/2026-05-18-bmad-output-stale.md`._
