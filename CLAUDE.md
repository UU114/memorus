# lurus-memorus (2b-svc-memorus)

AI Agent 记忆引擎 — mem0 fork + ACE v2.0 自适应上下文引擎。自动蒸馏、语义去重、时间衰退、团队联邦。Platform 产品组 (P0)。

> **Python 例外**：公司后端标准是 Go，本服务因依赖 mem0/ONNX Python 生态用 Python；对外以 REST/MCP 接口形式提供，边界清晰。

- Namespace / Port: `lurus-system` / 8880
- Transport: REST API + MCP (stdio)
- Storage: SQLite (local) / Qdrant (K8s optional) — PVC-backed
- Repo: github.com/hanmahong5-arch/lurus-memorus (private)
- Deployment: **直接 kubectl apply，ArgoCD 显示 Unknown**（不影响运行）

## Tech Stack

| Layer | Choice |
|-------|--------|
| Runtime | Python 3.12 |
| Core | mem0ai, pydantic v2, click |
| Embeddings | ONNX (offline) / OpenAI (cloud) |
| API | FastAPI + uvicorn |
| MCP | `mcp[fastmcp]` |
| Test | pytest, pytest-asyncio, pytest-benchmark |
| Lint | ruff, mypy |

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
docker build -t ghcr.io/hanmahong5-arch/lurus-memorus:main .
```

## REST API (OpenAPI: `api/openapi.yaml`)

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/memories` | Add memory |
| GET | `/memories/search` | Semantic search |
| GET | `/memories` | List all |
| GET/DELETE | `/memories/{id}` | Get / delete |
| GET | `/status` | KB statistics |

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
| `OPENAI_API_KEY` | ACE+LLM mode | LLM for reflector/distiller |
| `MEMORUS_DATA_DIR` | No | SQLite dir (default `/data`) |
| `MEMORUS_ACE_ENABLED` | No | Enable ACE pipeline (default false) |

## BMAD

| Resource | Path |
|----------|------|
| PRD | `./_bmad-output/planning-artifacts/prd.md` |
| Architecture | `./_bmad-output/planning-artifacts/architecture.md` |
| Sprint Plan | `./_bmad-output/planning-artifacts/sprint-plan.md` |
| Sprint Status | `./_bmad-output/planning-artifacts/sprint-status.yaml` |
| Stories | `./_bmad-output/planning-artifacts/stories/` |
