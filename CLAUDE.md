# lurus-memorus

AI Agent 记忆引擎 — mem0 fork + ACE v2.0 自适应上下文引擎。
支持自动蒸馏知识、语义去重、时间衰退、团队联邦同步，提供 REST API 和 MCP Server 两种接入方式。

> Python 例外说明：公司标准后端用 Go，本服务因依赖 mem0/ONNX Python 生态而使用 Python。
> 内部以 REST API 形式供其他服务（Go/TypeScript）调用，对外接口边界清晰。

## BMAD

| Resource | Path |
|----------|------|
| PRD | `./_bmad-output/planning-artifacts/prd.md` |
| Architecture | `./_bmad-output/planning-artifacts/architecture.md` |
| Sprint Plan | `./_bmad-output/planning-artifacts/sprint-plan.md` |
| Sprint Status | `./_bmad-output/planning-artifacts/sprint-status.yaml` |
| Stories | `./_bmad-output/planning-artifacts/stories/` |

## Commands

```bash
# Install (local development)
pip install -e ".[all]"

# Run tests
pytest tests/ -v

# Run REST API (dev)
MEMORUS_API_KEY=dev-secret memorus-api --host 0.0.0.0 --port 8880

# Run MCP server (stdio transport)
memorus-mcp

# CLI usage
memorus status
memorus search "trading strategy"
memorus learn "Always check RSI before entry"
memorus sweep

# Docker build
docker build -t ghcr.io/hanmahong5-arch/lurus-memorus:main .

# Production run
docker run -e MEMORUS_API_KEY=$SECRET -e OPENAI_API_KEY=$KEY \
  -p 8880:8880 -v memorus-data:/data \
  ghcr.io/hanmahong5-arch/lurus-memorus:main
```

## Tech Stack

| Layer | Choice |
|-------|--------|
| Runtime | Python 3.12 |
| Core | mem0ai, pydantic v2, click |
| Storage | SQLite (local) / Qdrant (K8s optional) |
| Embeddings | ONNX (offline) / OpenAI (cloud) |
| API | FastAPI + uvicorn |
| MCP | mcp[fastmcp] |
| Test | pytest, pytest-asyncio, pytest-benchmark |
| Lint | ruff, mypy |

## Directory Structure

```
lurus-memorus/
├── memorus/              # Python package
│   ├── core/             # ACE engine (reflector/curator/decay/generator)
│   ├── team/             # Federation layer
│   └── ext/              # REST API, MCP server, agent tools
├── tests/                # Unit + integration + benchmark tests
├── examples/             # Usage examples (25 files)
├── deploy/               # K8s manifests
├── _bmad-output/         # BMAD planning artifacts
├── Dockerfile
└── pyproject.toml
```

## Service Endpoints (REST API)

| Method | Path | Description |
|--------|------|-------------|
| POST | /memories | Add memory |
| GET | /memories/search | Semantic search |
| GET | /memories | List all |
| GET | /memories/{id} | Get by ID |
| DELETE | /memories/{id} | Delete |
| GET | /status | KB statistics |

Auth: `X-API-Key` header (= `MEMORUS_API_KEY` env).

## Integration Points

| Consumer | Method | user_id mapping |
|----------|--------|----------------|
| lurus-lucrum | REST `/memories` | Zitadel sub |
| lurus-creator | REST `/memories` (scope: project:{id}) | local project ID |
| lurus-switch | embedded lib (local SQLite) | device user |
| Claude Code | MCP (`memorus-mcp`) | developer |

## Environment Variables

| Var | Required | Description |
|-----|----------|-------------|
| `MEMORUS_API_KEY` | Yes | REST API auth key |
| `OPENAI_API_KEY` | ACE+LLM mode | LLM for reflector/distiller |
| `MEMORUS_DATA_DIR` | No | SQLite data dir (default: /data) |
| `MEMORUS_ACE_ENABLED` | No | Enable ACE pipeline (default: false) |
