"""Memorus REST API — expose Memory over HTTP with FastAPI.

Requires: pip install memorus[api]
Run:      memorus-api --no-auth   (development)
          MEMORUS_API_KEY=secret memorus-api   (production)

Backend: the Memory class is obtained from the backend-aware top-level
``memorus`` package, so ``MEMORUS_BACKEND=rust`` routes every endpoint through
the Rust core (every method used here — add/search/get_all/get/delete/status —
is implemented by the Rust shim). See doc/migration-rust-source-of-truth.md.

DEPRECATION: per the locked migration decision, MCP/REST entry points will be
unified onto the Rust ``memorus-server``; this Python REST API is transitional.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastapi import Depends, FastAPI, HTTPException, Request, Security
    from fastapi.security import APIKeyHeader
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]
    BaseModel = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Pydantic models (only defined when FastAPI is available)
# ---------------------------------------------------------------------------

if BaseModel is not None:

    class AddMemoryRequest(BaseModel):
        content: str
        user_id: str | None = None
        metadata: dict[str, Any] | None = None

    class AddMemoryResponse(BaseModel):
        results: dict[str, Any]

    class SearchQuery(BaseModel):
        query: str
        user_id: str | None = None
        limit: int = Field(default=100, ge=1, le=1000)

    class StatusResponse(BaseModel):
        status: dict[str, Any]

    class DeleteResponse(BaseModel):
        status: str
        memory_id: str


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False) if FastAPI else None
_NO_AUTH = False


def _verify_api_key(
    api_key: str | None = Security(_api_key_header) if _api_key_header else None,
) -> str | None:
    """Validate the API key header.

    Fail-closed: if MEMORUS_API_KEY is set, requests without a valid key are rejected.
    If --no-auth mode is active, all requests pass through.
    """
    if _NO_AUTH:
        return None
    expected = os.environ.get("MEMORUS_API_KEY")
    if expected is None:
        # Should not reach here — main() blocks startup without key or --no-auth
        raise HTTPException(status_code=500, detail="Server misconfigured: no API key")
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _get_memory_dep(request: Request) -> Any:
    """FastAPI dependency: retrieve Memory from app state."""
    return request.app.state.memory


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    """Create a FastAPI application with Memorus endpoints.

    Raises:
        ImportError: If ``fastapi`` is not installed.
    """
    if FastAPI is None:
        raise ImportError(
            "REST API requires 'fastapi' and 'uvicorn'. "
            "Install with: pip install memorus[api]"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Backend-aware import: honours MEMORUS_BACKEND (python default / rust).
        from memorus import Memory

        app.state.memory = Memory(config=config)
        yield
        app.state.memory = None

    app = FastAPI(
        title="Memorus API",
        description="Memorus Memory REST API",
        version="0.2.1",
        lifespan=lifespan,
    )

    # ---- Endpoints --------------------------------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        # Unauthenticated liveness probe for container orchestrators.
        return {"status": "ok"}

    @app.post("/memories", response_model=AddMemoryResponse)
    async def create_memory(
        body: AddMemoryRequest,
        _key: str | None = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(
            memory.add,
            body.content,
            user_id=body.user_id,
            metadata=body.metadata,
        )
        return {"results": result}

    @app.get("/memories/search")
    async def search_memories(
        query: str,
        user_id: str | None = None,
        limit: int = 100,
        _key: str | None = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            memory.search, query, user_id=user_id, limit=limit
        )

    @app.get("/memories")
    async def list_memories(
        user_id: str | None = None,
        limit: int = 100,
        _key: str | None = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if user_id is not None:
            kwargs["user_id"] = user_id
        if limit != 100:
            kwargs["limit"] = limit
        return await asyncio.to_thread(memory.get_all, **kwargs)

    @app.get("/memories/{memory_id}")
    async def get_memory(
        memory_id: str,
        _key: str | None = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, Any]:
        return await asyncio.to_thread(memory.get, memory_id)

    @app.delete("/memories/{memory_id}", response_model=DeleteResponse)
    async def delete_memory(
        memory_id: str,
        _key: str | None = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, str]:
        await asyncio.to_thread(memory.delete, memory_id)
        return {"status": "deleted", "memory_id": memory_id}

    @app.get("/status", response_model=StatusResponse)
    async def get_status(
        user_id: str | None = None,
        _key: str | None = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(memory.status, user_id=user_id)
        return {"status": result}

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_config_from_env() -> dict[str, Any] | None:
    """Build a Memorus config dict from environment variables.

    Container/K8s deployments configure the service purely via env. Returns
    None when OPENAI_API_KEY is absent so local runs keep library defaults
    (offline ONNX path). Produced shape is the standard flat config dict
    accepted by ``Memory`` on both backends.

    Supported variables:
      OPENAI_API_KEY          — API key for embedder + LLM (enables this path)
      OPENAI_BASE_URL         — OpenAI-compatible endpoint override
      MEMORUS_EMBEDDING_MODEL — embedding model (default: text-embedding-3-small)
      MEMORUS_EMBEDDING_DIMS  — embedding dimensions (default: 1536)
      MEMORUS_LLM_MODEL       — LLM model (default: gpt-4o-mini)
      MEMORUS_DATA_DIR        — persistent data directory (default: /data)
      MEMORUS_ACE_ENABLED     — "false"/"0" disables the ACE pipeline
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    embed_model = os.environ.get("MEMORUS_EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model = os.environ.get("MEMORUS_LLM_MODEL", "gpt-4o-mini")
    data_dir = os.environ.get("MEMORUS_DATA_DIR", "/data")
    try:
        dims = int(os.environ.get("MEMORUS_EMBEDDING_DIMS", "1536"))
    except ValueError:
        logger.warning("Invalid MEMORUS_EMBEDDING_DIMS, using 1536")
        dims = 1536

    embedder_cfg: dict[str, Any] = {
        "model": embed_model,
        "embedding_dims": dims,
        "api_key": api_key,
    }
    llm_cfg: dict[str, Any] = {"model": llm_model, "api_key": api_key}
    if base_url:
        embedder_cfg["openai_base_url"] = base_url
        llm_cfg["openai_base_url"] = base_url

    config: dict[str, Any] = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "memorus",
                "path": f"{data_dir}/qdrant",
                "embedding_model_dims": dims,
            },
        },
        "embedder": {"provider": "openai", "config": embedder_cfg},
        "llm": {"provider": "openai", "config": llm_cfg},
    }
    ace_env = os.environ.get("MEMORUS_ACE_ENABLED", "").strip().lower()
    if ace_env in ("false", "0", "no", "off"):
        config["ace_enabled"] = False
    logger.info(
        "Config from env: embed=%s dims=%d llm=%s base_url=%s data_dir=%s",
        embed_model, dims, llm_model, base_url or "(default)", data_dir,
    )
    return config


def main() -> None:
    """Entry point for ``memorus-api`` console script."""
    if FastAPI is None:
        print(
            "ERROR: REST API requires 'fastapi' and 'uvicorn'. "
            "Install with: pip install memorus[api]",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Memorus REST API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable API key authentication (development only)",
    )
    args = parser.parse_args()

    # Fail-closed: require API key unless --no-auth is explicit
    global _NO_AUTH
    if args.no_auth:
        _NO_AUTH = True
        logger.warning("Authentication disabled via --no-auth flag")
    elif not os.environ.get("MEMORUS_API_KEY"):
        print(
            "ERROR: MEMORUS_API_KEY environment variable is required.\n"
            "Set it or pass --no-auth for local development.",
            file=sys.stderr,
        )
        sys.exit(1)

    import uvicorn

    uvicorn.run(create_app(_build_config_from_env()), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
