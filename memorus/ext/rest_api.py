"""Memorus REST API — expose Memory over HTTP with FastAPI.

Requires: pip install memorus[api]
Run:      memorus-api --no-auth   (development)
          MEMORUS_API_KEY=secret memorus-api   (production)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Optional

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
        user_id: Optional[str] = None
        metadata: Optional[dict[str, Any]] = None

    class AddMemoryResponse(BaseModel):
        results: dict[str, Any]

    class SearchQuery(BaseModel):
        query: str
        user_id: Optional[str] = None
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
    api_key: Optional[str] = Security(_api_key_header) if _api_key_header else None,
) -> Optional[str]:
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


def create_app(config: Optional[Any] = None) -> FastAPI:
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
        from memorus.core.memory import Memory

        app.state.memory = Memory(config=config)
        yield
        app.state.memory = None

    app = FastAPI(
        title="Memorus API",
        description="Memorus Memory REST API",
        version="0.2.1",
        lifespan=lifespan,
    )

    # ---- Health check (no auth required) -----------------------------------

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    # ---- Endpoints --------------------------------------------------------

    @app.post("/memories", response_model=AddMemoryResponse)
    async def create_memory(
        body: AddMemoryRequest,
        _key: Optional[str] = Depends(_verify_api_key),
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
        user_id: Optional[str] = None,
        limit: int = 100,
        _key: Optional[str] = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            memory.search, query, user_id=user_id, limit=limit
        )

    @app.get("/memories")
    async def list_memories(
        user_id: Optional[str] = None,
        limit: int = 100,
        _key: Optional[str] = Depends(_verify_api_key),
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
        _key: Optional[str] = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, Any]:
        return await asyncio.to_thread(memory.get, memory_id)

    @app.delete("/memories/{memory_id}", response_model=DeleteResponse)
    async def delete_memory(
        memory_id: str,
        _key: Optional[str] = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, str]:
        await asyncio.to_thread(memory.delete, memory_id)
        return {"status": "deleted", "memory_id": memory_id}

    @app.get("/status", response_model=StatusResponse)
    async def get_status(
        user_id: Optional[str] = None,
        _key: Optional[str] = Depends(_verify_api_key),
        memory: Any = Depends(_get_memory_dep),
    ) -> dict[str, Any]:
        result = await asyncio.to_thread(memory.status, user_id=user_id)
        return {"status": result}

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_mem0_config_from_env() -> Optional[Any]:
    """Build a mem0 MemoryConfig from environment variables.

    Constructs a proper MemoryConfig object for mem0 1.0.x which requires
    structured config objects rather than raw dicts.

    Supported variables:
      OPENAI_API_KEY          — API key for embedder + LLM backends (required)
      OPENAI_BASE_URL         — Override base URL (for OpenAI-compatible endpoints)
      MEMORUS_EMBEDDING_MODEL — Embedding model name (default: text-embedding-3-small)
      MEMORUS_EMBEDDING_DIMS  — Embedding output dimensions (default: 1536)
      MEMORUS_LLM_MODEL       — LLM model name for memory extraction (default: gpt-4o-mini)
      MEMORUS_DATA_DIR        — Persistent data directory (default: /data)
      MEMORUS_ACE_ENABLED     — Enable ACE pipeline (default: false)
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    embed_model = os.environ.get("MEMORUS_EMBEDDING_MODEL", "text-embedding-3-small")
    llm_model = os.environ.get("MEMORUS_LLM_MODEL", "gpt-4o-mini")
    dims_str = os.environ.get("MEMORUS_EMBEDDING_DIMS", "1536")
    data_dir = os.environ.get("MEMORUS_DATA_DIR", "/data")

    try:
        dims = int(dims_str)
    except ValueError:
        logger.warning("Invalid MEMORUS_EMBEDDING_DIMS=%r, using 1536", dims_str)
        dims = 1536

    if not api_key:
        logger.warning(
            "OPENAI_API_KEY not set — returning None, mem0 will use its defaults."
        )
        return None

    try:
        from mem0.configs.base import MemoryConfig

        embedder_cfg: dict[str, Any] = {
            "model": embed_model,
            "embedding_dims": dims,
            "api_key": api_key,
        }
        if base_url:
            embedder_cfg["openai_base_url"] = base_url

        llm_cfg: dict[str, Any] = {"model": llm_model, "api_key": api_key}
        if base_url:
            llm_cfg["openai_base_url"] = base_url

        mem_config = MemoryConfig(
            vector_store={
                "provider": "qdrant",
                "config": {
                    "collection_name": "memorus",
                    "path": f"{data_dir}/qdrant",
                    "embedding_model_dims": dims,
                },
            },
            embedder={"provider": "openai", "config": embedder_cfg},
            llm={"provider": "openai", "config": llm_cfg},
            history_db_path=f"{data_dir}/.mem0/history.db",
        )
        logger.info(
            "mem0 MemoryConfig built: embed=%s dims=%d llm=%s base_url=%s qdrant=%s/qdrant",
            embed_model, dims, llm_model, base_url or "(default)", data_dir,
        )
        return mem_config
    except Exception as exc:
        logger.error("Failed to build MemoryConfig: %s — returning None", exc)
        return None


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

    mem0_config = _build_mem0_config_from_env()
    uvicorn.run(create_app(config=mem0_config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
