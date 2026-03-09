"""MemX REST API — expose Memory over HTTP with FastAPI.

Requires: pip install memx[api]
Run:      memx-api --no-auth   (development)
          MEMX_API_KEY=secret memx-api   (production)
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

    Fail-closed: if MEMX_API_KEY is set, requests without a valid key are rejected.
    If --no-auth mode is active, all requests pass through.
    """
    if _NO_AUTH:
        return None
    expected = os.environ.get("MEMX_API_KEY")
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


def create_app(config: Optional[dict[str, Any]] = None) -> FastAPI:
    """Create a FastAPI application with MemX endpoints.

    Raises:
        ImportError: If ``fastapi`` is not installed.
    """
    if FastAPI is None:
        raise ImportError(
            "REST API requires 'fastapi' and 'uvicorn'. "
            "Install with: pip install memx[api]"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from memx.core.memory import Memory

        app.state.memory = Memory(config=config)
        yield
        app.state.memory = None

    app = FastAPI(
        title="MemX API",
        description="MemX Memory REST API",
        version="0.2.1",
        lifespan=lifespan,
    )

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


def main() -> None:
    """Entry point for ``memx-api`` console script."""
    if FastAPI is None:
        print(
            "ERROR: REST API requires 'fastapi' and 'uvicorn'. "
            "Install with: pip install memx[api]",
            file=sys.stderr,
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="MemX REST API server")
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
    elif not os.environ.get("MEMX_API_KEY"):
        print(
            "ERROR: MEMX_API_KEY environment variable is required.\n"
            "Set it or pass --no-auth for local development.",
            file=sys.stderr,
        )
        sys.exit(1)

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
