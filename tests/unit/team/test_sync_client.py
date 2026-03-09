"""Tests for AceSyncClient — HTTP client for ACE Sync Server."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.team.sync_client import (
    AceSyncClient,
    BulletIndexEntry,
    IndexResponse,
    SyncAuthError,
    SyncConnectionError,
    SyncError,
    SyncRateLimitError,
    TaxonomyResponse,
    TaxonomyTag,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str = "",
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or ""
    resp.headers = headers or {}
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


@pytest.fixture
def client() -> AceSyncClient:
    """Create AceSyncClient with mocked httpx."""
    return AceSyncClient(
        server_url="https://sync.example.com",
        auth_token="test-token-123",
        max_retries=2,
        retry_backoff=0.01,  # fast retries for tests
    )


@pytest.fixture
def apikey_client() -> AceSyncClient:
    """Create AceSyncClient with API key auth."""
    return AceSyncClient(
        server_url="https://sync.example.com",
        auth_token="my-api-key",
        auth_type="apikey",
        max_retries=0,
        retry_backoff=0.01,
        team_id="team-42",
    )


# ---------------------------------------------------------------------------
# 1. pull_index — normal response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_index_normal(client: AceSyncClient) -> None:
    """pull_index returns parsed IndexResponse with bullets and cursor."""
    mock_data = {
        "bullets": [
            {"id": "b1", "updated_at": "2026-01-15T10:00:00Z", "status": "approved"},
            {"id": "b2", "updated_at": "2026-01-16T12:30:00Z", "status": "staging"},
        ],
        "cursor": "next-page-token",
    }
    mock_resp = _mock_response(json_data=mock_data)
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.pull_index()

    assert isinstance(result, IndexResponse)
    assert len(result.bullets) == 2
    assert result.bullets[0].id == "b1"
    assert result.bullets[0].status == "approved"
    assert result.cursor == "next-page-token"


@pytest.mark.asyncio
async def test_pull_index_with_since_and_tags(client: AceSyncClient) -> None:
    """pull_index passes since and tags as query parameters."""
    mock_resp = _mock_response(json_data={"bullets": [], "cursor": None})
    client._client.request = AsyncMock(return_value=mock_resp)

    since = datetime(2026, 1, 1, tzinfo=timezone.utc)
    await client.pull_index(since=since, tags=["python", "docker"])

    call_kwargs = client._client.request.call_args
    params = call_kwargs.kwargs.get("params") or call_kwargs[1].get("params", {})
    assert params["since"] == since.isoformat()
    assert params["tags"] == "python,docker"


@pytest.mark.asyncio
async def test_pull_index_empty_response(client: AceSyncClient) -> None:
    """pull_index handles empty bullet list."""
    mock_resp = _mock_response(json_data={"bullets": [], "cursor": None})
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.pull_index()

    assert result.bullets == []
    assert result.cursor is None


# ---------------------------------------------------------------------------
# 2. fetch_bullets — batch fetch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_bullets_normal(client: AceSyncClient) -> None:
    """fetch_bullets returns list of bullet dicts."""
    mock_data = {
        "bullets": [
            {"id": "b1", "content": "Use pytest fixtures"},
            {"id": "b2", "content": "Always pin dependencies"},
        ],
    }
    mock_resp = _mock_response(json_data=mock_data)
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.fetch_bullets(["b1", "b2"])

    assert len(result) == 2
    assert result[0]["id"] == "b1"


@pytest.mark.asyncio
async def test_fetch_bullets_empty_ids(client: AceSyncClient) -> None:
    """fetch_bullets with empty ID list returns empty without server call."""
    client._client.request = AsyncMock()

    result = await client.fetch_bullets([])

    assert result == []
    client._client.request.assert_not_called()


# ---------------------------------------------------------------------------
# 3. pull_taxonomy — response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_taxonomy_normal(client: AceSyncClient) -> None:
    """pull_taxonomy returns parsed TaxonomyResponse."""
    mock_data = {
        "tags": [
            {"name": "python", "aliases": ["py"], "parent": None},
            {"name": "testing", "aliases": [], "parent": "python"},
        ],
    }
    mock_resp = _mock_response(json_data=mock_data)
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.pull_taxonomy()

    assert isinstance(result, TaxonomyResponse)
    assert len(result.tags) == 2
    assert result.tags[0].name == "python"
    assert result.tags[0].aliases == ["py"]
    assert result.tags[1].parent == "python"


# ---------------------------------------------------------------------------
# 4. Auth header — API key vs Bearer
# ---------------------------------------------------------------------------


def test_bearer_auth_header(client: AceSyncClient) -> None:
    """Bearer auth sets Authorization header."""
    headers = client._build_headers()
    assert headers["Authorization"] == "Bearer test-token-123"
    assert "X-API-Key" not in headers


def test_apikey_auth_header(apikey_client: AceSyncClient) -> None:
    """API key auth sets X-API-Key header."""
    headers = apikey_client._build_headers()
    assert headers["X-API-Key"] == "my-api-key"
    assert "Authorization" not in headers


def test_team_id_header(apikey_client: AceSyncClient) -> None:
    """team_id is sent as X-Team-Id header."""
    headers = apikey_client._build_headers()
    assert headers["X-Team-Id"] == "team-42"


def test_invalid_auth_type_raises() -> None:
    """Unsupported auth_type raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported auth_type"):
        AceSyncClient(
            server_url="https://example.com",
            auth_token="token",
            auth_type="oauth",
        )


# ---------------------------------------------------------------------------
# 5. Timeout and retry behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_on_server_error(client: AceSyncClient) -> None:
    """Server 500 errors trigger retries up to max_retries."""
    error_resp = _mock_response(status_code=500, text="Internal Server Error")
    ok_resp = _mock_response(json_data={"bullets": [], "cursor": None})

    client._client.request = AsyncMock(side_effect=[error_resp, ok_resp])

    result = await client.pull_index()
    assert result.bullets == []
    assert client._client.request.call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted_raises(client: AceSyncClient) -> None:
    """All retries exhausted raises SyncError."""
    error_resp = _mock_response(status_code=500, text="down")

    client._client.request = AsyncMock(return_value=error_resp)

    with pytest.raises(SyncError, match="Server error 500"):
        await client.pull_index()

    # max_retries=2 means 3 total attempts
    assert client._client.request.call_count == 3


# ---------------------------------------------------------------------------
# 6. Server unreachable exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_error_raises(client: AceSyncClient) -> None:
    """Network failures raise SyncConnectionError after retries."""
    import httpx

    client._client.request = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(SyncConnectionError, match="Server unreachable"):
        await client.pull_index()

    assert client._client.request.call_count == 3


@pytest.mark.asyncio
async def test_timeout_raises_connection_error(client: AceSyncClient) -> None:
    """Timeout raises SyncConnectionError."""
    import httpx

    client._client.request = AsyncMock(
        side_effect=httpx.ReadTimeout("Read timed out")
    )

    with pytest.raises(SyncConnectionError, match="Server unreachable"):
        await client.pull_index()


# ---------------------------------------------------------------------------
# 7. 429 Rate Limit backoff
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_retry(client: AceSyncClient) -> None:
    """429 triggers retry with backoff."""
    rate_resp = _mock_response(status_code=429, text="Too Many Requests")
    ok_resp = _mock_response(json_data={"tags": []})

    client._client.request = AsyncMock(side_effect=[rate_resp, ok_resp])

    result = await client.pull_taxonomy()
    assert isinstance(result, TaxonomyResponse)
    assert client._client.request.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_with_retry_after(client: AceSyncClient) -> None:
    """429 with Retry-After header uses that delay."""
    rate_resp = _mock_response(
        status_code=429, text="slow down", headers={"Retry-After": "0.01"}
    )
    ok_resp = _mock_response(json_data={"tags": []})

    client._client.request = AsyncMock(side_effect=[rate_resp, ok_resp])

    result = await client.pull_taxonomy()
    assert isinstance(result, TaxonomyResponse)


@pytest.mark.asyncio
async def test_rate_limit_exhausted_raises(client: AceSyncClient) -> None:
    """All retries exhausted on 429 raises SyncRateLimitError."""
    rate_resp = _mock_response(status_code=429, text="Too Many Requests")

    client._client.request = AsyncMock(return_value=rate_resp)

    with pytest.raises(SyncRateLimitError, match="Rate limit exceeded"):
        await client.pull_taxonomy()


# ---------------------------------------------------------------------------
# 8. 401 Unauthorized — immediate failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auth_error_no_retry(client: AceSyncClient) -> None:
    """401 raises SyncAuthError immediately without retry."""
    auth_resp = _mock_response(status_code=401, text="Unauthorized")
    client._client.request = AsyncMock(return_value=auth_resp)

    with pytest.raises(SyncAuthError, match="Authentication failed"):
        await client.pull_index()

    # No retries on 401
    assert client._client.request.call_count == 1


# ---------------------------------------------------------------------------
# 9. Large batch auto-chunking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_bullets_auto_chunking(client: AceSyncClient) -> None:
    """fetch_bullets splits > 100 IDs into multiple requests."""
    ids = [f"b{i}" for i in range(250)]

    def make_resp(request_mock: Any, path: str, **kwargs: Any) -> MagicMock:
        body = kwargs.get("json", {})
        chunk_ids = body.get("ids", [])
        return _mock_response(
            json_data={"bullets": [{"id": bid} for bid in chunk_ids]}
        )

    client._client.request = AsyncMock(side_effect=make_resp)

    result = await client.fetch_bullets(ids)

    # 250 IDs -> 3 chunks (100 + 100 + 50)
    assert client._client.request.call_count == 3
    assert len(result) == 250


# ---------------------------------------------------------------------------
# 10. Response validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invalid_json_raises(client: AceSyncClient) -> None:
    """Invalid JSON in response raises SyncError."""
    bad_resp = MagicMock()
    bad_resp.status_code = 200
    bad_resp.json.side_effect = ValueError("Expecting value")
    bad_resp.text = "not-json"
    bad_resp.headers = {}

    client._client.request = AsyncMock(return_value=bad_resp)

    with pytest.raises(SyncError, match="Invalid JSON"):
        await client.pull_index()


@pytest.mark.asyncio
async def test_invalid_index_response_raises(client: AceSyncClient) -> None:
    """Pydantic validation error for malformed data raises."""
    # Missing required 'id' field in bullet entry
    mock_resp = _mock_response(json_data={
        "bullets": [{"status": "approved"}],  # missing id and updated_at
    })
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(Exception):  # Pydantic ValidationError
        await client.pull_index()


@pytest.mark.asyncio
async def test_fetch_bullets_invalid_response_type(client: AceSyncClient) -> None:
    """Non-list 'bullets' in fetch response raises SyncError."""
    mock_resp = _mock_response(json_data={"bullets": "not-a-list"})
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(SyncError, match="Expected 'bullets' list"):
        await client.fetch_bullets(["b1"])


@pytest.mark.asyncio
async def test_client_error_4xx_raises(client: AceSyncClient) -> None:
    """Non-401/429 4xx errors raise SyncError without retry."""
    resp_404 = _mock_response(status_code=404, text="Not Found")
    client._client.request = AsyncMock(return_value=resp_404)

    with pytest.raises(SyncError, match="Client error 404"):
        await client.pull_index()

    assert client._client.request.call_count == 1


# ---------------------------------------------------------------------------
# Context manager and close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_close(client: AceSyncClient) -> None:
    """close() calls aclose on underlying httpx client."""
    client._client.aclose = AsyncMock()
    await client.close()
    client._client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_context_manager(client: AceSyncClient) -> None:
    """Client works as async context manager."""
    client._client.aclose = AsyncMock()
    async with client as c:
        assert c is client
    client._client.aclose.assert_awaited_once()


# ---------------------------------------------------------------------------
# httpx not installed
# ---------------------------------------------------------------------------


def test_httpx_import_error() -> None:
    """Clear error when httpx is not installed."""
    import memx.team.sync_client as mod

    original = mod._httpx
    try:
        mod._httpx = None  # reset cache
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx is required"):
                mod._get_httpx()
    finally:
        mod._httpx = original


# ---------------------------------------------------------------------------
# Pydantic model unit tests
# ---------------------------------------------------------------------------


def test_bullet_index_entry_model() -> None:
    """BulletIndexEntry validates correctly."""
    entry = BulletIndexEntry(
        id="b1",
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status="approved",
    )
    assert entry.id == "b1"
    assert entry.status == "approved"


def test_taxonomy_tag_defaults() -> None:
    """TaxonomyTag has correct defaults."""
    tag = TaxonomyTag(name="python")
    assert tag.aliases == []
    assert tag.parent is None
