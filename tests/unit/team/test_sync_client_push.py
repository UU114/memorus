"""Tests for AceSyncClient push (write) methods — STORY-068."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from memx.team.sync_client import (
    AceSyncClient,
    ConflictError,
    NominateResponse,
    NotFoundError,
    PermanentError,
    RetryableError,
    SupersedeResponse,
    SyncAuthError,
    VoteResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(
    status_code: int = 200,
    json_data: dict[str, Any] | None = None,
    text: str = "",
    headers: dict[str, str] | None = None,
    content: bytes = b"{}",
) -> MagicMock:
    """Create a mock httpx Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text or ""
    resp.headers = headers or {}
    resp.content = content
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("No JSON")
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client() -> AceSyncClient:
    """Create AceSyncClient with mocked httpx."""
    return AceSyncClient(
        server_url="https://sync.example.com",
        auth_token="test-token-123",
        max_retries=0,
        retry_backoff=0.01,
    )


@pytest.fixture
def sample_bullet() -> dict[str, Any]:
    """A minimal sanitized bullet dict."""
    return {
        "text": "Python prefers snake_case for variable names",
        "tags": ["python", "style"],
    }


# ---------------------------------------------------------------------------
# 1. nominate_bullet — normal success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nominate_bullet_success(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """nominate_bullet returns NominateResponse on 200."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-001", "status": "staging"}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.nominate_bullet(sample_bullet)

    assert isinstance(result, NominateResponse)
    assert result.id == "b-001"
    assert result.status == "staging"
    # Verify request payload
    client._client.request.assert_called_once_with(
        "POST",
        "/api/v1/bullets/nominate",
        json={"bullet": sample_bullet, "priority": "normal"},
    )


# ---------------------------------------------------------------------------
# 2. nominate_bullet — urgent priority
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nominate_bullet_urgent(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """nominate_bullet with priority='urgent' sends correct payload."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-002", "status": "staging"}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.nominate_bullet(sample_bullet, priority="urgent")

    assert result.id == "b-002"
    client._client.request.assert_called_once_with(
        "POST",
        "/api/v1/bullets/nominate",
        json={"bullet": sample_bullet, "priority": "urgent"},
    )


# ---------------------------------------------------------------------------
# 3. nominate_bullet — duplicate (409 Conflict)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nominate_bullet_duplicate_conflict(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """nominate_bullet raises ConflictError with existing_id on 409."""
    mock_resp = _mock_response(
        409, json_data={"id": "b-existing"}, content=b'{"id": "b-existing"}'
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(ConflictError) as exc_info:
        await client.nominate_bullet(sample_bullet)

    assert exc_info.value.existing_id == "b-existing"
    assert "Already exists" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. cast_vote — up/down success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cast_vote_up(client: AceSyncClient) -> None:
    """cast_vote('up') returns VoteResponse with counts."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-001", "upvotes": 5, "downvotes": 1}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.cast_vote("b-001", "up")

    assert isinstance(result, VoteResponse)
    assert result.id == "b-001"
    assert result.upvotes == 5
    assert result.downvotes == 1


@pytest.mark.asyncio
async def test_cast_vote_down(client: AceSyncClient) -> None:
    """cast_vote('down') returns VoteResponse."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-001", "upvotes": 3, "downvotes": 4}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.cast_vote("b-001", "down")

    assert result.downvotes == 4
    client._client.request.assert_called_once_with(
        "POST",
        "/api/v1/bullets/b-001/vote",
        json={"vote": "down"},
    )


# ---------------------------------------------------------------------------
# 5. cast_vote — invalid vote value
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cast_vote_invalid_value(client: AceSyncClient) -> None:
    """cast_vote raises ValueError for invalid vote strings."""
    with pytest.raises(ValueError, match="Invalid vote"):
        await client.cast_vote("b-001", "maybe")


# ---------------------------------------------------------------------------
# 6. cast_vote — bullet not found (404)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cast_vote_not_found(client: AceSyncClient) -> None:
    """cast_vote raises NotFoundError on 404."""
    mock_resp = _mock_response(404, text="Not Found")
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(NotFoundError, match="Not found"):
        await client.cast_vote("b-nonexistent", "up")


# ---------------------------------------------------------------------------
# 7. propose_supersede — success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_supersede_success(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """propose_supersede returns SupersedeResponse on 200."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-new-001", "status": "pending_review"}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.propose_supersede("b-old-001", sample_bullet)

    assert isinstance(result, SupersedeResponse)
    assert result.id == "b-new-001"
    assert result.status == "pending_review"
    client._client.request.assert_called_once_with(
        "POST",
        "/api/v1/bullets/supersede",
        json={
            "origin_id": "b-old-001",
            "new_bullet": sample_bullet,
            "priority": "normal",
        },
    )


# ---------------------------------------------------------------------------
# 8. propose_supersede — urgent priority
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_supersede_urgent(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """propose_supersede with priority='urgent' sends correct payload."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-new-002", "status": "pending_review"}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = await client.propose_supersede(
        "b-old-001", sample_bullet, priority="urgent"
    )

    assert result.id == "b-new-002"
    call_kwargs = client._client.request.call_args
    assert call_kwargs[1]["json"]["priority"] == "urgent"


# ---------------------------------------------------------------------------
# 9. RetryableError on network failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retryable_error_on_connect_failure(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """Network failures raise RetryableError."""
    import httpx

    client._client.request = AsyncMock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    with pytest.raises(RetryableError) as exc_info:
        await client.nominate_bullet(sample_bullet)

    assert "Connection refused" in str(exc_info.value)


@pytest.mark.asyncio
async def test_retryable_error_on_server_error(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """5xx responses raise RetryableError."""
    mock_resp = _mock_response(503, text="Service Unavailable")
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(RetryableError, match="Server error 503"):
        await client.nominate_bullet(sample_bullet)


@pytest.mark.asyncio
async def test_retryable_error_on_rate_limit(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """429 responses raise RetryableError with retry_after."""
    mock_resp = _mock_response(
        429, text="Too Many Requests", headers={"Retry-After": "2.5"}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(RetryableError) as exc_info:
        await client.nominate_bullet(sample_bullet)

    assert exc_info.value.retry_after == 2.5


# ---------------------------------------------------------------------------
# 10. PermanentError on bad request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_permanent_error_on_bad_request(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """4xx (non-special) responses raise PermanentError."""
    mock_resp = _mock_response(422, text="Unprocessable Entity")
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(PermanentError, match="Client error 422"):
        await client.nominate_bullet(sample_bullet)


@pytest.mark.asyncio
async def test_auth_error_on_401(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """401 raises SyncAuthError."""
    mock_resp = _mock_response(401, text="Unauthorized")
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(SyncAuthError):
        await client.nominate_bullet(sample_bullet)


# ---------------------------------------------------------------------------
# 11. Sync wrappers work
# ---------------------------------------------------------------------------


def test_nominate_bullet_sync(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """nominate_bullet_sync delegates to async version."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-sync-001", "status": "staging"}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = client.nominate_bullet_sync(sample_bullet)

    assert isinstance(result, NominateResponse)
    assert result.id == "b-sync-001"


def test_cast_vote_sync(client: AceSyncClient) -> None:
    """cast_vote_sync delegates to async version."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-001", "upvotes": 1, "downvotes": 0}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = client.cast_vote_sync("b-001", "up")

    assert isinstance(result, VoteResponse)
    assert result.upvotes == 1


def test_propose_supersede_sync(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """propose_supersede_sync delegates to async version."""
    mock_resp = _mock_response(
        200, json_data={"id": "b-sup-001", "status": "pending_review"}
    )
    client._client.request = AsyncMock(return_value=mock_resp)

    result = client.propose_supersede_sync("b-old", sample_bullet)

    assert isinstance(result, SupersedeResponse)
    assert result.status == "pending_review"


# ---------------------------------------------------------------------------
# 12. Response model validation
# ---------------------------------------------------------------------------


def test_nominate_response_model() -> None:
    """NominateResponse validates fields correctly."""
    resp = NominateResponse(id="x", status="staging")
    assert resp.id == "x"
    assert resp.status == "staging"


def test_vote_response_model() -> None:
    """VoteResponse validates fields correctly."""
    resp = VoteResponse(id="x", upvotes=3, downvotes=1)
    assert resp.upvotes == 3


def test_supersede_response_model() -> None:
    """SupersedeResponse validates fields correctly."""
    resp = SupersedeResponse(id="x", status="pending_review")
    assert resp.status == "pending_review"


def test_nominate_response_rejects_missing_fields() -> None:
    """NominateResponse rejects missing required fields."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        NominateResponse(id="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nominate_bullet_invalid_priority(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """nominate_bullet rejects invalid priority values."""
    with pytest.raises(ValueError, match="Invalid priority"):
        await client.nominate_bullet(sample_bullet, priority="low")


@pytest.mark.asyncio
async def test_propose_supersede_invalid_priority(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """propose_supersede rejects invalid priority values."""
    with pytest.raises(ValueError, match="Invalid priority"):
        await client.propose_supersede("b-1", sample_bullet, priority="high")


@pytest.mark.asyncio
async def test_conflict_error_no_body(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """ConflictError with empty response body sets existing_id=None."""
    mock_resp = _mock_response(409, json_data={}, content=b"{}")
    client._client.request = AsyncMock(return_value=mock_resp)

    with pytest.raises(ConflictError) as exc_info:
        await client.nominate_bullet(sample_bullet)

    assert exc_info.value.existing_id is None


@pytest.mark.asyncio
async def test_retryable_error_on_timeout(
    client: AceSyncClient, sample_bullet: dict[str, Any]
) -> None:
    """ReadTimeout raises RetryableError."""
    import httpx

    client._client.request = AsyncMock(
        side_effect=httpx.ReadTimeout("Read timed out")
    )

    with pytest.raises(RetryableError, match="Read timed out"):
        await client.nominate_bullet(sample_bullet)


# ---------------------------------------------------------------------------
# Exception hierarchy checks
# ---------------------------------------------------------------------------


def test_retryable_error_is_sync_error() -> None:
    """RetryableError inherits from SyncError."""
    from memx.team.sync_client import SyncError

    err = RetryableError("test")
    assert isinstance(err, SyncError)


def test_permanent_error_is_sync_error() -> None:
    """PermanentError inherits from SyncError."""
    from memx.team.sync_client import SyncError

    err = PermanentError("test")
    assert isinstance(err, SyncError)


def test_conflict_error_is_permanent() -> None:
    """ConflictError inherits from PermanentError."""
    err = ConflictError("dup", existing_id="b-1")
    assert isinstance(err, PermanentError)
    assert err.existing_id == "b-1"


def test_not_found_error_is_permanent() -> None:
    """NotFoundError inherits from PermanentError."""
    err = NotFoundError("gone")
    assert isinstance(err, PermanentError)
