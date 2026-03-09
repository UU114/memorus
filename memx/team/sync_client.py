"""AceSyncClient — HTTP client for pulling team knowledge from ACE Sync Server.

Supports incremental bullet index pulls, batch bullet fetching, and taxonomy
retrieval. Uses httpx for async HTTP with configurable timeout, retry, and
authentication (API key or Bearer token).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy httpx import — the module stays importable even without httpx
# ---------------------------------------------------------------------------

_httpx = None  # type: Any


def _get_httpx() -> Any:
    """Import httpx lazily, raising a clear error if unavailable."""
    global _httpx
    if _httpx is not None:
        return _httpx
    try:
        import httpx  # type: ignore[import-untyped]

        _httpx = httpx
        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for AceSyncClient. "
            "Install it with: pip install memx[team]"
        )


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class SyncError(Exception):
    """Base exception for sync client errors."""


class SyncConnectionError(SyncError):
    """Server unreachable or network timeout."""


class SyncAuthError(SyncError):
    """Authentication failed (401)."""


class SyncRateLimitError(SyncError):
    """Rate limit exceeded (429)."""


class RetryableError(SyncError):
    """Network failure, can be retried."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PermanentError(SyncError):
    """Non-retryable error (auth, bad request, conflict)."""


class ConflictError(PermanentError):
    """409 Conflict — duplicate nomination."""

    def __init__(self, message: str, existing_id: str | None = None) -> None:
        super().__init__(message)
        self.existing_id = existing_id


class NotFoundError(PermanentError):
    """404 Not Found — bullet doesn't exist."""


# ---------------------------------------------------------------------------
# Response models (Pydantic)
# ---------------------------------------------------------------------------


class BulletIndexEntry(BaseModel):
    """Single entry in the bullet index response."""

    id: str
    updated_at: datetime
    status: str


class IndexResponse(BaseModel):
    """Response from pull_index endpoint."""

    bullets: list[BulletIndexEntry] = Field(default_factory=list)
    cursor: Optional[str] = None


class TaxonomyTag(BaseModel):
    """Single tag in the taxonomy."""

    name: str
    aliases: list[str] = Field(default_factory=list)
    parent: Optional[str] = None


class TaxonomyResponse(BaseModel):
    """Response from pull_taxonomy endpoint."""

    tags: list[TaxonomyTag] = Field(default_factory=list)


class NominateResponse(BaseModel):
    """Response from nominate_bullet endpoint."""

    id: str
    status: str


class VoteResponse(BaseModel):
    """Response from cast_vote endpoint."""

    id: str
    upvotes: int
    downvotes: int


class SupersedeResponse(BaseModel):
    """Response from propose_supersede endpoint."""

    id: str
    status: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TIMEOUT = 30
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF = 1.0
_FETCH_CHUNK_SIZE = 100  # max IDs per fetch_bullets request


# ---------------------------------------------------------------------------
# AceSyncClient
# ---------------------------------------------------------------------------


class AceSyncClient:
    """HTTP client for pulling team knowledge from ACE Sync Server.

    Args:
        server_url: Base URL of the ACE Sync Server (no trailing slash).
        auth_token: API key or Bearer token for authentication.
        auth_type: ``"bearer"`` (default) or ``"apikey"``.
        timeout: Request timeout in seconds.
        max_retries: Number of retries on transient failures (5xx, timeouts, 429).
        retry_backoff: Base backoff multiplier in seconds for exponential backoff.
        team_id: Optional team identifier sent as ``X-Team-Id`` header.
    """

    def __init__(
        self,
        server_url: str,
        auth_token: str,
        *,
        auth_type: str = "bearer",
        timeout: int = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_RETRIES,
        retry_backoff: float = _DEFAULT_BACKOFF,
        team_id: str | None = None,
    ) -> None:
        httpx = _get_httpx()

        self._server_url = server_url.rstrip("/")
        self._auth_token = auth_token
        self._auth_type = auth_type.lower()
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._team_id = team_id

        headers = self._build_headers()
        self._client = httpx.AsyncClient(
            base_url=self._server_url,
            headers=headers,
            timeout=httpx.Timeout(timeout),
        )

    # -- header helpers -----------------------------------------------------

    def _build_headers(self) -> dict[str, str]:
        """Build default headers including auth."""
        headers: dict[str, str] = {"Accept": "application/json"}

        if self._auth_type == "bearer":
            headers["Authorization"] = f"Bearer {self._auth_token}"
        elif self._auth_type == "apikey":
            headers["X-API-Key"] = self._auth_token
        else:
            raise ValueError(f"Unsupported auth_type: {self._auth_type!r}")

        if self._team_id:
            headers["X-Team-Id"] = self._team_id

        return headers

    # -- low-level request with retry --------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute an HTTP request with retry and error handling.

        Returns parsed JSON response as dict.
        Raises SyncConnectionError, SyncAuthError, SyncRateLimitError, or SyncError.
        """
        httpx = _get_httpx()
        last_exc: Exception | None = None

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json=json_body,
                )
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    delay = self._retry_backoff * (2 ** attempt)
                    logger.warning(
                        "Connection error on %s %s (attempt %d/%d), retrying in %.1fs: %s",
                        method, path, attempt + 1, self._max_retries + 1, delay, exc,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise SyncConnectionError(
                    f"Server unreachable after {self._max_retries + 1} attempts: {exc}"
                ) from exc

            # Handle status codes
            if response.status_code == 401:
                raise SyncAuthError(
                    f"Authentication failed (401): {response.text}"
                )

            if response.status_code == 429:
                if attempt < self._max_retries:
                    # Use Retry-After header if present, otherwise exponential backoff
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = float(retry_after)
                        except ValueError:
                            delay = self._retry_backoff * (2 ** attempt)
                    else:
                        delay = self._retry_backoff * (2 ** attempt)
                    logger.warning(
                        "Rate limited (429) on %s %s (attempt %d/%d), "
                        "retrying in %.1fs",
                        method, path, attempt + 1, self._max_retries + 1, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise SyncRateLimitError(
                    f"Rate limit exceeded after {self._max_retries + 1} attempts"
                )

            if response.status_code >= 500:
                last_exc = SyncError(
                    f"Server error {response.status_code}: {response.text}"
                )
                if attempt < self._max_retries:
                    delay = self._retry_backoff * (2 ** attempt)
                    logger.warning(
                        "Server error %d on %s %s (attempt %d/%d), "
                        "retrying in %.1fs",
                        response.status_code, method, path,
                        attempt + 1, self._max_retries + 1, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise last_exc

            if response.status_code >= 400:
                raise SyncError(
                    f"Client error {response.status_code}: {response.text}"
                )

            # Success — parse JSON
            try:
                return response.json()  # type: ignore[no-any-return]
            except Exception as exc:
                raise SyncError(
                    f"Invalid JSON response from {method} {path}: {exc}"
                ) from exc

        # Should not reach here, but safety net
        raise SyncConnectionError(
            f"Request failed after {self._max_retries + 1} attempts"
        ) from last_exc

    # -- public async API ---------------------------------------------------

    async def pull_index(
        self,
        *,
        since: datetime | None = None,
        tags: list[str] | None = None,
    ) -> IndexResponse:
        """Pull incremental bullet index from the server.

        Args:
            since: Only return bullets updated after this timestamp.
            tags: Filter by tag names.

        Returns:
            IndexResponse with bullet entries and optional cursor.
        """
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since.isoformat()
        if tags:
            params["tags"] = ",".join(tags)

        data = await self._request("GET", "/api/v1/bullets/index", params=params)
        return IndexResponse.model_validate(data)

    async def fetch_bullets(self, ids: list[str]) -> list[dict[str, Any]]:
        """Batch fetch full Bullet data (with vectors) by IDs.

        Automatically chunks into batches of 100 to avoid oversized requests.

        Args:
            ids: List of bullet IDs to fetch.

        Returns:
            List of bullet dicts (TeamBullet-compatible).
        """
        if not ids:
            return []

        all_bullets: list[dict[str, Any]] = []

        # Auto-chunk large batches
        for i in range(0, len(ids), _FETCH_CHUNK_SIZE):
            chunk = ids[i : i + _FETCH_CHUNK_SIZE]
            data = await self._request(
                "POST", "/api/v1/bullets/fetch", json_body={"ids": chunk}
            )
            bullets = data.get("bullets", [])
            if not isinstance(bullets, list):
                raise SyncError(
                    f"Expected 'bullets' list in response, got {type(bullets).__name__}"
                )
            all_bullets.extend(bullets)

        return all_bullets

    async def pull_taxonomy(self) -> TaxonomyResponse:
        """Fetch the team Tag Taxonomy.

        Returns:
            TaxonomyResponse with all tags.
        """
        data = await self._request("GET", "/api/v1/taxonomy")
        return TaxonomyResponse.model_validate(data)

    # -- push-specific request helper ----------------------------------------

    async def _push_request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a push request with push-specific error handling.

        Handles 409 Conflict, 404 Not Found, and converts connection errors
        to RetryableError for caller-side retry logic.
        """
        httpx = _get_httpx()
        try:
            response = await self._client.request(
                method, path, json=json_body,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout) as exc:
            raise RetryableError(str(exc)) from exc

        if response.status_code == 401:
            raise SyncAuthError("Authentication failed (401)")
        if response.status_code == 404:
            raise NotFoundError(f"Not found: {path}")
        if response.status_code == 409:
            data = response.json() if response.content else {}
            raise ConflictError("Already exists", existing_id=data.get("id"))
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RetryableError(
                "Rate limited",
                retry_after=float(retry_after) if retry_after else None,
            )
        if response.status_code >= 500:
            raise RetryableError(f"Server error {response.status_code}")
        if response.status_code >= 400:
            raise PermanentError(
                f"Client error {response.status_code}: {response.text}"
            )

        return response.json()  # type: ignore[no-any-return]

    # -- public push API ----------------------------------------------------

    _VALID_PRIORITIES = {"normal", "urgent"}
    _VALID_VOTES = {"up", "down"}

    async def nominate_bullet(
        self,
        bullet: dict[str, Any],
        *,
        priority: str = "normal",
    ) -> NominateResponse:
        """Upload a sanitized bullet to team Staging.

        Args:
            bullet: Sanitized bullet dict to nominate.
            priority: ``"normal"`` (default) or ``"urgent"``.

        Returns:
            NominateResponse with id and status.

        Raises:
            ConflictError: If the bullet already exists (409).
            RetryableError: On network or server failure.
            ValueError: If priority is invalid.
        """
        if priority not in self._VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority {priority!r}, must be one of {self._VALID_PRIORITIES}"
            )
        data = await self._push_request(
            "POST",
            "/api/v1/bullets/nominate",
            json_body={"bullet": bullet, "priority": priority},
        )
        return NominateResponse.model_validate(data)

    async def cast_vote(
        self,
        bullet_id: str,
        vote: str,
    ) -> VoteResponse:
        """Vote on a team bullet (up/down).

        Args:
            bullet_id: ID of the bullet to vote on.
            vote: ``"up"`` or ``"down"``.

        Returns:
            VoteResponse with updated vote counts.

        Raises:
            NotFoundError: If bullet_id does not exist (404).
            RetryableError: On network or server failure.
            ValueError: If vote is invalid.
        """
        if vote not in self._VALID_VOTES:
            raise ValueError(
                f"Invalid vote {vote!r}, must be one of {self._VALID_VOTES}"
            )
        data = await self._push_request(
            "POST",
            f"/api/v1/bullets/{bullet_id}/vote",
            json_body={"vote": vote},
        )
        return VoteResponse.model_validate(data)

    async def propose_supersede(
        self,
        origin_id: str,
        new_bullet: dict[str, Any],
        *,
        priority: str = "normal",
    ) -> SupersedeResponse:
        """Submit a correction for an existing team bullet.

        Args:
            origin_id: ID of the original bullet to supersede.
            new_bullet: Replacement bullet dict.
            priority: ``"normal"`` (default) or ``"urgent"``.

        Returns:
            SupersedeResponse with id and status.

        Raises:
            NotFoundError: If origin_id does not exist (404).
            RetryableError: On network or server failure.
            ValueError: If priority is invalid.
        """
        if priority not in self._VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority {priority!r}, must be one of {self._VALID_PRIORITIES}"
            )
        data = await self._push_request(
            "POST",
            "/api/v1/bullets/supersede",
            json_body={
                "origin_id": origin_id,
                "new_bullet": new_bullet,
                "priority": priority,
            },
        )
        return SupersedeResponse.model_validate(data)

    async def report_override_deviation(
        self, event: dict[str, Any]
    ) -> None:
        """Report a mandatory override deviation to the team server.

        This is a fire-and-forget audit endpoint. Failures are logged
        but never raised, so retrieval is never blocked.

        Args:
            event: Audit event dict with type, bullet_id, reason, expires.
        """
        try:
            await self._push_request(
                "POST",
                "/api/v1/audit/override-deviation",
                json_body=event,
            )
            logger.debug("Override deviation reported: %s", event.get("bullet_id"))
        except Exception:
            logger.warning(
                "Failed to report override deviation for bullet %s, ignoring",
                event.get("bullet_id", "unknown"),
            )

    def report_override_deviation_sync(
        self, event: dict[str, Any]
    ) -> None:
        """Synchronous wrapper for :meth:`report_override_deviation`."""
        try:
            self._run_sync(self.report_override_deviation(event))
        except Exception:
            logger.warning(
                "Sync report_override_deviation failed for %s, ignoring",
                event.get("bullet_id", "unknown"),
            )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    # -- context manager support --------------------------------------------

    async def __aenter__(self) -> AceSyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # -- sync wrappers for non-async callers --------------------------------

    def _run_sync(self, coro: Any) -> Any:
        """Run an async coroutine synchronously.

        Detects whether an event loop is already running and handles accordingly.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already inside an async context — create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            return asyncio.run(coro)

    def pull_index_sync(
        self,
        *,
        since: datetime | None = None,
        tags: list[str] | None = None,
    ) -> IndexResponse:
        """Synchronous wrapper for :meth:`pull_index`."""
        return self._run_sync(self.pull_index(since=since, tags=tags))

    def fetch_bullets_sync(self, ids: list[str]) -> list[dict[str, Any]]:
        """Synchronous wrapper for :meth:`fetch_bullets`."""
        return self._run_sync(self.fetch_bullets(ids))

    def pull_taxonomy_sync(self) -> TaxonomyResponse:
        """Synchronous wrapper for :meth:`pull_taxonomy`."""
        return self._run_sync(self.pull_taxonomy())

    def nominate_bullet_sync(
        self,
        bullet: dict[str, Any],
        *,
        priority: str = "normal",
    ) -> NominateResponse:
        """Synchronous wrapper for :meth:`nominate_bullet`."""
        return self._run_sync(self.nominate_bullet(bullet, priority=priority))

    def cast_vote_sync(
        self,
        bullet_id: str,
        vote: str,
    ) -> VoteResponse:
        """Synchronous wrapper for :meth:`cast_vote`."""
        return self._run_sync(self.cast_vote(bullet_id, vote))

    def propose_supersede_sync(
        self,
        origin_id: str,
        new_bullet: dict[str, Any],
        *,
        priority: str = "normal",
    ) -> SupersedeResponse:
        """Synchronous wrapper for :meth:`propose_supersede`."""
        return self._run_sync(
            self.propose_supersede(origin_id, new_bullet, priority=priority)
        )
