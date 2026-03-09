"""Tests for memx.ext.rest_api."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if fastapi not installed
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def mock_memory():
    mem = MagicMock()
    mem.search.return_value = {"results": []}
    mem.add.return_value = {"results": [{"id": "abc"}]}
    mem.get_all.return_value = {"results": []}
    mem.get.return_value = {"id": "abc", "memory": "test"}
    mem.delete.return_value = None
    mem.status.return_value = {"total": 5}
    return mem


@pytest.fixture()
def app_no_auth(mock_memory):
    """Create an app with auth disabled, injecting mock memory via dependency override."""
    import memx.ext.rest_api as mod

    mod._NO_AUTH = True
    app = mod.create_app()
    app.dependency_overrides[mod._get_memory_dep] = lambda: mock_memory
    yield app
    app.dependency_overrides.clear()
    mod._NO_AUTH = False


@pytest.fixture()
def client(app_no_auth):
    return TestClient(app_no_auth)


class TestEndpoints:
    """Test all REST API endpoints."""

    def test_create_memory(self, client, mock_memory):
        resp = client.post("/memories", json={"content": "hello", "user_id": "u1"})
        assert resp.status_code == 200
        mock_memory.add.assert_called_once_with("hello", user_id="u1", metadata=None)

    def test_search_memories(self, client, mock_memory):
        resp = client.get("/memories/search", params={"query": "test", "limit": 10})
        assert resp.status_code == 200
        mock_memory.search.assert_called_once_with("test", user_id=None, limit=10)

    def test_list_memories(self, client, mock_memory):
        resp = client.get("/memories", params={"user_id": "u1"})
        assert resp.status_code == 200
        mock_memory.get_all.assert_called_once_with(user_id="u1")

    def test_get_memory(self, client, mock_memory):
        resp = client.get("/memories/abc")
        assert resp.status_code == 200
        mock_memory.get.assert_called_once_with("abc")

    def test_delete_memory(self, client, mock_memory):
        resp = client.delete("/memories/abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "deleted"
        mock_memory.delete.assert_called_once_with("abc")

    def test_status(self, client, mock_memory):
        resp = client.get("/status")
        assert resp.status_code == 200
        assert resp.json()["status"]["total"] == 5


class TestAuth:
    """Test authentication behavior."""

    def _make_authed_client(self, mock_memory):
        """Create a TestClient with auth enabled and mock memory injected."""
        import memx.ext.rest_api as mod

        mod._NO_AUTH = False
        app = mod.create_app()
        app.dependency_overrides[mod._get_memory_dep] = lambda: mock_memory
        return TestClient(app)

    def test_no_auth_mode_allows_requests(self, client):
        """--no-auth mode allows unauthenticated requests."""
        resp = client.get("/status")
        assert resp.status_code == 200

    def test_valid_api_key_accepted(self, mock_memory):
        """Valid API key passes authentication."""
        with patch.dict(os.environ, {"MEMX_API_KEY": "secret123"}):
            tc = self._make_authed_client(mock_memory)
            resp = tc.get("/status", headers={"X-API-Key": "secret123"})
        assert resp.status_code == 200

    def test_invalid_api_key_rejected(self, mock_memory):
        """Invalid API key returns 401."""
        with patch.dict(os.environ, {"MEMX_API_KEY": "secret123"}):
            tc = self._make_authed_client(mock_memory)
            resp = tc.get("/status", headers={"X-API-Key": "wrong"})
        assert resp.status_code == 401

    def test_missing_api_key_rejected(self, mock_memory):
        """Missing API key returns 401 when auth is enabled."""
        with patch.dict(os.environ, {"MEMX_API_KEY": "secret123"}):
            tc = self._make_authed_client(mock_memory)
            resp = tc.get("/status")
        assert resp.status_code == 401


class TestImportError:
    """Test behavior when FastAPI is not installed."""

    def test_create_app_raises_without_fastapi(self):
        with patch("memx.ext.rest_api.FastAPI", None):
            from memx.ext.rest_api import create_app

            with pytest.raises(ImportError, match="fastapi"):
                create_app()
