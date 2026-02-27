"""Shared test fixtures for MemX."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


@pytest.fixture()
def mock_mem0_memory() -> MagicMock:
    """Create a mock mem0 Memory instance for testing."""
    mock = MagicMock()
    mock.add.return_value = {"id": "test-id", "memory": "test memory"}
    mock.search.return_value = {"results": []}
    mock.get_all.return_value = {"memories": []}
    mock.get.return_value = {"id": "test-id", "memory": "test memory"}
    mock.update.return_value = {"id": "test-id", "memory": "updated"}
    mock.delete.return_value = None
    mock.delete_all.return_value = None
    return mock


@pytest.fixture()
def sample_messages() -> list[dict[str, Any]]:
    """Sample chat messages for testing."""
    return [
        {"role": "user", "content": "I prefer dark mode in all my editors."},
        {"role": "assistant", "content": "Noted! I'll remember your preference for dark mode."},
    ]
