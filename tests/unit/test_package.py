"""Smoke tests for memx package structure."""

import memx
from memx.core.exceptions import ConfigurationError, EngineError, MemXError, PipelineError


def test_version() -> None:
    assert memx.__version__ == "1.0.0"


def test_exception_hierarchy() -> None:
    assert issubclass(ConfigurationError, MemXError)
    assert issubclass(PipelineError, MemXError)
    assert issubclass(EngineError, MemXError)
    assert issubclass(MemXError, Exception)
