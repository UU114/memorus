"""Smoke tests for memx package structure."""

import memx
from memx.exceptions import ConfigurationError, EngineError, MemXError, PipelineError


def test_version() -> None:
    assert memx.__version__ == "0.1.0"


def test_exception_hierarchy() -> None:
    assert issubclass(ConfigurationError, MemXError)
    assert issubclass(PipelineError, MemXError)
    assert issubclass(EngineError, MemXError)
    assert issubclass(MemXError, Exception)
