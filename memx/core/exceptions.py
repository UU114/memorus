"""MemX custom exception hierarchy."""


class MemXError(Exception):
    """Base exception for all MemX errors."""


class ConfigurationError(MemXError):
    """Invalid configuration."""


class PipelineError(MemXError):
    """Error in ingest/retrieval pipeline."""


class EngineError(MemXError):
    """Error in engine execution."""


class DaemonError(MemXError):
    """Error in daemon lifecycle or IPC."""


class DaemonUnavailableError(DaemonError, ConnectionError):
    """Raised when the daemon is not running or unreachable."""
