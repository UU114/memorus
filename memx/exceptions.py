"""MemX custom exception hierarchy."""


class MemXError(Exception):
    """Base exception for all MemX errors."""


class ConfigurationError(MemXError):
    """Invalid configuration."""


class PipelineError(MemXError):
    """Error in ingest/retrieval pipeline."""


class EngineError(MemXError):
    """Error in engine execution."""
