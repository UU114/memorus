"""MemX: Adaptive Context Engine on top of mem0."""

__version__ = "0.2.1"

# Re-export core API for backward compatibility
from memx.core.memory import Memory
from memx.core.async_memory import AsyncMemory

__all__ = ["Memory", "AsyncMemory"]
