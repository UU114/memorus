"""Memorus: Adaptive Context Engine on top of mem0."""

import os

__version__ = "0.2.1"

# Backend selection. Defaults to the pure-Python engine so importing memorus
# has zero behavior change unless MEMORUS_BACKEND=rust is set explicitly.
_BACKEND = os.environ.get("MEMORUS_BACKEND", "python").strip().lower()

if _BACKEND == "rust":
    # Rust-backed adapters delegate to the compiled `memorus_r` extension.
    try:
        import memorus_r  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MEMORUS_BACKEND=rust requires the compiled 'memorus_r' extension. "
            "Install it with: pip install memorus[rust]"
        ) from exc

    from memorus.core._rust_backend import (
        RustBackedAsyncMemory as AsyncMemory,
    )
    from memorus.core._rust_backend import (
        RustBackedMemory as Memory,
    )
else:
    # Default: pure-Python engines (unchanged).
    from memorus.core.async_memory import AsyncMemory
    from memorus.core.memory import Memory

__all__ = ["Memory", "AsyncMemory"]
