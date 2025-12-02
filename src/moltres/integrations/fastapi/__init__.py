"""FastAPI integration package for Moltres."""

from __future__ import annotations

__all__ = [
    "register_exception_handlers",
    "get_db",
    "get_async_db",
    "create_db_dependency",
    "create_async_db_dependency",
    "handle_moltres_errors",
    "FASTAPI_AVAILABLE",
]

try:
    from .core import (
        FASTAPI_AVAILABLE,
        create_async_db_dependency,
        create_db_dependency,
        get_async_db,
        get_db,
        handle_moltres_errors,
        register_exception_handlers,
    )
except ImportError:
    # FastAPI not available or import failed
    FASTAPI_AVAILABLE = False
    register_exception_handlers = None  # type: ignore[assignment]
    get_db = None  # type: ignore[assignment]
    get_async_db = None  # type: ignore[assignment]
    create_db_dependency = None  # type: ignore[assignment]
    create_async_db_dependency = None  # type: ignore[assignment]
    handle_moltres_errors = None  # type: ignore[assignment]
