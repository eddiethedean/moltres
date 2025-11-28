"""Django integration package for Moltres."""

from __future__ import annotations

__all__ = ["MoltresExceptionMiddleware", "get_moltres_db"]

# Import from the django_module.py file in the parent directory
try:
    from ..django_module import MoltresExceptionMiddleware, get_moltres_db
except ImportError:
    # Django not available or import failed
    MoltresExceptionMiddleware = None  # type: ignore[assignment, misc]
    get_moltres_db = None  # type: ignore[assignment]
