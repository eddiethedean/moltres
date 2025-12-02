"""Django integration package for Moltres."""

from __future__ import annotations

__all__ = ["MoltresExceptionMiddleware", "get_moltres_db", "DJANGO_AVAILABLE"]

try:
    from .core import DJANGO_AVAILABLE, MoltresExceptionMiddleware, get_moltres_db
except ImportError:
    # Django not available or import failed
    DJANGO_AVAILABLE = False
    MoltresExceptionMiddleware = None  # type: ignore[assignment, misc]
    get_moltres_db = None  # type: ignore[assignment]
