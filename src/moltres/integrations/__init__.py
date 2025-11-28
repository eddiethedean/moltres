"""Moltres integrations with popular frameworks."""

from __future__ import annotations

__all__ = ["fastapi", "django"]

# Optional Django integration - only import if available
try:
    from . import django as django_integration

    __all__.append("django_integration")
except ImportError:
    django_integration = None  # type: ignore[assignment]
