"""Prefect integration package for Moltres."""

from __future__ import annotations

__all__ = [
    "moltres_query",
    "moltres_to_table",
    "moltres_data_quality",
    "ETLPipeline",
]

try:
    from .core import ETLPipeline, moltres_data_quality, moltres_query, moltres_to_table
except ImportError:
    # Prefect not available or import failed
    moltres_query = None  # type: ignore[assignment, misc]
    moltres_to_table = None  # type: ignore[assignment, misc]
    moltres_data_quality = None  # type: ignore[assignment, misc]
    ETLPipeline = None  # type: ignore[assignment, misc]
