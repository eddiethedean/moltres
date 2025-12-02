"""Airflow integration package for Moltres."""

from __future__ import annotations

__all__ = [
    "MoltresQueryOperator",
    "MoltresToTableOperator",
    "MoltresDataQualityOperator",
    "ETLPipeline",
]

try:
    from .core import (
        ETLPipeline,
        MoltresDataQualityOperator,
        MoltresQueryOperator,
        MoltresToTableOperator,
    )
except ImportError:
    # Airflow not available or import failed
    MoltresQueryOperator = None  # type: ignore[assignment, misc]
    MoltresToTableOperator = None  # type: ignore[assignment, misc]
    MoltresDataQualityOperator = None  # type: ignore[assignment, misc]
    ETLPipeline = None  # type: ignore[assignment, misc]
