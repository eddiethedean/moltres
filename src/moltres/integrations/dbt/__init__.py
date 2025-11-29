"""dbt integration for Moltres.

This module provides integration with dbt (data build tool) for using
Moltres DataFrames in dbt Python models.

Key features:
- Get Moltres :class:`Database` instances from dbt connections
- Reference dbt models and sources as Moltres DataFrames
- Helper functions for common dbt patterns
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "get_moltres_connection",
    "moltres_ref",
    "moltres_source",
    "moltres_var",
]

# Graceful degradation
try:
    import dbt

    DBT_AVAILABLE = True
except ImportError:
    DBT_AVAILABLE = False
    # Create stubs for type checking
    dbt: Any = None  # type: ignore[no-redef]

from .adapter import get_moltres_connection
from .helpers import moltres_ref, moltres_source, moltres_var
