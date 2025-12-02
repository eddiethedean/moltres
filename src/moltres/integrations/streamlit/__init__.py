"""Streamlit integration package for Moltres."""

from __future__ import annotations

__all__ = [
    "moltres_dataframe",
    "query_builder",
    "cached_query",
    "clear_moltres_cache",
    "invalidate_query_cache",
    "get_db_from_session",
    "init_db_connection",
    "close_db_connection",
    "visualize_query",
    "display_moltres_error",
    "STREAMLIT_AVAILABLE",
]

try:
    from .core import (
        STREAMLIT_AVAILABLE,
        cached_query,
        clear_moltres_cache,
        close_db_connection,
        display_moltres_error,
        get_db_from_session,
        init_db_connection,
        invalidate_query_cache,
        moltres_dataframe,
        query_builder,
        visualize_query,
    )
except ImportError:
    # Streamlit not available or import failed
    STREAMLIT_AVAILABLE = False
    moltres_dataframe = None  # type: ignore[assignment]
    query_builder = None  # type: ignore[assignment]
    cached_query = None  # type: ignore[assignment]
    clear_moltres_cache = None  # type: ignore[assignment]
    invalidate_query_cache = None  # type: ignore[assignment]
    get_db_from_session = None  # type: ignore[assignment]
    init_db_connection = None  # type: ignore[assignment]
    close_db_connection = None  # type: ignore[assignment]
    visualize_query = None  # type: ignore[assignment]
    display_moltres_error = None  # type: ignore[assignment]
