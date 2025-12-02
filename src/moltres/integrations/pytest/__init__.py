"""Pytest integration package for Moltres."""

from __future__ import annotations

__all__ = [
    "moltres_db",
    "moltres_async_db",
    "test_data",
    "create_test_df",
    "assert_dataframe_equal",
    "assert_schema_equal",
    "assert_query_results",
    "_test_data_fixture",
    "query_logger",
]

try:
    from .fixtures import (
        _test_data_fixture,
        assert_dataframe_equal,
        assert_query_results,
        assert_schema_equal,
        create_test_df,
        moltres_async_db,
        moltres_db,
        test_data,
    )
    from .plugin import query_logger
except ImportError:
    # Pytest not available or import failed
    moltres_db = None  # type: ignore[assignment]
    moltres_async_db = None  # type: ignore[assignment]
    test_data = None  # type: ignore[assignment]
    _test_data_fixture = None  # type: ignore[assignment]
    create_test_df = None  # type: ignore[assignment]
    assert_dataframe_equal = None  # type: ignore[assignment]
    assert_schema_equal = None  # type: ignore[assignment]
    assert_query_results = None  # type: ignore[assignment]
    query_logger = None  # type: ignore[assignment]
