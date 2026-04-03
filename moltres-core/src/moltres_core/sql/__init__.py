"""SQL connection, dialect, and query execution (extracted from Moltres)."""

from __future__ import annotations

from moltres_core.sql.async_connection import AsyncConnectionManager
from moltres_core.sql.async_execution import AsyncQueryExecutor
from moltres_core.sql.connection import ConnectionManager
from moltres_core.sql.dialects import DialectSpec, get_dialect
from moltres_core.sql.execution import (
    QueryExecutor,
    QueryResult,
    register_performance_hook,
    unregister_performance_hook,
)

__all__ = [
    "AsyncConnectionManager",
    "AsyncQueryExecutor",
    "ConnectionManager",
    "DialectSpec",
    "QueryExecutor",
    "QueryResult",
    "get_dialect",
    "register_performance_hook",
    "unregister_performance_hook",
]
