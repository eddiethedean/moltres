"""Execution engine components (facade over :mod:`moltres_core.sql`)."""

from __future__ import annotations

from moltres_core.sql import (
    ConnectionManager,
    DialectSpec,
    QueryExecutor,
    QueryResult,
    get_dialect,
    register_performance_hook,
    unregister_performance_hook,
)

__all__ = [
    "ConnectionManager",
    "DialectSpec",
    "QueryExecutor",
    "QueryResult",
    "get_dialect",
    "register_performance_hook",
    "unregister_performance_hook",
]
