"""Execution engine components."""

from __future__ import annotations

from .connection import ConnectionManager
from .dialects import DialectSpec, get_dialect
from .execution import (
    QueryExecutor,
    QueryResult,
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
