"""Moltres SQL execution core and pydantable-compatible engine."""

from __future__ import annotations

from moltres_core.config import EngineConfig
from moltres_core.engine import MoltresPydantableEngine
from moltres_core.plan import SqlPlan, SqlRootData
from moltres_core.sql import (
    AsyncConnectionManager,
    AsyncQueryExecutor,
    ConnectionManager,
    DialectSpec,
    QueryExecutor,
    QueryResult,
    get_dialect,
    register_performance_hook,
    unregister_performance_hook,
)

from . import embedded_protocol

__all__ = [
    "AsyncConnectionManager",
    "AsyncQueryExecutor",
    "ConnectionManager",
    "DialectSpec",
    "EngineConfig",
    "MoltresPydantableEngine",
    "QueryExecutor",
    "QueryResult",
    "SqlPlan",
    "SqlRootData",
    "get_dialect",
    "register_performance_hook",
    "unregister_performance_hook",
    "embedded_protocol",
]

__version__ = "1.0.0"
