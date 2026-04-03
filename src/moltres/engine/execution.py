"""Query execution helpers (moltres facade over :mod:`moltres_core.sql`).

Translates :mod:`moltres_core.exceptions` into :mod:`moltres.utils.exceptions` so
framework integrations (FastAPI, etc.) keep matching registered handlers.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Type, Union

from moltres_core.exceptions import ExecutionError as _CoreExecutionError
from moltres_core.exceptions import QueryTimeoutError as _CoreQueryTimeoutError
from moltres_core.sql.execution import (
    QueryResult,
    register_performance_hook,
    unregister_performance_hook,
)
from moltres_core.sql.execution import QueryExecutor as _CoreQueryExecutor
from moltres.utils.exceptions import ExecutionError, QueryTimeoutError


class QueryExecutor(_CoreQueryExecutor):
    """Same as :class:`moltres_core.sql.execution.QueryExecutor` with public Moltres exceptions."""

    def fetch(
        self,
        stmt: Union[str, Any],
        params: Optional[Dict[str, Any]] = None,
        connection: Any = None,
        model: Optional[Type[Any]] = None,
    ) -> QueryResult:
        try:
            return super().fetch(stmt, params=params, connection=connection, model=model)
        except _CoreQueryTimeoutError as exc:
            timeout = exc.context.get("timeout_seconds") if exc.context else None
            raise QueryTimeoutError(
                exc.message, timeout=timeout, context=dict(exc.context)
            ) from exc
        except _CoreExecutionError as exc:
            raise ExecutionError(
                exc.message, suggestion=exc.suggestion, context=dict(exc.context)
            ) from exc

    def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        transaction: Any = None,
    ) -> QueryResult:
        try:
            return super().execute(sql, params=params, transaction=transaction)
        except _CoreQueryTimeoutError as exc:
            timeout = exc.context.get("timeout_seconds") if exc.context else None
            raise QueryTimeoutError(
                exc.message, timeout=timeout, context=dict(exc.context)
            ) from exc
        except _CoreExecutionError as exc:
            raise ExecutionError(
                exc.message, suggestion=exc.suggestion, context=dict(exc.context)
            ) from exc

    def execute_many(
        self,
        sql: str,
        params_list: Sequence[Dict[str, Any]],
        transaction: Any = None,
    ) -> QueryResult:
        try:
            return super().execute_many(sql, params_list, transaction=transaction)
        except _CoreQueryTimeoutError as exc:
            timeout = exc.context.get("timeout_seconds") if exc.context else None
            raise QueryTimeoutError(
                exc.message, timeout=timeout, context=dict(exc.context)
            ) from exc
        except _CoreExecutionError as exc:
            raise ExecutionError(
                exc.message, suggestion=exc.suggestion, context=dict(exc.context)
            ) from exc


__all__ = [
    "QueryExecutor",
    "QueryResult",
    "register_performance_hook",
    "unregister_performance_hook",
]
