"""Execution helpers for running compiled SQL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

from sqlalchemy import text

from ..config import EngineConfig
from .connection import ConnectionManager


@dataclass
class QueryResult:
    rows: Any
    rowcount: Optional[int]


class QueryExecutor:
    """Thin abstraction over SQL execution for future extensibility."""

    def __init__(self, connection_manager: ConnectionManager, config: EngineConfig):
        self._connections = connection_manager
        self._config = config

    def fetch(self, sql: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        with self._connections.connect() as conn:
            result = conn.execute(text(sql), params or {})
            rows = result.fetchall()
            columns = list(result.keys())
            payload = self._format_rows(rows, columns)
            return QueryResult(rows=payload, rowcount=result.rowcount)

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        with self._connections.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return QueryResult(rows=None, rowcount=result.rowcount)

    def _format_rows(self, rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> Any:
        fmt = self._config.fetch_format
        if fmt == "records":
            return [dict(zip(columns, row)) for row in rows]
        if fmt == "pandas":
            try:
                import pandas as pd  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Pandas support requested but pandas is not installed") from exc
            return pd.DataFrame(rows, columns=columns)
        if fmt == "polars":
            try:
                import polars as pl  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("Polars support requested but polars is not installed") from exc
            return pl.DataFrame(rows, schema=columns)
        raise ValueError(f"Unknown fetch format '{fmt}'")
