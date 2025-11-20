"""Table access primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from ..config import MoltresConfig
from ..engine.connection import ConnectionManager
from ..engine.dialects import DialectSpec, get_dialect
from ..engine.execution import QueryExecutor, QueryResult
from ..expressions.column import Column
from ..logical.plan import LogicalPlan
from ..sql.compiler import compile_plan


@dataclass
class TableHandle:
    """Lightweight handle representing a table reference."""

    name: str
    database: "Database"

    def select(self, *columns: str) -> "DataFrame":
        from ..dataframe.dataframe import DataFrame

        return DataFrame.from_table(self, columns=list(columns))

    def insert(self, rows: Sequence[Mapping[str, object]]) -> int:
        from .mutations import insert_rows

        return insert_rows(self, rows)

    def update(self, *, where: Column, set: Mapping[str, object]) -> int:
        from .mutations import update_rows

        return update_rows(self, where=where, values=set)

    def delete(self, where: Column) -> int:
        from .mutations import delete_rows

        return delete_rows(self, where=where)


class Database:
    """Entry-point object returned by ``moltres.connect``."""

    def __init__(self, config: MoltresConfig):
        self.config = config
        self._connections = ConnectionManager(config.engine)
        self._executor = QueryExecutor(self._connections, config.engine)
        self._dialect = get_dialect(self._dialect_name)

    @property
    def connection_manager(self) -> ConnectionManager:
        return self._connections

    @property
    def executor(self) -> QueryExecutor:
        return self._executor

    def table(self, name: str) -> TableHandle:
        return TableHandle(name=name, database=self)

    # -------------------------------------------------------------- query utils
    def compile_plan(self, plan: LogicalPlan) -> str:
        return compile_plan(plan, dialect=self._dialect)

    def execute_plan(self, plan: LogicalPlan) -> QueryResult:
        sql = self.compile_plan(plan)
        return self._executor.fetch(sql)

    def execute_sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        return self._executor.fetch(sql, params=params)

    @property
    def dialect(self) -> DialectSpec:
        return self._dialect

    # ----------------------------------------------------------------- internals
    @property
    def _dialect_name(self) -> str:
        if self.config.engine.dialect:
            return self.config.engine.dialect
        return self.config.engine.dsn.split(":", 1)[0]
