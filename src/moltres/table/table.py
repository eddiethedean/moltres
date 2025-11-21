"""Table access primitives."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..config import MoltresConfig

if TYPE_CHECKING:
    from ..dataframe.dataframe import DataFrame
    from ..dataframe.reader import DataFrameReader
from ..engine.connection import ConnectionManager
from ..engine.dialects import DialectSpec, get_dialect
from ..engine.execution import QueryExecutor, QueryResult
from ..expressions.column import Column
from ..logical.plan import LogicalPlan
from ..sql.compiler import compile_plan
from ..sql.ddl import compile_create_table, compile_drop_table
from .schema import ColumnDef, TableSchema


@dataclass
class TableHandle:
    """Lightweight handle representing a table reference."""

    name: str
    database: Database

    def select(self, *columns: str) -> DataFrame:
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
        """Get a handle to a table in the database.

        Args:
            name: Name of the table

        Returns:
            TableHandle for the specified table

        Raises:
            ValidationError: If table name is invalid
        """
        from ..sql.builders import quote_identifier
        from ..utils.exceptions import ValidationError

        if not name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(name, self._dialect.quote_char)
        return TableHandle(name=name, database=self)

    @property
    def read(self) -> DataFrameReader:
        """Return a DataFrameReader for reading from tables and files."""
        from ..dataframe.reader import DataFrameReader

        return DataFrameReader(self)

    # -------------------------------------------------------------- DDL operations
    def create_table(
        self,
        name: str,
        columns: Sequence[ColumnDef],
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
    ) -> TableHandle:
        """Create a new table with the specified schema.

        Args:
            name: Name of the table to create
            columns: Sequence of ColumnDef objects defining the table schema
            if_not_exists: If True, don't error if table already exists (default: True)
            temporary: If True, create a temporary table (default: False)

        Returns:
            TableHandle for the newly created table

        Raises:
            ValidationError: If table name or columns are invalid
            ExecutionError: If table creation fails
        """
        from ..utils.exceptions import ValidationError

        if not columns:
            raise ValidationError(f"Cannot create table '{name}' with no columns")

        schema = TableSchema(
            name=name, columns=columns, if_not_exists=if_not_exists, temporary=temporary
        )
        sql = compile_create_table(schema, self._dialect)
        self._executor.execute(sql)
        return self.table(name)

    def drop_table(self, name: str, *, if_exists: bool = True) -> None:
        """Drop a table by name.

        Args:
            name: Name of the table to drop
            if_exists: If True, don't error if table doesn't exist (default: True)

        Raises:
            ValidationError: If table name is invalid
            ExecutionError: If table dropping fails (when if_exists=False and table doesn't exist)
        """
        sql = compile_drop_table(name, self._dialect, if_exists=if_exists)
        self._executor.execute(sql)

    # -------------------------------------------------------------- query utils
    def compile_plan(self, plan: LogicalPlan) -> str:
        return compile_plan(plan, dialect=self._dialect)

    def execute_plan(self, plan: LogicalPlan) -> QueryResult:
        sql = self.compile_plan(plan)
        return self._executor.fetch(sql)

    def execute_plan_stream(self, plan: LogicalPlan) -> Iterator[list[dict[str, object]]]:
        """Execute a plan and return an iterator of row chunks."""
        sql = self.compile_plan(plan)
        return self._executor.fetch_stream(sql)

    def execute_sql(self, sql: str, params: dict[str, Any] | None = None) -> QueryResult:
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
