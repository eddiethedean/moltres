"""Table access primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from ..config import MoltresConfig

if TYPE_CHECKING:
    from ..dataframe.dataframe import DataFrame
    from ..dataframe.reader import DataLoader
    from ..io.records import Records
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
    database: "Database"

    def select(self, *columns: str) -> "DataFrame":
        from ..dataframe.dataframe import DataFrame

        return DataFrame.from_table(self, columns=list(columns))

    def insert(self, rows: Union[Sequence[Mapping[str, object]], "Records"]) -> int:
        from .mutations import insert_rows

        return insert_rows(self, rows)

    def update(self, *, where: Column, set: Mapping[str, object]) -> int:
        from .mutations import update_rows

        return update_rows(self, where=where, values=set)

    def delete(self, where: Column) -> int:
        from .mutations import delete_rows

        return delete_rows(self, where=where)

    def merge(
        self,
        rows: Union[Sequence[Mapping[str, object]], "Records"],
        *,
        on: Sequence[str],
        when_matched: Optional[Mapping[str, object]] = None,
        when_not_matched: Optional[Mapping[str, object]] = None,
    ) -> int:
        """Merge (upsert) rows into the table with conflict resolution.

        This implements MERGE/UPSERT operations with dialect-specific SQL:
        - PostgreSQL: INSERT ... ON CONFLICT ... DO UPDATE
        - SQLite: INSERT ... ON CONFLICT ... DO UPDATE
        - MySQL: INSERT ... ON DUPLICATE KEY UPDATE

        Args:
            rows: Sequence of row dictionaries to merge
            on: Sequence of column names that form the conflict key (primary key or unique constraint)
            when_matched: Optional dictionary of column updates when a conflict occurs
                         If None, no update is performed (insert only if not exists)
            when_not_matched: Optional dictionary of default values when inserting new rows
                             If None, uses values from rows

        Returns:
            Number of rows affected (inserted or updated)

        Example:
            >>> # Upsert users by email
            >>> table.merge(
            ...     [{"email": "user@example.com", "name": "Updated Name"}],
            ...     on=["email"],
            ...     when_matched={"name": "Updated Name", "updated_at": "NOW()"}
            ... )
        """
        from .mutations import merge_rows

        return merge_rows(
            self, rows, on=on, when_matched=when_matched, when_not_matched=when_not_matched
        )


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

    def close(self) -> None:
        """Close all database connections and dispose of the engine.

        This should be called when done with the database connection,
        especially for ephemeral test databases.

        Note: After calling close(), the Database instance should not be used.
        """
        if hasattr(self._connections, "_engine") and self._connections._engine is not None:
            try:
                self._connections._engine.dispose(close=True)
            except Exception:
                # Ignore errors during disposal (e.g., if already disposed)
                pass
            finally:
                self._connections._engine = None

    def table(self, name: str) -> TableHandle:
        """Get a handle to a table in the database.

        Args:
            name: Name of the table

        Returns:
            TableHandle for the specified table

        Raises:
            ValidationError: If table name is invalid
        """
        from ..utils.exceptions import ValidationError
        from ..sql.builders import quote_identifier

        if not name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(name, self._dialect.quote_char)
        return TableHandle(name=name, database=self)

    @property
    def load(self) -> "DataLoader":
        """Return a DataLoader for loading data from files and tables as Records.

        Note: For SQL operations on tables, use db.table(name).select() instead.
        """
        from ..dataframe.reader import DataLoader

        return DataLoader(self)

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
            name=name,
            columns=columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
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
    def compile_plan(self, plan: LogicalPlan) -> Any:
        """Compile a logical plan to a SQLAlchemy Select statement."""

        return compile_plan(plan, dialect=self._dialect)

    def execute_plan(self, plan: LogicalPlan) -> QueryResult:
        stmt = self.compile_plan(plan)
        return self._executor.fetch(stmt)

    def execute_plan_stream(self, plan: LogicalPlan) -> Iterator[List[Dict[str, object]]]:
        """Execute a plan and return an iterator of row chunks."""
        stmt = self.compile_plan(plan)
        return self._executor.fetch_stream(stmt)

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
        # Extract dialect from DSN, normalizing driver variants (e.g., "mysql+pymysql" -> "mysql")
        dsn = self.config.engine.dsn
        if not dsn:
            return "ansi"
        dialect_part = dsn.split(":", 1)[0]
        # Normalize driver variants: "mysql+pymysql" -> "mysql", "postgresql+psycopg2" -> "postgresql"
        if "+" in dialect_part:
            dialect_part = dialect_part.split("+", 1)[0]
        return dialect_part
