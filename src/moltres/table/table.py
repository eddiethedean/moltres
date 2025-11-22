"""Table access primitives."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union

from ..config import MoltresConfig

if TYPE_CHECKING:
    from ..dataframe.dataframe import DataFrame
    from ..dataframe.reader import DataLoader
    from ..io.records import Records
    from .actions import (
        CreateTableOperation,
        DeleteMutation,
        DropTableOperation,
        InsertMutation,
        MergeMutation,
        UpdateMutation,
    )
    from .batch import OperationBatch
from sqlalchemy.engine import Connection

from ..engine.connection import ConnectionManager
from ..engine.dialects import DialectSpec, get_dialect
from ..engine.execution import QueryExecutor, QueryResult
from ..expressions.column import Column
from ..logical.plan import LogicalPlan
from ..sql.compiler import compile_plan
from .schema import ColumnDef


@dataclass
class TableHandle:
    """Lightweight handle representing a table reference."""

    name: str
    database: "Database"

    def select(self, *columns: str) -> "DataFrame":
        from ..dataframe.dataframe import DataFrame

        return DataFrame.from_table(self, columns=list(columns))

    def insert(self, rows: Union[Sequence[Mapping[str, object]], "Records"]) -> "InsertMutation":
        """Create a lazy insert operation.

        Args:
            rows: Sequence of row dictionaries to insert

        Returns:
            InsertMutation that executes on collect()

        Example:
            >>> mutation = table.insert([{"id": 1, "name": "Alice"}])
            >>> rowcount = mutation.collect()  # Executes the insert
        """
        from .actions import InsertMutation
        from .batch import get_active_batch

        op = InsertMutation(handle=self, rows=rows)
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    def update(self, *, where: Column, set: Mapping[str, object]) -> "UpdateMutation":
        """Create a lazy update operation.

        Args:
            where: Column expression for the WHERE clause
            set: Dictionary of column names to new values

        Returns:
            UpdateMutation that executes on collect()

        Example:
            >>> mutation = table.update(where=col("id") == 1, set={"name": "Bob"})
            >>> rowcount = mutation.collect()  # Executes the update
        """
        from .actions import UpdateMutation
        from .batch import get_active_batch

        op = UpdateMutation(handle=self, where=where, values=set)
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    def delete(self, where: Column) -> "DeleteMutation":
        """Create a lazy delete operation.

        Args:
            where: Column expression for the WHERE clause

        Returns:
            DeleteMutation that executes on collect()

        Example:
            >>> mutation = table.delete(where=col("id") == 1)
            >>> rowcount = mutation.collect()  # Executes the delete
        """
        from .actions import DeleteMutation
        from .batch import get_active_batch

        op = DeleteMutation(handle=self, where=where)
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    def merge(
        self,
        rows: Union[Sequence[Mapping[str, object]], "Records"],
        *,
        on: Sequence[str],
        when_matched: Optional[Mapping[str, object]] = None,
        when_not_matched: Optional[Mapping[str, object]] = None,
    ) -> "MergeMutation":
        """Create a lazy merge (upsert) operation.

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
            MergeMutation that executes on collect()

        Example:
            >>> mutation = table.merge(
            ...     [{"email": "user@example.com", "name": "Updated Name"}],
            ...     on=["email"],
            ...     when_matched={"name": "Updated Name", "updated_at": "NOW()"}
            ... )
            >>> rowcount = mutation.collect()  # Executes the merge
        """
        from .actions import MergeMutation
        from .batch import get_active_batch

        op = MergeMutation(
            handle=self,
            rows=rows,
            on=on,
            when_matched=when_matched,
            when_not_matched=when_not_matched,
        )
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op


class Transaction:
    """Transaction context for grouping multiple operations."""

    def __init__(self, database: "Database", connection: Connection):
        self.database = database
        self.connection = connection
        self._committed = False
        self._rolled_back = False

    def commit(self) -> None:
        """Explicitly commit the transaction."""
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        self.database.connection_manager.commit_transaction(self.connection)
        self._committed = True

    def rollback(self) -> None:
        """Explicitly rollback the transaction."""
        if self._committed or self._rolled_back:
            raise RuntimeError("Transaction already committed or rolled back")
        self.database.connection_manager.rollback_transaction(self.connection)
        self._rolled_back = True

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            # Exception occurred, rollback
            if not self._rolled_back and not self._committed:
                self.rollback()
        else:
            # No exception, commit
            if not self._committed and not self._rolled_back:
                self.commit()


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
    ) -> "CreateTableOperation":
        """Create a lazy create table operation.

        Args:
            name: Name of the table to create
            columns: Sequence of ColumnDef objects defining the table schema
            if_not_exists: If True, don't error if table already exists (default: True)
            temporary: If True, create a temporary table (default: False)

        Returns:
            CreateTableOperation that executes on collect()

        Raises:
            ValidationError: If table name or columns are invalid

        Example:
            >>> op = db.create_table("users", [column("id", "INTEGER")])
            >>> table = op.collect()  # Executes the CREATE TABLE
        """
        from ..utils.exceptions import ValidationError
        from .actions import CreateTableOperation

        # Validate early (at operation creation time)
        if not columns:
            raise ValidationError(f"Cannot create table '{name}' with no columns")

        op = CreateTableOperation(
            database=self,
            name=name,
            columns=columns,
            if_not_exists=if_not_exists,
            temporary=temporary,
        )
        # Add to active batch if one exists
        from .batch import get_active_batch

        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    def drop_table(self, name: str, *, if_exists: bool = True) -> "DropTableOperation":
        """Create a lazy drop table operation.

        Args:
            name: Name of the table to drop
            if_exists: If True, don't error if table doesn't exist (default: True)

        Returns:
            DropTableOperation that executes on collect()

        Example:
            >>> op = db.drop_table("users")
            >>> op.collect()  # Executes the DROP TABLE
        """
        from .actions import DropTableOperation
        from .batch import get_active_batch

        op = DropTableOperation(database=self, name=name, if_exists=if_exists)
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

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

    def batch(self) -> "OperationBatch":
        """Create a batch context for grouping multiple operations.

        All operations within the batch context are executed together in a single transaction
        when the context exits. If any exception occurs, all operations are rolled back.

        Returns:
            OperationBatch context manager

        Example:
            >>> with db.batch():
            ...     db.create_table("users", [...])
            ...     table.insert([...])
            ...     # All operations execute together on exit
        """
        from .batch import OperationBatch

        return OperationBatch(self)

    @contextmanager
    def transaction(self) -> Iterator[Transaction]:
        """Create a transaction context for grouping multiple operations.

        All operations within the transaction context share the same transaction.
        If any exception occurs, the transaction is automatically rolled back.
        Otherwise, it is committed on successful exit.

        Yields:
            Transaction object that can be used for explicit commit/rollback

        Example:
            >>> with db.transaction() as txn:
            ...     table.insert([...]).collect()
            ...     table.update(...).collect()
            ...     # If any operation fails, all are rolled back
            ...     # Otherwise, all are committed on exit
        """
        connection = self._connections.begin_transaction()
        txn = Transaction(self, connection)
        try:
            yield txn
            if not txn._committed and not txn._rolled_back:
                txn.commit()
        except Exception:
            if not txn._rolled_back:
                txn.rollback()
            raise

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
