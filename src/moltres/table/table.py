"""Table access primitives."""

from __future__ import annotations

import atexit
from contextlib import contextmanager
import logging
import signal
import weakref
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    overload,
    Sequence,
    Type,
    Union,
)

from ..config import MoltresConfig

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    from sqlalchemy.orm import DeclarativeBase
    from ..dataframe.dataframe import DataFrame
    from ..dataframe.reader import DataLoader, ReadAccessor
    from ..expressions.column import Column
    from ..io.records import LazyRecords, Records
    from ..utils.inspector import ColumnInfo
    from .actions import (
        CreateIndexOperation,
        CreateTableOperation,
        DropIndexOperation,
        DropTableOperation,
    )
    from .batch import OperationBatch
    from .schema import (
        CheckConstraint,
        ForeignKeyConstraint,
        TableSchema,
        UniqueConstraint,
    )
from sqlalchemy.engine import Connection

from ..engine.connection import ConnectionManager
from ..engine.dialects import DialectSpec, get_dialect
from ..engine.execution import QueryExecutor, QueryResult
from ..logical.plan import LogicalPlan
from ..sql.compiler import compile_plan
from .schema import ColumnDef

logger = logging.getLogger(__name__)
_ACTIVE_DATABASES: "weakref.WeakSet[Database]" = weakref.WeakSet()


@dataclass
class TableHandle:
    """Lightweight handle representing a table reference."""

    name: str
    database: "Database"
    model: Optional[Type["DeclarativeBase"]] = None

    @property
    def model_class(self) -> Optional[Type["DeclarativeBase"]]:
        """Get the SQLAlchemy model class if this handle was created from a model.

        Returns:
            SQLAlchemy model class or None if handle was created from table name
        """
        return self.model

    def select(self, *columns: str) -> "DataFrame":
        from ..dataframe.dataframe import DataFrame

        return DataFrame.from_table(self, columns=list(columns))


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
        self._ephemeral_tables: set[str] = set()
        self._closed = False
        _ACTIVE_DATABASES.add(self)

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
        self._close_resources()

    def _close_resources(self) -> None:
        if self._closed:
            return
        self._cleanup_ephemeral_tables()
        engine = getattr(self._connections, "_engine", None)
        if engine is not None:
            try:
                engine.dispose(close=True)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Error disposing engine during close: %s", exc)
            finally:
                self._connections._engine = None
        self._closed = True
        _ACTIVE_DATABASES.discard(self)

    def _register_ephemeral_table(self, name: str) -> None:
        self._ephemeral_tables.add(name)

    def _unregister_ephemeral_table(self, name: str) -> None:
        self._ephemeral_tables.discard(name)

    def _cleanup_ephemeral_tables(self) -> None:
        if not self._ephemeral_tables:
            return
        for table_name in list(self._ephemeral_tables):
            try:
                self.drop_table(table_name, if_exists=True).collect()
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("Failed to drop ephemeral table %s: %s", table_name, exc)
        self._ephemeral_tables.clear()

    @overload
    def table(self, name: str) -> TableHandle:
        """Get a handle to a table in the database from table name."""
        ...

    @overload
    def table(self, model_class: Type["DeclarativeBase"]) -> TableHandle:
        """Get a handle to a table in the database from SQLAlchemy model class."""
        ...

    def table(  # type: ignore[misc]
        self, name_or_model: Union[str, Type["DeclarativeBase"]]
    ) -> TableHandle:
        """Get a handle to a table in the database.

        Args:
            name_or_model: Name of the table, or SQLAlchemy model class

        Returns:
            TableHandle for the specified table

        Raises:
            ValidationError: If table name is invalid
            ValueError: If model_class is not a valid SQLAlchemy model

        Example:
            >>> users = db.table("users")
            >>> df = users.select("id", "name")

            >>> # Or with SQLAlchemy model
            >>> from sqlalchemy.orm import DeclarativeBase
            >>> class User(Base):
            ...     __tablename__ = "users"
            >>> users = db.table(User)
            >>> df = users.select("id", "name")
        """
        from ..utils.exceptions import ValidationError
        from ..sql.builders import quote_identifier
        from .sqlalchemy_integration import (
            is_sqlalchemy_model,
            get_model_table_name,
        )

        # Check if argument is a SQLAlchemy model
        if is_sqlalchemy_model(name_or_model):
            model_class: Type["DeclarativeBase"] = name_or_model  # type: ignore[assignment]
            table_name = get_model_table_name(model_class)
            # Validate table name format
            quote_identifier(table_name, self._dialect.quote_char)
            return TableHandle(name=table_name, database=self, model=model_class)
        else:
            # Type narrowing: after is_sqlalchemy_model check, this must be str
            table_name = name_or_model  # type: ignore[assignment]
            if not table_name:
                raise ValidationError("Table name cannot be empty")
            # Validate table name format
            quote_identifier(table_name, self._dialect.quote_char)
            return TableHandle(name=table_name, database=self)

    def insert(
        self,
        table_name: str,
        rows: Union[
            Sequence[Mapping[str, object]],
            "Records",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
    ) -> int:
        """Insert rows into a table.

        Convenience method for inserting data into a table.

        Args:
            table_name: Name of the table to insert into
            rows: Sequence of row dictionaries, Records, pandas DataFrame, polars DataFrame, or polars LazyFrame

        Returns:
            Number of rows inserted

        Raises:
            ValidationError: If table name is invalid or rows are empty

        Example:
            >>> db.insert("users", [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}])
            >>> # Or with a DataFrame
            >>> import pandas as pd
            >>> df = pd.DataFrame([{"id": 3, "name": "Charlie"}])
            >>> db.insert("users", df)
        """
        from .mutations import insert_rows

        handle = self.table(table_name)
        return insert_rows(handle, rows)

    def update(
        self,
        table_name: str,
        *,
        where: "Column",
        set: Mapping[str, object],  # noqa: A002
    ) -> int:
        """Update rows in a table.

        Convenience method for updating data in a table.

        Args:
            table_name: Name of the table to update
            where: Column expression for the WHERE clause
            set: Dictionary of column names to new values

        Returns:
            Number of rows updated

        Raises:
            ValidationError: If table name is invalid or set dictionary is empty

        Example:
            >>> from moltres import col
            >>> db.update("users", where=col("id") == 1, set={"name": "Alice Updated"})
        """
        from .mutations import update_rows

        handle = self.table(table_name)
        return update_rows(handle, where=where, values=set)

    def delete(self, table_name: str, *, where: "Column") -> int:
        """Delete rows from a table.

        Convenience method for deleting data from a table.

        Args:
            table_name: Name of the table to delete from
            where: Column expression for the WHERE clause

        Returns:
            Number of rows deleted

        Raises:
            ValidationError: If table name is invalid

        Example:
            >>> from moltres import col
            >>> db.delete("users", where=col("id") == 1)
        """
        from .mutations import delete_rows

        handle = self.table(table_name)
        return delete_rows(handle, where=where)

    def merge(
        self,
        table_name: str,
        rows: Union[
            Sequence[Mapping[str, object]],
            "Records",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
        *,
        on: Sequence[str],
        when_matched: Optional[Mapping[str, object]] = None,
        when_not_matched: Optional[Mapping[str, object]] = None,
    ) -> int:
        """Merge (upsert) rows into a table.

        Convenience method for merging data into a table with conflict resolution.

        Args:
            table_name: Name of the table to merge into
            rows: Sequence of row dictionaries, Records, pandas DataFrame, polars DataFrame, or polars LazyFrame
            on: Sequence of column names that form the conflict key
            when_matched: Optional dictionary of column updates when a conflict occurs
            when_not_matched: Optional dictionary of default values when inserting new rows

        Returns:
            Number of rows affected (inserted or updated)

        Raises:
            ValidationError: If table name is invalid, rows are empty, or on columns are invalid

        Example:
            >>> db.merge(
            ...     "users",
            ...     [{"id": 1, "name": "Alice", "email": "alice@example.com"}],
            ...     on=["id"],
            ...     when_matched={"name": "Alice Updated"}
            ... )
        """
        from .mutations import merge_rows

        handle = self.table(table_name)
        return merge_rows(
            handle, rows, on=on, when_matched=when_matched, when_not_matched=when_not_matched
        )

    @property
    def load(self) -> "DataLoader":
        """Return a DataLoader for loading data from files and tables as DataFrames.

        Note: For SQL operations on tables, use db.table(name).select() instead.
        """
        from ..dataframe.reader import DataLoader

        return DataLoader(self)

    @property
    def read(self) -> "ReadAccessor":
        """Return a ReadAccessor for accessing read operations.

        Use db.read.records.* for Records-based reads (backward compatibility).
        Use db.load.* for DataFrame-based reads (PySpark-style).
        """
        from ..dataframe.reader import ReadAccessor

        return ReadAccessor(self)

    def sql(self, sql: str, **params: object) -> "DataFrame":
        """Execute a SQL query and return a DataFrame.

        Similar to PySpark's `spark.sql()`, this method accepts a raw SQL string
        and returns a lazy DataFrame that can be chained with further operations.
        The SQL dialect is determined by the database connection.

        Args:
            sql: SQL query string to execute
            **params: Optional named parameters for parameterized queries.
                     Use `:param_name` syntax in SQL and pass values as kwargs.

        Returns:
            Lazy DataFrame that can be chained with further operations

        Example:
            >>> # Basic SQL query
            >>> df = db.sql("SELECT * FROM users WHERE age > 18")
            >>> results = df.collect()

            >>> # Parameterized query
            >>> df = db.sql("SELECT * FROM users WHERE id = :id AND status = :status",
            ...             id=1, status="active")
            >>> results = df.collect()

            >>> # Chaining operations
            >>> df = db.sql("SELECT * FROM orders").where(col("amount") > 100).limit(10)
            >>> results = df.collect()
        """
        from ..dataframe.dataframe import DataFrame
        from ..logical import operators

        # Convert params dict to the format expected by RawSQL
        params_dict = params if params else None
        plan = operators.raw_sql(sql, params_dict)
        return DataFrame(plan=plan, database=self)

    # -------------------------------------------------------------- DDL operations
    @overload
    def create_table(
        self,
        name: str,
        columns: Sequence[ColumnDef],
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
        constraints: Optional[
            Sequence[Union["UniqueConstraint", "CheckConstraint", "ForeignKeyConstraint"]]
        ] = None,
    ) -> "CreateTableOperation":
        """Create a lazy create table operation from table name and columns."""
        ...

    @overload
    def create_table(
        self,
        model_class: Type["DeclarativeBase"],
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
    ) -> "CreateTableOperation":
        """Create a lazy create table operation from SQLAlchemy model class."""
        ...

    def create_table(  # type: ignore[misc]
        self,
        name_or_model: Union[str, Type["DeclarativeBase"]],
        columns: Optional[Sequence[ColumnDef]] = None,
        *,
        if_not_exists: bool = True,
        temporary: bool = False,
        constraints: Optional[
            Sequence[Union["UniqueConstraint", "CheckConstraint", "ForeignKeyConstraint"]]
        ] = None,
    ) -> "CreateTableOperation":
        """Create a lazy create table operation.

        Args:
            name_or_model: Name of the table to create, or SQLAlchemy model class
            columns: Sequence of ColumnDef objects defining the table schema (required if name_or_model is str)
            if_not_exists: If True, don't error if table already exists (default: True)
            temporary: If True, create a temporary table (default: False)
            constraints: Optional sequence of constraint objects (UniqueConstraint, CheckConstraint, ForeignKeyConstraint).
                        Ignored if model_class is provided (constraints are extracted from model).

        Returns:
            CreateTableOperation that executes on collect()

        Raises:
            ValidationError: If table name or columns are invalid
            ValueError: If model_class is not a valid SQLAlchemy model

        Example:
            >>> from moltres.table.schema import column, unique, check, foreign_key
            >>> op = db.create_table(
            ...     "users",
            ...     [column("id", "INTEGER", primary_key=True), column("email", "TEXT")],
            ...     constraints=[unique("email"), check("id > 0", name="ck_positive_id")]
            ... )
            >>> table = op.collect()  # Executes the CREATE TABLE

            >>> # Or with SQLAlchemy model
            >>> from sqlalchemy.orm import DeclarativeBase
            >>> class User(Base):
            ...     __tablename__ = "users"
            ...     id = Column(Integer, primary_key=True)
            >>> op = db.create_table(User)
            >>> table = op.collect()
        """
        from ..utils.exceptions import ValidationError
        from .actions import CreateTableOperation
        from .sqlalchemy_integration import (
            is_sqlalchemy_model,
            model_to_schema,
        )

        # Check if first argument is a SQLAlchemy model
        if is_sqlalchemy_model(name_or_model):
            # Model-based creation
            model_class: Type["DeclarativeBase"] = name_or_model  # type: ignore[assignment]
            schema = model_to_schema(model_class)

            op = CreateTableOperation(
                database=self,
                name=schema.name,
                columns=schema.columns,
                if_not_exists=if_not_exists,
                temporary=temporary,
                constraints=schema.constraints,
                model=model_class,
            )
        else:
            # Traditional string + columns creation
            table_name: str = name_or_model  # type: ignore[assignment]
            if columns is None:
                raise ValidationError("columns parameter is required when creating table from name")

            # Validate early (at operation creation time)
            if not columns:
                raise ValidationError(f"Cannot create table '{table_name}' with no columns")

            op = CreateTableOperation(
                database=self,
                name=table_name,
                columns=columns,
                if_not_exists=if_not_exists,
                temporary=temporary,
                constraints=constraints or (),
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

    def create_index(
        self,
        name: str,
        table: str,
        columns: Union[str, Sequence[str]],
        *,
        unique: bool = False,
        if_not_exists: bool = True,
    ) -> "CreateIndexOperation":
        """Create a lazy create index operation.

        Args:
            name: Name of the index to create
            table: Name of the table to create the index on
            columns: Column name(s) to index (single string or sequence)
            unique: If True, create a UNIQUE index (default: False)
            if_not_exists: If True, don't error if index already exists (default: True)

        Returns:
            CreateIndexOperation that executes on collect()

        Example:
            >>> op = db.create_index("idx_email", "users", "email")
            >>> op.collect()  # Executes the CREATE INDEX
            >>> # Multi-column index
            >>> op2 = db.create_index("idx_name_age", "users", ["name", "age"], unique=True)
        """
        from .actions import CreateIndexOperation
        from .batch import get_active_batch

        op = CreateIndexOperation(
            database=self,
            name=name,
            table_name=table,
            columns=columns,
            unique=unique,
            if_not_exists=if_not_exists,
        )
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    def drop_index(
        self,
        name: str,
        table: Optional[str] = None,
        *,
        if_exists: bool = True,
    ) -> "DropIndexOperation":
        """Create a lazy drop index operation.

        Args:
            name: Name of the index to drop
            table: Optional table name (required for some dialects like MySQL)
            if_exists: If True, don't error if index doesn't exist (default: True)

        Returns:
            DropIndexOperation that executes on collect()

        Example:
            >>> op = db.drop_index("idx_email", "users")
            >>> op.collect()  # Executes the DROP INDEX
        """
        from .actions import DropIndexOperation
        from .batch import get_active_batch

        op = DropIndexOperation(
            database=self,
            name=name,
            table_name=table,
            if_exists=if_exists,
        )
        batch = get_active_batch()
        if batch is not None:
            batch.add(op)
        return op

    # -------------------------------------------------------------- schema inspection
    def get_table_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of table names in the database.

        Args:
            schema: Optional schema name (for multi-schema databases like PostgreSQL).
                    If None, uses default schema.

        Returns:
            List of table names

        Raises:
            ValueError: If database connection is not available
            RuntimeError: If inspection fails

        Example:
            >>> tables = db.get_table_names()
            >>> # Returns: ['users', 'orders', 'products']
        """
        from ..utils.inspector import get_table_names

        return get_table_names(self, schema=schema)

    def get_view_names(self, schema: Optional[str] = None) -> List[str]:
        """Get list of view names in the database.

        Args:
            schema: Optional schema name (for multi-schema databases like PostgreSQL).
                    If None, uses default schema.

        Returns:
            List of view names

        Raises:
            ValueError: If database connection is not available
            RuntimeError: If inspection fails

        Example:
            >>> views = db.get_view_names()
            >>> # Returns: ['active_users_view', 'order_summary_view']
        """
        from ..utils.inspector import get_view_names

        return get_view_names(self, schema=schema)

    def get_columns(self, table_name: str) -> List["ColumnInfo"]:
        """Get column information for a table.

        Args:
            table_name: Name of the table to inspect

        Returns:
            List of ColumnInfo objects with column metadata

        Raises:
            ValidationError: If table name is invalid
            ValueError: If database connection is not available
            RuntimeError: If table does not exist or cannot be inspected

        Example:
            >>> columns = db.get_columns("users")
            >>> # Returns: [ColumnInfo(name='id', type_name='INTEGER', ...), ...]
        """
        from ..utils.exceptions import ValidationError
        from ..utils.inspector import get_table_columns
        from ..sql.builders import quote_identifier

        if not table_name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(table_name, self._dialect.quote_char)

        return get_table_columns(self, table_name)

    def reflect_table(self, name: str, schema: Optional[str] = None) -> "TableSchema":
        """Reflect a single table from the database.

        Args:
            name: Name of the table to reflect
            schema: Optional schema name (for multi-schema databases like PostgreSQL).
                    If None, uses default schema.

        Returns:
            TableSchema object with table metadata

        Raises:
            ValidationError: If table name is invalid
            ValueError: If database connection is not available
            RuntimeError: If table does not exist or reflection fails

        Example:
            >>> schema = db.reflect_table("users")
            >>> # Returns: TableSchema(name='users', columns=[ColumnDef(...), ...])
        """
        from ..utils.exceptions import ValidationError
        from ..utils.inspector import reflect_table
        from ..sql.builders import quote_identifier
        from .schema import TableSchema

        if not name:
            raise ValidationError("Table name cannot be empty")
        # Validate table name format
        quote_identifier(name, self._dialect.quote_char)

        reflected = reflect_table(self, name, schema=schema)
        column_defs = reflected[name]

        return TableSchema(name=name, columns=column_defs)

    def reflect(
        self, schema: Optional[str] = None, views: bool = False
    ) -> Dict[str, "TableSchema"]:
        """Reflect entire database schema.

        Args:
            schema: Optional schema name (for multi-schema databases like PostgreSQL).
                    If None, uses default schema.
            views: If True, also reflect views (default: False)

        Returns:
            Dictionary mapping table/view names to TableSchema objects

        Raises:
            ValueError: If database connection is not available
            RuntimeError: If reflection fails

        Example:
            >>> schemas = db.reflect()
            >>> # Returns: {'users': TableSchema(...), 'orders': TableSchema(...)}
        """
        from ..utils.inspector import reflect_database
        from .schema import TableSchema

        reflected = reflect_database(self, schema=schema, views=views)

        # Convert to TableSchema objects
        result: Dict[str, TableSchema] = {}
        for table_name, column_defs in reflected.items():
            result[table_name] = TableSchema(name=table_name, columns=column_defs)

        return result

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

    def explain(self, sql: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Get the execution plan for a SQL query.

        Args:
            sql: SQL query string
            params: Optional query parameters

        Returns:
            Execution plan as a string (dialect-specific)

        Example:
            >>> plan = db.explain("SELECT * FROM users WHERE id = :id", params={"id": 1})
            >>> print(plan)
        """
        dialect_name = self._dialect.name
        if dialect_name == "postgresql":
            explain_sql = f"EXPLAIN ANALYZE {sql}"
        elif dialect_name == "mysql":
            explain_sql = f"EXPLAIN {sql}"
        elif dialect_name == "sqlite":
            explain_sql = f"EXPLAIN QUERY PLAN {sql}"
        else:
            explain_sql = f"EXPLAIN {sql}"

        result = self._executor.fetch(explain_sql, params=params)
        return "\n".join(" ".join(str(cell) for cell in row) for row in result.rows)

    def show_tables(self, schema: Optional[str] = None) -> None:
        """Print a formatted list of tables in the database.

        Convenience method for interactive exploration.

        Args:
            schema: Optional schema name

        Example:
            >>> db.show_tables()
            Tables in database:
            - users
            - orders
            - products
        """
        tables = self.get_table_names(schema=schema)
        if tables:
            print("Tables in database:")
            for table in sorted(tables):
                print(f"  - {table}")
        else:
            print("No tables found in database.")

    def show_schema(self, table_name: str) -> None:
        """Print a formatted schema for a table.

        Convenience method for interactive exploration.

        Args:
            table_name: Name of the table

        Example:
            >>> db.show_schema("users")
            Schema for table 'users':
            - id: INTEGER (primary_key=True)
            - name: TEXT
            - email: TEXT
        """
        from ..utils.exceptions import ValidationError

        if not table_name:
            raise ValidationError("Table name cannot be empty")

        columns = self.get_columns(table_name)
        if columns:
            print(f"Schema for table '{table_name}':")
            for col_info in columns:
                attrs = []
                if col_info.primary_key:
                    attrs.append("primary_key=True")
                if col_info.nullable is False:
                    attrs.append("nullable=False")
                if col_info.default is not None:
                    attrs.append(f"default={col_info.default}")
                attr_str = f" ({', '.join(attrs)})" if attrs else ""
                print(f"  - {col_info.name}: {col_info.type_name}{attr_str}")
        else:
            print(f"No columns found for table '{table_name}'.")

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
            ...     df.write.insertInto("table")
            ...     df.write.update("table", where=..., set={...})
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

    def createDataFrame(
        self,
        data: Union[
            Sequence[dict[str, object]],
            Sequence[tuple],
            Records,
            "LazyRecords",
            "pd.DataFrame",
            "pl.DataFrame",
            "pl.LazyFrame",
        ],
        schema: Optional[Sequence[ColumnDef]] = None,
        pk: Optional[Union[str, Sequence[str]]] = None,
        auto_pk: Optional[Union[str, Sequence[str]]] = None,
    ) -> "DataFrame":
        """Create a DataFrame from Python data (list of dicts, list of tuples, Records, LazyRecords, pandas DataFrame, polars DataFrame, or polars LazyFrame).

        Creates a temporary table, inserts the data, and returns a DataFrame querying from that table.
        If LazyRecords is provided, it will be auto-materialized.
        If pandas/polars DataFrame or LazyFrame is provided, it will be converted to Records with lazy conversion.

        Args:
            data: Input data in one of supported formats:
                - List of dicts: [{"col1": val1, "col2": val2}, ...]
                - List of tuples: Requires schema parameter with column names
                - Records object: Extracts data and schema if available
                - LazyRecords object: Auto-materializes and extracts data and schema
                - pandas DataFrame: Converts to Records with schema preservation
                - polars DataFrame: Converts to Records with schema preservation
                - polars LazyFrame: Materializes and converts to Records with schema preservation
            schema: Optional explicit schema. If not provided, schema is inferred from data.
            pk: Optional column name(s) to mark as primary key. Can be a single string or sequence of strings for composite keys.
            auto_pk: Optional column name(s) to create as auto-incrementing primary key. Can specify same name as pk to make an existing column auto-incrementing.

        Returns:
            DataFrame querying from the created temporary table

        Raises:
            ValueError: If data is empty and no schema provided, or if primary key requirements are not met
            ValidationError: If list of tuples provided without schema, or other validation errors

        Example:
            >>> # Create DataFrame from list of dicts
            >>> df = db.createDataFrame([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], pk="id")
            >>> # Create DataFrame with auto-incrementing primary key
            >>> df = db.createDataFrame([{"name": "Alice"}, {"name": "Bob"}], auto_pk="id")
            >>> # Create DataFrame from Records
            >>> from moltres.io.records import Records
            >>> records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
            >>> df = db.createDataFrame(records, pk="id")
            >>> # Create DataFrame from LazyRecords (auto-materializes)
            >>> lazy_records = db.read.records.csv("data.csv")
            >>> df = db.createDataFrame(lazy_records, pk="id")
            >>> # Create DataFrame from pandas DataFrame
            >>> import pandas as pd
            >>> pdf = pd.DataFrame([{"id": 1, "name": "Alice"}])
            >>> df = db.createDataFrame(pdf, pk="id")
            >>> # Create DataFrame from polars DataFrame
            >>> import polars as pl
            >>> plf = pl.DataFrame([{"id": 1, "name": "Alice"}])
            >>> df = db.createDataFrame(plf, pk="id")
        """
        from ..dataframe.create_dataframe import (
            ensure_primary_key,
            generate_unique_table_name,
            get_schema_from_records,
            normalize_data_to_rows,
        )
        from ..dataframe.dataframe import DataFrame
        from ..dataframe.readers.schema_inference import infer_schema_from_rows
        from ..io.records import (
            LazyRecords,
            Records,
            _is_pandas_dataframe,
            _is_polars_dataframe,
            _is_polars_lazyframe,
            _dataframe_to_records,
        )
        from ..utils.exceptions import ValidationError

        # Convert DataFrame to Records if needed
        if _is_pandas_dataframe(data) or _is_polars_dataframe(data) or _is_polars_lazyframe(data):
            data = _dataframe_to_records(data, database=self)

        # Normalize data to list of dicts
        # Handle LazyRecords by auto-materializing
        if isinstance(data, LazyRecords):
            materialized_records = data.collect()  # Auto-materialize
            rows = normalize_data_to_rows(materialized_records)
            # Use schema from Records if available and no explicit schema provided
            if schema is None:
                schema = get_schema_from_records(materialized_records)
        elif isinstance(data, Records):
            rows = normalize_data_to_rows(data)
            # Use schema from Records if available and no explicit schema provided
            if schema is None:
                schema = get_schema_from_records(data)
        elif isinstance(data, list) and data and isinstance(data[0], tuple):
            # Handle list of tuples - requires schema
            if schema is None:
                from ..utils.exceptions import ValidationError

                raise ValidationError(
                    "List of tuples requires a schema with column names. "
                    "Provide schema parameter or use list of dicts instead."
                )
            # Convert tuples to dicts using schema column names
            column_names = [col.name for col in schema]
            rows = []
            for row_tuple in data:
                if len(row_tuple) != len(column_names):
                    raise ValueError(
                        f"Tuple length {len(row_tuple)} does not match schema column count {len(column_names)}"
                    )
                rows.append(dict(zip(column_names, row_tuple)))
        else:
            rows = normalize_data_to_rows(data)

        # Validate data is not empty (unless schema provided)
        if not rows and schema is None:
            raise ValueError("Cannot create DataFrame from empty data without a schema")

        # Infer or use schema
        if schema is None:
            if not rows:
                raise ValueError("Cannot infer schema from empty data. Provide schema parameter.")
            inferred_schema_list = list(infer_schema_from_rows(rows))
        else:
            inferred_schema_list = list(schema)

        # Ensure primary key
        inferred_schema_list, new_auto_increment_cols = ensure_primary_key(
            inferred_schema_list,
            pk=pk,
            auto_pk=auto_pk,
            dialect_name=self._dialect_name,
            require_primary_key=False,
        )

        # Generate unique table name
        table_name = generate_unique_table_name()

        # Always use persistent staging tables so later operations (which may run on a different
        # pooled connection) can still access the data. Ephemeral cleanup happens via close().
        use_temp_tables = False
        table_handle = self.create_table(
            table_name,
            inferred_schema_list,
            temporary=use_temp_tables,
            if_not_exists=True,
        ).collect()
        if not use_temp_tables:
            self._register_ephemeral_table(table_name)

        # Insert data (exclude new auto-increment columns from INSERT)
        if rows:
            # Filter rows to only include columns that exist in schema and are not new auto-increment columns
            filtered_rows = []
            for row in rows:
                filtered_row = {
                    k: v
                    for k, v in row.items()
                    if k not in new_auto_increment_cols
                    and any(col.name == k for col in inferred_schema_list)
                }
                filtered_rows.append(filtered_row)

            records_to_insert = Records(_data=filtered_rows, _database=self)
            records_to_insert.insert_into(table_handle)

        # Return DataFrame querying from the temporary table
        return DataFrame.from_table(table_handle)

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


def _cleanup_all_databases() -> None:
    """Best-effort cleanup for any Database instances left open at exit.

    This is called on normal interpreter shutdown and on signal handlers
    for crash scenarios (SIGTERM, SIGINT).
    """
    for db in list(_ACTIVE_DATABASES):
        try:
            db._close_resources()
        except Exception as exc:  # pragma: no cover - atexit safeguard
            logger.debug("Database cleanup during interpreter shutdown failed: %s", exc)


def _signal_handler(signum: int, frame: Any) -> None:
    """Handle signals (SIGTERM, SIGINT) by cleaning up databases before exit."""
    logger.info("Received signal %d, cleaning up databases...", signum)
    _cleanup_all_databases()
    # Re-raise the signal with default handler
    signal.signal(signum, signal.SIG_DFL)
    import os

    os.kill(os.getpid(), signum)


# Register signal handlers for crash scenarios (only on main thread)
try:
    # Check if we can register signal handlers (main thread only)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
except (ValueError, OSError):
    # Signal handlers can only be registered on the main thread
    # This is expected in some contexts (e.g., subprocesses, threads)
    pass


def _force_database_cleanup_for_tests() -> None:
    """Helper used by tests to simulate crash/GC cleanup."""
    _cleanup_all_databases()


atexit.register(_cleanup_all_databases)
