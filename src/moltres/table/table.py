"""Table access primitives."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Union

from ..config import MoltresConfig

if TYPE_CHECKING:
    from ..dataframe.dataframe import DataFrame
    from ..dataframe.reader import DataLoader, ReadAccessor
    from ..io.records import LazyRecords, Records
    from .actions import (
        CreateTableOperation,
        DropTableOperation,
    )
    from .batch import OperationBatch
from sqlalchemy.engine import Connection

from ..engine.connection import ConnectionManager
from ..engine.dialects import DialectSpec, get_dialect
from ..engine.execution import QueryExecutor, QueryResult
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

    # ----------------------------------------------------------------- internals
    def createDataFrame(
        self,
        data: Union[Sequence[dict[str, object]], Sequence[tuple], Records, "LazyRecords"],
        schema: Optional[Sequence[ColumnDef]] = None,
        pk: Optional[Union[str, Sequence[str]]] = None,
        auto_pk: Optional[Union[str, Sequence[str]]] = None,
    ) -> "DataFrame":
        """Create a DataFrame from Python data (list of dicts, list of tuples, Records, or LazyRecords).

        Creates a temporary table, inserts the data, and returns a DataFrame querying from that table.
        If LazyRecords is provided, it will be auto-materialized.

        Args:
            data: Input data in one of supported formats:
                - List of dicts: [{"col1": val1, "col2": val2}, ...]
                - List of tuples: Requires schema parameter with column names
                - Records object: Extracts data and schema if available
                - LazyRecords object: Auto-materializes and extracts data and schema
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
        """
        from ..dataframe.create_dataframe import (
            ensure_primary_key,
            generate_unique_table_name,
            get_schema_from_records,
            normalize_data_to_rows,
        )
        from ..dataframe.dataframe import DataFrame
        from ..dataframe.readers.schema_inference import infer_schema_from_rows
        from ..io.records import LazyRecords, Records
        from ..utils.exceptions import ValidationError

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
        )

        # Generate unique table name
        table_name = generate_unique_table_name()

        # Create temporary table
        table_handle = self.create_table(
            table_name,
            inferred_schema_list,
            temporary=True,
            if_not_exists=True,
        ).collect()

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
