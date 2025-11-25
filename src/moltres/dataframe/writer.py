"""DataFrame write operations."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

from ..expressions.column import Column
from ..table.mutations import delete_rows, insert_rows, update_rows
from ..table.schema import ColumnDef
from .dataframe import DataFrame

if TYPE_CHECKING:
    from ..table.table import Database


class DataFrameWriter:
    """Builder for writing DataFrames to tables."""

    def __init__(self, df: DataFrame):
        self._df = df
        self._mode: str = "append"
        self._table_name: Optional[str] = None
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}
        self._partition_by: Optional[Sequence[str]] = None
        self._stream_override: Optional[bool] = None
        self._primary_key: Optional[Sequence[str]] = None
        self._format: Optional[str] = None
        self._bucket_by: Optional[tuple[int, Sequence[str]]] = None
        self._sort_by: Optional[Sequence[str]] = None

    def mode(self, mode: str) -> "DataFrameWriter":
        """Set the write mode (append, overwrite, ignore, error_if_exists)."""
        normalized = mode.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
        if normalized == "errorifexists":
            normalized = "error_if_exists"
        elif normalized == "ignore":
            normalized = "ignore"
        elif normalized not in {"append", "overwrite"}:
            if normalized != "error_if_exists":
                raise ValueError(
                    f"Invalid mode '{mode}'. Must be one of: append, overwrite, ignore, error_if_exists"
                )
        self._mode = normalized
        return self

    def option(self, key: str, value: object) -> "DataFrameWriter":
        """Set a write option (e.g., header=True for CSV, compression='gzip' for Parquet)."""
        self._options[key] = value
        return self

    def options(self, *args: Mapping[str, object], **kwargs: object) -> "DataFrameWriter":
        """Set multiple write options at once."""
        if len(args) > 1:
            raise TypeError("options() accepts at most one positional mapping argument")
        if args:
            for key, value in args[0].items():
                self._options[str(key)] = value
        for key, value in kwargs.items():
            self._options[key] = value
        return self

    def format(self, format_name: str) -> "DataFrameWriter":
        """Specify the output format for save()."""
        self._format = format_name.strip().lower()
        return self

    def stream(self, enabled: bool = True) -> "DataFrameWriter":
        """Enable or disable streaming mode (chunked writing for large DataFrames)."""
        self._stream_override = bool(enabled)
        return self

    def partitionBy(self, *columns: str) -> "DataFrameWriter":
        """Partition data by the given columns when writing to files."""
        self._partition_by = columns if columns else None
        return self

    partition_by = partitionBy

    def schema(self, schema: Sequence[ColumnDef]) -> "DataFrameWriter":
        """Set an explicit schema for the target table."""
        self._schema = schema
        return self

    def primaryKey(self, *columns: str) -> "DataFrameWriter":
        """Specify primary key columns for the target table.

        Args:
            *columns: Column names to use as primary key

        Returns:
            Self for method chaining
        """
        self._primary_key = columns if columns else None
        return self

    primary_key = primaryKey

    def bucketBy(self, num_buckets: int, *columns: str) -> "DataFrameWriter":
        """PySpark-compatible bucketing hook (metadata only for now)."""
        if num_buckets <= 0:
            raise ValueError("num_buckets must be a positive integer")
        if not columns:
            raise ValueError("bucketBy() requires at least one column name")
        self._bucket_by = (num_buckets, tuple(columns))
        return self

    bucket_by = bucketBy

    def sortBy(self, *columns: str) -> "DataFrameWriter":
        """PySpark-compatible sortBy hook (metadata only for now)."""
        if not columns:
            raise ValueError("sortBy() requires at least one column name")
        self._sort_by = tuple(columns)
        return self

    sort_by = sortBy

    def save_as_table(self, name: str, primary_key: Optional[Sequence[str]] = None) -> None:
        """Write the DataFrame to a table with the given name.

        Args:
            name: Name of the target table
            primary_key: Optional sequence of column names to use as primary key.
                        If provided, overrides any primary key set via .primaryKey()
        """
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        # Use parameter if provided, otherwise use field
        if primary_key is not None:
            self._primary_key = primary_key
        self._table_name = name
        self._execute_write()

    saveAsTable = save_as_table  # PySpark-style alias

    def insertInto(self, table_name: str) -> None:
        """Insert DataFrame into an existing table (table must already exist)."""
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        db = self._df.database
        if not self._table_exists(db, table_name):
            raise ValueError(
                f"Table '{table_name}' does not exist. Use save_as_table() to create it."
            )

        if self._bucket_by or self._sort_by:
            raise NotImplementedError(
                "bucketBy/sortBy are not yet supported when writing to tables."
            )

        # Get active transaction if available
        transaction = db.connection_manager.active_transaction
        table_handle = db.table(table_name)

        use_stream = self._should_stream_materialization()
        rows, chunk_iter = self._collect_rows(use_stream)
        if use_stream:
            if rows:
                insert_rows(table_handle, rows, transaction=transaction)
            if chunk_iter is not None:
                for chunk in chunk_iter:
                    if chunk:
                        insert_rows(table_handle, chunk, transaction=transaction)
        elif rows:
            insert_rows(table_handle, rows, transaction=transaction)

    insert_into = insertInto

    def update(
        self,
        table_name: str,
        *,
        where: Column,
        set: Mapping[str, object],
    ) -> None:
        """Update rows in a table matching the WHERE condition.

        Executes immediately (eager execution like PySpark writes).

        Args:
            table_name: Name of the table to update
            where: Column expression for the WHERE clause
            set: Dictionary of column names to new values

        Example:
            >>> df = db.table("users").select()
            >>> df.write.update("users", where=col("id") == 1, set={"name": "Bob"})
        """
        if self._df.database is None:
            raise RuntimeError("Cannot update table without an attached Database")

        db = self._df.database
        if not self._table_exists(db, table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        # Get active transaction if available
        transaction = db.connection_manager.active_transaction

        # Use the mutation helper function
        table_handle = db.table(table_name)
        update_rows(table_handle, where=where, values=set, transaction=transaction)

    def delete(
        self,
        table_name: str,
        *,
        where: Column,
    ) -> None:
        """Delete rows from a table matching the WHERE condition.

        Executes immediately (eager execution like PySpark writes).

        Args:
            table_name: Name of the table to delete from
            where: Column expression for the WHERE clause

        Example:
            >>> df = db.table("users").select()
            >>> df.write.delete("users", where=col("id") == 1)
        """
        if self._df.database is None:
            raise RuntimeError("Cannot delete from table without an attached Database")

        db = self._df.database
        if not self._table_exists(db, table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        # Get active transaction if available
        transaction = db.connection_manager.active_transaction

        # Use the mutation helper function
        table_handle = db.table(table_name)
        delete_rows(table_handle, where=where, transaction=transaction)

    def save(self, path: str, format: Optional[str] = None) -> None:
        """Save DataFrame to a file or directory in the specified format."""
        format_to_use = format or self._format
        if format_to_use is None:
            # Infer format from file extension
            ext = Path(path).suffix.lower()
            format_map = {
                ".csv": "csv",
                ".json": "json",
                ".jsonl": "jsonl",
                ".txt": "text",
                ".parquet": "parquet",
            }
            format_to_use = format_map.get(ext)
            if format_to_use is None:
                raise ValueError(
                    f"Cannot infer format from path '{path}'. Specify format explicitly."
                )

        format_lower = format_to_use.lower()
        if format_lower == "csv":
            self._save_csv(path)
        elif format_lower == "json":
            self._save_json(path)
        elif format_lower == "jsonl":
            self._save_jsonl(path)
        elif format_lower == "text":
            self._save_text(path)
        elif format_lower == "parquet":
            self._save_parquet(path)
        elif format_lower == "orc":
            raise NotImplementedError(
                "ORC write support is not yet available. "
                "Consider contributing an implementation or using parquet/csv/json."
            )
        else:
            raise ValueError(
                f"Unsupported format '{format_to_use}'. Supported: csv, json, jsonl, text, parquet"
            )

    def csv(self, path: str) -> None:
        """Save DataFrame as CSV file."""
        self._save_csv(path)

    def json(self, path: str) -> None:
        """Save DataFrame as JSON file (array of objects)."""
        self._save_json(path)

    def jsonl(self, path: str) -> None:
        """Save DataFrame as JSONL file (one JSON object per line)."""
        self._save_jsonl(path)

    def text(self, path: str) -> None:
        """Save DataFrame as text file (expects a single 'value' column)."""
        self._save_text(path)

    def orc(self, path: str) -> None:
        """PySpark-style ORC helper (not yet implemented)."""
        raise NotImplementedError(
            "ORC output is not yet supported. Use parquet or contribute ORC writer support."
        )

    def parquet(self, path: str) -> None:
        """Save DataFrame as Parquet file."""
        self._save_parquet(path)

    def _execute_write(self) -> None:
        """Execute the write operation based on mode and table existence."""
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        db = self._df.database
        table_name = self._table_name
        if table_name is None:
            raise ValueError("Table name must be specified via save_as_table()")

        if self._bucket_by or self._sort_by:
            raise NotImplementedError(
                "bucketBy/sortBy are not yet supported when writing to tables."
            )

        # Check if table exists
        table_exists = self._table_exists(db, table_name)

        if self._mode == "error_if_exists" and table_exists:
            raise ValueError(f"Table '{table_name}' already exists and mode is 'error_if_exists'")
        if self._mode == "ignore" and table_exists:
            return

        # Check if we can use INSERT INTO ... SELECT optimization
        if self._can_use_insert_select():
            # Try to infer schema from plan
            schema = self._schema or self._infer_schema_from_plan()

            if schema is None:
                # Can't infer schema, fall back to materialization
                # Continue with existing materialization code below
                pass
            else:
                # Use optimized path
                self._execute_insert_select(schema)
                return

        # Fall back to existing materialization approach
        use_stream = self._should_stream_materialization()
        rows, chunk_iter = self._collect_rows(use_stream)

        # Infer or get schema (uses rows if needed, but allows empty if schema is explicit)
        try:
            schema = self._infer_or_get_schema(rows, force_nullable=use_stream)
        except ValueError as e:
            # Check if this is a primary key validation error - if so, re-raise it
            if "Primary key columns" in str(e) or "do not exist in schema" in str(e):
                raise
            # Empty DataFrame without explicit schema - try to infer from plan
            if self._schema is None:
                # For empty DataFrames, we can't reliably infer, so require explicit schema
                raise ValueError(
                    "Cannot infer schema from empty DataFrame. "
                    "Provide explicit schema via .schema([ColumnDef(...), ...])"
                )
            schema = self._schema

        # Handle overwrite mode
        if self._mode == "overwrite":
            if table_exists:
                db.drop_table(table_name, if_exists=True).collect()
            db.create_table(table_name, schema, if_not_exists=False).collect()
        elif not table_exists:
            # Create table if it doesn't exist
            db.create_table(table_name, schema, if_not_exists=True).collect()

        # Insert data (if any)
        table_handle = db.table(table_name)
        transaction = db.connection_manager.active_transaction
        if use_stream:
            if rows:
                insert_rows(table_handle, rows, transaction=transaction)
            if chunk_iter is not None:
                for chunk in chunk_iter:
                    if chunk:
                        insert_rows(table_handle, chunk, transaction=transaction)
        elif rows:
            insert_rows(table_handle, rows, transaction=transaction)

    def _infer_schema_from_plan(self) -> Optional[Sequence[ColumnDef]]:
        """Infer schema from DataFrame plan without materializing data.

        Returns:
            Inferred schema or None if inference is not possible
        """
        if self._df.database is None:
            return None

        try:
            from ..sql.compiler import compile_plan

            # Compile the plan to a SELECT statement
            select_stmt = compile_plan(self._df.plan, dialect=self._df.database.dialect)

            # Execute the query with LIMIT 1 to get a sample row for schema inference
            # This is much more efficient than materializing all data
            sample_stmt = select_stmt.limit(1)
            sample_result = self._df.database.executor.fetch(sample_stmt)

            # Get column names from the result
            # The result.rows is a list of dicts, so we can get column names from keys
            if sample_result.rows and len(sample_result.rows) > 0:
                sample_row = sample_result.rows[0]
                column_names = list(sample_row.keys())

                # Infer types from sample data
                column_defs = []
                for col_name in column_names:
                    value = sample_row.get(col_name)
                    col_type = self._infer_type_from_value(value)
                    # Check if column is nullable (if value is None, it's nullable)
                    nullable = value is None
                    column_defs.append(
                        ColumnDef(name=col_name, type_name=col_type, nullable=nullable)
                    )
                return column_defs
            else:
                # No rows returned, but we can still infer schema from the SELECT statement
                # Extract column names from the SELECT statement
                column_names = []
                for col_expr in select_stmt.selected_columns:
                    # Try to get column name from the expression
                    if hasattr(col_expr, "name") and col_expr.name:
                        column_names.append(col_expr.name)
                    elif hasattr(col_expr, "key") and col_expr.key:
                        column_names.append(col_expr.key)
                    elif hasattr(col_expr, "_label") and col_expr._label:
                        column_names.append(col_expr._label)
                    else:
                        # Fallback: use string representation and try to extract name
                        col_str = str(col_expr)
                        # Remove quotes and extract name
                        col_str = col_str.strip("\"'")
                        if "." in col_str:
                            column_names.append(col_str.split(".")[-1])
                        else:
                            column_names.append(col_str)

                if not column_names:
                    return None

                # Use TEXT as default type for empty result sets
                return [
                    ColumnDef(name=col_name, type_name="TEXT", nullable=True)
                    for col_name in column_names
                ]

        except Exception:
            # If inference fails, return None to fall back to materialization
            return None

    def _execute_insert_select(self, schema: Sequence[ColumnDef]) -> None:
        """Execute write using INSERT INTO ... SELECT optimization."""
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        db = self._df.database
        table_name = self._table_name
        if table_name is None:
            raise ValueError("Table name must be specified via save_as_table()")

        # Apply primary key flags to schema if specified
        final_schema = list(schema)
        if self._primary_key:
            # Validate that all primary key columns exist
            schema_column_names = {col.name for col in final_schema}
            missing_cols = [col for col in self._primary_key if col not in schema_column_names]
            if missing_cols:
                raise ValueError(
                    f"Primary key columns {missing_cols} do not exist in schema. "
                    f"Available columns: {sorted(schema_column_names)}"
                )

            # Apply primary key flags
            primary_key_set = set(self._primary_key)
            final_schema = [
                ColumnDef(
                    name=col.name,
                    type_name=col.type_name,
                    nullable=col.nullable,
                    default=col.default,
                    primary_key=col.name in primary_key_set,
                )
                for col in final_schema
            ]

        # Check if table exists
        table_exists = self._table_exists(db, table_name)

        # Handle overwrite/ignore/error modes
        if self._mode == "error_if_exists" and table_exists:
            raise ValueError(f"Table '{table_name}' already exists and mode is 'error_if_exists'")
        if self._mode == "ignore" and table_exists:
            return
        if self._mode == "overwrite":
            if table_exists:
                db.drop_table(table_name, if_exists=True).collect()
            db.create_table(table_name, final_schema, if_not_exists=False).collect()
        elif not table_exists:
            # Create table if it doesn't exist
            db.create_table(table_name, final_schema, if_not_exists=True).collect()

        # Compile DataFrame plan to SELECT statement
        from ..sql.compiler import compile_plan

        select_stmt = compile_plan(self._df.plan, dialect=db.dialect)

        # Get column names from schema
        column_names = [col.name for col in final_schema]

        # Execute INSERT INTO ... SELECT using SQLAlchemy statement directly
        # This ensures parameters from WHERE clauses are properly handled
        from sqlalchemy import insert, types as sa_types
        from sqlalchemy.schema import MetaData, Table, Column

        metadata = MetaData()
        table = Table(table_name, metadata)

        # Add columns to the table if specified (needed for from_select)
        if column_names:
            for col_name in column_names:
                table.append_column(Column(col_name, sa_types.String()))
            insert_stmt = insert(table).from_select(
                [table.c[col] for col in column_names], select_stmt
            )
        else:
            # Insert all columns
            col_names = [
                col.name if hasattr(col, "name") else str(col)
                for col in select_stmt.selected_columns
            ]
            for col_name in col_names:
                table.append_column(Column(col_name, sa_types.String()))
            insert_stmt = insert(table).from_select(
                [table.c[col] for col in col_names], select_stmt
            )

        # Execute the SQLAlchemy statement directly (handles parameters automatically)
        with db.connection_manager.connect() as conn:
            conn.execute(insert_stmt)
            conn.commit()

    def _can_use_insert_select(self) -> bool:
        """Check if we can use INSERT INTO ... SELECT optimization.

        Returns:
            True if optimization is possible, False otherwise
        """
        # DataFrame must have a database connection
        if self._df.database is None:
            return False

        # Not in streaming mode (streaming requires materialization for chunking)
        if self._stream_override:
            return False

        # Mode must not be "error_if_exists" (need to check table existence first,
        # which requires materialization path)
        if self._mode == "error_if_exists":
            return False

        # Plan must be compilable to SQL (all operations are SQL-compatible)
        # We can check this by trying to compile the plan
        try:
            from ..sql.compiler import compile_plan

            compile_plan(self._df.plan, dialect=self._df.database.dialect)
            return True
        except Exception:
            # If compilation fails, can't use optimization
            return False

    def _table_exists(self, db: "Database", table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            # Use a simple query that should work across dialects
            if db.dialect.name == "sqlite":
                sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=:name"
                result = db.execute_sql(sql, params={"name": table_name})
                return len(result.rows) > 0
            elif db.dialect.name == "postgresql":
                sql = "SELECT tablename FROM pg_tables WHERE tablename=:name"
                result = db.execute_sql(sql, params={"name": table_name})
                return len(result.rows) > 0
            else:
                # Generic approach: try to select from the table with LIMIT 0
                quote = db.dialect.quote_char
                sql = f"SELECT * FROM {quote}{table_name}{quote} LIMIT 0"
                db.execute_sql(sql)
                return True
        except Exception:
            return False

    def _infer_or_get_schema(
        self, rows: List[dict[str, object]], *, force_nullable: bool = False
    ) -> Sequence[ColumnDef]:
        """Infer schema from DataFrame plan or use provided schema."""
        if self._schema is not None:
            schema = list(self._schema)
        else:
            # Infer schema from collected data (most reliable)
            if not rows:
                raise ValueError(
                    "Cannot infer schema from empty DataFrame. Provide explicit schema via .schema()"
                )

            # Use first row to infer types, but check all rows for None values
            sample = rows[0]
            columns: List[ColumnDef] = []

            for key, value in sample.items():
                # Check if any row has None for this column (or force nullable when streaming)
                has_nulls = True if force_nullable else any(row.get(key) is None for row in rows)
                col_type = self._infer_type_from_value(value)
                columns.append(ColumnDef(name=key, type_name=col_type, nullable=has_nulls))

            schema = columns

        # Apply primary key flags if specified
        if self._primary_key:
            # Validate that all primary key columns exist
            schema_column_names = {col.name for col in schema}
            missing_cols = [col for col in self._primary_key if col not in schema_column_names]
            if missing_cols:
                raise ValueError(
                    f"Primary key columns {missing_cols} do not exist in schema. "
                    f"Available columns: {sorted(schema_column_names)}"
                )

            # Apply primary key flags
            primary_key_set = set(self._primary_key)
            schema = [
                ColumnDef(
                    name=col.name,
                    type_name=col.type_name,
                    nullable=col.nullable,
                    default=col.default,
                    primary_key=col.name in primary_key_set,
                )
                for col in schema
            ]

        return schema

    def _should_stream_materialization(self) -> bool:
        """Default to streaming inserts unless user explicitly disabled it."""
        if self._stream_override is not None:
            return self._stream_override
        return True

    def _should_stream_output(self) -> bool:
        """Use chunked output by default unless user explicitly disables it."""
        if self._stream_override is not None:
            return self._stream_override
        return True

    def _collect_rows(
        self, use_stream: bool
    ) -> tuple[List[dict[str, object]], Optional[Iterator[List[dict[str, object]]]]]:
        """Collect rows, optionally streaming in chunks."""
        if use_stream:
            chunk_iter = self._df.collect(stream=True)
            try:
                first_chunk = next(chunk_iter)
            except StopIteration:
                return [], chunk_iter
            return first_chunk, chunk_iter
        rows = self._df.collect()
        return rows, None

    def _ensure_file_layout_supported(self) -> None:
        """Raise if unsupported bucketing/sorting metadata is set for file sinks."""
        if self._bucket_by or self._sort_by:
            raise NotImplementedError(
                "bucketBy/sortBy metadata is not yet supported when writing to files."
            )

    def _prepare_file_target(self, path_obj: Path) -> bool:
        """Apply mode semantics (overwrite/ignore/error) for file outputs."""
        if self._mode == "ignore" and path_obj.exists():
            return False
        if self._mode == "error_if_exists" and path_obj.exists():
            raise ValueError(f"Target '{path_obj}' already exists (mode=error_if_exists)")
        if self._mode == "overwrite" and path_obj.exists():
            if path_obj.is_dir():
                shutil.rmtree(path_obj)
            else:
                path_obj.unlink()
        return True

    def _infer_type_from_value(self, value: object) -> str:
        """Infer SQL type from a Python value."""
        if value is None:
            return "TEXT"  # Can't infer from None
        if isinstance(value, bool):
            return "INTEGER"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "REAL"
        if isinstance(value, str):
            return "TEXT"
        return "TEXT"  # Default fallback

    def _save_csv(self, path: str) -> None:
        """Save DataFrame as CSV file."""
        self._ensure_file_layout_supported()

        if self._partition_by:
            rows = self._df.collect()
            if not rows and not self._schema:
                return
            headers = (
                [col.name for col in self._schema]
                if self._schema and not rows
                else (list(rows[0].keys()) if rows else [])
            )
            self._save_partitioned(path, "csv", rows, headers)
            return

        if self._should_stream_output():
            self._save_csv_stream(path)
            return

        rows = self._df.collect()
        if not rows:
            # Create empty file with headers if we have schema
            if self._schema:
                headers = [col.name for col in self._schema]
            else:
                headers = []
        else:
            headers = list(rows[0].keys())

        # Write CSV file
        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        header = cast(bool, self._options.get("header", True))
        delimiter = cast(str, self._options.get("delimiter", ","))

        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
            if header:
                writer.writeheader()
            writer.writerows(rows)

    def _save_csv_stream(self, path: str) -> None:
        """Save DataFrame as CSV file in streaming mode."""
        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        header = self._options.get("header", True)
        delimiter = self._options.get("delimiter", ",")

        chunk_iter = self._df.collect(stream=True)
        first_chunk: List[Dict[str, object]] = []
        try:
            first_chunk = next(chunk_iter)
        except StopIteration:
            pass

        # Determine headers
        if first_chunk:
            headers = list(first_chunk[0].keys())
        elif self._schema:
            headers = [col.name for col in self._schema]
        else:
            headers = []

        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=cast(str, delimiter))
            if header:
                writer.writeheader()
            if first_chunk:
                writer.writerows(first_chunk)
            for chunk in chunk_iter:
                if chunk:
                    writer.writerows(chunk)

    def _save_json(self, path: str) -> None:
        """Save DataFrame as JSON file (array of objects)."""
        self._ensure_file_layout_supported()
        indent_val = self._options.get("indent")

        if self._partition_by:
            rows = self._df.collect()
            self._save_partitioned(path, "json", rows, None)
            return

        use_stream = self._should_stream_output() and indent_val in (None, 0)

        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if use_stream:
            chunk_iter = self._df.collect(stream=True)
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write("[")
                first = True
                for chunk in chunk_iter:
                    for row in chunk:
                        if not first:
                            f.write(",\n")
                        else:
                            first = False
                        f.write(json.dumps(row, ensure_ascii=False))
                f.write("]")
            return

        rows = self._df.collect()
        with open(path_obj, "w", encoding="utf-8") as f:
            indent: Optional[Union[int, str]] = cast(Optional[Union[int, str]], indent_val)
            json.dump(rows, f, indent=indent, ensure_ascii=False)

    def _save_jsonl(self, path: str) -> None:
        """Save DataFrame as JSONL file (one JSON object per line)."""
        self._ensure_file_layout_supported()
        if self._should_stream_output():
            self._save_jsonl_stream(path)
            return

        rows = self._df.collect()

        # Handle partitioning
        if self._partition_by:
            self._save_partitioned(path, "jsonl", rows, None)
            return

        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_jsonl_stream(self, path: str) -> None:
        """Save DataFrame as JSONL file in streaming mode."""
        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            chunk_iter = self._df.collect(stream=True)
            for chunk in chunk_iter:
                for row in chunk:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_text(self, path: str) -> None:
        """Save DataFrame as text file (one value per line)."""
        self._ensure_file_layout_supported()
        column = cast(str, self._options.get("column", "value"))
        line_sep = cast(str, self._options.get("lineSep", "\n"))
        encoding = cast(str, self._options.get("encoding", "utf-8"))

        if self._partition_by:
            raise NotImplementedError("partitionBy()+text() is not supported yet.")

        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if self._should_stream_output():
            chunk_iter = self._df.collect(stream=True)
            with open(path_obj, "w", encoding=encoding) as f:
                for chunk in chunk_iter:
                    for row in chunk:
                        if column not in row:
                            raise ValueError(
                                f"Column '{column}' not found in row while writing text output"
                            )
                        f.write(f"{row[column]}{line_sep}")
            return

        rows = self._df.collect()
        with open(path_obj, "w", encoding=encoding) as f:
            for row in rows:
                if column not in row:
                    raise ValueError(
                        f"Column '{column}' not found in row while writing text output"
                    )
                f.write(f"{row[column]}{line_sep}")

    def _save_parquet(self, path: str) -> None:
        """Save DataFrame as Parquet file."""
        self._ensure_file_layout_supported()
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pandas. Install with: pip install pandas"
            ) from exc

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pyarrow. Install with: pip install pyarrow"
            ) from exc

        pa_mod = cast(Any, pa)
        pq_mod = cast(Any, pq)

        if self._partition_by:
            rows = self._df.collect()
            self._save_partitioned(path, "parquet", rows, None)
            return

        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        compression = cast(str, self._options.get("compression", "snappy"))
        if self._should_stream_output():
            chunk_iter = self._df.collect(stream=True)
            try:
                first_chunk = next(chunk_iter)
            except StopIteration:
                return

            table = pa_mod.Table.from_pandas(pd.DataFrame(first_chunk))
            with pq_mod.ParquetWriter(
                str(path_obj), table.schema, compression=compression
            ) as writer:
                writer.write_table(table)
                for chunk in chunk_iter:
                    if not chunk:
                        continue
                    writer.write_table(pa_mod.Table.from_pandas(pd.DataFrame(chunk)))
            return

        rows = self._df.collect()
        if not rows:
            return
        table = pa_mod.Table.from_pandas(pd.DataFrame(rows))
        pq_mod.write_table(table, str(path_obj), compression=compression)

    def _save_partitioned(
        self,
        base_path: str,
        format: str,
        rows: List[dict[str, object]],
        headers: Optional[List[str]],
    ) -> None:
        """Save data partitioned by specified columns."""
        self._ensure_file_layout_supported()
        if not rows:
            return

        # Group rows by partition columns
        partitions: Dict[tuple, List[dict[str, object]]] = {}
        partition_cols = self._partition_by or []
        for row in rows:
            partition_key = tuple(str(row.get(col, "")) for col in partition_cols)
            if partition_key not in partitions:
                partitions[partition_key] = []
            partitions[partition_key].append(row)

        # Write each partition to a subdirectory
        base_path_obj = Path(base_path)
        partition_root = base_path_obj.parent / base_path_obj.stem
        if not self._prepare_file_target(partition_root):
            return
        partition_root.parent.mkdir(parents=True, exist_ok=True)

        for partition_key, partition_rows in partitions.items():
            # Create partition directory path
            partition_parts = [f"{col}={val}" for col, val in zip(partition_cols, partition_key)]
            partition_dir = base_path_obj.parent / base_path_obj.stem
            for part in partition_parts:
                partition_dir = partition_dir / part
            partition_dir.mkdir(parents=True, exist_ok=True)

            # Determine filename
            if format == "csv":
                filename = partition_dir / "data.csv"
                self._write_csv_file(filename, partition_rows, headers)
            elif format == "json":
                filename = partition_dir / "data.json"
                self._write_json_file(filename, partition_rows)
            elif format == "jsonl":
                filename = partition_dir / "data.jsonl"
                self._write_jsonl_file(filename, partition_rows)
            elif format == "parquet":
                filename = partition_dir / "data.parquet"
                self._write_parquet_file(filename, partition_rows)

    def _write_csv_file(
        self, path: Path, rows: List[dict[str, object]], headers: Optional[List[str]]
    ) -> None:
        """Helper to write CSV file."""
        if not rows and headers:
            headers_to_use = headers
        elif rows:
            headers_to_use = list(rows[0].keys())
        else:
            headers_to_use = []

        header = cast(bool, self._options.get("header", True))
        delimiter = cast(str, self._options.get("delimiter", ","))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers_to_use, delimiter=delimiter)
            if header:
                writer.writeheader()
            writer.writerows(rows)

    def _write_json_file(self, path: Path, rows: List[dict[str, object]]) -> None:
        """Helper to write JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            indent_val = self._options.get("indent", None)
            indent: Optional[Union[int, str]] = cast(Optional[Union[int, str]], indent_val)
            json.dump(rows, f, indent=indent, ensure_ascii=False)

    def _write_jsonl_file(self, path: Path, rows: List[dict[str, object]]) -> None:
        """Helper to write JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_parquet_file(self, path: Path, rows: List[dict[str, object]]) -> None:
        """Helper to write Parquet file."""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame(rows)
        compression = cast(str, self._options.get("compression", "snappy"))
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path), compression=compression)
