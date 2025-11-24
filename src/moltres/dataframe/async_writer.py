"""Async DataFrame write operations."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    cast,
)

try:
    import aiofiles  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "Async writing requires aiofiles. Install with: pip install moltres[async]"
    ) from exc

from ..expressions.column import Column
from ..table.async_mutations import delete_rows_async, insert_rows_async, update_rows_async
from ..table.schema import ColumnDef
from .async_dataframe import AsyncDataFrame

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase


class AsyncDataFrameWriter:
    """Builder for writing AsyncDataFrames to tables and files."""

    def __init__(self, df: AsyncDataFrame):
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

    def mode(self, mode: str) -> "AsyncDataFrameWriter":
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

    def option(self, key: str, value: object) -> "AsyncDataFrameWriter":
        """Set a write option (e.g., header=True for CSV, compression='gzip' for Parquet)."""
        self._options[key] = value
        return self

    def options(self, *args: Mapping[str, object], **kwargs: object) -> "AsyncDataFrameWriter":
        """Set multiple write options at once."""
        if len(args) > 1:
            raise TypeError("options() accepts at most one positional mapping argument")
        if args:
            for key, value in args[0].items():
                self._options[str(key)] = value
        for key, value in kwargs.items():
            self._options[key] = value
        return self

    def format(self, format_name: str) -> "AsyncDataFrameWriter":
        """Specify the output format for save()."""
        self._format = format_name.strip().lower()
        return self

    def stream(self, enabled: bool = True) -> "AsyncDataFrameWriter":
        """Enable or disable streaming mode (chunked writing for large DataFrames)."""
        self._stream_override = bool(enabled)
        return self

    def partitionBy(self, *columns: str) -> "AsyncDataFrameWriter":
        """Partition data by the given columns when writing to files."""
        self._partition_by = columns if columns else None
        return self

    partition_by = partitionBy

    def schema(self, schema: Sequence[ColumnDef]) -> "AsyncDataFrameWriter":
        """Set an explicit schema for the target table."""
        self._schema = schema
        return self

    def primaryKey(self, *columns: str) -> "AsyncDataFrameWriter":
        """Specify primary key columns for the target table.

        Args:
            *columns: Column names to use as primary key

        Returns:
            Self for method chaining
        """
        self._primary_key = columns if columns else None
        return self

    primary_key = primaryKey

    def bucketBy(self, num_buckets: int, *columns: str) -> "AsyncDataFrameWriter":
        """PySpark-compatible bucketing hook (metadata only)."""
        if num_buckets <= 0:
            raise ValueError("num_buckets must be a positive integer")
        if not columns:
            raise ValueError("bucketBy() requires at least one column name")
        self._bucket_by = (num_buckets, tuple(columns))
        return self

    bucket_by = bucketBy

    def sortBy(self, *columns: str) -> "AsyncDataFrameWriter":
        """PySpark-compatible sortBy hook (metadata only)."""
        if not columns:
            raise ValueError("sortBy() requires at least one column name")
        self._sort_by = tuple(columns)
        return self

    sort_by = sortBy

    async def save_as_table(self, name: str, primary_key: Optional[Sequence[str]] = None) -> None:
        """Write the AsyncDataFrame to a table with the given name.

        Args:
            name: Name of the target table
            primary_key: Optional sequence of column names to use as primary key.
                        If provided, overrides any primary key set via .primaryKey()
        """
        if self._df.database is None:
            raise RuntimeError("Cannot write AsyncDataFrame without an attached AsyncDatabase")

        # Use parameter if provided, otherwise use field
        if primary_key is not None:
            self._primary_key = primary_key
        self._table_name = name
        await self._execute_write()

    saveAsTable = save_as_table  # PySpark-style alias

    async def insertInto(self, table_name: str) -> None:
        """Insert AsyncDataFrame into an existing table (table must already exist)."""
        if self._df.database is None:
            raise RuntimeError("Cannot write AsyncDataFrame without an attached AsyncDatabase")

        db = self._df.database
        if not await self._table_exists(db, table_name):
            raise ValueError(
                f"Table '{table_name}' does not exist. Use save_as_table() to create it."
            )

        if self._bucket_by or self._sort_by:
            raise NotImplementedError(
                "bucketBy/sortBy are not yet supported when writing to tables."
            )

        # Get active transaction if available
        transaction = db.connection_manager.active_transaction
        table_handle = await db.table(table_name)

        use_stream = self._should_stream_materialization()
        rows, chunk_iter = await self._collect_rows(use_stream)
        if use_stream:
            if rows:
                await insert_rows_async(table_handle, rows, transaction=transaction)
            if chunk_iter is not None:
                async for chunk in chunk_iter:
                    if chunk:
                        await insert_rows_async(table_handle, chunk, transaction=transaction)
        elif rows:
            await insert_rows_async(table_handle, rows, transaction=transaction)

    insert_into = insertInto

    async def update(
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
            >>> df = await db.table("users").select()
            >>> await df.write.update("users", where=col("id") == 1, set={"name": "Bob"})
        """
        if self._df.database is None:
            raise RuntimeError("Cannot update table without an attached AsyncDatabase")

        db = self._df.database
        if not await self._table_exists(db, table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        # Get active transaction if available
        transaction = db.connection_manager.active_transaction

        # Use the mutation helper function
        table_handle = await db.table(table_name)
        await update_rows_async(table_handle, where=where, values=set, transaction=transaction)

    async def delete(
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
            >>> df = await db.table("users").select()
            >>> await df.write.delete("users", where=col("id") == 1)
        """
        if self._df.database is None:
            raise RuntimeError("Cannot delete from table without an attached AsyncDatabase")

        db = self._df.database
        if not await self._table_exists(db, table_name):
            raise ValueError(f"Table '{table_name}' does not exist")

        # Get active transaction if available
        transaction = db.connection_manager.active_transaction

        # Use the mutation helper function
        table_handle = await db.table(table_name)
        await delete_rows_async(table_handle, where=where, transaction=transaction)

    async def save(self, path: str, format: Optional[str] = None) -> None:
        """Save AsyncDataFrame to a file or directory in the specified format."""
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
            await self._save_csv(path)
        elif format_lower == "json":
            await self._save_json(path)
        elif format_lower == "jsonl":
            await self._save_jsonl(path)
        elif format_lower == "text":
            await self._save_text(path)
        elif format_lower == "parquet":
            await self._save_parquet(path)
        elif format_lower == "orc":
            raise NotImplementedError("ORC write support is not yet available for async writers.")
        else:
            raise ValueError(
                f"Unsupported format '{format_to_use}'. Supported: csv, json, jsonl, text, parquet"
            )

    async def csv(self, path: str) -> None:
        """Save AsyncDataFrame as CSV file."""
        await self._save_csv(path)

    async def json(self, path: str) -> None:
        """Save AsyncDataFrame as JSON file (array of objects)."""
        await self._save_json(path)

    async def jsonl(self, path: str) -> None:
        """Save AsyncDataFrame as JSONL file (one JSON object per line)."""
        await self._save_jsonl(path)

    async def text(self, path: str) -> None:
        """Save AsyncDataFrame as text file (expects a single 'value' column)."""
        await self._save_text(path)

    async def orc(self, path: str) -> None:
        """PySpark-style ORC helper (not yet implemented)."""
        raise NotImplementedError(
            "Async ORC output is not supported. Use parquet or contribute ORC support."
        )

    async def parquet(self, path: str) -> None:
        """Save AsyncDataFrame as Parquet file."""
        await self._save_parquet(path)

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

    async def _infer_schema_from_plan(self) -> Optional[Sequence[ColumnDef]]:
        """Infer schema from AsyncDataFrame plan without materializing data.

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
            sample_result = await self._df.database.executor.fetch(sample_stmt)

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

    def _should_stream_output(self) -> bool:
        """Use chunked output by default unless user explicitly disables it."""
        if self._stream_override is not None:
            return self._stream_override
        return True

    def _should_stream_materialization(self) -> bool:
        """Default to streaming inserts unless user explicitly disabled it."""
        if self._stream_override is not None:
            return self._stream_override
        return True

    async def _collect_rows(
        self, use_stream: bool
    ) -> tuple[List[Dict[str, object]], Optional[AsyncIterator[List[Dict[str, object]]]]]:
        """Collect rows, optionally streaming in chunks."""
        if use_stream:
            chunk_iter = await self._df.collect(stream=True)
            try:
                first_chunk = await chunk_iter.__anext__()
            except StopAsyncIteration:
                return [], chunk_iter
            return first_chunk, chunk_iter
        rows = await self._df.collect()
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

    async def _execute_insert_select(self, schema: Sequence[ColumnDef]) -> None:
        """Execute write using INSERT INTO ... SELECT optimization."""
        if self._df.database is None:
            raise RuntimeError("Cannot write AsyncDataFrame without an attached AsyncDatabase")

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
        table_exists = await self._table_exists(db, table_name)

        # Handle overwrite/ignore/error modes
        if self._mode == "error_if_exists" and table_exists:
            raise ValueError(f"Table '{table_name}' already exists and mode is 'error_if_exists'")
        if self._mode == "ignore" and table_exists:
            return
        if self._mode == "overwrite":
            try:
                await db.drop_table(table_name, if_exists=True).collect()
            except Exception:
                pass  # Ignore errors if table doesn't exist
            await db.create_table(table_name, final_schema, if_not_exists=False).collect()
        elif not table_exists:
            # Create table if it doesn't exist
            await db.create_table(table_name, final_schema, if_not_exists=True).collect()

        # Compile DataFrame plan to SELECT statement
        from ..sql.compiler import compile_plan

        select_stmt = compile_plan(self._df.plan, dialect=db.dialect)

        # Get column names from schema
        column_names = [col.name for col in final_schema]

        # Compile INSERT INTO ... SELECT statement
        from ..sql.ddl import compile_insert_select

        insert_sql = compile_insert_select(
            target_table=table_name,
            select_stmt=select_stmt,
            dialect=db.dialect,
            columns=column_names,
        )

        # Execute the compiled SQL directly
        await db.executor.execute(insert_sql)

    async def _execute_write(self) -> None:
        """Execute the write operation based on mode and table existence."""
        if self._df.database is None:
            raise RuntimeError("Cannot write AsyncDataFrame without an attached AsyncDatabase")

        db = self._df.database
        table_name = self._table_name
        if table_name is None:
            raise ValueError("Table name must be specified via save_as_table()")

        # Check if table exists
        table_exists = await self._table_exists(db, table_name)

        if self._bucket_by or self._sort_by:
            raise NotImplementedError(
                "bucketBy/sortBy are not yet supported when writing to tables."
            )

        if self._mode == "error_if_exists" and table_exists:
            raise ValueError(f"Table '{table_name}' already exists and mode is 'error_if_exists'")
        if self._mode == "ignore" and table_exists:
            return

        # Check if we can use INSERT INTO ... SELECT optimization
        if self._can_use_insert_select():
            # Try to infer schema from plan
            schema = self._schema or await self._infer_schema_from_plan()

            if schema is None:
                # Can't infer schema, fall back to materialization
                # Continue with existing materialization code below
                pass
            else:
                # Use optimized path
                await self._execute_insert_select(schema)
                return

        # Fall back to existing materialization approach
        use_stream = self._should_stream_materialization()
        rows, chunk_iter = await self._collect_rows(use_stream)

        # Infer or get schema
        try:
            schema = self._infer_or_get_schema(rows, force_nullable=use_stream)
        except ValueError:
            # Empty AsyncDataFrame without explicit schema
            if self._schema is None:
                raise ValueError(
                    "Cannot infer schema from empty AsyncDataFrame. "
                    "Provide explicit schema via .schema([ColumnDef(...), ...])"
                )
            schema = self._schema

        # Handle overwrite mode
        if self._mode == "overwrite":
            # Drop and recreate table
            try:
                await db.drop_table(table_name, if_exists=True).collect()
            except Exception:
                pass  # Ignore errors if table doesn't exist

        # Create table if needed
        if not table_exists or self._mode == "overwrite":
            await db.create_table(table_name, schema, if_not_exists=False).collect()

        # Insert data
        table_handle = await db.table(table_name)
        transaction = db.connection_manager.active_transaction
        if use_stream and chunk_iter:
            # Stream inserts
            if rows:  # Insert first chunk
                await insert_rows_async(table_handle, rows, transaction=transaction)
            async for chunk in chunk_iter:
                if chunk:
                    await insert_rows_async(table_handle, chunk, transaction=transaction)
        else:
            if rows:
                await insert_rows_async(table_handle, rows, transaction=transaction)

    async def _table_exists(self, db: "AsyncDatabase", table_name: str) -> bool:
        """Check if a table exists in the database."""
        # Try to query the table - if it doesn't exist, we'll get an error
        try:
            await db.table(table_name)
            # Try to compile a simple query to verify table exists
            from ..logical import operators

            plan = operators.scan(table_name)
            db.compile_plan(plan)
            # Just compile, don't execute - if table doesn't exist, compilation might fail
            # For now, we'll try a simple approach
            return True  # Simplified - in production, query information_schema
        except Exception:
            return False

    def _infer_or_get_schema(
        self, rows: List[Dict[str, object]], *, force_nullable: bool = False
    ) -> Sequence[ColumnDef]:
        """Infer schema from rows or use explicit schema."""
        if self._schema:
            schema = list(self._schema)
        else:
            if not rows:
                raise ValueError("Cannot infer schema from empty data")
            from .readers.schema_inference import infer_schema_from_rows

            schema = list(infer_schema_from_rows(rows))

            if force_nullable:
                schema = [
                    ColumnDef(
                        name=col.name,
                        type_name=col.type_name,
                        nullable=True,
                        default=col.default,
                        primary_key=col.primary_key,
                        precision=col.precision,
                        scale=col.scale,
                    )
                    for col in schema
                ]

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

    async def _save_csv(self, path: str) -> None:
        """Save AsyncDataFrame as CSV file."""
        self._ensure_file_layout_supported()
        if self._partition_by:
            raise NotImplementedError("partitionBy()+csv() is not supported for async writes.")
        header = cast(bool, self._options.get("header", True))
        delimiter = cast(str, self._options.get("delimiter", ","))

        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if self._should_stream_output():
            chunk_iter = await self._df.collect(stream=True)
            first_chunk = True
            async with aiofiles.open(path, "w", encoding="utf-8", newline="") as f:
                async for chunk in chunk_iter:
                    if not chunk:
                        continue
                    if first_chunk and header:
                        fieldnames = list(chunk[0].keys())
                        await f.write(delimiter.join(fieldnames) + "\n")
                        first_chunk = False
                    for row in chunk:
                        values = [str(row.get(col, "")) for col in chunk[0].keys()]
                        await f.write(delimiter.join(values) + "\n")
            return

        rows = await self._df.collect()
        if not rows:
            return
        async with aiofiles.open(path, "w", encoding="utf-8", newline="") as f:
            if header:
                fieldnames = list(rows[0].keys())
                await f.write(delimiter.join(fieldnames) + "\n")
            for row in rows:
                values = [str(row.get(col, "")) for col in rows[0].keys()]
                await f.write(delimiter.join(values) + "\n")

    async def _save_json(self, path: str) -> None:
        """Save AsyncDataFrame as JSON file (array of objects)."""
        self._ensure_file_layout_supported()
        if self._partition_by:
            raise NotImplementedError("partitionBy()+json() is not supported for async writes.")

        indent = cast(Optional[int], self._options.get("indent"))
        use_stream = self._should_stream_output() and indent in (None, 0)
        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if use_stream:
            chunk_iter = await self._df.collect(stream=True)
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write("[")
                first = True
                async for chunk in chunk_iter:
                    for row in chunk:
                        if not first:
                            await f.write(",\n")
                        else:
                            first = False
                        await f.write(json.dumps(row, default=str))
                await f.write("]")
            return

        rows = await self._df.collect()
        content = json.dumps(rows, indent=indent, default=str)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)

    async def _save_jsonl(self, path: str) -> None:
        """Save AsyncDataFrame as JSONL file (one JSON object per line)."""
        self._ensure_file_layout_supported()
        if self._partition_by:
            raise NotImplementedError("partitionBy()+jsonl() is not supported for async writes.")

        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if self._should_stream_output():
            chunk_iter = await self._df.collect(stream=True)
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                async for chunk in chunk_iter:
                    for row in chunk:
                        await f.write(json.dumps(row, default=str) + "\n")
        else:
            rows = await self._df.collect()
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    await f.write(json.dumps(row, default=str) + "\n")

    async def _save_text(self, path: str) -> None:
        """Save AsyncDataFrame as text file (one value per line)."""
        self._ensure_file_layout_supported()
        if self._partition_by:
            raise NotImplementedError("partitionBy()+text() is not supported for async writes.")
        column = cast(str, self._options.get("column", "value"))
        line_sep = cast(str, self._options.get("lineSep", "\n"))
        encoding = cast(str, self._options.get("encoding", "utf-8"))

        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        if self._should_stream_output():
            chunk_iter = await self._df.collect(stream=True)
            async with aiofiles.open(path, "w", encoding=encoding) as f:
                async for chunk in chunk_iter:
                    for row in chunk:
                        if column not in row:
                            raise ValueError(
                                f"Column '{column}' not found in row while writing text output"
                            )
                        await f.write(f"{row[column]}{line_sep}")
            return

        rows = await self._df.collect()
        async with aiofiles.open(path, "w", encoding=encoding) as f:
            for row in rows:
                if column not in row:
                    raise ValueError(
                        f"Column '{column}' not found in row while writing text output"
                    )
                await f.write(f"{row[column]}{line_sep}")

    async def _save_parquet(self, path: str) -> None:
        """Save AsyncDataFrame as Parquet file."""
        self._ensure_file_layout_supported()
        if self._partition_by:
            raise NotImplementedError("partitionBy()+parquet() is not supported for async writes.")
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

        path_obj = Path(path)
        if not self._prepare_file_target(path_obj):
            return
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        compression = cast(str, self._options.get("compression", "snappy"))
        if self._should_stream_output():
            chunk_iter = await self._df.collect(stream=True)
            try:
                first_chunk = await chunk_iter.__anext__()
            except StopAsyncIteration:
                return

            table = pa_mod.Table.from_pandas(pd.DataFrame(first_chunk))
            with pq_mod.ParquetWriter(
                str(path_obj), table.schema, compression=compression
            ) as writer:
                writer.write_table(table)
                async for chunk in chunk_iter:
                    if not chunk:
                        continue
                    writer.write_table(pa_mod.Table.from_pandas(pd.DataFrame(chunk)))
            return

        rows = await self._df.collect()
        if not rows:
            return
        table = pa_mod.Table.from_pandas(pd.DataFrame(rows))
        pq_mod.write_table(table, str(path_obj), compression=compression)
