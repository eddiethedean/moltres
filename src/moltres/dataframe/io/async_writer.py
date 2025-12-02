"""Async :class:`DataFrame` write operations."""

from __future__ import annotations

import json
import logging
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
    Union,
    cast,
)

logger = logging.getLogger(__name__)

try:
    import aiofiles  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "Async writing requires aiofiles. Install with: pip install moltres[async]"
    ) from exc

from ...expressions.column import Column, LiteralValue  # noqa: E402
from ...table.async_mutations import delete_rows_async, insert_rows_async, update_rows_async  # noqa: E402
from ...table.schema import ColumnDef  # noqa: E402
from ...utils.exceptions import CompilationError, ExecutionError, ValidationError  # noqa: E402
from ..core.async_dataframe import AsyncDataFrame  # noqa: E402

if TYPE_CHECKING:
    from ...table.async_table import AsyncDatabase
else:
    AsyncDatabase = Any


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

    def mode(self, mode: str) -> AsyncDataFrameWriter:
        """Set the write mode (append, overwrite, ignore, error_if_exists)."""
        from ..helpers.writer_helpers import build_mode_setter

        return build_mode_setter(self, mode)

    def option(self, key: str, value: object) -> AsyncDataFrameWriter:
        """Set a write option (e.g., header=True for CSV, compression='gzip' for Parquet)."""
        from ..helpers.writer_helpers import build_option_setter

        return build_option_setter(self, key, value)

    def options(self, *args: Mapping[str, object], **kwargs: object) -> AsyncDataFrameWriter:
        """Set multiple write options at once."""
        from ..helpers.writer_helpers import build_options_setter

        return build_options_setter(self, *args, **kwargs)

    def format(self, format_name: str) -> AsyncDataFrameWriter:
        """Specify the output format for save()."""
        from ..helpers.writer_helpers import build_format_setter

        return build_format_setter(self, format_name)

    def stream(self, enabled: bool = True) -> AsyncDataFrameWriter:
        """Enable or disable streaming mode (chunked writing for large DataFrames)."""
        from ..helpers.writer_helpers import build_stream_setter

        return build_stream_setter(self, enabled)

    def partitionBy(self, *columns: str) -> AsyncDataFrameWriter:
        """Partition data by the given columns when writing to files."""
        from ..helpers.writer_helpers import build_partition_by_setter

        return build_partition_by_setter(self, *columns)

    partition_by = partitionBy

    def schema(self, schema: Sequence[ColumnDef]) -> AsyncDataFrameWriter:
        """Set an explicit schema for the target table."""
        from ..helpers.writer_helpers import build_schema_setter

        return build_schema_setter(self, schema)

    def primaryKey(self, *columns: str) -> AsyncDataFrameWriter:
        """Specify primary key columns for the target table.

        Args:
            *columns: :class:`Column` names to use as primary key

        Returns:
            Self for method chaining
        """
        from ..helpers.writer_helpers import build_primary_key_setter

        return build_primary_key_setter(self, *columns)

    primary_key = primaryKey

    def bucketBy(self, num_buckets: int, *columns: str) -> AsyncDataFrameWriter:
        """PySpark-compatible bucketing hook (metadata only)."""
        from ..helpers.writer_helpers import build_bucket_by_setter

        return build_bucket_by_setter(self, num_buckets, *columns)

    bucket_by = bucketBy

    def sortBy(self, *columns: str) -> AsyncDataFrameWriter:
        """PySpark-compatible sortBy hook (metadata only)."""
        from ..helpers.writer_helpers import build_sort_by_setter

        return build_sort_by_setter(self, *columns)

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
                "bucketBy/sortBy are not yet supported when writing to tables. "
                "Alternative: Use ORDER BY in your query before writing, or sort data in memory before insertion. "
                "See https://github.com/eddiethedean/moltres/issues for feature requests."
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
        set: Mapping[str, Union[Column, LiteralValue]],
    ) -> None:
        """Update rows in a table matching the WHERE condition.

        Executes immediately (eager execution like PySpark writes).

        Args:
            table_name: Name of the table to update
            where: :class:`Column` expression for the WHERE clause
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
            where: :class:`Column` expression for the WHERE clause

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
            raise NotImplementedError(
                "ORC write support is not yet available for async writers. "
                "Alternative: Use parquet format instead: await df.write.parquet('path/to/file.parquet'). "
                "Parquet provides similar columnar storage benefits. "
                "See https://github.com/eddiethedean/moltres/issues to request ORC support."
            )
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
            "Async ORC output is not supported. "
            "Alternative: Use parquet format (await df.write.parquet()) which provides similar columnar storage. "
            "To contribute ORC support, see https://github.com/eddiethedean/moltres/blob/main/CONTRIBUTING.md"
        )

    async def parquet(self, path: str) -> None:
        """Save AsyncDataFrame as Parquet file."""
        await self._save_parquet(path)

    def _can_use_insert_select(self) -> bool:
        """Check if we can use INSERT INTO ... SELECT optimization.

        Returns:
            True if optimization is possible, False otherwise
        """
        from ..helpers.writer_helpers import can_use_insert_select

        # Check if plan is compilable
        plan_compilable = False
        if self._df.database is not None:
            try:
                from ...sql.compiler import compile_plan

                compile_plan(self._df.plan, dialect=self._df.database.dialect)
                plan_compilable = True
            except (CompilationError, ValidationError) as e:
                logger.debug("Plan compilation failed, cannot use optimization: %s", e)
            except Exception as e:
                logger.debug("Unexpected error during plan compilation check: %s", e)

        return can_use_insert_select(
            has_database=self._df.database is not None,
            stream_override=self._stream_override,
            mode=self._mode,
            plan_compilable=plan_compilable,
        )

    async def _infer_schema_from_plan(self) -> Optional[Sequence[ColumnDef]]:
        """Infer schema from AsyncDataFrame plan without materializing data.

        Returns:
            Inferred schema or None if inference is not possible
        """
        if self._df.database is None:
            return None

        try:
            from ...sql.compiler import compile_plan

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
                from ..helpers.writer_helpers import infer_schema_from_sample_row

                return infer_schema_from_sample_row(sample_row, column_names)
            else:
                # No rows returned, but we can still infer schema from the SELECT statement
                from ..helpers.writer_helpers import infer_schema_from_select_stmt

                return infer_schema_from_select_stmt(select_stmt)

        except (CompilationError, ExecutionError, ValidationError) as e:
            # If inference fails, return None to fall back to materialization
            logger.debug("Schema inference from plan failed: %s", e)
            return None
        except (AttributeError, KeyError, TypeError, ValueError) as e:
            # Common errors when accessing DataFrame/result attributes
            logger.debug("Schema inference failed due to attribute/type error: %s", e)
            return None
        except Exception as e:
            # Catch any other unexpected errors
            logger.warning(
                "Unexpected error during schema inference from plan: %s", e, exc_info=True
            )
            return None

    def _infer_type_from_value(self, value: object) -> str:
        """Infer SQL type from a Python value."""
        from ..helpers.writer_helpers import infer_type_from_value

        return infer_type_from_value(value)

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
        from ..helpers.writer_helpers import ensure_file_layout_supported

        ensure_file_layout_supported(self._bucket_by, self._sort_by)

    def _prepare_file_target(self, path_obj: Path) -> bool:
        """Apply mode semantics (overwrite/ignore/error) for file outputs."""
        from ..helpers.writer_helpers import prepare_file_target

        return prepare_file_target(path_obj, self._mode)

    async def _execute_insert_select(self, schema: Sequence[ColumnDef]) -> None:
        """Execute write using INSERT INTO ... SELECT optimization."""
        if self._df.database is None:
            raise RuntimeError("Cannot write AsyncDataFrame without an attached AsyncDatabase")

        db = self._df.database
        table_name = self._table_name
        if table_name is None:
            raise ValueError("Table name must be specified via save_as_table()")

        # Apply primary key flags to schema if specified
        from ..helpers.writer_helpers import apply_primary_key_to_schema

        final_schema = apply_primary_key_to_schema(schema, self._primary_key)

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
            except (ExecutionError, ValidationError) as e:
                # Ignore errors if table doesn't exist (expected in some cases)
                logger.debug("Error dropping table (may not exist): %s", e)
            except Exception as e:
                # Log unexpected errors but continue
                logger.warning("Unexpected error dropping table: %s", e)
            await db.create_table(table_name, final_schema, if_not_exists=False).collect()
        elif not table_exists:
            # Create table if it doesn't exist
            await db.create_table(table_name, final_schema, if_not_exists=True).collect()

        # Compile DataFrame plan to SELECT statement
        from ...sql.compiler import compile_plan

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
        async with db.connection_manager.connect() as conn:
            await conn.execute(insert_stmt)
            await conn.commit()

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
                "bucketBy/sortBy are not yet supported when writing to tables. "
                "Alternative: Use ORDER BY in your query before writing, or sort data in memory before insertion. "
                "See https://github.com/eddiethedean/moltres/issues for feature requests."
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
            except (ExecutionError, ValidationError) as e:
                # Ignore errors if table doesn't exist (expected in some cases)
                logger.debug("Error dropping table (may not exist): %s", e)
            except Exception as e:
                # Log unexpected errors but continue
                logger.warning("Unexpected error dropping table: %s", e)

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

    async def _table_exists(self, db: AsyncDatabase, table_name: str) -> bool:
        """Check if a table exists in the database."""
        # Query database metadata to check if table exists
        dialect_name = db._dialect_name
        if dialect_name == "sqlite":
            # Query sqlite_master for SQLite - use string formatting for SQLite's ? placeholders
            try:
                # For SQLite, we can use string formatting since table_name is validated
                from ...sql.builders import quote_identifier

                quoted_name = quote_identifier(table_name, '"')
                sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name={quoted_name}"
                result = await db.executor.fetch(sql)
                return result.rows is not None and len(result.rows) > 0
            except (ExecutionError, ValidationError) as e:
                logger.debug("Error checking table existence (SQLite): %s", e)
                return False
            except Exception as e:
                logger.debug("Unexpected error checking table existence (SQLite): %s", e)
                return False
        else:
            # For other databases, query information_schema
            try:
                # PostgreSQL, MySQL, etc. use information_schema
                if dialect_name in {"postgresql", "mysql", "mariadb"}:
                    if dialect_name == "postgresql":
                        # For PostgreSQL, use pg_tables which respects search_path
                        # This handles custom schemas set via search_path in the DSN
                        sql = "SELECT tablename FROM pg_tables WHERE schemaname = current_schema() AND tablename = :table_name"
                        params = {"table_name": table_name}
                    else:
                        # For MySQL/MariaDB, query information_schema
                        sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = DATABASE() AND table_name = :table_name"
                        params = {"table_name": table_name}
                    result = await db.executor.fetch(sql, params=params)
                    return result.rows is not None and len(result.rows) > 0
                else:
                    # Fallback: try to access the table
                    try:
                        await db.table(table_name)
                        return True
                    except (ValidationError, ExecutionError) as e:
                        logger.debug("Error accessing table (fallback check): %s", e)
                        return False
                    except Exception as e:
                        logger.debug("Unexpected error accessing table (fallback check): %s", e)
                        return False
            except (ExecutionError, ValidationError) as e:
                logger.debug("Error checking table existence: %s", e)
                return False
            except Exception as e:
                logger.debug("Unexpected error checking table existence: %s", e)
                return False

    def _infer_or_get_schema(
        self, rows: List[Dict[str, object]], *, force_nullable: bool = False
    ) -> Sequence[ColumnDef]:
        """Infer schema from rows or use explicit schema."""
        from ..helpers.writer_helpers import apply_primary_key_to_schema

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
        return apply_primary_key_to_schema(schema, self._primary_key)

    async def _save_csv(self, path: str) -> None:
        """Save AsyncDataFrame as CSV file."""
        self._ensure_file_layout_supported()
        if self._partition_by:
            raise NotImplementedError(
                "partitionBy()+csv() is not supported for async writes. "
                "Alternative: Write without partitioning, or use parquet format which supports partitioning. "
                "See https://github.com/eddiethedean/moltres/issues for feature requests."
            )
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
            raise NotImplementedError(
                "partitionBy()+json() is not supported for async writes. "
                "Alternative: Write without partitioning, or use parquet format which supports partitioning. "
                "See https://github.com/eddiethedean/moltres/issues for feature requests."
            )

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
            raise NotImplementedError(
                "partitionBy()+jsonl() is not supported for async writes. "
                "Alternative: Write without partitioning, or use parquet format which supports partitioning. "
                "See https://github.com/eddiethedean/moltres/issues for feature requests."
            )

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
            raise NotImplementedError(
                "partitionBy()+text() is not supported for async writes. "
                "Alternative: Write without partitioning, or use parquet format which supports partitioning. "
                "See https://github.com/eddiethedean/moltres/issues for feature requests."
            )
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
            raise NotImplementedError(
                "partitionBy()+parquet() is not supported for async writes. "
                "Alternative: Write without partitioning, or use synchronous writes (df.write.parquet()) which support partitioning. "
                "See https://github.com/eddiethedean/moltres/issues for feature requests."
            )
        from ...utils.optional_deps import get_pandas, get_pyarrow, get_pyarrow_parquet

        pd = get_pandas(required=True)
        pa = get_pyarrow(required=True)
        pq = get_pyarrow_parquet(required=True)

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
