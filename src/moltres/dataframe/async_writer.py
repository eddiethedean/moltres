"""Async DataFrame write operations."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Optional, cast

try:
    import aiofiles  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "Async writing requires aiofiles. Install with: pip install moltres[async]"
    ) from exc

from sqlalchemy.exc import SQLAlchemyError

from ..table.schema import ColumnDef
from ..utils.exceptions import ExecutionError
from .async_dataframe import AsyncDataFrame

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase

logger = logging.getLogger(__name__)


class AsyncDataFrameWriter:
    """Builder for writing AsyncDataFrames to tables and files."""

    def __init__(self, df: AsyncDataFrame):
        self._df = df
        self._mode: str = "append"
        self._table_name: str | None = None
        self._schema: Sequence[ColumnDef] | None = None
        self._options: dict[str, object] = {}
        self._partition_by: Sequence[str] | None = None
        self._stream: bool = False

    def mode(self, mode: str) -> AsyncDataFrameWriter:
        """Set the write mode: 'append', 'overwrite', or 'error_if_exists'."""
        if mode not in ("append", "overwrite", "error_if_exists"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: append, overwrite, error_if_exists"
            )
        self._mode = mode
        return self

    def option(self, key: str, value: object) -> AsyncDataFrameWriter:
        """Set a write option (e.g., header=True for CSV, compression='gzip' for Parquet)."""
        self._options[key] = value
        return self

    def stream(self, enabled: bool = True) -> AsyncDataFrameWriter:
        """Enable or disable streaming mode (chunked writing for large DataFrames)."""
        self._stream = enabled
        return self

    def partitionBy(self, *columns: str) -> AsyncDataFrameWriter:
        """Partition data by the given columns when writing to files."""
        self._partition_by = columns if columns else None
        return self

    partition_by = partitionBy

    def schema(self, schema: Sequence[ColumnDef]) -> AsyncDataFrameWriter:
        """Set an explicit schema for the target table."""
        self._schema = schema
        return self

    async def save_as_table(self, name: str) -> None:
        """Write the AsyncDataFrame to a table with the given name."""
        if self._df.database is None:
            raise RuntimeError("Cannot write AsyncDataFrame without an attached AsyncDatabase")

        self._table_name = name
        await self._execute_write()

    async def insertInto(self, table_name: str) -> None:
        """Insert AsyncDataFrame into an existing table (table must already exist)."""
        if self._df.database is None:
            raise RuntimeError("Cannot write AsyncDataFrame without an attached AsyncDatabase")

        db = self._df.database
        if not await self._table_exists(db, table_name):
            raise ValueError(
                f"Table '{table_name}' does not exist. Use save_as_table() to create it."
            )

        if self._stream:
            # Stream inserts in batches
            table = await db.table(table_name)
            chunk_iter = cast(
                "AsyncIterator[list[dict[str, object]]]", await self._df.collect(stream=True)
            )
            async for chunk in chunk_iter:
                if chunk:
                    await table.insert(chunk)
        else:
            rows = cast("list[dict[str, object]]", await self._df.collect())
            if rows:
                table = await db.table(table_name)
                await table.insert(rows)

    insert_into = insertInto

    async def save(self, path: str, format: str | None = None) -> None:
        """Save AsyncDataFrame to a file or directory in the specified format."""
        if format is None:
            # Infer format from file extension
            ext = Path(path).suffix.lower()
            format_map = {
                ".csv": "csv",
                ".json": "json",
                ".jsonl": "jsonl",
                ".parquet": "parquet",
            }
            format_str = format_map.get(ext, "csv")  # Default to csv for unknown extensions
        else:
            format_str = format.lower()
        if format == "csv":
            await self._save_csv(path)
        elif format == "json":
            await self._save_json(path)
        elif format == "jsonl":
            await self._save_jsonl(path)
        elif format == "parquet":
            await self._save_parquet(path)
        else:
            raise ValueError(f"Unsupported format '{format}'. Supported: csv, json, jsonl, parquet")

    async def csv(self, path: str) -> None:
        """Save AsyncDataFrame as CSV file."""
        await self._save_csv(path)

    async def json(self, path: str) -> None:
        """Save AsyncDataFrame as JSON file (array of objects)."""
        await self._save_json(path)

    async def jsonl(self, path: str) -> None:
        """Save AsyncDataFrame as JSONL file (one JSON object per line)."""
        await self._save_jsonl(path)

    async def parquet(self, path: str) -> None:
        """Save AsyncDataFrame as Parquet file."""
        await self._save_parquet(path)

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

        if self._mode == "error_if_exists" and table_exists:
            raise ValueError(f"Table '{table_name}' already exists and mode is 'error_if_exists'")

        # Collect data from AsyncDataFrame
        chunk_iter: AsyncIterator[list[dict[str, object]]] | None = None
        if self._stream:
            chunk_iter = cast(
                "AsyncIterator[list[dict[str, object]]]", await self._df.collect(stream=True)
            )
            try:
                first_chunk = await chunk_iter.__anext__()
                rows = first_chunk
            except StopAsyncIteration:
                rows = []
        else:
            rows = cast("list[dict[str, object]]", await self._df.collect())

        # Infer or get schema
        try:
            schema = self._infer_or_get_schema(rows)
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
                await db.drop_table(table_name, if_exists=True)
            except (ExecutionError, SQLAlchemyError) as exc:
                # Log but ignore errors when dropping with if_exists=True
                # This can happen if table doesn't exist or other transient errors
                logger.debug("Error dropping table '%s' (if_exists=True): %s", table_name, exc)

        # Create table if needed
        if not table_exists or self._mode == "overwrite":
            await db.create_table(table_name, schema, if_not_exists=False)

        # Insert data
        table = await db.table(table_name)
        if self._stream and chunk_iter:
            # Stream inserts
            if rows:  # Insert first chunk
                await table.insert(rows)
            async for chunk in chunk_iter:
                if chunk:
                    await table.insert(chunk)
        elif rows:
            await table.insert(rows)

    async def _table_exists(self, db: AsyncDatabase, table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            # Use a simple query that should work across dialects
            if db.dialect.name == "sqlite":
                sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=:name"
                result = await db.execute_sql(sql, params={"name": table_name})
                return len(result.rows) > 0
            if db.dialect.name == "postgresql":
                sql = "SELECT tablename FROM pg_tables WHERE tablename=:name"
                result = await db.execute_sql(sql, params={"name": table_name})
                return len(result.rows) > 0
            # Generic approach: try to select from the table with LIMIT 0
            quote = db.dialect.quote_char
            sql = f"SELECT * FROM {quote}{table_name}{quote} LIMIT 0"
            await db.execute_sql(sql)
            return True
        except (ExecutionError, SQLAlchemyError) as exc:
            logger.debug("Table existence check failed for '%s': %s", table_name, exc)
            return False

    def _infer_or_get_schema(self, rows: list[dict[str, object]]) -> Sequence[ColumnDef]:
        """Infer schema from rows or use explicit schema."""
        if self._schema:
            return self._schema
        if not rows:
            raise ValueError("Cannot infer schema from empty data")
        from .readers.schema_inference import infer_schema_from_rows

        return infer_schema_from_rows(rows)

    async def _save_csv(self, path: str) -> None:
        """Save AsyncDataFrame as CSV file."""
        header = cast("bool", self._options.get("header", True))
        delimiter = cast("str", self._options.get("delimiter", ","))

        # Collect data
        if self._stream:
            chunk_iter = cast(
                "AsyncIterator[list[dict[str, object]]]", await self._df.collect(stream=True)
            )
            first_chunk = True
            async with aiofiles.open(path, "w", encoding="utf-8", newline="") as f:
                async for chunk in chunk_iter:
                    if not chunk:
                        continue
                    if first_chunk and header:
                        # Write header
                        fieldnames = list(chunk[0].keys())
                        await f.write(delimiter.join(fieldnames) + "\n")
                        first_chunk = False
                    # Write rows
                    for row in chunk:
                        values = [str(row.get(col, "")) for col in chunk[0].keys()]
                        await f.write(delimiter.join(values) + "\n")
        else:
            rows = cast("list[dict[str, object]]", await self._df.collect())
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
        indent = cast("Optional[int]", self._options.get("indent"))
        if self._stream:
            chunk_iter = cast(
                "AsyncIterator[list[dict[str, object]]]", await self._df.collect(stream=True)
            )
            all_rows = []
            async for chunk in chunk_iter:
                all_rows.extend(chunk)
            rows = all_rows
        else:
            rows = cast("list[dict[str, object]]", await self._df.collect())

        content = json.dumps(rows, indent=indent, default=str)
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(content)

    async def _save_jsonl(self, path: str) -> None:
        """Save AsyncDataFrame as JSONL file (one JSON object per line)."""
        if self._stream:
            chunk_iter = cast(
                "AsyncIterator[list[dict[str, object]]]", await self._df.collect(stream=True)
            )
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                async for chunk in chunk_iter:
                    for row in chunk:
                        await f.write(json.dumps(row, default=str) + "\n")
        else:
            rows = cast("list[dict[str, object]]", await self._df.collect())
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    await f.write(json.dumps(row, default=str) + "\n")

    async def _save_parquet(self, path: str) -> None:
        """Save AsyncDataFrame as Parquet file."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pandas. Install with: pip install pandas"
            ) from exc

        try:
            import pyarrow as pa  # type: ignore[import]
            import pyarrow.parquet as pq  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pyarrow. Install with: pip install pyarrow"
            ) from exc

        # Collect data
        if self._stream:
            chunk_iter = cast(
                "AsyncIterator[list[dict[str, object]]]", await self._df.collect(stream=True)
            )
            all_rows = []
            async for chunk in chunk_iter:
                all_rows.extend(chunk)
            rows = all_rows
        else:
            rows = cast("list[dict[str, object]]", await self._df.collect())

        if not rows:
            return

        # Convert to pandas DataFrame and save
        import asyncio

        def _save_parquet_sync() -> None:
            df = pd.DataFrame(rows)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path)

        await asyncio.to_thread(_save_parquet_sync)
