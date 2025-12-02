"""Common helper functions for file I/O operations.

This module contains shared logic for routing file format reads to the
appropriate reader functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Literal, Optional, Sequence, Union, cast, overload

if TYPE_CHECKING:
    from ...io.records import AsyncRecords, Records
    from ...table.async_table import AsyncDatabase
    from ...table.schema import ColumnDef
    from ...table.table import Database


@overload
def route_file_read(
    format_name: str,
    path: str,
    database: "Database",
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
    *,
    async_mode: Literal[False] = False,
) -> "Records": ...


@overload
def route_file_read(
    format_name: str,
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
    *,
    async_mode: Literal[True],
) -> "AsyncRecords": ...


def route_file_read(
    format_name: str,
    path: str,
    database: Union["Database", "AsyncDatabase"],
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
    *,
    async_mode: bool = False,
) -> Union["Records", "AsyncRecords"]:
    """Route file read to the appropriate reader function based on format.

    Args:
        format_name: File format ("csv", "json", "jsonl", "parquet", "text")
        path: Path to the file
        database: :class:`Database` instance (sync or async)
        schema: Optional schema definition
        options: Read options dictionary
        column_name: Optional column name for text files
        async_mode: If True, use async readers; if False, use sync readers

    Returns:
        :class:`Records` or :class:`AsyncRecords` object (depending on async_mode)

    Raises:
        ValueError: If format is unsupported
        ImportError: If required dependencies are missing (e.g., pyarrow for parquet)
    """
    if async_mode:
        return _route_async_file_read(
            format_name, path, cast("AsyncDatabase", database), schema, options, column_name
        )  # type: ignore[return-value]
    else:
        return _route_sync_file_read(
            format_name, path, cast("Database", database), schema, options, column_name
        )


def _route_sync_file_read(
    format_name: str,
    path: str,
    database: "Database",
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
) -> "Records":
    """Route to sync file reader."""
    from ..io.readers import (
        read_csv,
        read_json,
        read_jsonl,
        read_parquet,
        read_text,
    )

    if format_name == "csv":
        return read_csv(path, database, schema, options)
    elif format_name == "json":
        return read_json(path, database, schema, options)
    elif format_name == "jsonl":
        return read_jsonl(path, database, schema, options)
    elif format_name == "parquet":
        return read_parquet(path, database, schema, options)
    elif format_name == "text":
        return read_text(path, database, schema, options, column_name or "value")
    else:
        raise ValueError(f"Unsupported file format: {format_name}")


async def _route_async_file_read(
    format_name: str,
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
) -> "AsyncRecords":
    """Route to async file reader."""
    from ..io.readers.async_csv_reader import read_csv
    from ..io.readers.async_json_reader import read_json, read_jsonl
    from ..io.readers.async_text_reader import read_text

    if format_name == "csv":
        return await read_csv(path, database, schema, options)
    elif format_name == "json":
        return await read_json(path, database, schema, options)
    elif format_name == "jsonl":
        return await read_jsonl(path, database, schema, options)
    elif format_name == "parquet":
        # Lazy import for parquet
        try:
            from ..io.readers.async_parquet_reader import read_parquet
        except ImportError:
            raise ImportError("Parquet support requires pyarrow. Install with: pip install pyarrow")
        return await read_parquet(path, database, schema, options)
    elif format_name == "text":
        return await read_text(path, database, schema, options, column_name or "value")
    else:
        raise ValueError(f"Unsupported file format: {format_name}")


@overload
def route_file_read_streaming(
    format_name: str,
    path: str,
    database: "Database",
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
    *,
    async_mode: Literal[False] = False,
) -> "Records": ...


@overload
def route_file_read_streaming(
    format_name: str,
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
    *,
    async_mode: Literal[True],
) -> "AsyncRecords": ...


def route_file_read_streaming(
    format_name: str,
    path: str,
    database: Union["Database", "AsyncDatabase"],
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
    *,
    async_mode: bool = False,
) -> Union["Records", "AsyncRecords"]:
    """Route file read to the appropriate streaming reader function based on format.

    Args:
        format_name: File format ("csv", "json", "jsonl", "parquet", "text")
        path: Path to the file
        database: :class:`Database` instance (sync or async)
        schema: Optional schema definition
        options: Read options dictionary
        column_name: Optional column name for text files
        async_mode: If True, use async readers; if False, use sync readers

    Returns:
        :class:`Records` or :class:`AsyncRecords` object with streaming generator (depending on async_mode)

    Raises:
        ValueError: If format is unsupported
        ImportError: If required dependencies are missing (e.g., pyarrow for parquet)
    """
    if async_mode:
        return _route_async_file_read_streaming(
            format_name, path, cast("AsyncDatabase", database), schema, options, column_name
        )  # type: ignore[return-value]
    else:
        return _route_sync_file_read_streaming(
            format_name, path, cast("Database", database), schema, options, column_name
        )


def _route_sync_file_read_streaming(
    format_name: str,
    path: str,
    database: "Database",
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
) -> "Records":
    """Route to sync streaming file reader."""
    from ..io.readers import (
        read_csv_stream,
        read_json_stream,
        read_jsonl_stream,
        read_parquet_stream,
        read_text_stream,
    )

    if format_name == "csv":
        return read_csv_stream(path, database, schema, options)
    elif format_name == "json":
        return read_json_stream(path, database, schema, options)
    elif format_name == "jsonl":
        return read_jsonl_stream(path, database, schema, options)
    elif format_name == "parquet":
        return read_parquet_stream(path, database, schema, options)
    elif format_name == "text":
        return read_text_stream(path, database, schema, options, column_name or "value")
    else:
        raise ValueError(f"Unsupported file format: {format_name}")


async def _route_async_file_read_streaming(
    format_name: str,
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence["ColumnDef"]],
    options: Dict[str, object],
    column_name: Optional[str] = None,
) -> "AsyncRecords":
    """Route to async streaming file reader."""
    from ..io.readers.async_csv_reader import read_csv_stream
    from ..io.readers.async_json_reader import read_json_stream, read_jsonl_stream
    from ..io.readers.async_text_reader import read_text_stream

    if format_name == "csv":
        return await read_csv_stream(path, database, schema, options)
    elif format_name == "json":
        return await read_json_stream(path, database, schema, options)
    elif format_name == "jsonl":
        return await read_jsonl_stream(path, database, schema, options)
    elif format_name == "parquet":
        # Lazy import for parquet
        try:
            from ..io.readers.async_parquet_reader import read_parquet_stream
        except ImportError:
            raise ImportError("Parquet support requires pyarrow. Install with: pip install pyarrow")
        return await read_parquet_stream(path, database, schema, options)
    elif format_name == "text":
        return await read_text_stream(path, database, schema, options, column_name or "value")
    else:
        raise ValueError(f"Unsupported file format: {format_name}")
