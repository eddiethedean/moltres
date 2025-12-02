"""Async text file reader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    cast,
)

from ....io.records import AsyncRecords
from ....table.schema import ColumnDef
from .compression import read_compressed_async

if TYPE_CHECKING:
    from ....table.async_table import AsyncDatabase


async def read_text(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
    column_name: str = "value",
) -> AsyncRecords:
    """Read text file line-by-line asynchronously and return :class:`AsyncRecords`.

    Args:
        path: Path to text file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema (unused, always TEXT)
        options: Reader options (unused for text)
        column_name: Name of the column to create (default: "value")

    Returns:
        :class:`AsyncRecords` containing the text file lines

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    wholetext = cast(bool, options.get("wholetext", False))
    line_sep = cast(Optional[str], options.get("lineSep", None))

    content = await read_compressed_async(path, compression=compression, encoding=encoding)
    rows: List[Dict[str, object]] = []

    if wholetext:
        # Read entire file as single value
        rows.append({column_name: content})
    else:
        # Read line by line
        if line_sep:
            lines = content.split(line_sep)
            for line in lines:
                rows.append({column_name: line})
        else:
            for line in content.splitlines(keepends=True):
                rows.append({column_name: line.rstrip("\n\r")})

    if not rows:
        return _create_async_records_from_schema(database, [], [])

    schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]
    return _create_async_records_from_data(database, rows, schema)


async def read_text_stream(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
    column_name: str = "value",
) -> AsyncRecords:
    """Read text file asynchronously in streaming mode (chunked).

    Args:
        path: Path to text file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema (unused, always TEXT)
        options: Reader options (chunk_size)
        column_name: Name of the column to create (default: "value")

    Returns:
        :class:`AsyncRecords` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    chunk_size = int(cast(int, options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    wholetext = cast(bool, options.get("wholetext", False))
    line_sep = cast(Optional[str], options.get("lineSep", None))

    async def _chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        content = await read_compressed_async(path, compression=compression, encoding=encoding)
        chunk: List[Dict[str, object]] = []

        if wholetext:
            # Read entire file as single value
            yield [{column_name: content}]
        else:
            # Read line by line
            if line_sep:
                lines = content.split(line_sep)
                for line in lines:
                    chunk.append({column_name: line})
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk
            else:
                for line in content.splitlines(keepends=True):
                    chunk.append({column_name: line.rstrip("\n\r")})
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk

    # Read first chunk
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = await first_chunk_gen.__anext__()
    except StopAsyncIteration:
        schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]
        return _create_async_records_from_schema(database, schema, [])

    schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]

    async def _typed_chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        yield first_chunk
        async for chunk in first_chunk_gen:
            yield chunk

    return _create_async_records_from_stream(database, _typed_chunk_generator, schema)


def _create_async_records_from_data(
    database: "AsyncDatabase", rows: List[Dict[str, object]], schema: Optional[Sequence[ColumnDef]]
) -> AsyncRecords:
    """Create :class:`AsyncRecords` from materialized data."""
    return AsyncRecords(_data=rows, _schema=schema, _database=database)


def _create_async_records_from_schema(
    database: "AsyncDatabase", schema: Sequence[ColumnDef], rows: List[Dict[str, object]]
) -> AsyncRecords:
    """Create :class:`AsyncRecords` with explicit schema but no data."""
    return AsyncRecords(_data=rows, _schema=schema, _database=database)


def _create_async_records_from_stream(
    database: "AsyncDatabase",
    chunk_generator: Callable[[], AsyncIterator[List[Dict[str, object]]]],
    schema: Sequence[ColumnDef],
) -> AsyncRecords:
    """Create :class:`AsyncRecords` from streaming generator."""
    return AsyncRecords(_generator=chunk_generator, _schema=schema, _database=database)
