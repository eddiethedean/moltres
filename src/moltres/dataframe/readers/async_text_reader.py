"""Async text file reader implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    cast,
)

try:
    import aiofiles  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "Async text reading requires aiofiles. Install with: pip install moltres[async]"
    ) from exc

from ...table.schema import ColumnDef
from ..async_dataframe import AsyncDataFrame

if TYPE_CHECKING:
    from ...table.async_table import AsyncDatabase


async def read_text(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
    column_name: str = "value",
) -> AsyncDataFrame:
    """Read text file line-by-line asynchronously and return AsyncDataFrame.

    Args:
        path: Path to text file
        database: AsyncDatabase instance
        schema: Optional explicit schema (unused, always TEXT)
        options: Reader options (unused for text)
        column_name: Name of the column to create (default: "value")

    Returns:
        AsyncDataFrame containing the text file lines

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    rows: list[dict[str, object]] = []
    async with aiofiles.open(path_obj, encoding="utf-8") as f:
        async for line in f:
            rows.append({column_name: line.rstrip("\n\r")})

    if not rows:
        return _create_async_dataframe_from_schema(database, [], [])

    return _create_async_dataframe_from_data(database, rows)


async def read_text_stream(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
    column_name: str = "value",
) -> AsyncDataFrame:
    """Read text file asynchronously in streaming mode (chunked).

    Args:
        path: Path to text file
        database: AsyncDatabase instance
        schema: Optional explicit schema (unused, always TEXT)
        options: Reader options (chunk_size)
        column_name: Name of the column to create (default: "value")

    Returns:
        AsyncDataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    chunk_size = int(cast("int", options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    async def _chunk_generator() -> AsyncIterator[list[dict[str, object]]]:
        chunk: list[dict[str, object]] = []
        async with aiofiles.open(path_obj, encoding="utf-8") as f:
            async for line in f:
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
        return _create_async_dataframe_from_schema(database, schema, [])

    schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]

    async def _typed_chunk_generator() -> AsyncIterator[list[dict[str, object]]]:
        yield first_chunk
        async for chunk in first_chunk_gen:
            yield chunk

    return _create_async_dataframe_from_stream(database, _typed_chunk_generator, schema)


def _create_async_dataframe_from_data(
    database: AsyncDatabase, rows: list[dict[str, object]]
) -> AsyncDataFrame:
    """Create AsyncDataFrame from materialized data."""
    from ...logical.plan import TableScan

    return AsyncDataFrame(
        plan=TableScan(table="__memory__"), database=database, _materialized_data=rows
    )


def _create_async_dataframe_from_schema(
    database: AsyncDatabase, schema: Sequence[ColumnDef], rows: list[dict[str, object]]
) -> AsyncDataFrame:
    """Create AsyncDataFrame with explicit schema but no data."""
    return _create_async_dataframe_from_data(database, rows)


def _create_async_dataframe_from_stream(
    database: AsyncDatabase,
    chunk_generator: Callable[[], AsyncIterator[list[dict[str, object]]]],
    schema: Sequence[ColumnDef],
) -> AsyncDataFrame:
    """Create AsyncDataFrame from streaming generator."""
    from ...logical.plan import TableScan

    return AsyncDataFrame(
        plan=TableScan(table="__stream__"),
        database=database,
        _stream_generator=chunk_generator,
        _stream_schema=schema,
    )
