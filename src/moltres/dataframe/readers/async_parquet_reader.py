"""Async Parquet file reader implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

try:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
except ImportError as exc:
    raise ImportError(
        "Async Parquet reading requires pyarrow. Install with: pip install pyarrow"
    ) from exc

# aiofiles is not directly used here, but required for async file operations
# The import check is handled by the caller

from ...table.schema import ColumnDef
from ..async_dataframe import AsyncDataFrame

if TYPE_CHECKING:
    from ...table.async_table import AsyncDatabase


async def read_parquet(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
) -> AsyncDataFrame:
    """Read Parquet file asynchronously and return AsyncDataFrame.

    Args:
        path: Path to Parquet file
        database: AsyncDatabase instance
        schema: Optional explicit schema (currently unused, schema from Parquet file is used)
        options: Reader options (unused for Parquet)

    Returns:
        AsyncDataFrame containing the Parquet data

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pandas or pyarrow are not installed
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    # For async, we'll read the file content first, then parse
    # Note: pyarrow doesn't have native async support, so we use asyncio.to_thread
    import asyncio

    def _read_parquet_sync() -> list[dict[str, object]]:
        table = pq.read_table(str(path_obj))
        try:
            import pandas as pd  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pandas. Install with: pip install pandas"
            ) from exc
        df = table.to_pandas()
        return cast(list[dict[str, object]], df.to_dict("records"))

    rows = await asyncio.to_thread(_read_parquet_sync)

    if not rows:
        if schema:
            return _create_async_dataframe_from_schema(database, schema, [])
        return _create_async_dataframe_from_data(database, [])

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(rows)

    # Convert values to appropriate types
    from .schema_inference import apply_schema_to_rows

    typed_rows = apply_schema_to_rows(rows, final_schema)

    return _create_async_dataframe_from_data(database, typed_rows)


async def read_parquet_stream(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
) -> AsyncDataFrame:
    """Read Parquet file asynchronously in streaming mode (row group by row group).

    Args:
        path: Path to Parquet file
        database: AsyncDatabase instance
        schema: Optional explicit schema
        options: Reader options (unused for Parquet)

    Returns:
        AsyncDataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pyarrow is not installed
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    import asyncio

    def _get_parquet_file() -> pq.ParquetFile:
        return pq.ParquetFile(str(path_obj))

    parquet_file = await asyncio.to_thread(_get_parquet_file)

    async def _chunk_generator() -> AsyncIterator[list[dict[str, object]]]:
        for i in range(parquet_file.num_row_groups):

            def _read_row_group(idx: int) -> list[dict[str, object]]:
                row_group = parquet_file.read_row_group(idx)
                try:
                    import pandas as pd  # noqa: F401
                except ImportError:
                    raise RuntimeError("Parquet requires pandas") from None
                df = row_group.to_pandas()
                return cast(list[dict[str, object]], df.to_dict("records"))

            rows = await asyncio.to_thread(_read_row_group, i)
            if rows:
                yield rows

    # Read first row group to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = await first_chunk_gen.__anext__()
    except StopAsyncIteration:
        if schema:
            return _create_async_dataframe_from_schema(database, schema, [])
        return _create_async_dataframe_from_data(database, [])

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(first_chunk)

    from .schema_inference import apply_schema_to_rows

    async def _typed_chunk_generator() -> AsyncIterator[list[dict[str, object]]]:
        yield apply_schema_to_rows(first_chunk, final_schema)
        async for chunk in first_chunk_gen:
            yield apply_schema_to_rows(chunk, final_schema)

    return _create_async_dataframe_from_stream(database, _typed_chunk_generator, final_schema)


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
