"""Async Parquet file reader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator, Callable, Dict, List, Optional, Sequence

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise ImportError(
        "Async Parquet reading requires pyarrow. Install with: pip install pyarrow"
    ) from exc

# aiofiles is not directly used here, but required for async file operations
# The import check is handled by the caller

from ...io.records import AsyncRecords
from ...table.schema import ColumnDef

if TYPE_CHECKING:
    from ...table.async_table import AsyncDatabase


async def read_parquet(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read Parquet file asynchronously and return AsyncRecords.

    Args:
        path: Path to Parquet file
        database: AsyncDatabase instance
        schema: Optional explicit schema (currently unused, schema from Parquet file is used)
        options: Reader options (unused for Parquet)

    Returns:
        AsyncRecords containing the Parquet data

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

    def _read_parquet_sync() -> List[Dict[str, object]]:
        table = pq.read_table(str(path_obj))
        try:
            import pandas as pd  # type: ignore[import-untyped]  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pandas. Install with: pip install pandas"
            ) from exc
        df = table.to_pandas()
        return df.to_dict("records")  # type: ignore[no-any-return]

    rows = await asyncio.to_thread(_read_parquet_sync)

    if not rows:
        if schema:
            return _create_async_records_from_schema(database, schema, [])
        return _create_async_records_from_data(database, [], schema)

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(rows)

    # Convert values to appropriate types
    from .schema_inference import apply_schema_to_rows

    typed_rows = apply_schema_to_rows(rows, final_schema)

    return _create_async_records_from_data(database, typed_rows, final_schema)


async def read_parquet_stream(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read Parquet file asynchronously in streaming mode (row group by row group).

    Args:
        path: Path to Parquet file
        database: AsyncDatabase instance
        schema: Optional explicit schema
        options: Reader options (unused for Parquet)

    Returns:
        AsyncRecords with streaming generator

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

    async def _chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        for i in range(parquet_file.num_row_groups):

            def _read_row_group(idx: int) -> List[Dict[str, object]]:
                row_group = parquet_file.read_row_group(idx)
                try:
                    import pandas as pd  # noqa: F401
                except ImportError:
                    raise RuntimeError("Parquet requires pandas") from None
                df = row_group.to_pandas()
                return df.to_dict("records")  # type: ignore[no-any-return]

            rows = await asyncio.to_thread(_read_row_group, i)
            if rows:
                yield rows

    # Read first row group to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = await first_chunk_gen.__anext__()
    except StopAsyncIteration:
        if schema:
            return _create_async_records_from_schema(database, schema, [])
        return _create_async_records_from_data(database, [], schema)

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(first_chunk)

    from .schema_inference import apply_schema_to_rows

    async def _typed_chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        yield apply_schema_to_rows(first_chunk, final_schema)
        async for chunk in first_chunk_gen:
            yield apply_schema_to_rows(chunk, final_schema)

    return _create_async_records_from_stream(database, _typed_chunk_generator, final_schema)


def _create_async_records_from_data(
    database: "AsyncDatabase", rows: List[Dict[str, object]], schema: Optional[Sequence[ColumnDef]]
) -> AsyncRecords:
    """Create AsyncRecords from materialized data."""
    return AsyncRecords(_data=rows, _schema=schema, _database=database)


def _create_async_records_from_schema(
    database: "AsyncDatabase", schema: Sequence[ColumnDef], rows: List[Dict[str, object]]
) -> AsyncRecords:
    """Create AsyncRecords with explicit schema but no data."""
    return AsyncRecords(_data=rows, _schema=schema, _database=database)


def _create_async_records_from_stream(
    database: "AsyncDatabase",
    chunk_generator: Callable[[], AsyncIterator[List[Dict[str, object]]]],
    schema: Sequence[ColumnDef],
) -> AsyncRecords:
    """Create AsyncRecords from streaming generator."""
    return AsyncRecords(_generator=chunk_generator, _schema=schema, _database=database)
