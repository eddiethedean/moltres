"""Async Parquet file reader implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator, Callable, Dict, List, Optional, Sequence, cast

# aiofiles is not directly used here, but required for async file operations
# The import check is handled by the caller

from ....io.records import AsyncRecords
from ....table.schema import ColumnDef

if TYPE_CHECKING:
    from ....table.async_table import AsyncDatabase


async def read_parquet(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read Parquet file asynchronously and return :class:`AsyncRecords`.

    Args:
        path: Path to Parquet file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema (currently unused, schema from Parquet file is used)
        options: Reader options (unused for Parquet)

    Returns:
        :class:`AsyncRecords` containing the Parquet data

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pandas or pyarrow are not installed
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    # Extract Parquet options
    merge_schema = cast(bool, options.get("mergeSchema", False))

    # Note: rebaseDatetimeInRead, datetimeRebaseMode, and int96RebaseMode options
    # are accepted but not yet implemented (PyArrow handles rebasing automatically)

    # For async, we'll read the file content first, then parse
    # Note: pyarrow doesn't have native async support, so we use asyncio.to_thread
    import asyncio

    def _read_parquet_sync() -> List[Dict[str, object]]:
        from ....utils.optional_deps import get_pyarrow_parquet

        pq = get_pyarrow_parquet(required=True)

        read_options: Dict[str, object] = {}
        if merge_schema:
            # Note: PyArrow's read_table doesn't directly support mergeSchema
            # This would require reading multiple files and merging schemas
            # For single file, this is a no-op
            pass

        table = pq.read_table(str(path_obj), **read_options)
        from ....utils.optional_deps import get_pandas

        get_pandas(required=True)
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
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema
        options: Reader options (unused for Parquet)

    Returns:
        :class:`AsyncRecords` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If pyarrow is not installed
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Parquet file not found: {path}")

    import asyncio
    from ....utils.optional_deps import get_pyarrow_parquet

    pq = get_pyarrow_parquet(required=True)

    def _get_parquet_file() -> Any:
        return pq.ParquetFile(str(path_obj))

    parquet_file = await asyncio.to_thread(_get_parquet_file)

    async def _chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        for i in range(parquet_file.num_row_groups):

            def _read_row_group(idx: int) -> List[Dict[str, object]]:
                from ....utils.optional_deps import get_pandas

                row_group = parquet_file.read_row_group(idx)
                get_pandas(required=True)
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
