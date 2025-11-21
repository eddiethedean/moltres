"""Async CSV file reader implementation."""

from __future__ import annotations

import csv
import io
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
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
        "Async CSV reading requires aiofiles. Install with: pip install moltres[async]"
    ) from exc

from ...table.schema import ColumnDef
from ..async_dataframe import AsyncDataFrame

if TYPE_CHECKING:
    from ...table.async_table import AsyncDatabase


async def read_csv(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
) -> AsyncDataFrame:
    """Read CSV file asynchronously and return AsyncDataFrame.

    Args:
        path: Path to CSV file
        database: AsyncDatabase instance
        schema: Optional explicit schema
        options: Reader options (header, delimiter, inferSchema)

    Returns:
        AsyncDataFrame containing the CSV data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    header = cast("bool", options.get("header", True))
    delimiter = cast("str", options.get("delimiter", ","))
    infer_schema = cast("bool", options.get("inferSchema", True))

    rows: list[dict[str, object]] = []

    async with aiofiles.open(path_obj, encoding="utf-8") as f:
        content = await f.read()

        # Parse CSV content
        csv_file = io.StringIO(content)
        if header and not schema:
            # Use DictReader when we have headers and no explicit schema
            dict_reader: Any = csv.DictReader(csv_file, delimiter=delimiter)
            for row in dict_reader:
                # Convert empty strings to None for nullable inference
                processed_row = {k: (None if v == "" else v) for k, v in row.items()}
                rows.append(processed_row)
        else:
            # Read without header or with explicit schema
            csv_reader: Any = csv.reader(csv_file, delimiter=delimiter)
            if header and schema:
                # Skip header row when we have explicit schema
                next(csv_reader, None)

            if schema:
                # Use schema column names
                for row_data in csv_reader:
                    if not row_data:  # Skip empty rows
                        continue
                    row_dict = {}
                    for i, col_def in enumerate(schema):
                        value = row_data[i] if i < len(row_data) else None
                        row_dict[col_def.name] = None if value == "" else value
                    rows.append(row_dict)
            else:
                # No header and no schema - can't determine column names
                raise ValueError(
                    "CSV file without header requires explicit schema. "
                    "Use .schema([ColumnDef(...), ...])"
                )

    if not rows:
        if schema:
            # Empty file with explicit schema
            return _create_async_dataframe_from_schema(database, schema, [])
        raise ValueError(f"CSV file is empty: {path}")

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    elif infer_schema:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(rows)
    else:
        # All columns as TEXT
        final_schema = [
            ColumnDef(name=col, type_name="TEXT", nullable=True) for col in rows[0].keys()
        ]

    # Convert values to appropriate types based on schema
    from .schema_inference import apply_schema_to_rows

    typed_rows = apply_schema_to_rows(rows, final_schema)

    return _create_async_dataframe_from_data(database, typed_rows)


async def read_csv_stream(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
) -> AsyncDataFrame:
    """Read CSV file asynchronously in streaming mode (chunked).

    Args:
        path: Path to CSV file
        database: AsyncDatabase instance
        schema: Optional explicit schema
        options: Reader options (header, delimiter, inferSchema, chunk_size)

    Returns:
        AsyncDataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    chunk_size = int(cast("int", options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    header = cast("bool", options.get("header", True))
    delimiter = cast("str", options.get("delimiter", ","))
    infer_schema = cast("bool", options.get("inferSchema", True))

    async def _chunk_generator() -> AsyncIterator[list[dict[str, object]]]:
        async with aiofiles.open(path_obj, encoding="utf-8") as f:
            # Read file in chunks for streaming
            content = await f.read()
            csv_file = io.StringIO(content)

            if header and not schema:
                reader = csv.DictReader(csv_file, delimiter=delimiter)
                chunk = []
                for row in reader:
                    processed_row = {k: (None if v == "" else v) for k, v in row.items()}
                    chunk.append(processed_row)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk
            else:
                reader_obj: Any = csv.reader(csv_file, delimiter=delimiter)
                if header and schema:
                    next(reader_obj, None)
                if schema:
                    chunk = []
                    for row_data in reader_obj:
                        if not row_data:
                            continue
                        row_dict: dict[str, object] = {}
                        for i, col_def in enumerate(schema):
                            value = row_data[i] if i < len(row_data) else None
                            row_dict[col_def.name] = None if value == "" else value
                        chunk.append(row_dict)
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    if chunk:
                        yield chunk
                else:
                    raise ValueError(
                        "CSV file without header requires explicit schema. "
                        "Use .schema([ColumnDef(...), ...])"
                    )

    # Read first chunk to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = await first_chunk_gen.__anext__()
    except StopAsyncIteration:
        if schema:
            return _create_async_dataframe_from_schema(database, schema, [])
        raise ValueError(f"CSV file is empty: {path}")

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    elif infer_schema:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(first_chunk)
    else:
        final_schema = [
            ColumnDef(name=col, type_name="TEXT", nullable=True) for col in first_chunk[0].keys()
        ]

    # Create generator that applies schema and yields chunks
    from .schema_inference import apply_schema_to_rows

    async def _typed_chunk_generator() -> AsyncIterator[list[dict[str, object]]]:
        # Yield first chunk (already read)
        yield apply_schema_to_rows(first_chunk, final_schema)
        # Yield remaining chunks
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
        _stream_generator=chunk_generator,  # Store the async generator callable
        _stream_schema=schema,
    )
