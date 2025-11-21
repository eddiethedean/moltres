"""Async JSON and JSONL file reader implementation."""

from __future__ import annotations

import json
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
        "Async JSON reading requires aiofiles. Install with: pip install moltres[async]"
    ) from exc

from ...table.schema import ColumnDef
from ..async_dataframe import AsyncDataFrame

if TYPE_CHECKING:
    from ...table.async_table import AsyncDatabase


async def read_json(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
) -> AsyncDataFrame:
    """Read JSON file (array of objects) asynchronously and return AsyncDataFrame.

    Args:
        path: Path to JSON file
        database: AsyncDatabase instance
        schema: Optional explicit schema
        options: Reader options (multiline)

    Returns:
        AsyncDataFrame containing the JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    multiline = cast("bool", options.get("multiline", False))

    async with aiofiles.open(path_obj, encoding="utf-8") as f:
        content = await f.read()

        if multiline:
            # Read as JSONL (one object per line)
            rows = []
            for line in content.splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            # Read as JSON array
            data = json.loads(content)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = [data]
            else:
                raise ValueError(f"JSON file must contain an array or object: {path}")

    if not rows:
        if schema:
            return _create_async_dataframe_from_schema(database, schema, [])
        raise ValueError(f"JSON file is empty: {path}")

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


async def read_jsonl(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
) -> AsyncDataFrame:
    """Read JSONL file (one JSON object per line) asynchronously and return AsyncDataFrame.

    Args:
        path: Path to JSONL file
        database: AsyncDatabase instance
        schema: Optional explicit schema
        options: Reader options (unused for JSONL)

    Returns:
        AsyncDataFrame containing the JSONL data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows = []
    async with aiofiles.open(path_obj, encoding="utf-8") as f:
        async for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        if schema:
            return _create_async_dataframe_from_schema(database, schema, [])
        raise ValueError(f"JSONL file is empty: {path}")

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


async def read_json_stream(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
) -> AsyncDataFrame:
    """Read JSON file asynchronously in streaming mode (chunked).

    Args:
        path: Path to JSON file
        database: AsyncDatabase instance
        schema: Optional explicit schema
        options: Reader options (multiline, chunk_size)

    Returns:
        AsyncDataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    chunk_size = int(cast("int", options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    multiline = cast("bool", options.get("multiline", False))

    async def _chunk_generator() -> AsyncIterator[list[dict[str, object]]]:
        async with aiofiles.open(path_obj, encoding="utf-8") as f:
            if multiline:
                chunk = []
                async for line in f:
                    line = line.strip()
                    if line:
                        chunk.append(json.loads(line))
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                if chunk:
                    yield chunk
            else:
                # For JSON arrays, we need to read the whole file
                # This is a limitation - JSON arrays can't be truly streamed
                content = await f.read()
                data = json.loads(content)
                if isinstance(data, list):
                    for i in range(0, len(data), chunk_size):
                        yield data[i : i + chunk_size]
                elif isinstance(data, dict):
                    yield [data]
                else:
                    raise ValueError(f"JSON file must contain an array or object: {path}")

    # Read first chunk to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = await first_chunk_gen.__anext__()
    except StopAsyncIteration:
        if schema:
            return _create_async_dataframe_from_schema(database, schema, [])
        raise ValueError(f"JSON file is empty: {path}")

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


async def read_jsonl_stream(
    path: str,
    database: AsyncDatabase,
    schema: Sequence[ColumnDef] | None,
    options: dict[str, object],
) -> AsyncDataFrame:
    """Read JSONL file asynchronously in streaming mode (chunked).

    Args:
        path: Path to JSONL file
        database: AsyncDatabase instance
        schema: Optional explicit schema
        options: Reader options (chunk_size)

    Returns:
        AsyncDataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    chunk_size = int(cast("int", options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    async def _chunk_generator() -> AsyncIterator[list[dict[str, object]]]:
        chunk = []
        async with aiofiles.open(path_obj, encoding="utf-8") as f:
            async for line in f:
                line = line.strip()
                if line:
                    chunk.append(json.loads(line))
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
            if chunk:
                yield chunk

    # Read first chunk to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = await first_chunk_gen.__anext__()
    except StopAsyncIteration:
        if schema:
            return _create_async_dataframe_from_schema(database, schema, [])
        raise ValueError(f"JSONL file is empty: {path}")

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
