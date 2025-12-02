"""Async JSON and JSONL file reader implementation."""

from __future__ import annotations

import json
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


async def read_json(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read JSON file (array of objects) asynchronously and return :class:`AsyncRecords`.

    Args:
        path: Path to JSON file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema
        options: Reader options (multiline)

    Returns:
        :class:`AsyncRecords` containing the JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    # Extract all options with defaults
    multiline = cast(bool, options.get("multiline", options.get("multiLine", False)))
    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    mode = cast(str, options.get("mode", "PERMISSIVE"))
    corrupt_column = cast(Optional[str], options.get("columnNameOfCorruptRecord", None))
    date_format = cast(Optional[str], options.get("dateFormat", None))
    timestamp_format = cast(Optional[str], options.get("timestampFormat", None))
    line_sep = cast(Optional[str], options.get("lineSep", None))
    drop_field_if_all_null = cast(bool, options.get("dropFieldIfAllNull", False))

    rows: List[Dict[str, object]] = []
    corrupt_records: List[Dict[str, object]] = []

    content = await read_compressed_async(path, compression=compression, encoding=encoding)

    if multiline:
        # Read as JSONL (one object per line)
        line_separator = line_sep if line_sep else "\n"
        if line_sep:
            lines = content.split(line_separator)
        else:
            lines = content.splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                if mode == "FAILFAST":
                    raise ValueError(f"Failed to parse JSON line: {e}") from e
                elif mode == "DROPMALFORMED":
                    continue
                else:  # PERMISSIVE
                    if corrupt_column:
                        corrupt_records.append({corrupt_column: line})
                    continue
    else:
        # Read as JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = [data]
            else:
                if mode == "FAILFAST":
                    raise ValueError(f"JSON file must contain an array or object: {path}")
                elif mode == "DROPMALFORMED":
                    rows = []
                else:  # PERMISSIVE
                    if corrupt_column:
                        corrupt_records.append({corrupt_column: content})
                    rows = []
        except json.JSONDecodeError as e:
            if mode == "FAILFAST":
                raise ValueError(f"Failed to parse JSON file: {e}") from e
            elif mode == "DROPMALFORMED":
                rows = []
            else:  # PERMISSIVE
                if corrupt_column:
                    corrupt_records.append({corrupt_column: content})
                rows = []

    # Drop fields that are all null if requested
    if drop_field_if_all_null and rows:
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(row.keys())

        fields_to_drop = []
        for key in all_keys:
            if all(row.get(key) is None for row in rows):
                fields_to_drop.append(key)

        for row in rows:
            for field in fields_to_drop:
                row.pop(field, None)

    if not rows:
        if schema:
            return _create_async_records_from_schema(database, schema, [])
        raise ValueError(f"JSON file is empty: {path}")

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    elif rows:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(
            rows, date_format=date_format, timestamp_format=timestamp_format
        )
    else:
        final_schema = []

    # Add corrupt records column to schema if needed
    if corrupt_column and corrupt_column not in [col.name for col in final_schema]:
        final_schema = list(final_schema) + [
            ColumnDef(name=corrupt_column, type_name="TEXT", nullable=True)
        ]
        for row in rows:
            if corrupt_column not in row:
                row[corrupt_column] = None
        rows.extend(corrupt_records)

    # Convert values to appropriate types
    from .schema_inference import apply_schema_to_rows

    typed_rows = apply_schema_to_rows(
        rows, final_schema, date_format=date_format, timestamp_format=timestamp_format
    )

    return _create_async_records_from_data(database, typed_rows, final_schema)


async def read_jsonl(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read JSONL file (one JSON object per line) asynchronously and return :class:`AsyncRecords`.

    Args:
        path: Path to JSONL file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema
        options: Reader options (unused for JSONL)

    Returns:
        :class:`AsyncRecords` containing the JSONL data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    compression = cast(Optional[str], options.get("compression", None))
    content = await read_compressed_async(path, compression=compression)
    rows = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))

    if not rows:
        if schema:
            return _create_async_records_from_schema(database, schema, [])
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

    return _create_async_records_from_data(database, typed_rows, final_schema)


async def read_json_stream(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read JSON file asynchronously in streaming mode (chunked).

    Args:
        path: Path to JSON file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema
        options: Reader options (multiline, chunk_size)

    Returns:
        :class:`AsyncRecords` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    chunk_size = int(cast(int, options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    multiline = cast(bool, options.get("multiline", False))
    compression = cast(Optional[str], options.get("compression", None))

    async def _chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        content = await read_compressed_async(path, compression=compression)
        if multiline:
            chunk = []
            for line in content.splitlines():
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
            return _create_async_records_from_schema(database, schema, [])
        raise ValueError(f"JSON file is empty: {path}")

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


async def read_jsonl_stream(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read JSONL file asynchronously in streaming mode (chunked).

    Args:
        path: Path to JSONL file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema
        options: Reader options (chunk_size)

    Returns:
        :class:`AsyncRecords` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    chunk_size = int(cast(int, options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    compression = cast(Optional[str], options.get("compression", None))

    async def _chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        content = await read_compressed_async(path, compression=compression)
        chunk = []
        for line in content.splitlines():
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
            return _create_async_records_from_schema(database, schema, [])
        raise ValueError(f"JSONL file is empty: {path}")

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
