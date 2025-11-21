"""JSON and JSONL file reader implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, cast

from ...table.schema import ColumnDef
from ..dataframe import DataFrame

if TYPE_CHECKING:
    from ...table.table import Database


def read_json(
    path: str,
    database: Database,
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> DataFrame:
    """Read JSON file (array of objects) and return DataFrame.

    Args:
        path: Path to JSON file
        database: Database instance
        schema: Optional explicit schema
        options: Reader options (multiline)

    Returns:
        DataFrame containing the JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    multiline = cast("bool", options.get("multiline", False))

    with open(path_obj, encoding="utf-8") as f:
        if multiline:
            # Read as JSONL (one object per line)
            rows = []
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            # Read as JSON array
            data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict):
                rows = [data]
            else:
                raise ValueError(f"JSON file must contain an array or object: {path}")

    if not rows:
        if schema:
            return _create_dataframe_from_schema(database, schema, [])
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

    return _create_dataframe_from_data(database, typed_rows)


def read_jsonl(
    path: str,
    database: Database,
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> DataFrame:
    """Read JSONL file (one JSON object per line) and return DataFrame.

    Args:
        path: Path to JSONL file
        database: Database instance
        schema: Optional explicit schema
        options: Reader options (unused for JSONL)

    Returns:
        DataFrame containing the JSONL data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    rows = []
    with open(path_obj, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        if schema:
            return _create_dataframe_from_schema(database, schema, [])
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

    return _create_dataframe_from_data(database, typed_rows)


def read_json_stream(
    path: str,
    database: Database,
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> DataFrame:
    """Read JSON file in streaming mode (chunked).

    Args:
        path: Path to JSON file
        database: Database instance
        schema: Optional explicit schema
        options: Reader options (multiline, chunk_size)

    Returns:
        DataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    chunk_size = int(cast("int", options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    multiline = cast("bool", options.get("multiline", False))

    def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
        with open(path_obj, encoding="utf-8") as f:
            if multiline:
                chunk = []
                for line in f:
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
                data = json.load(f)
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
        first_chunk = next(first_chunk_gen)
    except StopIteration:
        if schema:
            return _create_dataframe_from_schema(database, schema, [])
        raise ValueError(f"JSON file is empty: {path}")

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(first_chunk)

    from .schema_inference import apply_schema_to_rows

    def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
        yield apply_schema_to_rows(first_chunk, final_schema)
        for chunk in first_chunk_gen:
            yield apply_schema_to_rows(chunk, final_schema)

    return _create_dataframe_from_stream(database, _typed_chunk_generator, final_schema)


def read_jsonl_stream(
    path: str,
    database: Database,
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> DataFrame:
    """Read JSONL file in streaming mode (chunked).

    Args:
        path: Path to JSONL file
        database: Database instance
        schema: Optional explicit schema
        options: Reader options (chunk_size)

    Returns:
        DataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    chunk_size = int(cast("int", options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
        chunk = []
        with open(path_obj, encoding="utf-8") as f:
            for line in f:
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
        first_chunk = next(first_chunk_gen)
    except StopIteration:
        if schema:
            return _create_dataframe_from_schema(database, schema, [])
        raise ValueError(f"JSONL file is empty: {path}")

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    else:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(first_chunk)

    from .schema_inference import apply_schema_to_rows

    def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
        yield apply_schema_to_rows(first_chunk, final_schema)
        for chunk in first_chunk_gen:
            yield apply_schema_to_rows(chunk, final_schema)

    return _create_dataframe_from_stream(database, _typed_chunk_generator, final_schema)


def _create_dataframe_from_data(database: Database, rows: List[Dict[str, object]]) -> DataFrame:
    """Create DataFrame from materialized data."""
    from ...logical.plan import TableScan

    return DataFrame(plan=TableScan(table="__memory__"), database=database, _materialized_data=rows)


def _create_dataframe_from_schema(
    database: Database, schema: Sequence[ColumnDef], rows: List[Dict[str, object]]
) -> DataFrame:
    """Create DataFrame with explicit schema but no data."""
    return _create_dataframe_from_data(database, rows)


def _create_dataframe_from_stream(
    database: Database,
    chunk_generator: Callable[[], Iterator[List[Dict[str, object]]]],
    schema: Sequence[ColumnDef],
) -> DataFrame:
    """Create DataFrame from streaming generator.

    Args:
        database: Database instance
        chunk_generator: Callable that returns an iterator of chunks
        schema: Schema for the data
    """
    from ...logical.plan import TableScan

    return DataFrame(
        plan=TableScan(table="__stream__"),
        database=database,
        _stream_generator=chunk_generator,
        _stream_schema=schema,
    )
