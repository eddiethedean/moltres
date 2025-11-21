"""CSV file reader implementation."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

from ...table.schema import ColumnDef
from ..dataframe import DataFrame

if TYPE_CHECKING:
    from ...table.table import Database


def read_csv(
    path: str,
    database: Database,
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> DataFrame:
    """Read CSV file and return DataFrame.

    Args:
        path: Path to CSV file
        database: Database instance
        schema: Optional explicit schema
        options: Reader options (header, delimiter, inferSchema)

    Returns:
        DataFrame containing the CSV data

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

    rows: List[Dict[str, object]] = []

    with open(path_obj, encoding="utf-8") as f:
        if header and not schema:
            # Use DictReader when we have headers and no explicit schema
            dict_reader: Any = csv.DictReader(f, delimiter=delimiter)
            for row in dict_reader:
                # Convert empty strings to None for nullable inference
                processed_row = {k: (None if v == "" else v) for k, v in row.items()}
                rows.append(processed_row)
        else:
            # Read without header or with explicit schema
            csv_reader: Any = csv.reader(f, delimiter=delimiter)
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
            return _create_dataframe_from_schema(database, schema, [])
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

    return _create_dataframe_from_data(database, typed_rows)


def read_csv_stream(
    path: str,
    database: Database,
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> DataFrame:
    """Read CSV file in streaming mode (chunked).

    Args:
        path: Path to CSV file
        database: Database instance
        schema: Optional explicit schema
        options: Reader options (header, delimiter, inferSchema, chunk_size)

    Returns:
        DataFrame with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    chunk_size = int(cast("Any", options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    header = cast("bool", options.get("header", True))
    delimiter = cast("str", options.get("delimiter", ","))
    infer_schema = cast("bool", options.get("inferSchema", True))

    def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
        with open(path_obj, encoding="utf-8") as f:
            if header and not schema:
                reader = csv.DictReader(f, delimiter=delimiter)
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
                reader_obj: Any = csv.reader(f, delimiter=delimiter)
                if header and schema:
                    next(reader_obj, None)
                if schema:
                    chunk = []
                    for row_data in reader_obj:
                        if not row_data:
                            continue
                        row_dict: Dict[str, object] = {}
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
        first_chunk = next(first_chunk_gen)
    except StopIteration:
        if schema:
            return _create_dataframe_from_schema(database, schema, [])
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

    def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
        # Yield first chunk (already read)
        yield apply_schema_to_rows(first_chunk, final_schema)
        # Yield remaining chunks
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
