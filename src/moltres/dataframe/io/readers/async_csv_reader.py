"""Async CSV file reader implementation."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
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


async def read_csv(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read CSV file asynchronously and return :class:`AsyncRecords`.

    Args:
        path: Path to CSV file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema
        options: Reader options (header, delimiter, inferSchema)

    Returns:
        :class:`AsyncRecords` containing the CSV data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Extract all options with defaults (same as sync read_csv)
    header = cast(bool, options.get("header", True))
    delimiter = cast(str, options.get("delimiter", options.get("sep", ",")))
    infer_schema = cast(bool, options.get("inferSchema", True))
    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    quote = cast(str, options.get("quote", '"'))
    escape = cast(str, options.get("escape", "\\"))
    null_value = cast(str, options.get("nullValue", ""))
    nan_value = cast(str, options.get("nanValue", "NaN"))
    date_format = cast(Optional[str], options.get("dateFormat", None))
    timestamp_format = cast(Optional[str], options.get("timestampFormat", None))
    ignore_leading_whitespace = cast(bool, options.get("ignoreLeadingWhiteSpace", False))
    ignore_trailing_whitespace = cast(bool, options.get("ignoreTrailingWhiteSpace", False))
    comment = cast(Optional[str], options.get("comment", None))
    sampling_ratio = cast(float, options.get("samplingRatio", 1.0))
    mode = cast(str, options.get("mode", "PERMISSIVE"))
    corrupt_column = cast(Optional[str], options.get("columnNameOfCorruptRecord", None))

    rows: List[Dict[str, object]] = []
    corrupt_records: List[Dict[str, object]] = []

    content = await read_compressed_async(path, compression=compression, encoding=encoding)

    # Configure CSV reader with options
    csv_kwargs: Dict[str, Any] = {
        "delimiter": delimiter,
        "skipinitialspace": ignore_leading_whitespace or ignore_trailing_whitespace,
    }

    # Handle quote character
    if quote and len(quote) == 1:
        csv_kwargs["quotechar"] = quote
    else:
        csv_kwargs["quotechar"] = '"'

    # Handle escape character
    if escape and len(escape) == 1:
        csv_kwargs["escapechar"] = escape
    else:
        csv_kwargs["escapechar"] = None

    # Handle comment character
    if comment and len(comment) == 1:
        csv_kwargs["comment"] = comment

    def _process_value(value: str, col_name: Optional[str] = None) -> object:
        """Process a CSV value according to options."""
        if value is None:
            return None

        # Handle null values
        if value == null_value:
            return None

        # Handle NaN values
        if value == nan_value:
            return float("nan")

        # Handle whitespace
        if ignore_leading_whitespace:
            value = value.lstrip()
        if ignore_trailing_whitespace:
            value = value.rstrip()

        return value

    def _parse_row(row_data: List[str], column_names: List[str]) -> Optional[Dict[str, object]]:
        """Parse a CSV row with error handling based on mode."""
        try:
            row_dict: Dict[str, object] = {}
            for i, col_name in enumerate(column_names):
                raw_value = row_data[i] if i < len(row_data) else None
                processed_value: object = (
                    _process_value(raw_value, col_name) if raw_value is not None else None
                )
                row_dict[col_name] = processed_value

            # Add corrupt record column if specified
            if corrupt_column:
                row_dict[corrupt_column] = None

            return row_dict
        except Exception as e:
            if mode == "FAILFAST":
                raise ValueError(f"Failed to parse CSV row: {e}") from e
            elif mode == "DROPMALFORMED":
                return None
            else:  # PERMISSIVE
                if corrupt_column:
                    corrupt_row: Dict[str, object] = {corrupt_column: str(row_data)}
                    corrupt_records.append(corrupt_row)
                return None

    # Parse CSV content
    csv_file = io.StringIO(content)
    if header and not schema:
        # Use DictReader when we have headers and no explicit schema
        dict_reader: Any = csv.DictReader(csv_file, **csv_kwargs)
        column_names = dict_reader.fieldnames or []

        for row in dict_reader:
            try:
                processed_row: Dict[str, object] = {}
                for k, v in row.items():
                    processed_row[k] = _process_value(v, k)

                # Add corrupt record column if specified
                if corrupt_column and corrupt_column not in processed_row:
                    processed_row[corrupt_column] = None

                rows.append(processed_row)
            except Exception as e:
                if mode == "FAILFAST":
                    raise ValueError(f"Failed to parse CSV row: {e}") from e
                elif mode == "DROPMALFORMED":
                    continue
                else:  # PERMISSIVE
                    if corrupt_column:
                        corrupt_row: Dict[str, object] = {corrupt_column: str(row)}
                        corrupt_records.append(corrupt_row)
                    continue
    else:
        # Read without header or with explicit schema
        csv_reader: Any = csv.reader(csv_file, **csv_kwargs)
        if header and schema:
            # Skip header row when we have explicit schema
            next(csv_reader, None)

        if schema:
            # Use schema column names
            column_names = [col_def.name for col_def in schema]
            for row_data in csv_reader:
                if not row_data:  # Skip empty rows
                    continue
                parsed_row = _parse_row(row_data, column_names)
                if parsed_row is not None:
                    rows.append(parsed_row)
        else:
            # No header and no schema - can't determine column names
            raise ValueError(
                "CSV file without header requires explicit schema. Use .schema([ColumnDef(...), ...])"
            )

    if not rows:
        if schema:
            # Empty file with explicit schema
            return _create_async_records_from_schema(database, schema, [])
        raise ValueError(f"CSV file is empty: {path}")

    # Apply sampling ratio for schema inference
    sample_rows = rows
    if sampling_ratio < 1.0 and rows:
        sample_size = max(1, int(len(rows) * sampling_ratio))
        sample_rows = rows[:sample_size]

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    elif infer_schema and sample_rows:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(
            sample_rows, date_format=date_format, timestamp_format=timestamp_format
        )
    elif rows:
        # All columns as TEXT
        final_schema = [
            ColumnDef(name=col, type_name="TEXT", nullable=True) for col in rows[0].keys()
        ]
    else:
        # Empty file
        final_schema = []

    # Add corrupt records column to schema if needed
    if corrupt_column and corrupt_column not in [col.name for col in final_schema]:
        final_schema = list(final_schema) + [
            ColumnDef(name=corrupt_column, type_name="TEXT", nullable=True)
        ]
        # Add None values for corrupt column to all existing rows
        for row in rows:
            if corrupt_column not in row:
                row[corrupt_column] = None
        # Add corrupt records
        rows.extend(corrupt_records)

    # Convert values to appropriate types based on schema
    from .schema_inference import apply_schema_to_rows

    typed_rows = apply_schema_to_rows(
        rows, final_schema, date_format=date_format, timestamp_format=timestamp_format
    )

    return _create_async_records_from_data(database, typed_rows, final_schema)


async def read_csv_stream(
    path: str,
    database: "AsyncDatabase",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> AsyncRecords:
    """Read CSV file asynchronously in streaming mode (chunked).

    Args:
        path: Path to CSV file
        database: :class:`AsyncDatabase` instance
        schema: Optional explicit schema
        options: Reader options (header, delimiter, inferSchema, chunk_size)

    Returns:
        :class:`AsyncRecords` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    chunk_size = int(cast(int, options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # Extract all options with defaults (same as sync read_csv_stream)
    header = cast(bool, options.get("header", True))
    delimiter = cast(str, options.get("delimiter", options.get("sep", ",")))
    infer_schema = cast(bool, options.get("inferSchema", True))
    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    quote = cast(str, options.get("quote", '"'))
    escape = cast(str, options.get("escape", "\\"))
    null_value = cast(str, options.get("nullValue", ""))
    nan_value = cast(str, options.get("nanValue", "NaN"))
    date_format = cast(Optional[str], options.get("dateFormat", None))
    timestamp_format = cast(Optional[str], options.get("timestampFormat", None))
    ignore_leading_whitespace = cast(bool, options.get("ignoreLeadingWhiteSpace", False))
    ignore_trailing_whitespace = cast(bool, options.get("ignoreTrailingWhiteSpace", False))
    comment = cast(Optional[str], options.get("comment", None))
    mode = cast(str, options.get("mode", "PERMISSIVE"))
    corrupt_column = cast(Optional[str], options.get("columnNameOfCorruptRecord", None))

    def _process_value(value: str, col_name: Optional[str] = None) -> object:
        """Process a CSV value according to options."""
        if value is None:
            return None

        # Handle null values
        if value == null_value:
            return None

        # Handle NaN values
        if value == nan_value:
            return float("nan")

        # Handle whitespace
        if ignore_leading_whitespace:
            value = value.lstrip()
        if ignore_trailing_whitespace:
            value = value.rstrip()

        return value

    async def _chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        # Configure CSV reader with options
        csv_kwargs: Dict[str, Any] = {
            "delimiter": delimiter,
            "skipinitialspace": ignore_leading_whitespace or ignore_trailing_whitespace,
        }

        # Handle quote character
        if quote and len(quote) == 1:
            csv_kwargs["quotechar"] = quote
        else:
            csv_kwargs["quotechar"] = '"'

        # Handle escape character
        if escape and len(escape) == 1:
            csv_kwargs["escapechar"] = escape
        else:
            csv_kwargs["escapechar"] = None

        # Handle comment character
        if comment and len(comment) == 1:
            csv_kwargs["comment"] = comment

        # Read and decompress file content
        content = await read_compressed_async(path, compression=compression, encoding=encoding)
        csv_file = io.StringIO(content)

        if header and not schema:
            reader = csv.DictReader(csv_file, **csv_kwargs)
            chunk = []
            for row in reader:
                try:
                    processed_row: Dict[str, object] = {}
                    for k, v in row.items():
                        processed_row[k] = _process_value(v, k)

                    # Add corrupt record column if specified
                    if corrupt_column and corrupt_column not in processed_row:
                        processed_row[corrupt_column] = None

                    chunk.append(processed_row)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                except Exception as e:
                    if mode == "FAILFAST":
                        raise ValueError(f"Failed to parse CSV row: {e}") from e
                    elif mode == "DROPMALFORMED":
                        continue
                    else:  # PERMISSIVE
                        if corrupt_column:
                            corrupt_row: Dict[str, object] = {corrupt_column: str(row)}
                            chunk.append(corrupt_row)
                        continue
            if chunk:
                yield chunk
        else:
            reader_obj: Any = csv.reader(csv_file, **csv_kwargs)
            if header and schema:
                next(reader_obj, None)
            if schema:
                column_names = [col_def.name for col_def in schema]
                chunk = []
                for row_data in reader_obj:
                    if not row_data:
                        continue
                    try:
                        row_dict: Dict[str, object] = {}
                        for i, col_name in enumerate(column_names):
                            raw_value = row_data[i] if i < len(row_data) else None
                            processed_value: object = (
                                _process_value(raw_value, col_name)
                                if raw_value is not None
                                else None
                            )
                            row_dict[col_name] = processed_value

                        # Add corrupt record column if specified
                        if corrupt_column:
                            row_dict[corrupt_column] = None

                        chunk.append(row_dict)
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    except Exception as e:
                        if mode == "FAILFAST":
                            raise ValueError(f"Failed to parse CSV row: {e}") from e
                        elif mode == "DROPMALFORMED":
                            continue
                        else:  # PERMISSIVE
                            if corrupt_column:
                                corrupt_record: Dict[str, object] = {corrupt_column: str(row_data)}
                                chunk.append(corrupt_record)
                            continue
                if chunk:
                    yield chunk
            else:
                raise ValueError(
                    "CSV file without header requires explicit schema. Use .schema([ColumnDef(...), ...])"
                )

    # Read first chunk to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = await first_chunk_gen.__anext__()
    except StopAsyncIteration:
        if schema:
            return _create_async_records_from_schema(database, schema, [])
        raise ValueError(f"CSV file is empty: {path}")

    # Add corrupt records column to schema if needed
    if corrupt_column and first_chunk:
        if corrupt_column not in first_chunk[0]:
            # Add None values for corrupt column to all rows in first chunk
            for row in first_chunk:
                row[corrupt_column] = None

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    elif infer_schema and first_chunk:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(
            first_chunk, date_format=date_format, timestamp_format=timestamp_format
        )
    elif first_chunk:
        final_schema = [
            ColumnDef(name=col, type_name="TEXT", nullable=True) for col in first_chunk[0].keys()
        ]
    else:
        final_schema = []

    # Ensure corrupt column is in schema
    if corrupt_column and corrupt_column not in [col.name for col in final_schema]:
        final_schema = list(final_schema) + [
            ColumnDef(name=corrupt_column, type_name="TEXT", nullable=True)
        ]

    # Create generator that applies schema and yields chunks
    from .schema_inference import apply_schema_to_rows

    async def _typed_chunk_generator() -> AsyncIterator[List[Dict[str, object]]]:
        # Yield first chunk (already read)
        yield apply_schema_to_rows(
            first_chunk, final_schema, date_format=date_format, timestamp_format=timestamp_format
        )
        # Yield remaining chunks
        async for chunk in first_chunk_gen:
            yield apply_schema_to_rows(
                chunk, final_schema, date_format=date_format, timestamp_format=timestamp_format
            )

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
