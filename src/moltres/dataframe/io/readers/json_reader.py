"""JSON and JSONL file reader implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Sequence, cast

from ....io.records import Records
from ....table.schema import ColumnDef
from .compression import open_compressed

if TYPE_CHECKING:
    from ....table.table import Database


def read_json(
    path: str,
    database: "Database",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> Records:
    """Read JSON file (array of objects) and return :class:`Records`.

    Args:
        path: Path to JSON file
        database: :class:`Database` instance
        schema: Optional explicit schema
        options: Reader options (multiline)

    Returns:
        :class:`Records` containing the JSON data

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
    sampling_ratio = cast(float, options.get("samplingRatio", 1.0))
    line_sep = cast(Optional[str], options.get("lineSep", None))
    drop_field_if_all_null = cast(bool, options.get("dropFieldIfAllNull", False))
    # Note: Many JSON parsing options (allowComments, allowUnquotedFieldNames, etc.)
    # are not supported by Python's json module, so they are ignored

    rows: List[Dict[str, object]] = []
    corrupt_records: List[Dict[str, object]] = []

    with open_compressed(path, "r", compression=compression, encoding=encoding) as f:
        if multiline:
            # Read as JSONL (one object per line)
            line_separator = line_sep if line_sep else "\n"
            content = f.read()
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
                data = json.load(f)
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
                            corrupt_records.append({corrupt_column: str(data)})
                        rows = []
            except json.JSONDecodeError as e:
                if mode == "FAILFAST":
                    raise ValueError(f"Failed to parse JSON file: {e}") from e
                elif mode == "DROPMALFORMED":
                    rows = []
                else:  # PERMISSIVE
                    if corrupt_column:
                        # Try to read file content for corrupt record
                        f.seek(0)
                        try:
                            corrupt_content = f.read()
                            corrupt_records.append({corrupt_column: corrupt_content})
                        except Exception:
                            corrupt_records.append({corrupt_column: str(e)})
                    rows = []

    # Drop fields that are all null if requested
    if drop_field_if_all_null and rows:
        # Find fields that are all null
        all_keys: set[str] = set()
        for row in rows:
            all_keys.update(row.keys())

        fields_to_drop = []
        for key in all_keys:
            if all(row.get(key) is None for row in rows):
                fields_to_drop.append(key)

        # Remove dropped fields from rows
        for row in rows:
            for field in fields_to_drop:
                row.pop(field, None)

    if not rows:
        if schema:
            return _create_records_from_schema(database, schema, [])
        raise ValueError(f"JSON file is empty: {path}")

    # Apply sampling ratio for schema inference
    sample_rows = rows
    if sampling_ratio < 1.0 and rows:
        sample_size = max(1, int(len(rows) * sampling_ratio))
        sample_rows = rows[:sample_size]

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    elif sample_rows:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(
            sample_rows, date_format=date_format, timestamp_format=timestamp_format
        )
    else:
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

    # Convert values to appropriate types
    from .schema_inference import apply_schema_to_rows

    typed_rows = apply_schema_to_rows(
        rows, final_schema, date_format=date_format, timestamp_format=timestamp_format
    )

    return _create_records_from_data(database, typed_rows, final_schema)


def read_jsonl(
    path: str,
    database: "Database",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> Records:
    """Read JSONL file (one JSON object per line) and return :class:`Records`.

    Args:
        path: Path to JSONL file
        database: :class:`Database` instance
        schema: Optional explicit schema
        options: Reader options (unused for JSONL)

    Returns:
        :class:`Records` containing the JSONL data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

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

    with open_compressed(path, "r", compression=compression, encoding=encoding) as f:
        line_separator = line_sep if line_sep else "\n"
        content = f.read()
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
                    raise ValueError(f"Failed to parse JSONL line: {e}") from e
                elif mode == "DROPMALFORMED":
                    continue
                else:  # PERMISSIVE
                    if corrupt_column:
                        corrupt_records.append({corrupt_column: line})
                    continue

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
            return _create_records_from_schema(database, schema, [])
        raise ValueError(f"JSONL file is empty: {path}")

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

    return _create_records_from_data(database, typed_rows, final_schema)


def read_json_stream(
    path: str,
    database: "Database",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> Records:
    """Read JSON file in streaming mode (chunked).

    Args:
        path: Path to JSON file
        database: :class:`Database` instance
        schema: Optional explicit schema
        options: Reader options (multiline, chunk_size)

    Returns:
        :class:`Records` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    chunk_size = int(cast(int, options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    multiline = cast(bool, options.get("multiline", options.get("multiLine", False)))
    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    mode = cast(str, options.get("mode", "PERMISSIVE"))
    corrupt_column = cast(Optional[str], options.get("columnNameOfCorruptRecord", None))
    date_format = cast(Optional[str], options.get("dateFormat", None))
    timestamp_format = cast(Optional[str], options.get("timestampFormat", None))
    line_sep = cast(Optional[str], options.get("lineSep", None))
    drop_field_if_all_null = cast(bool, options.get("dropFieldIfAllNull", False))

    def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
        with open_compressed(path, "r", compression=compression, encoding=encoding) as f:
            if multiline:
                chunk = []
                line_separator = line_sep if line_sep else "\n"
                content = f.read()
                if line_sep:
                    lines = content.split(line_separator)
                else:
                    lines = content.splitlines()

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk.append(json.loads(line))
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    except json.JSONDecodeError as e:
                        if mode == "FAILFAST":
                            raise ValueError(f"Failed to parse JSON line: {e}") from e
                        elif mode == "DROPMALFORMED":
                            continue
                        else:  # PERMISSIVE
                            if corrupt_column:
                                corrupt_row = {corrupt_column: line}
                                chunk.append(corrupt_row)
                            continue
                if chunk:
                    yield chunk
            else:
                # For JSON arrays, we need to read the whole file
                # This is a limitation - JSON arrays can't be truly streamed
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for i in range(0, len(data), chunk_size):
                            yield data[i : i + chunk_size]
                    elif isinstance(data, dict):
                        yield [data]
                    else:
                        if mode == "FAILFAST":
                            raise ValueError(f"JSON file must contain an array or object: {path}")
                        elif mode == "DROPMALFORMED":
                            return
                        else:  # PERMISSIVE
                            if corrupt_column:
                                f.seek(0)
                                corrupt_content = f.read()
                                yield [{corrupt_column: corrupt_content}]
                            return
                except json.JSONDecodeError as e:
                    if mode == "FAILFAST":
                        raise ValueError(f"Failed to parse JSON file: {e}") from e
                    elif mode == "DROPMALFORMED":
                        return
                    else:  # PERMISSIVE
                        if corrupt_column:
                            f.seek(0)
                            try:
                                corrupt_content = f.read()
                                yield [{corrupt_column: corrupt_content}]
                            except Exception:
                                yield [{corrupt_column: str(e)}]
                        return

    # Read first chunk to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = next(first_chunk_gen)
    except StopIteration:
        if schema:
            return _create_records_from_schema(database, schema, [])
        raise ValueError(f"JSON file is empty: {path}")

    # Drop fields that are all null if requested
    if drop_field_if_all_null and first_chunk:
        all_keys: set[str] = set()
        for row in first_chunk:
            all_keys.update(row.keys())

        fields_to_drop = []
        for key in all_keys:
            if all(row.get(key) is None for row in first_chunk):
                fields_to_drop.append(key)

        for row in first_chunk:
            for field in fields_to_drop:
                row.pop(field, None)

    # Add corrupt records column to schema if needed
    if corrupt_column and first_chunk:
        if corrupt_column not in first_chunk[0]:
            for row in first_chunk:
                row[corrupt_column] = None

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    elif first_chunk:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(
            first_chunk, date_format=date_format, timestamp_format=timestamp_format
        )
    else:
        final_schema = []

    # Ensure corrupt column is in schema
    if corrupt_column and corrupt_column not in [col.name for col in final_schema]:
        final_schema = list(final_schema) + [
            ColumnDef(name=corrupt_column, type_name="TEXT", nullable=True)
        ]

    from .schema_inference import apply_schema_to_rows

    def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
        yield apply_schema_to_rows(
            first_chunk, final_schema, date_format=date_format, timestamp_format=timestamp_format
        )
        for chunk in first_chunk_gen:
            # Apply drop_field_if_all_null to each chunk
            if drop_field_if_all_null and chunk:
                all_keys: set[str] = set()
                for row in chunk:
                    all_keys.update(row.keys())

                fields_to_drop = []
                for key in all_keys:
                    if all(row.get(key) is None for row in chunk):
                        fields_to_drop.append(key)

                for row in chunk:
                    for field in fields_to_drop:
                        row.pop(field, None)

            yield apply_schema_to_rows(
                chunk, final_schema, date_format=date_format, timestamp_format=timestamp_format
            )

    return _create_records_from_stream(database, _typed_chunk_generator, final_schema)


def read_jsonl_stream(
    path: str,
    database: "Database",
    schema: Optional[Sequence[ColumnDef]],
    options: Dict[str, object],
) -> Records:
    """Read JSONL file in streaming mode (chunked).

    Args:
        path: Path to JSONL file
        database: :class:`Database` instance
        schema: Optional explicit schema
        options: Reader options (chunk_size)

    Returns:
        :class:`Records` with streaming generator

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    chunk_size = int(cast(int, options.get("chunk_size", 10000)))
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    compression = cast(Optional[str], options.get("compression", None))
    encoding = cast(str, options.get("encoding", "UTF-8"))
    mode = cast(str, options.get("mode", "PERMISSIVE"))
    corrupt_column = cast(Optional[str], options.get("columnNameOfCorruptRecord", None))
    date_format = cast(Optional[str], options.get("dateFormat", None))
    timestamp_format = cast(Optional[str], options.get("timestampFormat", None))
    line_sep = cast(Optional[str], options.get("lineSep", None))
    drop_field_if_all_null = cast(bool, options.get("dropFieldIfAllNull", False))

    def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
        chunk = []
        with open_compressed(path, "r", compression=compression, encoding=encoding) as f:
            line_separator = line_sep if line_sep else "\n"
            content = f.read()
            if line_sep:
                lines = content.split(line_separator)
            else:
                lines = content.splitlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk.append(json.loads(line))
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                except json.JSONDecodeError as e:
                    if mode == "FAILFAST":
                        raise ValueError(f"Failed to parse JSONL line: {e}") from e
                    elif mode == "DROPMALFORMED":
                        continue
                    else:  # PERMISSIVE
                        if corrupt_column:
                            corrupt_row = {corrupt_column: line}
                            chunk.append(corrupt_row)
                        continue
            if chunk:
                yield chunk

    # Read first chunk to infer schema
    first_chunk_gen = _chunk_generator()
    try:
        first_chunk = next(first_chunk_gen)
    except StopIteration:
        if schema:
            return _create_records_from_schema(database, schema, [])
        raise ValueError(f"JSONL file is empty: {path}")

    # Drop fields that are all null if requested
    if drop_field_if_all_null and first_chunk:
        all_keys: set[str] = set()
        for row in first_chunk:
            all_keys.update(row.keys())

        fields_to_drop = []
        for key in all_keys:
            if all(row.get(key) is None for row in first_chunk):
                fields_to_drop.append(key)

        for row in first_chunk:
            for field in fields_to_drop:
                row.pop(field, None)

    # Add corrupt records column to schema if needed
    if corrupt_column and first_chunk:
        if corrupt_column not in first_chunk[0]:
            for row in first_chunk:
                row[corrupt_column] = None

    # Infer or use explicit schema
    if schema:
        final_schema = schema
    elif first_chunk:
        from .schema_inference import infer_schema_from_rows

        final_schema = infer_schema_from_rows(
            first_chunk, date_format=date_format, timestamp_format=timestamp_format
        )
    else:
        final_schema = []

    # Ensure corrupt column is in schema
    if corrupt_column and corrupt_column not in [col.name for col in final_schema]:
        final_schema = list(final_schema) + [
            ColumnDef(name=corrupt_column, type_name="TEXT", nullable=True)
        ]

    from .schema_inference import apply_schema_to_rows

    def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
        yield apply_schema_to_rows(
            first_chunk, final_schema, date_format=date_format, timestamp_format=timestamp_format
        )
        for chunk in first_chunk_gen:
            # Apply drop_field_if_all_null to each chunk
            if drop_field_if_all_null and chunk:
                all_keys: set[str] = set()
                for row in chunk:
                    all_keys.update(row.keys())

                fields_to_drop = []
                for key in all_keys:
                    if all(row.get(key) is None for row in chunk):
                        fields_to_drop.append(key)

                for row in chunk:
                    for field in fields_to_drop:
                        row.pop(field, None)

            yield apply_schema_to_rows(
                chunk, final_schema, date_format=date_format, timestamp_format=timestamp_format
            )

    return _create_records_from_stream(database, _typed_chunk_generator, final_schema)


def _create_records_from_data(
    database: "Database", rows: List[Dict[str, object]], schema: Optional[Sequence[ColumnDef]]
) -> Records:
    """Create :class:`Records` from materialized data."""
    return Records(_data=rows, _schema=schema, _database=database)


def _create_records_from_schema(
    database: "Database", schema: Sequence[ColumnDef], rows: List[Dict[str, object]]
) -> Records:
    """Create :class:`Records` with explicit schema but no data."""
    return Records(_data=rows, _schema=schema, _database=database)


def _create_records_from_stream(
    database: "Database",
    chunk_generator: Callable[[], Iterator[List[Dict[str, object]]]],
    schema: Sequence[ColumnDef],
) -> Records:
    """Create :class:`Records` from streaming generator.

    Args:
        database: :class:`Database` instance
        chunk_generator: Callable that returns an iterator of chunks
        schema: Schema for the data
    """
    return Records(_generator=chunk_generator, _schema=schema, _database=database)
