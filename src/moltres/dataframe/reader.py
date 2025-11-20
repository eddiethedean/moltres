"""DataFrame read operations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, cast

from ..table.schema import ColumnDef
from .dataframe import DataFrame

if TYPE_CHECKING:
    from ..table.table import Database


class DataFrameReader:
    """Builder for reading DataFrames from tables and files."""

    def __init__(self, database: "Database"):
        self._database = database
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}

    def stream(self, enabled: bool = True) -> "DataFrameReader":
        """Enable or disable streaming mode (chunked reading for large files)."""
        self._options["stream"] = enabled
        return self

    def schema(self, schema: Sequence[ColumnDef]) -> "DataFrameReader":
        """Set an explicit schema for the data source."""
        self._schema = schema
        return self

    def option(self, key: str, value: object) -> "DataFrameReader":
        """Set a read option (e.g., header=True for CSV, multiline=True for JSON)."""
        self._options[key] = value
        return self

    def table(self, name: str) -> DataFrame:
        """Read from a database table."""
        return self._database.table(name).select()

    def csv(self, path: str) -> DataFrame:
        """Read a CSV file."""
        stream = cast(bool, self._options.get("stream", False))
        if stream:
            return self._read_csv_stream(path)
        return self._read_csv(path)

    def json(self, path: str) -> DataFrame:
        """Read a JSON file (array of objects)."""
        stream = cast(bool, self._options.get("stream", False))
        if stream:
            return self._read_json_stream(path)
        return self._read_json(path)

    def jsonl(self, path: str) -> DataFrame:
        """Read a JSONL file (one JSON object per line)."""
        stream = cast(bool, self._options.get("stream", False))
        if stream:
            return self._read_jsonl_stream(path)
        return self._read_jsonl(path)

    def parquet(self, path: str) -> DataFrame:
        """Read a Parquet file."""
        stream = cast(bool, self._options.get("stream", False))
        if stream:
            return self._read_parquet_stream(path)
        return self._read_parquet(path)

    def text(self, path: str, column_name: str = "value") -> DataFrame:
        """Read a text file as a single column (one line per row)."""
        stream = cast(bool, self._options.get("stream", False))
        if stream:
            return self._read_text_stream(path, column_name)
        return self._read_text(path, column_name)

    def format(self, source: str) -> "FormatReader":
        """Specify the data source format."""
        return FormatReader(self, source)

    # ---------------------------------------------------------------- CSV reader
    def _read_csv(self, path: str) -> DataFrame:
        """Read CSV file and return DataFrame."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        header = cast(bool, self._options.get("header", True))
        delimiter = cast(str, self._options.get("delimiter", ","))
        infer_schema = cast(bool, self._options.get("inferSchema", True))

        rows: List[Dict[str, object]] = []

        with open(path_obj, "r", encoding="utf-8") as f:
            if header and not self._schema:
                # Use DictReader when we have headers and no explicit schema
                dict_reader: Any = csv.DictReader(f, delimiter=delimiter)
                for row in dict_reader:
                    # Convert empty strings to None for nullable inference
                    processed_row = {k: (None if v == "" else v) for k, v in row.items()}
                    rows.append(processed_row)
            else:
                # Read without header or with explicit schema
                csv_reader: Any = csv.reader(f, delimiter=delimiter)
                if header and self._schema:
                    # Skip header row when we have explicit schema
                    next(csv_reader, None)

                if self._schema:
                    # Use schema column names
                    for row_data in csv_reader:
                        if not row_data:  # Skip empty rows
                            continue
                        row_dict = {}
                        for i, col_def in enumerate(self._schema):
                            value = row_data[i] if i < len(row_data) else None
                            row_dict[col_def.name] = None if value == "" else value
                        rows.append(row_dict)
                else:
                    # No header and no schema - can't determine column names
                    raise ValueError(
                        "CSV file without header requires explicit schema. Use .schema([ColumnDef(...), ...])"
                    )

        if not rows:
            if self._schema:
                # Empty file with explicit schema
                return self._create_dataframe_from_schema([])
            raise ValueError(f"CSV file is empty: {path}")

        # Infer or use explicit schema
        if self._schema:
            schema = self._schema
        elif infer_schema:
            schema = self._infer_schema_from_rows(rows)
        else:
            # All columns as TEXT
            schema = [
                ColumnDef(name=col, type_name="TEXT", nullable=True) for col in rows[0].keys()
            ]

        # Convert values to appropriate types based on schema
        typed_rows = self._apply_schema_to_rows(rows, schema)

        return self._create_dataframe_from_data(typed_rows)

    # ---------------------------------------------------------------- JSON reader
    def _read_json(self, path: str) -> DataFrame:
        """Read JSON file (array of objects) and return DataFrame."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        multiline = cast(bool, self._options.get("multiline", False))

        with open(path_obj, "r", encoding="utf-8") as f:
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
            if self._schema:
                return self._create_dataframe_from_schema([])
            raise ValueError(f"JSON file is empty: {path}")

        # Infer or use explicit schema
        if self._schema:
            schema = self._schema
        else:
            schema = self._infer_schema_from_rows(rows)

        # Convert values to appropriate types
        typed_rows = self._apply_schema_to_rows(rows, schema)

        return self._create_dataframe_from_data(typed_rows)

    def _read_jsonl(self, path: str) -> DataFrame:
        """Read JSONL file (one JSON object per line) and return DataFrame."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        rows = []
        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            if self._schema:
                return self._create_dataframe_from_schema([])
            raise ValueError(f"JSONL file is empty: {path}")

        # Infer or use explicit schema
        if self._schema:
            schema = self._schema
        else:
            schema = self._infer_schema_from_rows(rows)

        # Convert values to appropriate types
        typed_rows = self._apply_schema_to_rows(rows, schema)

        return self._create_dataframe_from_data(typed_rows)

    # ---------------------------------------------------------------- Parquet reader
    def _read_parquet(self, path: str) -> DataFrame:
        """Read Parquet file and return DataFrame."""
        try:
            import pandas as pd  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pandas. Install with: pip install pandas"
            ) from exc

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pyarrow. Install with: pip install pyarrow"
            ) from exc

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        # Read parquet file
        table = pq.read_table(str(path_obj))
        df = table.to_pandas()

        # Convert to list of dicts
        rows = df.to_dict("records")

        if not rows:
            if self._schema:
                return self._create_dataframe_from_schema([])
            return self._create_dataframe_from_data([])

        # Infer or use explicit schema
        if self._schema:
            schema = self._schema
        else:
            schema = self._infer_schema_from_rows(rows)

        # Convert values to appropriate types
        typed_rows = self._apply_schema_to_rows(rows, schema)

        return self._create_dataframe_from_data(typed_rows)

    # ---------------------------------------------------------------- Text reader
    def _read_text(self, path: str, column_name: str) -> DataFrame:
        """Read text file line-by-line and return DataFrame."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Text file not found: {path}")

        rows: List[Dict[str, object]] = []
        with open(path_obj, "r", encoding="utf-8") as f:
            for line in f:
                rows.append({column_name: line.rstrip("\n\r")})

        if not rows:
            return self._create_dataframe_from_schema([])

        return self._create_dataframe_from_data(rows)

    # ---------------------------------------------------------------- Schema inference
    def _infer_schema_from_rows(self, rows: List[Dict[str, object]]) -> Sequence[ColumnDef]:
        """Infer schema from sample rows."""
        if not rows:
            raise ValueError("Cannot infer schema from empty data")

        sample = rows[0]
        columns: List[ColumnDef] = []

        for key, value in sample.items():
            # Check if any row has None for this column
            has_nulls = any(row.get(key) is None for row in rows)
            # Sample multiple rows to better infer type (especially for string numbers)
            sample_values = [row.get(key) for row in rows[:100] if row.get(key) is not None]
            col_type = self._infer_type_from_values(sample_values, value)
            columns.append(ColumnDef(name=key, type_name=col_type, nullable=has_nulls))

        return columns

    def _infer_type_from_values(self, sample_values: List[object], first_value: object) -> str:
        """Infer SQL type from sample values, trying to parse strings as numbers."""
        if first_value is None:
            return "TEXT"  # Can't infer from None

        # If already a Python type, use it directly
        if isinstance(first_value, bool):
            return "INTEGER"
        if isinstance(first_value, int):
            return "INTEGER"
        if isinstance(first_value, float):
            return "REAL"

        # If it's a string, try to infer type by attempting to parse
        if isinstance(first_value, str):
            # Try to parse as integer
            all_integers = True
            for val in sample_values[:10]:  # Sample first 10 non-null values
                if val is None:
                    continue
                try:
                    int(str(val))
                except (ValueError, TypeError):
                    all_integers = False
                    break

            if all_integers and sample_values:
                return "INTEGER"

            # Try to parse as float
            all_floats = True
            for val in sample_values[:10]:
                if val is None:
                    continue
                try:
                    float(str(val))
                except (ValueError, TypeError):
                    all_floats = False
                    break

            if all_floats and sample_values:
                return "REAL"

        return "TEXT"  # Default fallback

    def _infer_type_from_value(self, value: object) -> str:
        """Infer SQL type from a single Python value (legacy method)."""
        if value is None:
            return "TEXT"
        if isinstance(value, bool):
            return "INTEGER"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "REAL"
        if isinstance(value, str):
            return "TEXT"
        return "TEXT"

    def _apply_schema_to_rows(
        self, rows: List[Dict[str, object]], schema: Sequence[ColumnDef]
    ) -> List[Dict[str, object]]:
        """Apply schema type conversions to rows."""
        typed_rows = []

        for row in rows:
            typed_row: Dict[str, object] = {}
            for col_def in schema:
                value = row.get(col_def.name)
                if value is None:
                    typed_row[col_def.name] = None
                elif col_def.type_name == "INTEGER":
                    try:
                        typed_row[col_def.name] = int(cast(Any, value))
                    except (ValueError, TypeError):
                        typed_row[col_def.name] = value
                elif col_def.type_name == "REAL":
                    try:
                        typed_row[col_def.name] = float(cast(Any, value))
                    except (ValueError, TypeError):
                        typed_row[col_def.name] = value
                else:
                    typed_row[col_def.name] = str(value) if value is not None else None
            typed_rows.append(typed_row)

        return typed_rows

    def _create_dataframe_from_data(self, rows: List[Dict[str, object]]) -> DataFrame:
        """Create DataFrame from materialized data."""
        # For file-based readers, we create a DataFrame with materialized data
        # Since files aren't in SQL, we can't use logical plans
        from ..logical.plan import TableScan

        # Create a DataFrame with materialized data stored in the _materialized_data field
        return DataFrame(
            plan=TableScan(table="__memory__"), database=self._database, _materialized_data=rows
        )

    def _create_dataframe_from_schema(self, rows: List[Dict[str, object]]) -> DataFrame:
        """Create DataFrame with explicit schema but no data."""
        return self._create_dataframe_from_data(rows)

    # ---------------------------------------------------------------- Streaming readers
    def _read_csv_stream(self, path: str) -> DataFrame:
        """Read CSV file in streaming mode (chunked)."""
        chunk_size = int(cast(Any, self._options.get("chunk_size", 10000)))
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        header = cast(bool, self._options.get("header", True))
        delimiter = cast(str, self._options.get("delimiter", ","))
        infer_schema = cast(bool, self._options.get("inferSchema", True))

        def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
            with open(path_obj, "r", encoding="utf-8") as f:
                if header and not self._schema:
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
                    if header and self._schema:
                        next(reader_obj, None)
                    if self._schema:
                        chunk = []
                        for row_data in reader_obj:
                            if not row_data:
                                continue
                            row_dict: Dict[str, object] = {}
                            for i, col_def in enumerate(self._schema):
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
                            "CSV file without header requires explicit schema. Use .schema([ColumnDef(...), ...])"
                        )

        # Read first chunk to infer schema
        first_chunk_gen = _chunk_generator()
        try:
            first_chunk = next(first_chunk_gen)
        except StopIteration:
            if self._schema:
                return self._create_dataframe_from_schema([])
            raise ValueError(f"CSV file is empty: {path}")

        # Infer or use explicit schema
        if self._schema:
            schema = self._schema
        elif infer_schema:
            schema = self._infer_schema_from_rows(first_chunk)
        else:
            schema = [
                ColumnDef(name=col, type_name="TEXT", nullable=True)
                for col in first_chunk[0].keys()
            ]

        # Create generator that applies schema and yields chunks
        def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
            # Yield first chunk (already read)
            yield self._apply_schema_to_rows(first_chunk, schema)
            # Yield remaining chunks
            for chunk in first_chunk_gen:
                yield self._apply_schema_to_rows(chunk, schema)

        return self._create_dataframe_from_stream(_typed_chunk_generator, schema)

    def _read_json_stream(self, path: str) -> DataFrame:
        """Read JSON file in streaming mode (chunked)."""
        chunk_size = int(cast(Any, self._options.get("chunk_size", 10000)))
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        multiline = cast(bool, self._options.get("multiline", False))

        def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
            with open(path_obj, "r", encoding="utf-8") as f:
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
            if self._schema:
                return self._create_dataframe_from_schema([])
            raise ValueError(f"JSON file is empty: {path}")

        # Infer or use explicit schema
        if self._schema:
            schema = self._schema
        else:
            schema = self._infer_schema_from_rows(first_chunk)

        def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
            yield self._apply_schema_to_rows(first_chunk, schema)
            for chunk in first_chunk_gen:
                yield self._apply_schema_to_rows(chunk, schema)

        return self._create_dataframe_from_stream(_typed_chunk_generator, schema)

    def _read_jsonl_stream(self, path: str) -> DataFrame:
        """Read JSONL file in streaming mode (chunked)."""
        chunk_size = int(cast(Any, self._options.get("chunk_size", 10000)))
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
            chunk = []
            with open(path_obj, "r", encoding="utf-8") as f:
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
            if self._schema:
                return self._create_dataframe_from_schema([])
            raise ValueError(f"JSONL file is empty: {path}")

        # Infer or use explicit schema
        if self._schema:
            schema = self._schema
        else:
            schema = self._infer_schema_from_rows(first_chunk)

        def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
            yield self._apply_schema_to_rows(first_chunk, schema)
            for chunk in first_chunk_gen:
                yield self._apply_schema_to_rows(chunk, schema)

        return self._create_dataframe_from_stream(_typed_chunk_generator, schema)

    def _read_parquet_stream(self, path: str) -> DataFrame:
        """Read Parquet file in streaming mode (row group by row group)."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pyarrow. Install with: pip install pyarrow"
            ) from exc

        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        parquet_file = pq.ParquetFile(str(path_obj))

        def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
            for i in range(parquet_file.num_row_groups):
                row_group = parquet_file.read_row_group(i)
                df = row_group.to_pandas()
                rows = df.to_dict("records")
                if rows:
                    yield rows

        # Read first row group to infer schema
        first_chunk_gen = _chunk_generator()
        try:
            first_chunk = next(first_chunk_gen)
        except StopIteration:
            if self._schema:
                return self._create_dataframe_from_schema([])
            return self._create_dataframe_from_data([])

        # Infer or use explicit schema
        if self._schema:
            schema = self._schema
        else:
            schema = self._infer_schema_from_rows(first_chunk)

        def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
            yield self._apply_schema_to_rows(first_chunk, schema)
            for chunk in first_chunk_gen:
                yield self._apply_schema_to_rows(chunk, schema)

        return self._create_dataframe_from_stream(_typed_chunk_generator, schema)

    def _read_text_stream(self, path: str, column_name: str) -> DataFrame:
        """Read text file in streaming mode (chunked)."""
        chunk_size = int(cast(Any, self._options.get("chunk_size", 10000)))
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Text file not found: {path}")

        def _chunk_generator() -> Iterator[List[Dict[str, object]]]:
            chunk: List[Dict[str, object]] = []
            with open(path_obj, "r", encoding="utf-8") as f:
                for line in f:
                    chunk.append({column_name: line.rstrip("\n\r")})
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk

        # Read first chunk
        first_chunk_gen = _chunk_generator()
        try:
            first_chunk = next(first_chunk_gen)
        except StopIteration:
            schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]
            return self._create_dataframe_from_schema([])

        schema = [ColumnDef(name=column_name, type_name="TEXT", nullable=False)]

        def _typed_chunk_generator() -> Iterator[List[Dict[str, object]]]:
            yield cast(List[Dict[str, object]], first_chunk)
            for chunk in first_chunk_gen:
                yield cast(List[Dict[str, object]], chunk)

        return self._create_dataframe_from_stream(_typed_chunk_generator, schema)

    def _create_dataframe_from_stream(
        self,
        chunk_generator: Callable[[], Iterator[List[Dict[str, object]]]],
        schema: Sequence[ColumnDef],
    ) -> DataFrame:
        """Create DataFrame from streaming generator."""
        from ..logical.plan import TableScan

        # Store the generator function and schema in the DataFrame
        # When collect(stream=True) is called, it will use this generator
        return DataFrame(
            plan=TableScan(table="__stream__"),
            database=self._database,
            _stream_generator=chunk_generator,
            _stream_schema=schema,
        )


class FormatReader:
    """Builder for format-specific reads."""

    def __init__(self, reader: DataFrameReader, source: str):
        self._reader = reader
        self._source = source.lower()

    def load(self, path: str) -> DataFrame:
        """Load data from the specified path using the configured format."""
        if self._source == "csv":
            return self._reader.csv(path)
        elif self._source == "json":
            return self._reader.json(path)
        elif self._source == "jsonl":
            return self._reader.jsonl(path)
        elif self._source == "parquet":
            return self._reader.parquet(path)
        elif self._source == "text":
            return self._reader.text(path)
        else:
            raise ValueError(f"Unsupported format: {self._source}")
