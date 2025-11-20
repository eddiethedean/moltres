"""DataFrame write operations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Union, cast

from ..table.schema import ColumnDef
from .dataframe import DataFrame

if TYPE_CHECKING:
    from ..table.table import Database


class DataFrameWriter:
    """Builder for writing DataFrames to tables."""

    def __init__(self, df: DataFrame):
        self._df = df
        self._mode: str = "append"
        self._table_name: Optional[str] = None
        self._schema: Optional[Sequence[ColumnDef]] = None
        self._options: Dict[str, object] = {}
        self._partition_by: Optional[Sequence[str]] = None
        self._stream: bool = False

    def mode(self, mode: str) -> "DataFrameWriter":
        """Set the write mode: 'append', 'overwrite', or 'error_if_exists'."""
        if mode not in ("append", "overwrite", "error_if_exists"):
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: append, overwrite, error_if_exists"
            )
        self._mode = mode
        return self

    def option(self, key: str, value: object) -> "DataFrameWriter":
        """Set a write option (e.g., header=True for CSV, compression='gzip' for Parquet)."""
        self._options[key] = value
        return self

    def stream(self, enabled: bool = True) -> "DataFrameWriter":
        """Enable or disable streaming mode (chunked writing for large DataFrames)."""
        self._stream = enabled
        return self

    def partitionBy(self, *columns: str) -> "DataFrameWriter":
        """Partition data by the given columns when writing to files."""
        self._partition_by = columns if columns else None
        return self

    partition_by = partitionBy

    def schema(self, schema: Sequence[ColumnDef]) -> "DataFrameWriter":
        """Set an explicit schema for the target table."""
        self._schema = schema
        return self

    def save_as_table(self, name: str) -> None:
        """Write the DataFrame to a table with the given name."""
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        self._table_name = name
        self._execute_write()

    def insertInto(self, table_name: str) -> None:
        """Insert DataFrame into an existing table (table must already exist)."""
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        db = self._df.database
        if not self._table_exists(db, table_name):
            raise ValueError(
                f"Table '{table_name}' does not exist. Use save_as_table() to create it."
            )

        if self._stream:
            # Stream inserts in batches
            table = db.table(table_name)
            chunk_iter = cast(Iterator[List[Dict[str, object]]], self._df.collect(stream=True))
            for chunk in chunk_iter:
                if chunk:
                    table.insert(chunk)
        else:
            rows = cast(List[Dict[str, object]], self._df.collect())
            if rows:
                table = db.table(table_name)
                table.insert(rows)

    insert_into = insertInto

    def save(self, path: str, format: Optional[str] = None) -> None:
        """Save DataFrame to a file or directory in the specified format."""
        if format is None:
            # Infer format from file extension
            ext = Path(path).suffix.lower()
            format_map = {
                ".csv": "csv",
                ".json": "json",
                ".jsonl": "jsonl",
                ".parquet": "parquet",
            }
            format = format_map.get(ext, "csv")
            if format is None:
                raise ValueError(
                    f"Cannot infer format from path '{path}'. Specify format explicitly."
                )

        format = format.lower()
        if format == "csv":
            self._save_csv(path)
        elif format == "json":
            self._save_json(path)
        elif format == "jsonl":
            self._save_jsonl(path)
        elif format == "parquet":
            self._save_parquet(path)
        else:
            raise ValueError(f"Unsupported format '{format}'. Supported: csv, json, jsonl, parquet")

    def csv(self, path: str) -> None:
        """Save DataFrame as CSV file."""
        self._save_csv(path)

    def json(self, path: str) -> None:
        """Save DataFrame as JSON file (array of objects)."""
        self._save_json(path)

    def jsonl(self, path: str) -> None:
        """Save DataFrame as JSONL file (one JSON object per line)."""
        self._save_jsonl(path)

    def parquet(self, path: str) -> None:
        """Save DataFrame as Parquet file."""
        self._save_parquet(path)

    def _execute_write(self) -> None:
        """Execute the write operation based on mode and table existence."""
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        db = self._df.database
        table_name = self._table_name
        if table_name is None:
            raise ValueError("Table name must be specified via save_as_table()")

        # Check if table exists
        table_exists = self._table_exists(db, table_name)

        if self._mode == "error_if_exists" and table_exists:
            raise ValueError(f"Table '{table_name}' already exists and mode is 'error_if_exists'")

        # Collect data from DataFrame (we'll use this for both schema inference and insertion)
        chunk_iter: Optional[Iterator[List[Dict[str, object]]]] = None
        if self._stream:
            # For streaming, we need to peek at first chunk for schema inference
            chunk_iter = cast(Iterator[List[Dict[str, object]]], self._df.collect(stream=True))
            try:
                first_chunk = next(chunk_iter)
                rows = first_chunk
            except StopIteration:
                rows = []
        else:
            rows = cast(List[Dict[str, object]], self._df.collect())

        # Infer or get schema (uses rows if needed, but allows empty if schema is explicit)
        try:
            schema = self._infer_or_get_schema(rows)
        except ValueError:
            # Empty DataFrame without explicit schema - try to infer from plan
            if self._schema is None:
                # For empty DataFrames, we can't reliably infer, so require explicit schema
                raise ValueError(
                    "Cannot infer schema from empty DataFrame. "
                    "Provide explicit schema via .schema([ColumnDef(...), ...])"
                )
            schema = self._schema

        # Handle overwrite mode
        if self._mode == "overwrite":
            if table_exists:
                db.drop_table(table_name, if_exists=True)
            db.create_table(table_name, schema, if_not_exists=False)
        elif not table_exists:
            # Create table if it doesn't exist
            db.create_table(table_name, schema, if_not_exists=True)

        # Insert data (if any)
        if self._stream:
            # Streaming write: insert first chunk, then remaining chunks
            table = db.table(table_name)
            if rows:  # First chunk already read
                table.insert(rows)
            # Insert remaining chunks
            if chunk_iter is not None:
                for chunk in chunk_iter:
                    if chunk:
                        table.insert(chunk)
        elif rows:
            table = db.table(table_name)
            table.insert(rows)

    def _table_exists(self, db: "Database", table_name: str) -> bool:
        """Check if a table exists in the database."""
        try:
            # Use a simple query that should work across dialects
            if db.dialect.name == "sqlite":
                sql = "SELECT name FROM sqlite_master WHERE type='table' AND name=:name"
                result = db.execute_sql(sql, params={"name": table_name})
                return len(result.rows) > 0
            elif db.dialect.name == "postgresql":
                sql = "SELECT tablename FROM pg_tables WHERE tablename=:name"
                result = db.execute_sql(sql, params={"name": table_name})
                return len(result.rows) > 0
            else:
                # Generic approach: try to select from the table with LIMIT 0
                quote = db.dialect.quote_char
                sql = f"SELECT * FROM {quote}{table_name}{quote} LIMIT 0"
                db.execute_sql(sql)
                return True
        except Exception:
            return False

    def _infer_or_get_schema(self, rows: List[dict[str, object]]) -> Sequence[ColumnDef]:
        """Infer schema from DataFrame plan or use provided schema."""
        if self._schema is not None:
            return self._schema

        # Infer schema from collected data (most reliable)
        if not rows:
            raise ValueError(
                "Cannot infer schema from empty DataFrame. Provide explicit schema via .schema()"
            )

        # Use first row to infer types, but check all rows for None values
        sample = rows[0]
        columns: List[ColumnDef] = []

        for key, value in sample.items():
            # Check if any row has None for this column
            has_nulls = any(row.get(key) is None for row in rows)
            col_type = self._infer_type_from_value(value)
            columns.append(ColumnDef(name=key, type_name=col_type, nullable=has_nulls))

        return columns

    def _infer_type_from_value(self, value: object) -> str:
        """Infer SQL type from a Python value."""
        if value is None:
            return "TEXT"  # Can't infer from None
        if isinstance(value, bool):
            return "INTEGER"
        if isinstance(value, int):
            return "INTEGER"
        if isinstance(value, float):
            return "REAL"
        if isinstance(value, str):
            return "TEXT"
        return "TEXT"  # Default fallback

    def _save_csv(self, path: str) -> None:
        """Save DataFrame as CSV file."""
        if self._stream:
            self._save_csv_stream(path)
            return

        rows = cast(List[Dict[str, object]], self._df.collect())
        if not rows:
            # Create empty file with headers if we have schema
            if self._schema:
                headers = [col.name for col in self._schema]
            else:
                headers = []
        else:
            headers = list(rows[0].keys())

        # Handle partitioning
        if self._partition_by:
            self._save_partitioned(path, "csv", rows, headers)
            return

        # Write CSV file
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        header = cast(bool, self._options.get("header", True))
        delimiter = cast(str, self._options.get("delimiter", ","))

        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=cast(str, delimiter))
            if header:
                writer.writeheader()
            writer.writerows(rows)

    def _save_csv_stream(self, path: str) -> None:
        """Save DataFrame as CSV file in streaming mode."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        header = self._options.get("header", True)
        delimiter = self._options.get("delimiter", ",")

        chunk_iter = cast(Iterator[List[Dict[str, object]]], self._df.collect(stream=True))
        first_chunk: List[Dict[str, object]] = []
        try:
            first_chunk = next(chunk_iter)
        except StopIteration:
            pass

        # Determine headers
        if first_chunk:
            headers = list(first_chunk[0].keys())
        elif self._schema:
            headers = [col.name for col in self._schema]
        else:
            headers = []

        with open(path_obj, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=cast(str, delimiter))
            if header:
                writer.writeheader()
            if first_chunk:
                writer.writerows(first_chunk)
            for chunk in chunk_iter:
                if chunk:
                    writer.writerows(cast(List[Dict[str, object]], chunk))

    def _save_json(self, path: str) -> None:
        """Save DataFrame as JSON file (array of objects)."""
        rows = cast(List[Dict[str, object]], self._df.collect())

        # Handle partitioning
        if self._partition_by:
            self._save_partitioned(path, "json", rows, None)
            return

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            indent_val = self._options.get("indent", None)
            indent: Optional[Union[int, str]] = cast(Optional[Union[int, str]], indent_val)
            json.dump(rows, f, indent=indent, ensure_ascii=False)

    def _save_jsonl(self, path: str) -> None:
        """Save DataFrame as JSONL file (one JSON object per line)."""
        if self._stream:
            self._save_jsonl_stream(path)
            return

        rows = cast(List[Dict[str, object]], self._df.collect())

        # Handle partitioning
        if self._partition_by:
            self._save_partitioned(path, "jsonl", rows, None)
            return

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        rows_list = cast(List[Dict[str, object]], rows)
        with open(path_obj, "w", encoding="utf-8") as f:
            for row in rows_list:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_jsonl_stream(self, path: str) -> None:
        """Save DataFrame as JSONL file in streaming mode."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            chunk_iter = cast(Iterator[List[Dict[str, object]]], self._df.collect(stream=True))
            for chunk in chunk_iter:
                for row in chunk:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_parquet(self, path: str) -> None:
        """Save DataFrame as Parquet file."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pandas. Install with: pip install pandas"
            ) from exc

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Parquet format requires pyarrow. Install with: pip install pyarrow"
            ) from exc

        rows = cast(List[Dict[str, object]], self._df.collect())

        # Handle partitioning
        if self._partition_by:
            self._save_partitioned(path, "parquet", rows, None)
            return

        # Convert to pandas DataFrame
        df = pd.DataFrame(rows)  # type: ignore[call-overload]

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Get compression option
        compression = cast(str, self._options.get("compression", "snappy"))

        # Write parquet file
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path_obj), compression=compression)

    def _save_partitioned(
        self,
        base_path: str,
        format: str,
        rows: List[dict[str, object]],
        headers: Optional[List[str]],
    ) -> None:
        """Save data partitioned by specified columns."""
        if not rows:
            return

        # Group rows by partition columns
        partitions: Dict[tuple, List[dict[str, object]]] = {}
        partition_cols = self._partition_by or []
        for row in rows:
            partition_key = tuple(str(row.get(col, "")) for col in partition_cols)
            if partition_key not in partitions:
                partitions[partition_key] = []
            partitions[partition_key].append(row)

        # Write each partition to a subdirectory
        base_path_obj = Path(base_path)
        base_path_obj.parent.mkdir(parents=True, exist_ok=True)

        for partition_key, partition_rows in partitions.items():
            # Create partition directory path
            partition_parts = [f"{col}={val}" for col, val in zip(partition_cols, partition_key)]
            partition_dir = base_path_obj.parent / base_path_obj.stem
            for part in partition_parts:
                partition_dir = partition_dir / part
            partition_dir.mkdir(parents=True, exist_ok=True)

            # Determine filename
            if format == "csv":
                filename = partition_dir / "data.csv"
                self._write_csv_file(filename, partition_rows, headers)
            elif format == "json":
                filename = partition_dir / "data.json"
                self._write_json_file(filename, partition_rows)
            elif format == "jsonl":
                filename = partition_dir / "data.jsonl"
                self._write_jsonl_file(filename, partition_rows)
            elif format == "parquet":
                filename = partition_dir / "data.parquet"
                self._write_parquet_file(filename, partition_rows)

    def _write_csv_file(
        self, path: Path, rows: List[dict[str, object]], headers: Optional[List[str]]
    ) -> None:
        """Helper to write CSV file."""
        if not rows and headers:
            headers_to_use = headers
        elif rows:
            headers_to_use = list(rows[0].keys())
        else:
            headers_to_use = []

        header = cast(bool, self._options.get("header", True))
        delimiter = cast(str, self._options.get("delimiter", ","))

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers_to_use, delimiter=delimiter)
            if header:
                writer.writeheader()
            writer.writerows(rows)

    def _write_json_file(self, path: Path, rows: List[dict[str, object]]) -> None:
        """Helper to write JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            indent_val = self._options.get("indent", None)
            indent: Optional[Union[int, str]] = cast(Optional[Union[int, str]], indent_val)
            json.dump(rows, f, indent=indent, ensure_ascii=False)

    def _write_jsonl_file(self, path: Path, rows: List[dict[str, object]]) -> None:
        """Helper to write JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _write_parquet_file(self, path: Path, rows: List[dict[str, object]]) -> None:
        """Helper to write Parquet file."""
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        df = pd.DataFrame(rows)
        compression = cast(str, self._options.get("compression", "snappy"))
        table = pa.Table.from_pandas(df)
        pq.write_table(table, str(path), compression=compression)
