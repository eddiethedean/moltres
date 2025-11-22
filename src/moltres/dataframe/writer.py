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
        self._primary_key: Optional[Sequence[str]] = None

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

    def primaryKey(self, *columns: str) -> "DataFrameWriter":
        """Specify primary key columns for the target table.

        Args:
            *columns: Column names to use as primary key

        Returns:
            Self for method chaining
        """
        self._primary_key = columns if columns else None
        return self

    primary_key = primaryKey

    def save_as_table(self, name: str, primary_key: Optional[Sequence[str]] = None) -> None:
        """Write the DataFrame to a table with the given name.

        Args:
            name: Name of the target table
            primary_key: Optional sequence of column names to use as primary key.
                        If provided, overrides any primary key set via .primaryKey()
        """
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        # Use parameter if provided, otherwise use field
        if primary_key is not None:
            self._primary_key = primary_key
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
            chunk_iter = self._df.collect(stream=True)
            for chunk in chunk_iter:
                if chunk:
                    table.insert(chunk)
        else:
            rows = self._df.collect()
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

        # Check if we can use INSERT INTO ... SELECT optimization
        if self._can_use_insert_select():
            # Try to infer schema from plan
            schema = self._schema or self._infer_schema_from_plan()

            if schema is None:
                # Can't infer schema, fall back to materialization
                # Continue with existing materialization code below
                pass
            else:
                # Use optimized path
                self._execute_insert_select(schema)
                return

        # Fall back to existing materialization approach
        # Collect data from DataFrame (we'll use this for both schema inference and insertion)
        chunk_iter: Optional[Iterator[List[Dict[str, object]]]] = None
        if self._stream:
            # For streaming, we need to peek at first chunk for schema inference
            chunk_iter = self._df.collect(stream=True)
            try:
                first_chunk = next(chunk_iter)
                rows = first_chunk
            except StopIteration:
                rows = []
        else:
            rows = self._df.collect()

        # Infer or get schema (uses rows if needed, but allows empty if schema is explicit)
        try:
            schema = self._infer_or_get_schema(rows)
        except ValueError as e:
            # Check if this is a primary key validation error - if so, re-raise it
            if "Primary key columns" in str(e) or "do not exist in schema" in str(e):
                raise
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

    def _infer_schema_from_plan(self) -> Optional[Sequence[ColumnDef]]:
        """Infer schema from DataFrame plan without materializing data.

        Returns:
            Inferred schema or None if inference is not possible
        """
        if self._df.database is None:
            return None

        try:
            from ..sql.compiler import compile_plan

            # Compile the plan to a SELECT statement
            select_stmt = compile_plan(self._df.plan, dialect=self._df.database.dialect)

            # Execute the query with LIMIT 1 to get a sample row for schema inference
            # This is much more efficient than materializing all data
            sample_stmt = select_stmt.limit(1)
            sample_result = self._df.database.executor.fetch(sample_stmt)

            # Get column names from the result
            # The result.rows is a list of dicts, so we can get column names from keys
            if sample_result.rows and len(sample_result.rows) > 0:
                sample_row = sample_result.rows[0]
                column_names = list(sample_row.keys())

                # Infer types from sample data
                column_defs = []
                for col_name in column_names:
                    value = sample_row.get(col_name)
                    col_type = self._infer_type_from_value(value)
                    # Check if column is nullable (if value is None, it's nullable)
                    nullable = value is None
                    column_defs.append(
                        ColumnDef(name=col_name, type_name=col_type, nullable=nullable)
                    )
                return column_defs
            else:
                # No rows returned, but we can still infer schema from the SELECT statement
                # Extract column names from the SELECT statement
                column_names = []
                for col_expr in select_stmt.selected_columns:
                    # Try to get column name from the expression
                    if hasattr(col_expr, "name") and col_expr.name:
                        column_names.append(col_expr.name)
                    elif hasattr(col_expr, "key") and col_expr.key:
                        column_names.append(col_expr.key)
                    elif hasattr(col_expr, "_label") and col_expr._label:
                        column_names.append(col_expr._label)
                    else:
                        # Fallback: use string representation and try to extract name
                        col_str = str(col_expr)
                        # Remove quotes and extract name
                        col_str = col_str.strip("\"'")
                        if "." in col_str:
                            column_names.append(col_str.split(".")[-1])
                        else:
                            column_names.append(col_str)

                if not column_names:
                    return None

                # Use TEXT as default type for empty result sets
                return [
                    ColumnDef(name=col_name, type_name="TEXT", nullable=True)
                    for col_name in column_names
                ]

        except Exception:
            # If inference fails, return None to fall back to materialization
            return None

    def _execute_insert_select(self, schema: Sequence[ColumnDef]) -> None:
        """Execute write using INSERT INTO ... SELECT optimization."""
        if self._df.database is None:
            raise RuntimeError("Cannot write DataFrame without an attached Database")

        db = self._df.database
        table_name = self._table_name
        if table_name is None:
            raise ValueError("Table name must be specified via save_as_table()")

        # Apply primary key flags to schema if specified
        final_schema = list(schema)
        if self._primary_key:
            # Validate that all primary key columns exist
            schema_column_names = {col.name for col in final_schema}
            missing_cols = [col for col in self._primary_key if col not in schema_column_names]
            if missing_cols:
                raise ValueError(
                    f"Primary key columns {missing_cols} do not exist in schema. "
                    f"Available columns: {sorted(schema_column_names)}"
                )

            # Apply primary key flags
            primary_key_set = set(self._primary_key)
            final_schema = [
                ColumnDef(
                    name=col.name,
                    type_name=col.type_name,
                    nullable=col.nullable,
                    default=col.default,
                    primary_key=col.name in primary_key_set,
                )
                for col in final_schema
            ]

        # Check if table exists
        table_exists = self._table_exists(db, table_name)

        # Handle overwrite mode
        if self._mode == "overwrite":
            if table_exists:
                db.drop_table(table_name, if_exists=True)
            db.create_table(table_name, final_schema, if_not_exists=False)
        elif not table_exists:
            # Create table if it doesn't exist
            db.create_table(table_name, final_schema, if_not_exists=True)

        # Compile DataFrame plan to SELECT statement
        from ..sql.compiler import compile_plan

        select_stmt = compile_plan(self._df.plan, dialect=db.dialect)

        # Get column names from schema
        column_names = [col.name for col in final_schema]

        # Compile INSERT INTO ... SELECT statement
        from ..sql.ddl import compile_insert_select

        insert_sql = compile_insert_select(
            target_table=table_name,
            select_stmt=select_stmt,
            dialect=db.dialect,
            columns=column_names,
        )

        # Execute the compiled SQL directly
        db.executor.execute(insert_sql)

    def _can_use_insert_select(self) -> bool:
        """Check if we can use INSERT INTO ... SELECT optimization.

        Returns:
            True if optimization is possible, False otherwise
        """
        # DataFrame must have a database connection
        if self._df.database is None:
            return False

        # Not in streaming mode (streaming requires materialization for chunking)
        if self._stream:
            return False

        # Mode must not be "error_if_exists" (need to check table existence first,
        # which requires materialization path)
        if self._mode == "error_if_exists":
            return False

        # Plan must be compilable to SQL (all operations are SQL-compatible)
        # We can check this by trying to compile the plan
        try:
            from ..sql.compiler import compile_plan

            compile_plan(self._df.plan, dialect=self._df.database.dialect)
            return True
        except Exception:
            # If compilation fails, can't use optimization
            return False

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
            schema = list(self._schema)
        else:
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

            schema = columns

        # Apply primary key flags if specified
        if self._primary_key:
            # Validate that all primary key columns exist
            schema_column_names = {col.name for col in schema}
            missing_cols = [col for col in self._primary_key if col not in schema_column_names]
            if missing_cols:
                raise ValueError(
                    f"Primary key columns {missing_cols} do not exist in schema. "
                    f"Available columns: {sorted(schema_column_names)}"
                )

            # Apply primary key flags
            primary_key_set = set(self._primary_key)
            schema = [
                ColumnDef(
                    name=col.name,
                    type_name=col.type_name,
                    nullable=col.nullable,
                    default=col.default,
                    primary_key=col.name in primary_key_set,
                )
                for col in schema
            ]

        return schema

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

        rows = self._df.collect()
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
            writer = csv.DictWriter(f, fieldnames=headers, delimiter=delimiter)
            if header:
                writer.writeheader()
            writer.writerows(rows)

    def _save_csv_stream(self, path: str) -> None:
        """Save DataFrame as CSV file in streaming mode."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        header = self._options.get("header", True)
        delimiter = self._options.get("delimiter", ",")

        chunk_iter = self._df.collect(stream=True)
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
                    writer.writerows(chunk)

    def _save_json(self, path: str) -> None:
        """Save DataFrame as JSON file (array of objects)."""
        rows = self._df.collect()

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

        rows = self._df.collect()

        # Handle partitioning
        if self._partition_by:
            self._save_partitioned(path, "jsonl", rows, None)
            return

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_jsonl_stream(self, path: str) -> None:
        """Save DataFrame as JSONL file in streaming mode."""
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            chunk_iter = self._df.collect(stream=True)
            for chunk in chunk_iter:
                for row in chunk:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _save_parquet(self, path: str) -> None:
        """Save DataFrame as Parquet file."""
        try:
            import pandas as pd  # type: ignore[import-untyped]
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

        rows = self._df.collect()

        # Handle partitioning
        if self._partition_by:
            self._save_partitioned(path, "parquet", rows, None)
            return

        # Convert to pandas DataFrame
        df = pd.DataFrame(rows)

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
