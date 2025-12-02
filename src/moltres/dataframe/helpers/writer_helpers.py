"""Common helper functions for :class:`DataFrame` writer implementations.

This module contains shared logic used by both :class:`DataFrameWriter` and AsyncDataFrameWriter
to reduce code duplication.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, TypeVar

from ...table.schema import ColumnDef

if TYPE_CHECKING:
    from sqlalchemy.sql import Select
    from typing import Protocol

logger = logging.getLogger(__name__)

# Type variable for generic Writer operations
T = TypeVar("T", bound=Any)

if TYPE_CHECKING:

    class WriterProtocol(Protocol):
        """Protocol defining the interface that Writer classes must implement."""

        _mode: str
        _options: Dict[str, object]
        _format: Optional[str]
        _stream_override: Optional[bool]
        _partition_by: Optional[Sequence[str]]
        _schema: Optional[Sequence[ColumnDef]]
        _primary_key: Optional[Sequence[str]]
        _bucket_by: Optional[tuple[int, Sequence[str]]]
        _sort_by: Optional[Sequence[str]]

else:
    WriterProtocol = Any


def infer_type_from_value(value: object) -> str:
    """Infer SQL type from a Python value.

    Args:
        value: Python value to infer type from

    Returns:
        SQL type name string (e.g., "INTEGER", "REAL", "TEXT")
    """
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


def extract_column_names_from_select(select_stmt: "Select") -> List[str]:
    """Extract column names from a SQLAlchemy Select statement.

    Args:
        select_stmt: SQLAlchemy Select statement

    Returns:
        List of column name strings
    """
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
    return column_names


def infer_schema_from_sample_row(
    sample_row: Dict[str, object], column_names: Optional[List[str]] = None
) -> List[ColumnDef]:
    """Infer schema from a sample row of data.

    Args:
        sample_row: Dictionary representing a sample row
        column_names: Optional list of column names. If None, uses keys from sample_row.

    Returns:
        List of ColumnDef objects
    """
    if column_names is None:
        column_names = list(sample_row.keys())

    column_defs = []
    for col_name in column_names:
        value = sample_row.get(col_name)
        col_type = infer_type_from_value(value)
        # Check if column is nullable (if value is None, it's nullable)
        nullable = value is None
        column_defs.append(ColumnDef(name=col_name, type_name=col_type, nullable=nullable))
    return column_defs


def infer_schema_from_select_stmt(select_stmt: "Select") -> Optional[List[ColumnDef]]:
    """Infer schema from a SELECT statement when no rows are returned.

    Args:
        select_stmt: SQLAlchemy Select statement

    Returns:
        List of ColumnDef objects with TEXT type, or None if no columns found
    """
    column_names = extract_column_names_from_select(select_stmt)
    if not column_names:
        return None

    # Use TEXT as default type for empty result sets
    return [ColumnDef(name=col_name, type_name="TEXT", nullable=True) for col_name in column_names]


def ensure_file_layout_supported(
    bucket_by: Optional[tuple[int, Sequence[str]]], sort_by: Optional[Sequence[str]]
) -> None:
    """Raise if unsupported bucketing/sorting metadata is set for file sinks.

    Args:
        bucket_by: Optional tuple of (num_buckets, columns) for bucketing
        sort_by: Optional sequence of column names for sorting

    Raises:
        NotImplementedError: If bucket_by or sort_by is set
    """
    if bucket_by or sort_by:
        raise NotImplementedError(
            "bucketBy/sortBy metadata is not yet supported when writing to files. "
            "Alternative: Sort data using orderBy() before writing, or use partitioned writes with partitionBy(). "
            "See https://github.com/eddiethedean/moltres/issues for feature requests."
        )


def prepare_file_target(path_obj: Path, mode: str) -> bool:
    """Apply mode semantics (overwrite/ignore/error) for file outputs.

    Args:
        path_obj: Path object for the target file/directory
        mode: Write mode ("append", "overwrite", "ignore", "error_if_exists")

    Returns:
        True if write should proceed, False if should be skipped (ignore mode)

    Raises:
        ValueError: If mode is "error_if_exists" and path exists
    """
    import shutil

    if mode == "ignore" and path_obj.exists():
        return False
    if mode == "error_if_exists" and path_obj.exists():
        raise ValueError(f"Target '{path_obj}' already exists (mode=error_if_exists)")
    if mode == "overwrite" and path_obj.exists():
        if path_obj.is_dir():
            shutil.rmtree(path_obj)
        else:
            path_obj.unlink()
    return True


def can_use_insert_select(
    has_database: bool,
    stream_override: Optional[bool],
    mode: str,
    plan_compilable: bool,
) -> bool:
    """Check if we can use INSERT INTO ... SELECT optimization.

    Args:
        has_database: Whether :class:`DataFrame` has a database connection
        stream_override: Whether streaming is explicitly enabled/disabled
        mode: Write mode string
        plan_compilable: Whether the plan can be compiled to SQL

    Returns:
        True if optimization is possible, False otherwise
    """
    # DataFrame must have a database connection
    if not has_database:
        return False

    # Not in streaming mode (streaming requires materialization for chunking)
    if stream_override:
        return False

    # Mode must not be "error_if_exists" (need to check table existence first,
    # which requires materialization path)
    if mode == "error_if_exists":
        return False

    # Plan must be compilable to SQL
    return plan_compilable


def apply_primary_key_to_schema(
    schema: Sequence[ColumnDef], primary_key: Optional[Sequence[str]]
) -> List[ColumnDef]:
    """Apply primary key flags to schema if specified.

    Args:
        schema: Original schema definition
        primary_key: Optional sequence of column names to use as primary key

    Returns:
        New schema with primary key flags applied

    Raises:
        ValueError: If primary key columns don't exist in schema
    """
    if not primary_key:
        return list(schema)

    # Validate that all primary key columns exist
    schema_column_names = {col.name for col in schema}
    missing_cols = [col for col in primary_key if col not in schema_column_names]
    if missing_cols:
        raise ValueError(
            f"Primary key columns {missing_cols} do not exist in schema. "
            f"Available columns: {sorted(schema_column_names)}"
        )

    # Apply primary key flags
    primary_key_set = set(primary_key)
    return [
        ColumnDef(
            name=col.name,
            type_name=col.type_name,
            nullable=col.nullable,
            default=col.default,
            primary_key=col.name in primary_key_set,
        )
        for col in schema
    ]


def infer_or_get_schema(
    rows: List[Dict[str, object]],
    explicit_schema: Optional[Sequence[ColumnDef]],
    primary_key: Optional[Sequence[str]],
    force_nullable: bool = False,
) -> Sequence[ColumnDef]:
    """Infer schema from rows or use provided schema, applying primary key flags.

    Args:
        rows: List of row dictionaries
        explicit_schema: Optional explicit schema to use
        primary_key: Optional primary key columns
        force_nullable: If True, mark all columns as nullable

    Returns:
        Final schema with primary key flags applied

    Raises:
        ValueError: If rows are empty and no explicit schema provided
    """
    if explicit_schema is not None:
        schema = list(explicit_schema)
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
            # Check if any row has None for this column (or force nullable when streaming)
            has_nulls = True if force_nullable else any(row.get(key) is None for row in rows)
            col_type = infer_type_from_value(value)
            columns.append(ColumnDef(name=key, type_name=col_type, nullable=has_nulls))

        schema = columns

    # Apply primary key flags if specified
    return apply_primary_key_to_schema(schema, primary_key)


def build_mode_setter(writer: T, mode: str) -> T:
    """Normalize and set write mode for a Writer.

    Args:
        writer: Writer instance
        mode: Write mode string to normalize

    Returns:
        The same writer instance (for method chaining)

    Raises:
        ValueError: If mode is invalid
    """
    normalized = mode.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if normalized == "errorifexists":
        normalized = "error_if_exists"
    elif normalized == "ignore":
        normalized = "ignore"
    elif normalized not in {"append", "overwrite"}:
        if normalized != "error_if_exists":
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: append, overwrite, ignore, error_if_exists"
            )
    writer._mode = normalized
    return writer


def build_option_setter(writer: T, key: str, value: object) -> T:
    """Set a single write option for a Writer.

    Args:
        writer: Writer instance
        key: Option key
        value: Option value

    Returns:
        The same writer instance (for method chaining)
    """
    writer._options[key] = value
    return writer


def build_options_setter(writer: T, *args: Mapping[str, object], **kwargs: object) -> T:
    """Set multiple write options for a Writer.

    Args:
        writer: Writer instance
        *args: Optional single positional mapping argument
        **kwargs: Keyword arguments as options

    Returns:
        The same writer instance (for method chaining)

    Raises:
        TypeError: If more than one positional argument is provided
    """
    if len(args) > 1:
        raise TypeError("options() accepts at most one positional mapping argument")
    if args:
        for key, value in args[0].items():
            writer._options[str(key)] = value
    for key, value in kwargs.items():
        writer._options[key] = value
    return writer


def build_format_setter(writer: T, format_name: str) -> T:
    """Set output format for a Writer.

    Args:
        writer: Writer instance
        format_name: Format name to normalize

    Returns:
        The same writer instance (for method chaining)
    """
    writer._format = format_name.strip().lower()
    return writer


def build_stream_setter(writer: T, enabled: bool = True) -> T:
    """Set streaming mode for a Writer.

    Args:
        writer: Writer instance
        enabled: Whether streaming is enabled

    Returns:
        The same writer instance (for method chaining)
    """
    writer._stream_override = bool(enabled)
    return writer


def build_partition_by_setter(writer: T, *columns: str) -> T:
    """Set partition columns for a Writer.

    Args:
        writer: Writer instance
        *columns: Column names to partition by

    Returns:
        The same writer instance (for method chaining)
    """
    writer._partition_by = columns if columns else None
    return writer


def build_schema_setter(writer: T, schema: Sequence[ColumnDef]) -> T:
    """Set explicit schema for a Writer.

    Args:
        writer: Writer instance
        schema: Schema definition

    Returns:
        The same writer instance (for method chaining)
    """
    writer._schema = schema
    return writer


def build_primary_key_setter(writer: T, *columns: str) -> T:
    """Set primary key columns for a Writer.

    Args:
        writer: Writer instance
        *columns: Column names to use as primary key

    Returns:
        The same writer instance (for method chaining)
    """
    writer._primary_key = columns if columns else None
    return writer


def build_bucket_by_setter(writer: T, num_buckets: int, *columns: str) -> T:
    """Set bucketing configuration for a Writer.

    Args:
        writer: Writer instance
        num_buckets: Number of buckets
        *columns: Column names to bucket by

    Returns:
        The same writer instance (for method chaining)

    Raises:
        ValueError: If num_buckets is invalid or no columns provided
    """
    if num_buckets <= 0:
        raise ValueError("num_buckets must be a positive integer")
    if not columns:
        raise ValueError("bucketBy() requires at least one column name")
    writer._bucket_by = (num_buckets, tuple(columns))
    return writer


def build_sort_by_setter(writer: T, *columns: str) -> T:
    """Set sort columns for a Writer.

    Args:
        writer: Writer instance
        *columns: Column names to sort by

    Returns:
        The same writer instance (for method chaining)

    Raises:
        ValueError: If no columns provided
    """
    if not columns:
        raise ValueError("sortBy() requires at least one column name")
    writer._sort_by = tuple(columns)
    return writer
