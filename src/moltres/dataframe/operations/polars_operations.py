"""Common operations for Polars-style :class:`DataFrame` interfaces.

This module contains shared logic used by both :class:`PolarsDataFrame` and
AsyncPolarsDataFrame to reduce duplication and improve maintainability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    pass


def sql_type_to_polars_dtype(sql_type: str) -> str:
    """Map SQL type names to Polars dtype strings.

    Args:
        sql_type: SQL type name (e.g., "INTEGER", "TEXT", "VARCHAR(255)", "REAL")

    Returns:
        Polars dtype string (e.g., "Int64", "Utf8", "Float64")

    Example:
        >>> sql_type_to_polars_dtype("INTEGER")
        'Int64'
        >>> sql_type_to_polars_dtype("TEXT")
        'Utf8'
        >>> sql_type_to_polars_dtype("REAL")
        'Float64'
    """
    # Normalize the type name - remove parameters and convert to uppercase
    type_upper = sql_type.upper().strip()
    # Remove parameters if present (e.g., "VARCHAR(255)" -> "VARCHAR")
    if "(" in type_upper:
        type_upper = type_upper.split("(")[0].strip()

    # Remove parentheses suffix if present
    type_upper = type_upper.replace("()", "")

    # Map SQL types to Polars dtypes
    type_mapping: Dict[str, str] = {
        # Integer types
        "INTEGER": "Int64",
        "INT": "Int64",
        "BIGINT": "Int64",
        "SMALLINT": "Int32",
        "TINYINT": "Int8",
        "SERIAL": "Int64",
        "BIGSERIAL": "Int64",
        # Floating point types
        "REAL": "Float64",
        "FLOAT": "Float64",
        "DOUBLE": "Float64",
        "DOUBLE PRECISION": "Float64",
        "NUMERIC": "Float64",
        "DECIMAL": "Float64",
        "MONEY": "Float64",
        # Text types
        "TEXT": "Utf8",
        "VARCHAR": "Utf8",
        "CHAR": "Utf8",
        "CHARACTER": "Utf8",
        "STRING": "Utf8",
        "CLOB": "Utf8",
        # Binary types
        "BLOB": "Binary",
        "BYTEA": "Binary",
        "BINARY": "Binary",
        "VARBINARY": "Binary",
        # Boolean
        "BOOLEAN": "Boolean",
        "BOOL": "Boolean",
        # Date/Time types
        "DATE": "Date",
        "TIME": "Time",
        "TIMESTAMP": "Datetime",
        "DATETIME": "Datetime",
        "TIMESTAMPTZ": "Datetime",
        # JSON types
        "JSON": "Object",
        "JSONB": "Object",
        # UUID
        "UUID": "Utf8",
    }

    # Return mapped type or default to 'Utf8' for unknown types
    return type_mapping.get(type_upper, "Utf8")


def validate_columns_exist(
    column_names: Sequence[str],
    available_columns: Sequence[str],
    operation: str = "operation",
) -> None:
    """Validate that all specified columns exist in the :class:`DataFrame`.

    Args:
        column_names: List of column names to validate
        available_columns: List of available column names
        operation: Name of the operation being performed (for error messages)

    Raises:
        ValueError: If any column does not exist
    """
    from ...utils.validation import validate_columns_exist as validate

    available_set = set(available_columns)
    if available_set:
        validate(column_names, available_set, operation)


def normalize_join_how(how: str) -> str:
    """Normalize Polars join 'how' parameter to join type.

    Args:
        how: Join type string ('inner', 'left', 'right', 'outer', 'anti', 'semi', 'full', 'full_outer')

    Returns:
        Normalized join type ('inner', 'left', 'right', 'outer', 'anti', 'semi')
    """
    how_map = {
        "inner": "inner",
        "left": "left",
        "right": "right",
        "outer": "outer",
        "full": "outer",
        "full_outer": "outer",
        "anti": "anti",  # Polars-specific
        "semi": "semi",  # Polars-specific
    }
    return how_map.get(how.lower(), "inner")


def prepare_polars_join_keys(
    on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]],
    left_on: Optional[Union[str, Sequence[str]]],
    right_on: Optional[Union[str, Sequence[str]]],
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    left_validate_fn: Any,
    right_validate_fn: Any,
) -> List[Tuple[str, str]]:
    """Prepare join keys for Polars-style join operations (string-based).

    Args:
        on: :class:`Column` name(s) or list of tuples to join on
        left_on: :class:`Column` name(s) in left :class:`DataFrame`
        right_on: :class:`Column` name(s) in right :class:`DataFrame`
        left_columns: Available columns in left :class:`DataFrame`
        right_columns: Available columns in right :class:`DataFrame`
        left_validate_fn: Function to validate left :class:`DataFrame` columns
        right_validate_fn: Function to validate right :class:`DataFrame` columns

    Returns:
        List of (left_col, right_col) tuples for join keys

    Raises:
        ValueError: If keys cannot be determined or are invalid
        TypeError: If key types are incompatible
    """
    if on is not None:
        # Same column names in both DataFrames
        if isinstance(on, str):
            left_validate_fn([on], "join (left DataFrame)")
            right_validate_fn([on], "join (right DataFrame)")
            return [(on, on)]
        else:
            on_list = list(on)
            # Handle list of tuples or list of strings
            if on_list and isinstance(on_list[0], tuple):
                # Already in tuple format
                return [t for t in on_list if isinstance(t, tuple) and len(t) == 2]
            else:
                # List of strings - validate and convert to tuples
                str_cols = [c for c in on_list if isinstance(c, str)]
                if str_cols:
                    left_validate_fn(str_cols, "join (left DataFrame)")
                    right_validate_fn(str_cols, "join (right DataFrame)")
                return [(str(col), str(col)) for col in on_list if isinstance(col, str)]
    elif left_on is not None and right_on is not None:
        # Different column names
        if isinstance(left_on, str) and isinstance(right_on, str):
            left_validate_fn([left_on], "join (left DataFrame)")
            right_validate_fn([right_on], "join (right DataFrame)")
            return [(left_on, right_on)]
        elif isinstance(left_on, (list, tuple)) and isinstance(right_on, (list, tuple)):
            if len(left_on) != len(right_on):
                raise ValueError("left_on and right_on must have the same length")
            left_validate_fn(list(left_on), "join (left DataFrame)")
            right_validate_fn(list(right_on), "join (right DataFrame)")
            return list(zip(left_on, right_on))
        else:
            raise TypeError("left_on and right_on must both be str or both be sequences")
    else:
        raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")
