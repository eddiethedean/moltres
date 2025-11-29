"""Conversion utilities for :class:`Records`.

This module contains helper functions for converting between pandas/polars
DataFrames and :class:`Records`, including schema extraction and type conversion.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..table.schema import ColumnDef


def is_pandas_dataframe(obj: Any) -> bool:
    """Check if object is a pandas :class:`DataFrame` without importing pandas unnecessarily.

    Args:
        obj: Object to check

    Returns:
        True if object is a pandas :class:`DataFrame`, False otherwise
    """
    module = getattr(type(obj), "__module__", "")
    if "pandas" not in module:
        return False
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(obj, pd.DataFrame)


def is_polars_dataframe(obj: Any) -> bool:
    """Check if object is a polars :class:`DataFrame`.

    Args:
        obj: Object to check

    Returns:
        True if object is a polars :class:`DataFrame`, False otherwise
    """
    module = getattr(type(obj), "__module__", "")
    if "polars" not in module:
        return False
    try:
        import polars as pl
    except ImportError:
        return False
    return isinstance(obj, pl.DataFrame)


def is_polars_lazyframe(obj: Any) -> bool:
    """Check if object is a polars LazyFrame.

    Args:
        obj: Object to check

    Returns:
        True if object is a polars LazyFrame, False otherwise
    """
    module = getattr(type(obj), "__module__", "")
    if "polars" not in module:
        return False
    try:
        import polars as pl
    except ImportError:
        return False
    return isinstance(obj, pl.LazyFrame)


def convert_pandas_dtype_to_sql_type(dtype: Any) -> str:
    """Convert pandas dtype to SQL type name.

    Args:
        dtype: pandas dtype object

    Returns:
        SQL type name string
    """
    dtype_str = str(dtype)

    # Handle nullable integer types
    if dtype_str.startswith("Int"):
        return "INTEGER"

    # Handle standard types
    if dtype_str in ("int64", "int32", "int16", "int8", "int"):
        return "INTEGER"
    if dtype_str in ("float64", "float32", "float"):
        return "REAL"
    if dtype_str in ("bool", "boolean"):
        return "INTEGER"  # SQLite uses INTEGER for booleans
    if dtype_str.startswith("datetime"):
        return "TIMESTAMP"
    if dtype_str.startswith("date"):
        return "DATE"
    if dtype_str.startswith("timedelta"):
        return "TEXT"  # No standard SQL type for timedelta
    if dtype_str in ("object", "string", "str"):
        return "TEXT"

    # Default to TEXT for unknown types
    return "TEXT"


def convert_polars_type_to_sql_type(polars_type: Any) -> str:
    """Convert polars type to SQL type name.

    Args:
        polars_type: polars DataType object or string representation

    Returns:
        SQL type name string
    """
    type_str = str(polars_type).lower()

    # Handle polars types
    if "int" in type_str:
        return "INTEGER"
    if "float" in type_str or "f64" in type_str or "f32" in type_str:
        return "REAL"
    if "bool" in type_str:
        return "INTEGER"  # SQLite uses INTEGER for booleans
    if "datetime" in type_str or "timestamp" in type_str:
        return "TIMESTAMP"
    if "date" in type_str:
        return "DATE"
    if "duration" in type_str or "timedelta" in type_str:
        return "TEXT"  # No standard SQL type for duration
    if "str" in type_str or "string" in type_str or "utf8" in type_str:
        return "TEXT"
    if "null" in type_str:
        return "TEXT"  # Nullable type, default to TEXT

    # Default to TEXT for unknown types
    return "TEXT"


def extract_schema_from_pandas_dataframe(df: Any) -> Optional[List["ColumnDef"]]:
    """Extract schema from pandas :class:`DataFrame`.

    Args:
        df: pandas :class:`DataFrame`

    Returns:
        List of ColumnDef objects or None if extraction fails
    """
    try:
        from ..table.schema import ColumnDef

        columns = []
        for col_name, dtype in df.dtypes.items():
            sql_type = convert_pandas_dtype_to_sql_type(dtype)
            # Check if column has any nulls - convert numpy boolean to Python boolean
            nullable = bool(df[col_name].isna().any())
            columns.append(ColumnDef(name=str(col_name), type_name=sql_type, nullable=nullable))
        return columns
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.debug("Schema extraction from pandas DataFrame failed: %s", e)
        return None
    except Exception as e:
        logger.warning(
            "Unexpected error during pandas DataFrame schema extraction: %s", e, exc_info=True
        )
        return None


def extract_schema_from_polars_dataframe(df: Any) -> Optional[List["ColumnDef"]]:
    """Extract schema from polars :class:`DataFrame`.

    Args:
        df: polars :class:`DataFrame`

    Returns:
        List of ColumnDef objects or None if extraction fails
    """
    try:
        from ..table.schema import ColumnDef

        columns = []
        schema = df.schema
        for col_name, polars_type in schema.items():
            sql_type = convert_polars_type_to_sql_type(polars_type)
            # Check if type is nullable - convert to Python boolean
            nullable = bool(
                polars_type.is_nullable() if hasattr(polars_type, "is_nullable") else True
            )
            columns.append(ColumnDef(name=str(col_name), type_name=sql_type, nullable=nullable))
        return columns
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.debug("Schema extraction from polars DataFrame failed: %s", e)
        return None
    except Exception as e:
        logger.warning(
            "Unexpected error during polars DataFrame schema extraction: %s", e, exc_info=True
        )
        return None


def extract_schema_from_polars_lazyframe(lf: Any) -> Optional[List["ColumnDef"]]:
    """Extract schema from polars LazyFrame.

    Args:
        lf: polars LazyFrame

    Returns:
        List of ColumnDef objects or None if extraction fails
    """
    try:
        from ..table.schema import ColumnDef

        columns = []
        # Use collect_schema() to avoid performance warning
        schema = lf.collect_schema() if hasattr(lf, "collect_schema") else lf.schema
        for col_name, polars_type in schema.items():
            sql_type = convert_polars_type_to_sql_type(polars_type)
            # Check if type is nullable - convert to Python boolean
            nullable = bool(
                polars_type.is_nullable() if hasattr(polars_type, "is_nullable") else True
            )
            columns.append(ColumnDef(name=str(col_name), type_name=sql_type, nullable=nullable))
        return columns
    except (AttributeError, KeyError, TypeError, ValueError) as e:
        logger.debug("Schema extraction from polars LazyFrame failed: %s", e)
        return None
    except Exception as e:
        logger.warning(
            "Unexpected error during polars LazyFrame schema extraction: %s", e, exc_info=True
        )
        return None


def convert_dataframe_to_rows(df: Any) -> List[dict[str, object]]:
    """Convert pandas/polars :class:`DataFrame` or polars LazyFrame to list of dictionaries.

    Args:
        df: pandas :class:`DataFrame`, polars :class:`DataFrame`, or polars LazyFrame

    Returns:
        List of row dictionaries

    Raises:
        ValueError: If :class:`DataFrame` type is not supported
    """
    if is_pandas_dataframe(df):
        # Convert pandas DataFrame to list of dicts
        return df.to_dict("records")  # type: ignore[no-any-return]
    elif is_polars_dataframe(df):
        # Convert polars DataFrame to list of dicts
        return df.to_dicts()  # type: ignore[no-any-return]
    elif is_polars_lazyframe(df):
        # Materialize LazyFrame first, then convert
        materialized = df.collect()
        return materialized.to_dicts()  # type: ignore[no-any-return]
    else:
        raise ValueError(f"Unsupported DataFrame type: {type(df)}")


def extract_schema_from_dataframe(df: Any) -> Optional[List["ColumnDef"]]:
    """Extract schema from pandas/polars :class:`DataFrame` or polars LazyFrame.

    Args:
        df: pandas :class:`DataFrame`, polars :class:`DataFrame`, or polars LazyFrame

    Returns:
        List of ColumnDef objects or None if extraction fails
    """
    if is_pandas_dataframe(df):
        return extract_schema_from_pandas_dataframe(df)
    elif is_polars_dataframe(df):
        return extract_schema_from_polars_dataframe(df)
    elif is_polars_lazyframe(df):
        return extract_schema_from_polars_lazyframe(df)
    return None
