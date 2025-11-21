"""Schema inference utilities for file readers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, cast

from ...table.schema import ColumnDef


def infer_schema_from_rows(rows: list[dict[str, object]]) -> Sequence[ColumnDef]:
    """Infer schema from sample rows.

    Args:
        rows: List of row dictionaries to analyze

    Returns:
        Sequence of ColumnDef objects representing the inferred schema

    Raises:
        ValueError: If rows list is empty
    """
    if not rows:
        raise ValueError("Cannot infer schema from empty data")

    sample = rows[0]
    columns: list[ColumnDef] = []

    for key, value in sample.items():
        # Check if any row has None for this column
        has_nulls = any(row.get(key) is None for row in rows)
        # Sample multiple rows to better infer type (especially for string numbers)
        sample_values = [row.get(key) for row in rows[:100] if row.get(key) is not None]
        col_type = _infer_type_from_values(sample_values, value)
        columns.append(ColumnDef(name=key, type_name=col_type, nullable=has_nulls))

    return columns


def _infer_type_from_values(sample_values: list[object], first_value: object) -> str:
    """Infer SQL type from sample values, trying to parse strings as numbers.

    Args:
        sample_values: List of sample values (non-None)
        first_value: First value in the column

    Returns:
        SQL type name (INTEGER, REAL, or TEXT)
    """
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


def apply_schema_to_rows(
    rows: list[dict[str, object]], schema: Sequence[ColumnDef]
) -> list[dict[str, object]]:
    """Apply schema type conversions to rows.

    Args:
        rows: List of row dictionaries
        schema: Sequence of ColumnDef objects defining the schema

    Returns:
        List of rows with values converted to appropriate types
    """
    typed_rows = []

    for row in rows:
        typed_row: dict[str, object] = {}
        for col_def in schema:
            value = row.get(col_def.name)
            if value is None:
                typed_row[col_def.name] = None
            elif col_def.type_name == "INTEGER":
                try:
                    typed_row[col_def.name] = int(cast("Any", value))
                except (ValueError, TypeError):
                    typed_row[col_def.name] = value
            elif col_def.type_name == "REAL":
                try:
                    typed_row[col_def.name] = float(cast("Any", value))
                except (ValueError, TypeError):
                    typed_row[col_def.name] = value
            else:
                typed_row[col_def.name] = str(value) if value is not None else None
        typed_rows.append(typed_row)

    return typed_rows
