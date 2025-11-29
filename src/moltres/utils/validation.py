"""Validation utilities for pandas interface and :class:`DataFrame` operations."""

from __future__ import annotations

from typing import Sequence, Set
from ..utils.exceptions import ValidationError, _suggest_column_name


def validate_columns_exist(
    column_names: Sequence[str],
    available_columns: Set[str],
    operation: str = "operation",
) -> None:
    """Validate that all specified columns exist in the available columns.

    Args:
        column_names: List of column names to validate
        available_columns: Set of available column names
        operation: Name of the operation being performed (for error messages)

    Raises:
        ValidationError: If any column does not exist, with helpful suggestions
    """
    missing_columns = [col for col in column_names if col not in available_columns]
    if missing_columns:
        # Use the first missing column for suggestion
        first_missing = missing_columns[0]
        suggestion = _suggest_column_name(first_missing, list(available_columns))

        if len(missing_columns) == 1:
            message = f"Column '{first_missing}' does not exist in {operation}."
        else:
            message = f"Columns {missing_columns} do not exist in {operation}."

        raise ValidationError(
            message,
            suggestion=suggestion,
            context={
                "missing_columns": missing_columns,
                "available_columns": list(available_columns),
                "operation": operation,
            },
        )
