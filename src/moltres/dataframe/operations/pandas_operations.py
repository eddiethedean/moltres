"""Common operations for Pandas-style :class:`DataFrame` interfaces.

This module contains shared logic used by both :class:`PandasDataFrame` and
AsyncPandasDataFrame to reduce duplication and improve maintainability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Set, Tuple, Union

from ...expressions.column import Column
from ...utils.exceptions import PandasAPIError

if TYPE_CHECKING:
    pass


def parse_query_expression(
    expr: str, available_columns: Optional[Set[str]], df_plan: Any
) -> Column:
    """Parse a pandas-style query string into a :class:`Column` expression.

    Args:
        expr: Query string with pandas-style syntax
        available_columns: Set of available column names (for validation)
        df_plan: :class:`DataFrame` plan (for fallback column extraction)

    Returns:
        :class:`Column` expression representing the query predicate

    Raises:
        PandasAPIError: If the query string cannot be parsed
    """
    from ...expressions.sql_parser import parse_sql_expr

    # Get available column names for context and validation
    if available_columns is None:
        try:
            if hasattr(df_plan, "projections"):
                available_columns = set()
                for proj in df_plan.projections:
                    if isinstance(proj, Column) and proj.op == "column" and proj.args:
                        available_columns.add(str(proj.args[0]))
        except Exception:
            pass

    # Parse query string to Column expression
    try:
        predicate = parse_sql_expr(expr, available_columns)
    except ValueError as e:
        # Provide more helpful error message
        raise PandasAPIError(
            f"Failed to parse query expression: {expr}",
            suggestion=(
                f"Error: {str(e)}\n"
                "Query syntax should follow pandas-style syntax:\n"
                "  - Use '=' or '==' for equality: 'age == 25' or 'age = 25'\n"
                "  - Use 'and'/'or' keywords: 'age > 25 and status == \"active\"'\n"
                "  - Use comparison operators: >, <, >=, <=, !=, ==\n"
                f"{'  - Available columns: ' + ', '.join(sorted(available_columns)) if available_columns else ''}"
            ),
            context={
                "query": expr,
                "available_columns": list(available_columns) if available_columns else [],
            },
        ) from e

    return predicate


def normalize_merge_how(how: str) -> str:
    """Normalize pandas merge 'how' parameter to join type.

    Args:
        how: Join type string ('inner', 'left', 'right', 'outer', 'full', 'full_outer')

    Returns:
        Normalized join type ('inner', 'left', 'right', 'outer')
    """
    how_map = {
        "inner": "inner",
        "left": "left",
        "right": "right",
        "outer": "outer",
        "full": "outer",
        "full_outer": "outer",
    }
    return how_map.get(how.lower(), "inner")


def prepare_merge_keys(
    on: Optional[Union[str, Sequence[str]]],
    left_on: Optional[Union[str, Sequence[str]]],
    right_on: Optional[Union[str, Sequence[str]]],
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    left_validate_fn: Any,
    right_validate_fn: Any,
) -> list[Tuple[str, str]]:
    """Prepare join keys for merge operation.

    Args:
        on: :class:`Column` name(s) to join on (must exist in both DataFrames)
        left_on: :class:`Column` name(s) in left :class:`DataFrame`
        right_on: :class:`Column` name(s) in right :class:`DataFrame`
        left_columns: Available columns in left :class:`DataFrame` (for validation)
        right_columns: Available columns in right :class:`DataFrame` (for validation)
        left_validate_fn: Function to validate left :class:`DataFrame` columns
        right_validate_fn: Function to validate right :class:`DataFrame` columns

    Returns:
        List of (left_col, right_col) tuples for join keys

    Raises:
        ValueError: If keys cannot be determined or are invalid
        TypeError: If key types are incompatible
    """
    # Determine join keys
    if on is not None:
        # Same column names in both DataFrames
        if isinstance(on, str):
            # Validate columns exist in both DataFrames
            left_validate_fn([on], "merge (left DataFrame)")
            right_validate_fn([on], "merge (right DataFrame)")
            return [(on, on)]
        else:
            # Validate all columns exist
            left_validate_fn(list(on), "merge (left DataFrame)")
            right_validate_fn(list(on), "merge (right DataFrame)")
            return [(col, col) for col in on]
    elif left_on is not None and right_on is not None:
        # Different column names
        if isinstance(left_on, str) and isinstance(right_on, str):
            # Validate columns exist
            left_validate_fn([left_on], "merge (left DataFrame)")
            right_validate_fn([right_on], "merge (right DataFrame)")
            return [(left_on, right_on)]
        elif isinstance(left_on, (list, tuple)) and isinstance(right_on, (list, tuple)):
            if len(left_on) != len(right_on):
                raise ValueError("left_on and right_on must have the same length")
            # Validate all columns exist
            left_validate_fn(list(left_on), "merge (left DataFrame)")
            right_validate_fn(list(right_on), "merge (right DataFrame)")
            return list(zip(left_on, right_on))
        else:
            raise TypeError("left_on and right_on must both be str or both be sequences")
    else:
        raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")


def normalize_groupby_by(by: Union[str, Sequence[str]]) -> Tuple[str, ...]:
    """Normalize groupby 'by' parameter to tuple of column names.

    Args:
        by: :class:`Column` name(s) to group by

    Returns:
        Tuple of column name strings

    Raises:
        TypeError: If by is not str or sequence of str
    """
    if isinstance(by, str):
        return (by,)
    elif isinstance(by, (list, tuple)):
        return tuple(by)
    else:
        raise TypeError(f"by must be str or sequence of str, got {type(by)}")


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
    available_set = set(available_columns)
    missing = [col for col in column_names if col not in available_set]
    if missing:
        available_str = ", ".join(sorted(available_columns))
        raise ValueError(
            f"Column(s) {missing} not found in {operation}. Available columns: {available_str}"
        )
