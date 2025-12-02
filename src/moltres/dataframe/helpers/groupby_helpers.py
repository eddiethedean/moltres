"""Common helper functions for GroupBy implementations.

This module contains shared aggregation logic used across all GroupBy
implementations (sync, async, pandas, polars).
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from ...expressions.column import Column, col


def create_aggregation_from_string(
    column_name: str, func_name: str, alias: Optional[str] = None
) -> Column:
    """Create an aggregation :class:`Column` from a column name and function name string.

    This is a shared helper used by all GroupBy implementations.

    Args:
        column_name: Name of the column to aggregate
        func_name: Name of the aggregation function (e.g., "sum", "avg", "min", "max", "count")
        alias: Optional alias for the result column. If None, uses column_name for PySpark-style,
              or generates a descriptive alias for pandas-style.

    Returns:
        :class:`Column` expression for the aggregation

    Raises:
        ValueError: If the function name is not recognized
    """
    from ...expressions.functions import (
        avg,
        count,
        max as max_func,
        min as min_func,
        sum as sum_func,
        count_distinct,
    )

    func_map: Dict[str, Callable[[Column], Column]] = {
        "sum": sum_func,
        "avg": avg,
        "average": avg,  # Alias for avg
        "mean": avg,  # Pandas-style alias
        "min": min_func,
        "max": max_func,
        "count": count,
        "count_distinct": count_distinct,
        "nunique": count_distinct,  # Pandas-style alias
    }

    func_name_lower = func_name.lower()
    if func_name_lower not in func_map:
        raise ValueError(
            f"Unknown aggregation function: {func_name}. "
            f"Supported functions: {', '.join(func_map.keys())}"
        )

    agg_func = func_map[func_name_lower]
    agg_expr = agg_func(col(column_name))

    if alias:
        return agg_expr.alias(alias)
    else:
        # Default: alias with column name (PySpark-style)
        return agg_expr.alias(column_name)


def validate_aggregation(expr: Column) -> Column:
    """Validate that an expression is a valid aggregation.

    Args:
        expr: :class:`Column` expression to validate

    Returns:
        The validated column expression

    Raises:
        ValueError: If the expression is not a valid aggregation
    """
    if not expr.op.startswith("agg_"):
        raise ValueError(
            "Aggregation expressions must be created with moltres aggregate helpers "
            "(e.g., sum(), avg(), count(), min(), max())"
        )
    return expr


def extract_value_column(agg_expr: Column) -> str:
    """Extract the column name from an aggregation expression.

    Args:
        agg_expr: Aggregation :class:`Column` expression (e.g., sum(col("amount")))

    Returns:
        :class:`Column` name string (e.g., "amount")

    Raises:
        ValueError: If the column cannot be extracted
    """
    if not agg_expr.op.startswith("agg_"):
        raise ValueError("Expected an aggregation expression")

    if not agg_expr.args:
        raise ValueError("Aggregation expression must have arguments")

    # The first argument should be a Column with op="column"
    col_expr = agg_expr.args[0]
    if not isinstance(col_expr, Column):
        raise ValueError("Aggregation must operate on a column")

    if col_expr.op == "column":
        if not col_expr.args:
            raise ValueError("Column expression must have arguments")
        col_name = col_expr.args[0]
        if not isinstance(col_name, str):
            raise ValueError("Column name must be a string")
        return col_name
    else:
        raise ValueError(f"Cannot extract column name from expression: {col_expr.op}")


def extract_agg_func(agg_expr: Column) -> str:
    """Extract the aggregation function name from an aggregation expression.

    Args:
        agg_expr: Aggregation :class:`Column` expression (e.g., sum(col("amount")))

    Returns:
        Aggregation function name (e.g., "sum")
    """
    op = agg_expr.op
    if op == "agg_sum":
        return "sum"
    elif op == "agg_avg":
        return "avg"
    elif op == "agg_min":
        return "min"
    elif op == "agg_max":
        return "max"
    elif op == "agg_count" or op == "agg_count_star":
        return "count"
    elif op == "agg_count_distinct":
        return "count_distinct"
    else:
        # Default to sum if unknown
        return "sum"


def normalize_aggregations(
    aggregations: tuple, alias_with_column_name: bool = True, allow_empty: bool = False
) -> list[Column]:
    """Normalize aggregation expressions to :class:`Column` objects.

    Handles multiple input formats:
    - :class:`Column` expressions (passed through)
    - String column names (converted to sum(col(name)).alias(name))
    - Dictionary mapping column names to function names

    Args:
        aggregations: Tuple of aggregation expressions (:class:`Column`, str, or dict)
        alias_with_column_name: If True, alias string aggregations with column name.
                               If False, let the aggregation function generate its own alias.
        allow_empty: If True, allow empty aggregations (returns empty list).
                    If False, raise ValueError when no aggregations provided.

    Returns:
        List of normalized :class:`Column` expressions

    Raises:
        ValueError: If no aggregations provided (when allow_empty=False) or invalid types
    """
    if not aggregations:
        if allow_empty:
            return []
        raise ValueError("agg requires at least one aggregation expression")

    normalized_aggs = []
    for agg_expr in aggregations:
        if isinstance(agg_expr, str):
            # String column name - default to sum() and alias with column name
            from ...expressions.functions import sum as sum_func

            if alias_with_column_name:
                normalized_aggs.append(sum_func(col(agg_expr)).alias(agg_expr))
            else:
                normalized_aggs.append(sum_func(col(agg_expr)))
        elif isinstance(agg_expr, dict):
            # Dictionary syntax: {"column": "function"}
            for col_name, func_name in agg_expr.items():
                agg_col = create_aggregation_from_string(col_name, func_name)
                normalized_aggs.append(agg_col)
        elif isinstance(agg_expr, Column):
            # Already a Column expression
            normalized_aggs.append(agg_expr)
        else:
            raise ValueError(
                f"Invalid aggregation type: {type(agg_expr)}. "
                "Expected Column, str, or Dict[str, str]"
            )

    return normalized_aggs
