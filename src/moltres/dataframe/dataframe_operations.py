"""Complex DataFrame operations extracted for better maintainability.

This module contains complex operations like joins, set operations, pivots,
explode, and CTEs that are used by DataFrame and AsyncDataFrame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

from ..expressions.column import Column
from ..logical import operators
from .dataframe import DataFrame

if TYPE_CHECKING:
    pass


def join_dataframes(
    left: DataFrame,
    right: DataFrame,
    *,
    on: Optional[
        Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
    ] = None,
    how: str = "inner",
    lateral: bool = False,
    hints: Optional[Sequence[str]] = None,
) -> DataFrame:
    """Join two DataFrames.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Join condition
        how: Join type ("inner", "left", "right", "full", "cross")
        lateral: If True, create a LATERAL join
        hints: Optional sequence of join hints

    Returns:
        New DataFrame containing the join result
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before joining")
    if left.database is not right.database:
        raise ValueError("Cannot join DataFrames from different Database instances")
    # Cross joins don't require an 'on' clause
    if how.lower() == "cross":
        normalized_on = None
        condition = None
    else:
        normalized_condition = left._normalize_join_condition(on)
        if isinstance(normalized_condition, Column):
            # PySpark-style Column expression
            normalized_on = None
            condition = normalized_condition
        else:
            # Tuple-based join (backward compatible)
            normalized_on = normalized_condition
            condition = None
    hints_tuple = tuple(hints) if hints else None
    plan = operators.join(
        left.plan,
        right.plan,
        how=how.lower(),
        on=normalized_on,
        condition=condition,
        lateral=lateral,
        hints=hints_tuple,
    )
    return DataFrame(plan=plan, database=left.database)


def semi_join_dataframes(
    left: DataFrame,
    right: DataFrame,
    *,
    on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
) -> DataFrame:
    """Perform a semi-join between two DataFrames.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Join condition

    Returns:
        New DataFrame containing rows from left that have matches in right
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before semi_join")
    if left.database is not right.database:
        raise ValueError("Cannot semi_join DataFrames from different Database instances")
    normalized_condition = left._normalize_join_condition(on)
    if isinstance(normalized_condition, Column):
        raise ValueError("semi_join does not support Column expressions, use tuple syntax")
    normalized_on = normalized_condition
    plan = operators.semi_join(left.plan, right.plan, on=normalized_on)
    return DataFrame(plan=plan, database=left.database)


def anti_join_dataframes(
    left: DataFrame,
    right: DataFrame,
    *,
    on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
) -> DataFrame:
    """Perform an anti-join between two DataFrames.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Join condition

    Returns:
        New DataFrame containing rows from left that have no matches in right
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before anti_join")
    if left.database is not right.database:
        raise ValueError("Cannot anti_join DataFrames from different Database instances")
    normalized_condition = left._normalize_join_condition(on)
    if isinstance(normalized_condition, Column):
        raise ValueError("anti_join does not support Column expressions, use tuple syntax")
    normalized_on = normalized_condition
    plan = operators.anti_join(left.plan, right.plan, on=normalized_on)
    return DataFrame(plan=plan, database=left.database)


def union_dataframes(left: DataFrame, right: DataFrame, distinct: bool = True) -> DataFrame:
    """Union two DataFrames.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        distinct: If True, return distinct rows only (UNION). If False, return all rows (UNION ALL).

    Returns:
        New DataFrame containing the union result
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before union")
    if left.database is not right.database:
        raise ValueError("Cannot union DataFrames from different Database instances")
    plan = operators.union(left.plan, right.plan, distinct=distinct)
    return DataFrame(plan=plan, database=left.database)


def intersect_dataframes(left: DataFrame, right: DataFrame, distinct: bool = True) -> DataFrame:
    """Intersect two DataFrames.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        distinct: If True, return distinct rows only. If False, return all rows.

    Returns:
        New DataFrame containing the intersection result
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before intersect")
    if left.database is not right.database:
        raise ValueError("Cannot intersect DataFrames from different Database instances")
    plan = operators.intersect(left.plan, right.plan, distinct=distinct)
    return DataFrame(plan=plan, database=left.database)


def except_dataframes(left: DataFrame, right: DataFrame, distinct: bool = True) -> DataFrame:
    """Return rows from left DataFrame that are not in right DataFrame.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        distinct: If True, return distinct rows only. If False, return all rows.

    Returns:
        New DataFrame containing the except result
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before except")
    if left.database is not right.database:
        raise ValueError("Cannot except DataFrames from different Database instances")
    plan = operators.except_(left.plan, right.plan, distinct=distinct)
    return DataFrame(plan=plan, database=left.database)


def pivot_dataframe(
    df: DataFrame,
    pivot_column: str,
    value_column: str,
    agg_func: str = "sum",
    pivot_values: Optional[Sequence[str]] = None,
) -> DataFrame:
    """Pivot a DataFrame to reshape data from long to wide format.

    Args:
        df: DataFrame to pivot
        pivot_column: Column name to pivot on
        value_column: Column name containing values to aggregate
        agg_func: Aggregation function ("sum", "avg", "count", "min", "max")
        pivot_values: Optional sequence of pivot values (if None, will be inferred)

    Returns:
        New DataFrame with pivoted data
    """
    plan = operators.pivot(
        df.plan, pivot_column, value_column, agg_func=agg_func, pivot_values=pivot_values
    )
    return DataFrame(plan=plan, database=df.database)


def explode_dataframe(df: DataFrame, column: Union[Column, str], alias: str = "value") -> DataFrame:
    """Explode an array/JSON column into multiple rows.

    Args:
        df: DataFrame to explode
        column: Column expression or column name to explode
        alias: Alias for the exploded value column

    Returns:
        New DataFrame with exploded rows
    """
    if isinstance(column, str):
        from ..expressions.column import col

        col_expr = col(column)
    else:
        col_expr = column
    plan = operators.explode(df.plan, col_expr, alias=alias)
    return DataFrame(plan=plan, database=df.database)


def cte_dataframe(df: DataFrame, name: str) -> DataFrame:
    """Create a Common Table Expression (CTE) from a DataFrame.

    Args:
        df: DataFrame to convert to CTE
        name: Name for the CTE

    Returns:
        New DataFrame representing the CTE
    """
    plan = operators.cte(df.plan, name)
    return DataFrame(plan=plan, database=df.database)


def recursive_cte_dataframe(
    initial: DataFrame, name: str, recursive: DataFrame, union_all: bool = False
) -> DataFrame:
    """Create a Recursive Common Table Expression (WITH RECURSIVE) from DataFrames.

    Args:
        initial: Initial DataFrame (base case)
        name: Name for the recursive CTE
        recursive: DataFrame representing the recursive part (references the CTE)
        union_all: If True, use UNION ALL; if False, use UNION (distinct)

    Returns:
        New DataFrame representing the recursive CTE
    """
    if initial.database is None or recursive.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before recursive_cte")
    if initial.database is not recursive.database:
        raise ValueError("Cannot create recursive CTE from DataFrames in different Databases")
    plan = operators.recursive_cte(name, initial.plan, recursive.plan, union_all=union_all)
    return DataFrame(plan=plan, database=initial.database)
