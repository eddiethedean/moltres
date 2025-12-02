"""Shared helper functions for Pandas DataFrame interface operations.

This module contains shared logic used by both PandasDataFrame and AsyncPandasDataFrame
to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

from typing import (
    cast,
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
)

from ...expressions.column import Column, col

if TYPE_CHECKING:
    from ..core.dataframe import DataFrame
    from ..core.async_dataframe import AsyncDataFrame

# Type variable for generic Pandas DataFrame operations
P = TypeVar("P", bound="PandasDataFrameProtocol")

if TYPE_CHECKING:

    class PandasDataFrameProtocol(Protocol):
        """Protocol defining the interface that Pandas DataFrame classes must implement."""

        _df: Union["DataFrame", "AsyncDataFrame"]
        columns: List[str]

        def _validate_columns_exist(
            self, column_names: Sequence[str], operation: str = "operation"
        ) -> None:
            """Validate that all specified columns exist."""
            ...
else:
    PandasDataFrameProtocol = Any


def build_pandas_query_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    expr: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a query operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        expr: Query string with pandas-style syntax

    Returns:
        Resulting underlying DataFrame
    """
    from ..operations.pandas_operations import parse_query_expression

    # Get available column names for context and validation
    available_columns: Optional[set[str]] = None
    try:
        available_columns = set(pandas_df.columns)
    except Exception:
        pass

    # Parse query string to Column expression
    predicate = parse_query_expression(expr, available_columns, pandas_df._df.plan)

    # Apply filter
    return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.where(predicate))


def build_pandas_select_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    *columns: Union[str, Column],
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a select operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        *columns: Column names or Column expressions to select

    Returns:
        Resulting underlying DataFrame
    """
    # Validate column names if they're strings
    str_columns = [c for c in columns if isinstance(c, str)]
    if str_columns:
        pandas_df._validate_columns_exist(str_columns, "select")

    # Use underlying DataFrame's select
    selected_cols = [col(c) if isinstance(c, str) else c for c in columns]
    return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.select(*selected_cols))


def build_pandas_assign_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    **kwargs: Union[Column, Any],
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build an assign operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        **kwargs: Column name = value pairs where value can be a Column expression or literal

    Returns:
        Resulting underlying DataFrame
    """
    result_df = pandas_df._df
    for col_name, value in kwargs.items():
        if isinstance(value, Column):
            result_df = result_df.withColumn(col_name, value)
        else:
            # Literal value
            from ...expressions import lit

            result_df = result_df.withColumn(col_name, lit(value))
    return cast(Union["DataFrame", "AsyncDataFrame"], result_df)


def build_pandas_merge_operation(
    left_pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    right_pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    on: Optional[Union[str, Sequence[str]]] = None,
    left_on: Optional[Union[str, Sequence[str]]] = None,
    right_on: Optional[Union[str, Sequence[str]]] = None,
    how: str = "inner",
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a merge operation for Pandas DataFrame.

    Args:
        left_pandas_df: Left Pandas DataFrame instance
        right_pandas_df: Right Pandas DataFrame instance
        on: Column name(s) to join on (must exist in both DataFrames)
        left_on: Column name(s) in left DataFrame
        right_on: Column name(s) in right DataFrame
        how: Type of join ('inner', 'left', 'right', 'outer')

    Returns:
        Resulting underlying DataFrame
    """
    from ..operations.pandas_operations import normalize_merge_how, prepare_merge_keys

    # Normalize how parameter
    join_how = normalize_merge_how(how)

    # Determine join keys
    join_on = prepare_merge_keys(
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_columns=left_pandas_df.columns,
        right_columns=right_pandas_df.columns,
        left_validate_fn=left_pandas_df._validate_columns_exist,
        right_validate_fn=right_pandas_df._validate_columns_exist,
    )

    # Perform join
    return cast(
        Union["DataFrame", "AsyncDataFrame"],
        left_pandas_df._df.join(right_pandas_df._df, on=join_on, how=join_how),
    )


def build_pandas_cross_join_operation(
    left_pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    right_pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a cross join operation for Pandas DataFrame.

    Args:
        left_pandas_df: Left Pandas DataFrame instance
        right_pandas_df: Right Pandas DataFrame instance

    Returns:
        Resulting underlying DataFrame
    """
    return cast(
        Union["DataFrame", "AsyncDataFrame"], left_pandas_df._df.crossJoin(right_pandas_df._df)
    )


def build_pandas_rename_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    columns: Dict[str, str],
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a rename operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        columns: Dictionary mapping old names to new names

    Returns:
        Resulting underlying DataFrame
    """
    result_df = pandas_df._df
    for old_name, new_name in columns.items():
        result_df = result_df.withColumnRenamed(old_name, new_name)
    return cast(Union["DataFrame", "AsyncDataFrame"], result_df)


def build_pandas_drop_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    columns: Optional[Union[str, Sequence[str]]] = None,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a drop operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        columns: Column name(s) to drop

    Returns:
        Resulting underlying DataFrame
    """
    if columns is None:
        return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df)

    if isinstance(columns, str):
        cols_to_drop = [columns]
    else:
        cols_to_drop = list(columns)

    return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.drop(*cols_to_drop))


def build_pandas_sort_values_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    by: Union[str, Sequence[str]],
    ascending: Union[bool, Sequence[bool]] = True,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a sort_values operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        by: Column name(s) to sort by
        ascending: Sort order (True for ascending, False for descending)

    Returns:
        Resulting underlying DataFrame
    """
    if isinstance(by, str):
        columns = [by]
        ascending_list = [ascending] if isinstance(ascending, bool) else list(ascending)
    else:
        columns = list(by)
        if isinstance(ascending, bool):
            ascending_list = [ascending] * len(columns)
        else:
            ascending_list = list(ascending)
            if len(ascending_list) != len(columns):
                raise ValueError("ascending must have same length as by")

    # Validate columns exist
    pandas_df._validate_columns_exist(columns, "sort_values")

    # Build order_by list - use Column expressions with .desc() for descending
    order_by_cols = []
    for col_name, asc in zip(columns, ascending_list):
        col_expr = col(col_name)
        if not asc:
            # Descending order
            col_expr = col_expr.desc()
        order_by_cols.append(col_expr)

    return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.order_by(*order_by_cols))


def build_pandas_drop_duplicates_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    subset: Optional[Union[str, Sequence[str]]] = None,
    keep: str = "first",
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a drop_duplicates operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        subset: Column name(s) to consider for duplicates (None means all columns)
        keep: Which duplicate to keep ('first' or 'last')

    Returns:
        Resulting underlying DataFrame
    """
    from ...expressions import functions as F

    if subset is None:
        # Remove duplicates on all columns
        return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.distinct())
    else:
        # Validate subset columns exist
        if isinstance(subset, str):
            subset_cols = [subset]
        else:
            subset_cols = list(subset)

        pandas_df._validate_columns_exist(subset_cols, "drop_duplicates")

        # For subset-based deduplication, use GROUP BY
        grouped = pandas_df._df.group_by(*subset_cols)

        # Get all column names
        all_cols = pandas_df.columns
        other_cols = [col_name for col_name in all_cols if col_name not in subset_cols]

        if not other_cols:
            # If only grouping columns, distinct works fine
            return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.distinct())
        else:
            # Build aggregations for non-grouped columns only
            agg_exprs = []
            for col_name in other_cols:
                if keep == "last":
                    agg_exprs.append(F.max(col(col_name)).alias(col_name))
                else:  # keep == "first"
                    agg_exprs.append(F.min(col(col_name)).alias(col_name))

            return cast(Union["DataFrame", "AsyncDataFrame"], grouped.agg(*agg_exprs))


def build_pandas_limit_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    n: int,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a limit operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        n: Number of rows to return

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.limit(n))


def build_pandas_sample_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a sample operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        n: Number of rows to sample
        frac: Fraction of rows to sample (0.0 to 1.0)
        random_state: Random seed

    Returns:
        Resulting underlying DataFrame
    """
    if n is not None and frac is not None:
        raise ValueError("Cannot specify both 'n' and 'frac'")

    if n is not None:
        # Sample n rows - use fraction=1.0 then limit
        return cast(
            Union["DataFrame", "AsyncDataFrame"],
            pandas_df._df.sample(fraction=1.0, seed=random_state).limit(n),
        )
    elif frac is not None:
        return cast(
            Union["DataFrame", "AsyncDataFrame"],
            pandas_df._df.sample(fraction=frac, seed=random_state),
        )
    else:
        raise ValueError("Must specify either 'n' or 'frac'")


def build_pandas_append_operation(
    left_pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    right_pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build an append operation for Pandas DataFrame.

    Args:
        left_pandas_df: Left Pandas DataFrame instance
        right_pandas_df: Right Pandas DataFrame instance

    Returns:
        Resulting underlying DataFrame (union all)
    """
    return cast(
        Union["DataFrame", "AsyncDataFrame"], left_pandas_df._df.unionAll(right_pandas_df._df)
    )


def build_pandas_concat_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    others: Sequence[Any],  # Sequence[PandasDataFrameProtocol] - using Any for mypy compatibility
    axis: Union[int, str] = 0,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a concat operation for Pandas DataFrame.

    Args:
        pandas_df: First Pandas DataFrame instance
        others: Other Pandas DataFrame instances to concatenate
        axis: Concatenation axis (0 for vertical, 1 for horizontal)

    Returns:
        Resulting underlying DataFrame
    """
    if axis == 0 or axis == "index":
        # Vertical concatenation (union all)
        result_df = pandas_df._df
        for other in others:
            result_df = result_df.unionAll(other._df)
        return cast(Union["DataFrame", "AsyncDataFrame"], result_df)
    elif axis == 1 or axis == "columns":
        # Horizontal concatenation (cross join)
        result_df = pandas_df._df
        for other in others:
            result_df = result_df.crossJoin(other._df)
        return cast(Union["DataFrame", "AsyncDataFrame"], result_df)
    else:
        raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, 'index', or 'columns'")


def build_pandas_select_expr_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    *exprs: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a select_expr operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        *exprs: SQL expression strings

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.selectExpr(*exprs))


def build_pandas_cte_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    name: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a cte operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        name: Name for the CTE

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.cte(name))


def build_pandas_summary_operation(
    pandas_df: Any,  # PandasDataFrameProtocol - using Any for mypy compatibility
    *statistics: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a summary operation for Pandas DataFrame.

    Args:
        pandas_df: Pandas DataFrame instance
        *statistics: Statistics to compute

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union["DataFrame", "AsyncDataFrame"], pandas_df._df.summary(*statistics))
