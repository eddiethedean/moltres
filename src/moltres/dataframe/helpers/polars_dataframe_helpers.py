"""Shared helper functions for Polars DataFrame interface operations.

This module contains shared logic used by both PolarsDataFrame and AsyncPolarsDataFrame
to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from ...expressions.column import Column, col
from ...logical.plan import TableScan
from ...utils.typing import FillValue

# Import at runtime for cast() calls
from ..core.dataframe import DataFrame
from ..core.async_dataframe import AsyncDataFrame

# Type variable for generic Polars DataFrame operations - using Any bound for flexibility
P = TypeVar("P", bound=Any)

if TYPE_CHECKING:

    class PolarsDataFrameProtocol(Protocol):
        """Protocol defining the interface that Polars DataFrame classes must implement."""

        _df: Union["DataFrame", "AsyncDataFrame"]

        @property
        def columns(self) -> List[str]:
            """Get column names."""
            ...

        # Schema can be either a property (sync) or async method (async)
        schema: Union[List[Tuple[str, str]], Any]

        def _validate_columns_exist(
            self, column_names: Sequence[str], operation: str = "operation"
        ) -> None:
            """Validate that all specified columns exist."""
            ...

        def _with_dataframe(self, df: Union["DataFrame", "AsyncDataFrame"]) -> Any:
            """Create a new instance with a different underlying DataFrame."""
            ...
else:
    PolarsDataFrameProtocol = Any


def build_polars_select_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    *exprs: Union[str, Column],
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a select operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        *exprs: Column names or Column expressions to select

    Returns:
        Resulting underlying DataFrame
    """
    # Validate column names if they're strings
    str_columns = [e for e in exprs if isinstance(e, str)]
    if str_columns:
        polars_df._validate_columns_exist(str_columns, "select")

    # Use underlying DataFrame's select
    selected_cols = [col(e) if isinstance(e, str) else e for e in exprs]
    return cast(Union["DataFrame", "AsyncDataFrame"], polars_df._df.select(*selected_cols))


def build_polars_filter_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    predicate: Column,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a filter operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        predicate: Column expression for filtering condition

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union["DataFrame", "AsyncDataFrame"], polars_df._df.where(predicate))


def build_polars_with_columns_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    *exprs: Union[Column, Tuple[str, Column]],
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a with_columns operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        *exprs: Column expressions or (name, expression) tuples

    Returns:
        Resulting underlying DataFrame

    Raises:
        ValueError: If column expression doesn't have an alias
        TypeError: If expression type is invalid
    """
    result_df = polars_df._df
    for expr in exprs:
        if isinstance(expr, tuple) and len(expr) == 2:
            # (name, expression) tuple
            col_name, col_expr = expr
            if isinstance(col_expr, str):
                col_expr = col(col_expr)
            result_df = result_df.withColumn(col_name, col_expr)
        elif isinstance(expr, Column):
            # Column expression with alias
            if expr.alias_name:
                result_df = result_df.withColumn(expr.alias_name, expr)
            else:
                raise ValueError(
                    "Column expression in with_columns() must have an alias, "
                    "or use tuple (name, expression) format"
                )
        else:
            raise TypeError(
                f"with_columns() expects Column expressions or (name, Column) tuples, "
                f"got {type(expr)}"
            )
    return cast(Union["DataFrame", "AsyncDataFrame"], result_df)


def build_polars_drop_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    *columns: Union[str, Column],
) -> Union["DataFrame", "AsyncDataFrame", None]:
    """Build a drop operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        *columns: Column names to drop

    Returns:
        Resulting underlying DataFrame, or None if no columns to drop
    """
    if not columns:
        return None

    # Validate columns exist
    str_columns = [c for c in columns if isinstance(c, str)]
    if str_columns:
        polars_df._validate_columns_exist(str_columns, "drop")

    # If the underlying DataFrame is a TableScan, we need to select columns first
    # to create a Project operation, then drop will work
    if isinstance(polars_df._df.plan, TableScan):
        # Get all columns and select them to create a Project
        all_columns = polars_df.columns
        # Select all columns except the ones to drop
        cols_to_keep = [col for col in all_columns if col not in str_columns]
        if cols_to_keep:
            return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.select(*cols_to_keep))
        else:
            # All columns were dropped - return empty select
            return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.select())
    else:
        return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.drop(*columns))


def build_polars_rename_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    mapping: Dict[str, str],
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a rename operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        mapping: Dictionary mapping old names to new names

    Returns:
        Resulting underlying DataFrame
    """
    result_df = polars_df._df
    for old_name, new_name in mapping.items():
        result_df = result_df.withColumnRenamed(old_name, new_name)
    return cast(Union[DataFrame, AsyncDataFrame], result_df)


def build_polars_sort_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    *columns: Union[str, Column],
    descending: Union[bool, Sequence[bool]] = False,
) -> Union["DataFrame", "AsyncDataFrame", None]:
    """Build a sort operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        *columns: Column names or Column expressions to sort by
        descending: Sort order - single bool or sequence of bools for each column

    Returns:
        Resulting underlying DataFrame, or None if no columns to sort

    Raises:
        ValueError: If descending length doesn't match columns length
    """
    if not columns:
        return None

    # Validate column names if they're strings
    str_columns = [c for c in columns if isinstance(c, str)]
    if str_columns:
        polars_df._validate_columns_exist(str_columns, "sort")

    # Normalize descending parameter
    if isinstance(descending, bool):
        descending_list = [descending] * len(columns)
    else:
        descending_list = list(descending)
        if len(descending_list) != len(columns):
            raise ValueError("descending must have same length as columns")

    # Build order_by list
    order_by_cols = []
    for col_expr, desc in zip(columns, descending_list):
        if isinstance(col_expr, str):
            col_expr = col(col_expr)
        if desc:
            col_expr = col_expr.desc()
        order_by_cols.append(col_expr)

    return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.order_by(*order_by_cols))


def build_polars_limit_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    n: int,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a limit operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        n: Number of rows to return

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.limit(n))


def build_polars_tail_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    n: int = 5,
) -> Union["DataFrame", "AsyncDataFrame", None]:
    """Build a tail operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        n: Number of rows to return (default: 5)

    Returns:
        Resulting underlying DataFrame, or None if no columns

    Note:
        This is a simplified implementation. For proper tail() behavior with lazy
        evaluation, this method sorts all columns in descending order and takes
        the first n rows.
    """
    cols = polars_df.columns
    if not cols:
        return None

    # Sort by all columns in descending order, then limit
    sorted_df = polars_df._df
    for col_name in cols:
        sorted_df = sorted_df.order_by(col(col_name).desc())

    return cast(Union["DataFrame", "AsyncDataFrame"], sorted_df.limit(n))


def build_polars_sample_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    fraction: Optional[float] = None,
    n: Optional[int] = None,
    seed: Optional[int] = None,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a sample operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        fraction: Fraction of rows to sample (0.0 to 1.0)
        n: Number of rows to sample (if provided, fraction is ignored)
        seed: Random seed for reproducibility

    Returns:
        Resulting underlying DataFrame

    Raises:
        ValueError: If neither fraction nor n is provided
    """
    if n is not None:
        # When n is provided, sample all rows (fraction=1.0) then limit to n
        # This provides random sampling of n rows
        sampled_df = polars_df._df.sample(fraction=1.0, seed=seed)
        return cast(Union["DataFrame", "AsyncDataFrame"], sampled_df.limit(n))
    elif fraction is not None:
        return cast(
            Union[DataFrame, AsyncDataFrame], polars_df._df.sample(fraction=fraction, seed=seed)
        )
    else:
        raise ValueError("Either 'fraction' or 'n' must be provided to sample()")


def build_polars_drop_nulls_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    subset: Optional[Union[str, Sequence[str]]] = None,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a drop_nulls operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        subset: Column name(s) to check for nulls (None means all columns)

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.dropna(subset=subset))


def build_polars_fill_null_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    value: Optional[FillValue] = None,
    strategy: Optional[str] = None,
    limit: Optional[int] = None,
    subset: Optional[Union[str, Sequence[str]]] = None,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a fill_null operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        value: Value to fill nulls with
        strategy: Fill strategy (not fully supported)
        limit: Maximum number of consecutive nulls to fill (not fully supported)
        subset: Column name(s) to fill nulls in (None means all columns)

    Returns:
        Resulting underlying DataFrame

    Raises:
        NotImplementedError: If strategy or limit is provided (not yet implemented)
    """
    if strategy is not None:
        raise NotImplementedError("fill_null with strategy is not yet implemented")
    if limit is not None:
        raise NotImplementedError("fill_null with limit is not yet implemented")

    return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.fillna(value=value, subset=subset))


def build_polars_explode_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    columns: Union[str, Sequence[str]],
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build an explode operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        columns: Column name(s) to explode

    Returns:
        Resulting underlying DataFrame
    """
    if isinstance(columns, str):
        columns = [columns]
    polars_df._validate_columns_exist(columns, "explode")

    # Explode columns one at a time
    result_df = polars_df._df
    for col_name in columns:
        result_df = result_df.explode(col(col_name), alias=col_name)
    return cast(Union[DataFrame, AsyncDataFrame], result_df)


def build_polars_pivot_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    values: Union[str, Sequence[str]],
    columns: Optional[str],
    aggregate_function: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a pivot operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        values: Column(s) to aggregate
        columns: Column to use as columns (pivot column)
        aggregate_function: Aggregation function (e.g., 'sum', 'mean', 'count')

    Returns:
        Resulting underlying DataFrame

    Raises:
        ValueError: If columns parameter is None
    """
    if columns is None:
        raise ValueError("pivot() requires 'columns' parameter")

    # Use underlying DataFrame's pivot method
    # Note: DataFrame.pivot has different signature, so we need to adapt
    if isinstance(values, (list, tuple)) and len(values) > 0:
        value_col: str = str(values[0])
    else:
        value_col = str(values)

    return cast(
        Union[DataFrame, AsyncDataFrame],
        polars_df._df.pivot(
            pivot_column=columns,
            value_column=value_col,
            agg_func=aggregate_function,
            pivot_values=None,
        ),
    )


def build_polars_unique_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    subset: Optional[Union[str, Sequence[str]]],
    keep: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a unique operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        subset: Column name(s) to consider for duplicates (None means all columns)
        keep: Which duplicate to keep ('first' or 'last')

    Returns:
        Resulting underlying DataFrame
    """
    if subset is None:
        # Remove duplicates on all columns
        return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.distinct())

    # Validate subset columns exist
    if isinstance(subset, str):
        subset_cols = [subset]
    else:
        subset_cols = list(subset)

    polars_df._validate_columns_exist(subset_cols, "unique")

    # For subset-based deduplication, use GROUP BY
    grouped = polars_df._df.group_by(*subset_cols)

    # Get all column names
    all_cols = polars_df.columns
    other_cols = [col for col in all_cols if col not in subset_cols]

    from ...expressions import functions as F

    if not other_cols:
        # If only grouping columns, distinct works fine
        return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.distinct())
    else:
        # Build aggregations for non-grouped columns
        agg_exprs = []
        for col_name in other_cols:
            if keep == "last":
                agg_exprs.append(F.max(col(col_name)).alias(col_name))
            else:  # keep == "first"
                agg_exprs.append(F.min(col(col_name)).alias(col_name))

        return cast(Union["DataFrame", "AsyncDataFrame"], grouped.agg(*agg_exprs))


def build_polars_union_operation(
    left_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    right_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    distinct: bool,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a union operation for Polars DataFrame.

    Args:
        left_polars_df: Left Polars DataFrame instance
        right_polars_df: Right Polars DataFrame instance
        distinct: If True, return distinct rows only

    Returns:
        Resulting underlying DataFrame
    """
    if distinct:
        return cast(Union[DataFrame, AsyncDataFrame], left_polars_df._df.union(right_polars_df._df))
    else:
        return cast(
            Union[DataFrame, AsyncDataFrame], left_polars_df._df.unionAll(right_polars_df._df)
        )


def build_polars_intersect_operation(
    left_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    right_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build an intersect operation for Polars DataFrame.

    Args:
        left_polars_df: Left Polars DataFrame instance
        right_polars_df: Right Polars DataFrame instance

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union[DataFrame, AsyncDataFrame], left_polars_df._df.intersect(right_polars_df._df))


def build_polars_difference_operation(
    left_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    right_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a difference operation for Polars DataFrame.

    Args:
        left_polars_df: Left Polars DataFrame instance
        right_polars_df: Right Polars DataFrame instance

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union[DataFrame, AsyncDataFrame], left_polars_df._df.except_(right_polars_df._df))


def build_polars_cross_join_operation(
    left_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    right_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a cross join operation for Polars DataFrame.

    Args:
        left_polars_df: Left Polars DataFrame instance
        right_polars_df: Right Polars DataFrame instance

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union[DataFrame, AsyncDataFrame], left_polars_df._df.crossJoin(right_polars_df._df))


def build_polars_concat_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    others: Sequence[Any],  # Sequence[PolarsDataFrameProtocol] - using Any for mypy compatibility
    how: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a concat operation for Polars DataFrame.

    Args:
        polars_df: First Polars DataFrame instance
        others: Other Polars DataFrame instances to concatenate
        how: Concatenation mode ("vertical" or "diagonal")

    Returns:
        Resulting underlying DataFrame

    Raises:
        ValueError: If how is not "vertical" or "diagonal"
    """
    result_df = polars_df._df
    for other in others:
        # others can be either DataFrames directly or PolarsDataFrames with ._df attribute
        other_df = other._df if hasattr(other, "_df") else other
        if how == "vertical":
            # Vertical concatenation (union all)
            result_df = result_df.unionAll(other_df)
        elif how == "diagonal":
            # Diagonal concatenation (union all with different schemas)
            # For now, same as vertical
            result_df = result_df.unionAll(other_df)
        else:
            raise ValueError(f"Invalid 'how' parameter: {how}. Must be 'vertical' or 'diagonal'")
    return cast(Union[DataFrame, AsyncDataFrame], result_df)


def build_polars_hstack_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    others: Sequence[Any],  # Sequence[PolarsDataFrameProtocol] - using Any for mypy compatibility
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build an hstack operation for Polars DataFrame.

    Args:
        polars_df: First Polars DataFrame instance
        others: Other Polars DataFrame instances to stack horizontally

    Returns:
        Resulting underlying DataFrame
    """
    result_df = polars_df._df
    for other in others:
        # others can be either DataFrames directly or PolarsDataFrames with ._df attribute
        other_df = other._df if hasattr(other, "_df") else other
        result_df = result_df.crossJoin(other_df)
    return cast(Union[DataFrame, AsyncDataFrame], result_df)


def build_polars_select_expr_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    *exprs: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a select_expr operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        *exprs: SQL expression strings

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.selectExpr(*exprs))


def build_polars_cte_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    name: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a CTE operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        name: Name for the CTE

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.cte(name))


def build_polars_recursive_cte_operation(
    initial_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    name: str,
    recursive_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    union_all: bool,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a recursive CTE operation for Polars DataFrame.

    Args:
        initial_polars_df: Initial Polars DataFrame instance
        name: Name for the recursive CTE
        recursive_polars_df: Recursive Polars DataFrame instance
        union_all: If True, use UNION ALL; if False, use UNION (distinct)

    Returns:
        Resulting underlying DataFrame
    """
    return cast(
        Union["DataFrame", "AsyncDataFrame"],
        initial_polars_df._df.recursive_cte(name, recursive_polars_df._df, union_all=union_all),
    )


def build_polars_summary_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    *statistics: str,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a summary operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        *statistics: Statistics to compute

    Returns:
        Resulting underlying DataFrame
    """
    return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.summary(*statistics))


def build_polars_join_operation(
    left_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    right_polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]],
    how: str,
    left_on: Optional[Union[str, Sequence[str]]],
    right_on: Optional[Union[str, Sequence[str]]],
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a join operation for Polars DataFrame.

    Args:
        left_polars_df: Left Polars DataFrame instance
        right_polars_df: Right Polars DataFrame instance (must have columns attribute)
        on: Column name(s) to join on
        how: Type of join ('inner', 'left', 'right', 'outer', 'anti', 'semi')
        left_on: Column name(s) in left DataFrame
        right_on: Column name(s) in right DataFrame

    Returns:
        Resulting underlying DataFrame
    """
    from ..operations.polars_operations import normalize_join_how, prepare_polars_join_keys

    # Normalize how parameter
    join_how = normalize_join_how(how)

    # Determine join keys
    join_on = prepare_polars_join_keys(
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_columns=left_polars_df.columns,
        right_columns=right_polars_df.columns,
        left_validate_fn=left_polars_df._validate_columns_exist,
        right_validate_fn=right_polars_df._validate_columns_exist,
    )

    # Handle anti and semi joins (Polars-specific)
    # These methods require tuple syntax, not Column expressions
    if join_how == "anti":
        # Anti-join: rows in left that don't have matches in right
        # Convert join_on to the format expected by anti_join (list of tuples)
        on_param: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = (
            join_on if isinstance(join_on, list) else None
        )
        return cast(
            Union[DataFrame, AsyncDataFrame],
            left_polars_df._df.anti_join(right_polars_df._df, on=on_param),
        )
    elif join_how == "semi":
        # Semi-join: rows in left that have matches in right (no right columns)
        on_param_semi: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = (
            join_on if isinstance(join_on, list) else None
        )
        return cast(
            Union[DataFrame, AsyncDataFrame],
            left_polars_df._df.semi_join(right_polars_df._df, on=on_param_semi),
        )

    # Perform standard join
    return cast(
        Union[DataFrame, AsyncDataFrame],
        left_polars_df._df.join(right_polars_df._df, on=join_on, how=join_how),
    )


def build_polars_slice_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    offset: int,
    length: Optional[int] = None,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a slice operation for Polars DataFrame.

    Args:
        polars_df: Polars DataFrame instance
        offset: Starting row index
        length: Number of rows to return (if None, returns all remaining rows)

    Returns:
        Resulting underlying DataFrame

    Note:
        This is a simplified implementation. Full slice support requires OFFSET support
        in the underlying DataFrame, which may not be available.
    """
    if length is None:
        # Return all rows from offset onwards
        # This is a limitation - we use a large number as workaround
        return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.limit(offset + 1000000))
    else:
        # Use limit with offset calculation
        return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.limit(offset + length))


def build_polars_gather_every_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
    n: int,
    offset: int = 0,
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a gather_every operation for Polars DataFrame (sample every nth row).

    Args:
        polars_df: Polars DataFrame instance
        n: Sample every nth row
        offset: Starting offset

    Returns:
        Resulting underlying DataFrame

    Note:
        This implementation uses row number window function and modulo operation
        to sample rows. The row number column is added, filtered, then removed.
    """
    from ...expressions import functions as F

    # Add row number, filter by modulo, then remove row number
    # Use over() with empty partition/order for global row number
    row_num_col = F.row_number().over()

    # Add row number column
    df_with_row_num = polars_df._df.select(
        *[col(c) for c in polars_df.columns], row_num_col.alias("__row_num__")
    )

    # Filter by modulo
    filtered_df = df_with_row_num.where((col("__row_num__") - offset) % n == 0)

    # Drop row number column and return result
    result_df = filtered_df.select(*[col(c) for c in polars_df.columns])
    return cast(Union[DataFrame, AsyncDataFrame], result_df)


def build_polars_describe_operation(
    polars_df: Any,  # PolarsDataFrameProtocol - using Any for mypy compatibility
) -> Union["DataFrame", "AsyncDataFrame"]:
    """Build a describe operation for Polars DataFrame (descriptive statistics).

    Args:
        polars_df: Polars DataFrame instance

    Returns:
        Resulting underlying DataFrame with statistics

    Note:
        Standard deviation (std) may not be available in all databases (e.g., SQLite).
        In such cases, std will be omitted.
    """
    from ...expressions import functions as F

    # Get numeric columns
    numeric_cols = [c for c in polars_df.columns if _is_numeric_column(polars_df, c)]
    if not numeric_cols:
        return cast(Union["DataFrame", "AsyncDataFrame"], polars_df._df)

    stats_exprs = []
    for col_name in numeric_cols:
        col_expr = col(col_name)
        stats_exprs.extend(
            [
                F.count(col_expr).alias(f"{col_name}_count"),
                F.avg(col_expr).alias(f"{col_name}_mean"),
                F.min(col_expr).alias(f"{col_name}_min"),
                F.max(col_expr).alias(f"{col_name}_max"),
            ]
        )
        # Note: stddev is omitted as it's not supported by all databases (e.g., SQLite)

    return cast(Union[DataFrame, AsyncDataFrame], polars_df._df.select(*stats_exprs))


def _is_numeric_column(polars_df: PolarsDataFrameProtocol, col_name: str) -> bool:
    """Check if a column is numeric based on schema.

    Args:
        polars_df: Polars DataFrame instance (must have schema property)
        col_name: Column name to check

    Returns:
        True if the column is numeric, False otherwise
    """
    # Access schema through PolarsDataFrame's schema property which returns List[Tuple[str, str]]
    # Schema can be either a property (sync) or async method (async)
    schema: Any = getattr(polars_df, "schema", [])
    # Handle async schema - if it's a coroutine, we can't check it here
    # This function should only be called after schema is available
    if not isinstance(schema, list):
        schema = []
    for name, dtype in schema:
        if name == col_name:
            # Check if dtype is numeric
            numeric_dtypes = ["Int64", "Int32", "Int8", "Float64", "Float32"]
            return dtype in numeric_dtypes
    return False
