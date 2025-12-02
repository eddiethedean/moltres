"""Shared helper functions for Polars GroupBy operations.

This module contains shared logic used by both PolarsGroupBy and AsyncPolarsGroupBy
to reduce code duplication and improve maintainability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, List, Protocol, Tuple, TypeVar, Union

from ...expressions.column import Column, col

if TYPE_CHECKING:
    from ..core.dataframe import DataFrame
    from ..core.async_dataframe import AsyncDataFrame
    from ..groupby.groupby import GroupedDataFrame
    from ..groupby.async_groupby import AsyncGroupedDataFrame
    from ...logical.plan import LogicalPlan
else:
    GroupedDataFrame = Any
    AsyncGroupedDataFrame = Any
    DataFrame = Any
    AsyncDataFrame = Any
    LogicalPlan = Any

# Type variable for generic Polars GroupBy operations
G = TypeVar("G", bound="PolarsGroupByProtocol")

if TYPE_CHECKING:

    class PolarsGroupByProtocol(Protocol):
        """Protocol defining the interface that Polars GroupBy classes must implement."""

        _grouped: Union[GroupedDataFrame, AsyncGroupedDataFrame]

        def agg(
            self, *aggregations: Union[Column, str, dict[str, str]]
        ) -> Union[DataFrame, AsyncDataFrame]:
            """Apply aggregations to the grouped data."""
            ...
else:
    PolarsGroupByProtocol = Any


def build_polars_groupby_agg_with_dict_handling(
    polars_groupby: Any,  # PolarsGroupByProtocol - using Any for mypy compatibility
    *exprs: Union[Column, dict[str, str]],
) -> Tuple[List[Column], Union[GroupedDataFrame, AsyncGroupedDataFrame]]:
    """Build aggregations for Polars GroupBy with dictionary syntax support.

    Args:
        polars_groupby: Polars GroupBy instance
        *exprs: Column expressions or dictionaries mapping column names to function names

    Returns:
        Tuple of (normalized expressions list, grouped dataframe instance)
    """
    # Handle dictionary syntax
    normalized_exprs = []
    for expr in exprs:
        if isinstance(expr, dict):
            # Dictionary syntax: {"column": "function"}
            for col_name, func_name in expr.items():
                agg_col = polars_groupby._grouped._create_aggregation_from_string(
                    col_name, func_name
                )
                normalized_exprs.append(agg_col)
        else:
            # Column expression
            normalized_exprs.append(expr)

    return normalized_exprs, polars_groupby._grouped


def build_polars_groupby_column_aggregation(
    polars_groupby: Any,  # PolarsGroupByProtocol - using Any for mypy compatibility
    func_name: str,
    func_builder: Callable[[Column], Column],
    suffix: str,
    error_msg: str,
) -> List[Column]:
    """Build column-based aggregation for Polars GroupBy methods like mean, sum, etc.

    Args:
        polars_groupby: Polars GroupBy instance
        func_name: Name of the aggregation function (for error messages)
        func_builder: Function that takes a Column and returns an aggregated Column
        suffix: Suffix for aliases (e.g., "_mean", "_sum")
        error_msg: Error message to show if no columns found

    Returns:
        List of aggregation Column expressions

    Raises:
        NotImplementedError: If columns cannot be extracted
        ValueError: If no columns found for aggregation
    """
    # Extract columns from the grouped dataframe
    from ...logical.plan import Aggregate

    grouped = polars_groupby._grouped
    columns: List[str] = []

    # Try to get columns - sync and async have different structures
    if hasattr(grouped, "parent"):
        # Sync GroupBy - has parent DataFrame
        parent_df = grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                f"{func_name}() requires accessible columns - use agg() with explicit columns"
            )
    elif hasattr(grouped, "plan") and isinstance(grouped.plan, Aggregate) and grouped.plan.child:
        # Async GroupBy - extract from plan
        from ..core.async_dataframe import AsyncDataFrame

        child_plan = grouped.plan.child
        temp_df = AsyncDataFrame(plan=child_plan, database=getattr(grouped, "database", None))
        try:
            columns = temp_df._extract_column_names(child_plan)
        except Exception:
            raise NotImplementedError(
                f"{func_name}() requires accessible columns - use agg() with explicit columns"
            )
    else:
        raise NotImplementedError(
            f"{func_name}() requires accessible columns - use agg() with explicit columns"
        )

    # Build aggregation list
    agg_list = []
    for col_name in columns:
        try:
            agg_col = func_builder(col(col_name))
            agg_list.append(agg_col.alias(f"{col_name}{suffix}"))
        except Exception:
            pass

    if not agg_list:
        raise ValueError(error_msg)

    return agg_list
