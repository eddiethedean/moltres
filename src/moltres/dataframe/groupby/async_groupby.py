"""Async grouped :class:`DataFrame` operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

from ...expressions.column import Column, col
from ...logical import operators
from ...logical.plan import LogicalPlan
from ..core.async_dataframe import AsyncDataFrame

if TYPE_CHECKING:
    from ...table.async_table import AsyncDatabase


class AsyncGroupedDataFrame:
    """Represents a grouped :class:`DataFrame` for async aggregation operations."""

    def __init__(
        self,
        plan: LogicalPlan,
        database: Optional["AsyncDatabase"] = None,
    ):
        self.plan = plan
        self.database = database

    def agg(self, *aggregates: Union[Column, str, Dict[str, str]]) -> AsyncDataFrame:
        """Apply aggregate functions to the grouped data.

        Args:
            *aggregates: Aggregation expressions. Can be:
                - :class:`Column` expressions (e.g., sum(col("amount")))
                - String column names (e.g., "amount" - defaults to sum())
                - Dictionary mapping column names to aggregation functions
                  (e.g., {"amount": "sum", "price": "avg"})

        Returns:
            AsyncDataFrame with aggregated results

        Example:
            >>> from moltres import col
            >>> from moltres.expressions import functions as F
            >>> # Using :class:`Column` expressions
            >>> await df.group_by("category").agg(F.sum(col("amount")).alias("total"))

            >>> # Using string column names (defaults to sum)
            >>> await df.group_by("category").agg("amount", "price")

            >>> # Using dictionary syntax
            >>> await df.group_by("category").agg({"amount": "sum", "price": "avg"})
        """
        # Extract grouping keys from current plan
        from ...logical.plan import Aggregate

        if not isinstance(self.plan, Aggregate) or not self.plan.grouping:
            raise ValueError("GroupedDataFrame must have grouping columns")

        grouping = self.plan.grouping

        # Normalize all aggregations to Column expressions
        from ..helpers.groupby_helpers import normalize_aggregations

        # Allow empty aggregations for special cases like dropDuplicates
        normalized_aggs = normalize_aggregations(
            aggregates, alias_with_column_name=True, allow_empty=True
        )

        # If no aggregations, just return grouping columns (for dropDuplicates)
        if not normalized_aggs:
            # Select only grouping columns and apply distinct
            grouping_cols = list(grouping)
            new_plan = operators.project(self.plan.child, tuple(grouping_cols))
            new_plan = operators.distinct(new_plan)  # type: ignore[assignment]
        else:
            new_plan = operators.aggregate(
                self.plan.child, keys=grouping, aggregates=tuple(normalized_aggs)
            )  # type: ignore[assignment]
        return AsyncDataFrame(
            plan=new_plan,
            database=self.database,
        )

    @staticmethod
    def _create_aggregation_from_string(column_name: str, func_name: str) -> Column:
        """Create an aggregation :class:`Column` from a column name and function name string.

        Args:
            column_name: Name of the column to aggregate
            func_name: Name of the aggregation function (e.g., "sum", "avg", "min", "max", "count")

        Returns:
            :class:`Column` expression for the aggregation

        Raises:
            ValueError: If the function name is not recognized
        """
        from ..helpers.groupby_helpers import create_aggregation_from_string

        return create_aggregation_from_string(column_name, func_name)

    def pivot(
        self, pivot_col: str, values: Optional[Sequence[str]] = None
    ) -> "AsyncPivotedGroupedDataFrame":
        """Pivot the grouped data on a column.

        Args:
            pivot_col: :class:`Column` to pivot on (values become column headers)
            values: Optional list of specific values to pivot (if None, must be provided later or discovered)

        Returns:
            AsyncPivotedGroupedDataFrame that can be aggregated

        Example:
            >>> await df.group_by("category").pivot("status").agg("amount")
            >>> await df.group_by("category").pivot("status", values=["active", "inactive"]).agg("amount")
        """
        # Extract grouping keys from current plan
        from ...logical.plan import Aggregate

        if not isinstance(self.plan, Aggregate) or not self.plan.grouping:
            raise ValueError("GroupedDataFrame must have grouping columns")

        grouping = self.plan.grouping
        return AsyncPivotedGroupedDataFrame(
            plan=self.plan,
            grouping=grouping,
            pivot_column=pivot_col,
            pivot_values=tuple(values) if values else None,
            database=self.database,
        )


class AsyncPivotedGroupedDataFrame:
    """Represents an async :class:`DataFrame` grouped by columns with a pivot operation applied.

    This is returned by AsyncGroupedDataFrame.pivot() and provides aggregation methods
    that will create pivoted columns.
    """

    def __init__(
        self,
        plan: LogicalPlan,
        grouping: tuple[Column, ...],
        pivot_column: str,
        pivot_values: Optional[tuple[str, ...]],
        database: Optional["AsyncDatabase"] = None,
    ):
        self.plan = plan
        self.grouping = grouping
        self.pivot_column = pivot_column
        self.pivot_values = pivot_values
        self.database = database

    async def agg(self, *aggregations: Union[Column, str, Dict[str, str]]) -> AsyncDataFrame:
        """Apply aggregation functions to the pivoted grouped data.

        Args:
            *aggregations: One or more aggregation expressions. Can be:
                - :class:`Column` expressions (e.g., sum(col("amount")))
                - String column names (e.g., "amount" - defaults to sum())
                - Dictionary mapping column names to aggregation functions
                  (e.g., {"amount": "sum", "price": "avg"})

        Returns:
            AsyncDataFrame with pivoted aggregated results

        Raises:
            ValueError: If no aggregations are provided or if invalid
                aggregation expressions are used

        Example:
            >>> from moltres import col
            >>> from moltres.expressions import functions as F
            >>> # Using string column name
            >>> await df.group_by("category").pivot("status").agg("amount")

            >>> # Using :class:`Column` expression
            >>> await df.group_by("category").pivot("status").agg(F.sum(col("amount")))

            >>> # With specific pivot values
            >>> await df.group_by("category").pivot("status", values=["active", "inactive"]).agg("amount")
        """
        if not aggregations:
            raise ValueError("agg requires at least one aggregation expression")

        # Normalize all aggregations to Column expressions
        from ..helpers.groupby_helpers import (
            normalize_aggregations,
            validate_aggregation,
            extract_value_column,
            extract_agg_func,
        )

        normalized_aggs = normalize_aggregations(aggregations, alias_with_column_name=False)

        # For pivoted grouped data, we can only aggregate one column at a time
        if len(normalized_aggs) > 1:
            raise ValueError(
                "Pivoted grouped aggregation supports only one aggregation expression. "
                "Multiple aggregations are not supported with pivot."
            )

        agg_expr = normalized_aggs[0]
        validate_aggregation(agg_expr)

        # Extract the value column from the aggregation
        value_column = extract_value_column(agg_expr)

        # Extract the aggregation function name
        agg_func = extract_agg_func(agg_expr)

        # If pivot_values is not provided, infer them from the data (PySpark behavior)
        pivot_values = self.pivot_values
        if pivot_values is None:
            # Query distinct values from the pivot column
            # We need to use the child plan (before aggregation) to get distinct values

            plan_children = self.plan.children()
            if not plan_children:
                raise ValueError("Plan must have at least one child for pivot value inference")
            child_plan = plan_children[0]
            distinct_df = AsyncDataFrame(plan=child_plan, database=self.database)
            distinct_df = distinct_df.select(col(self.pivot_column)).distinct()
            distinct_rows = await distinct_df.collect()
            pivot_values = tuple(
                str(row[self.pivot_column])
                for row in distinct_rows
                if row[self.pivot_column] is not None
            )

            if not pivot_values:
                raise ValueError(
                    f"No distinct values found in pivot column '{self.pivot_column}'. "
                    "Please provide pivot_values explicitly."
                )

        # Create a GroupedPivot logical plan
        plan_children = self.plan.children()
        if not plan_children:
            raise ValueError("Plan must have at least one child for grouped pivot")
        child_plan = plan_children[0]
        plan = operators.grouped_pivot(
            child_plan,
            grouping=self.grouping,
            pivot_column=self.pivot_column,
            value_column=value_column,
            agg_func=agg_func,
            pivot_values=pivot_values,
        )
        return AsyncDataFrame(plan=plan, database=self.database)

    @staticmethod
    def _extract_value_column(agg_expr: Column) -> str:
        """Extract the column name from an aggregation expression.

        Args:
            agg_expr: Aggregation :class:`Column` expression (e.g., sum(col("amount")))

        Returns:
            :class:`Column` name string (e.g., "amount")

        Raises:
            ValueError: If the column cannot be extracted
        """
        from ..helpers.groupby_helpers import extract_value_column

        return extract_value_column(agg_expr)

    @staticmethod
    def _extract_agg_func(agg_expr: Column) -> str:
        """Extract the aggregation function name from an aggregation expression.

        Args:
            agg_expr: Aggregation :class:`Column` expression (e.g., sum(col("amount")))

        Returns:
            Aggregation function name (e.g., "sum")
        """
        from ..helpers.groupby_helpers import extract_agg_func

        return extract_agg_func(agg_expr)

    @staticmethod
    def _create_aggregation_from_string(column_name: str, func_name: str) -> Column:
        """Create an aggregation :class:`Column` from a column name and function name string.

        Args:
            column_name: Name of the column to aggregate
            func_name: Name of the aggregation function (e.g., "sum", "avg", "min", "max", "count")

        Returns:
            :class:`Column` expression for the aggregation

        Raises:
            ValueError: If the function name is not recognized
        """
        from ..helpers.groupby_helpers import create_aggregation_from_string

        return create_aggregation_from_string(column_name, func_name)

    @staticmethod
    def _validate_aggregation(expr: Column) -> Column:
        """Validate that an expression is a valid aggregation.

        Args:
            expr: :class:`Column` expression to validate

        Returns:
            The validated column expression

        Raises:
            ValueError: If the expression is not a valid aggregation
        """
        from ..helpers.groupby_helpers import validate_aggregation

        return validate_aggregation(expr)
