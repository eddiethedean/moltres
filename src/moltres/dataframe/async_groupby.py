"""Async grouped DataFrame operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence, Union

from ..expressions.column import Column, col
from ..logical import operators
from ..logical.plan import LogicalPlan
from .async_dataframe import AsyncDataFrame

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase


class AsyncGroupedDataFrame:
    """Represents a grouped DataFrame for async aggregation operations."""

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
                - Column expressions (e.g., sum(col("amount")))
                - String column names (e.g., "amount" - defaults to sum())
                - Dictionary mapping column names to aggregation functions
                  (e.g., {"amount": "sum", "price": "avg"})

        Returns:
            AsyncDataFrame with aggregated results

        Example:
            >>> from moltres import col
            >>> from moltres.expressions.functions import sum, avg
            >>> # Using Column expressions
            >>> await df.group_by("category").agg(sum(col("amount")).alias("total"))

            >>> # Using string column names (defaults to sum)
            >>> await df.group_by("category").agg("amount", "price")

            >>> # Using dictionary syntax
            >>> await df.group_by("category").agg({"amount": "sum", "price": "avg"})
        """
        # Extract grouping keys from current plan
        from ..logical.plan import Aggregate

        if not isinstance(self.plan, Aggregate) or not self.plan.grouping:
            raise ValueError("GroupedDataFrame must have grouping columns")

        grouping = self.plan.grouping

        # Normalize all aggregations to Column expressions
        normalized_aggs = []
        for agg_expr in aggregates:
            if isinstance(agg_expr, str):
                # String column name - default to sum() and alias with column name
                from ..expressions.functions import sum as sum_func

                normalized_aggs.append(sum_func(col(agg_expr)).alias(agg_expr))
            elif isinstance(agg_expr, dict):
                # Dictionary syntax: {"column": "function"}
                for col_name, func_name in agg_expr.items():
                    agg_col = self._create_aggregation_from_string(col_name, func_name)
                    normalized_aggs.append(agg_col)
            elif isinstance(agg_expr, Column):
                # Already a Column expression
                normalized_aggs.append(agg_expr)
            else:
                raise ValueError(
                    f"Invalid aggregation type: {type(agg_expr)}. "
                    "Expected Column, str, or Dict[str, str]"
                )

        new_plan = operators.aggregate(
            self.plan.child, keys=grouping, aggregates=tuple(normalized_aggs)
        )
        return AsyncDataFrame(
            plan=new_plan,
            database=self.database,
        )

    @staticmethod
    def _create_aggregation_from_string(column_name: str, func_name: str) -> Column:
        """Create an aggregation Column from a column name and function name string.

        Args:
            column_name: Name of the column to aggregate
            func_name: Name of the aggregation function (e.g., "sum", "avg", "min", "max", "count")

        Returns:
            Column expression for the aggregation

        Raises:
            ValueError: If the function name is not recognized
        """
        from ..expressions.functions import (
            avg,
            count,
            max as max_func,
            min as min_func,
            sum as sum_func,
            count_distinct,
        )
        from typing import Callable

        func_map: Dict[str, Callable[[Column], Column]] = {
            "sum": sum_func,
            "avg": avg,
            "average": avg,  # Alias for avg
            "min": min_func,
            "max": max_func,
            "count": count,
            "count_distinct": count_distinct,
        }

        func_name_lower = func_name.lower()
        if func_name_lower not in func_map:
            raise ValueError(
                f"Unknown aggregation function: {func_name}. "
                f"Supported functions: {', '.join(func_map.keys())}"
            )

        agg_func = func_map[func_name_lower]
        result = agg_func(col(column_name)).alias(column_name)
        return result

    def pivot(
        self, pivot_col: str, values: Optional[Sequence[str]] = None
    ) -> "AsyncPivotedGroupedDataFrame":
        """Pivot the grouped data on a column.

        Args:
            pivot_col: Column to pivot on (values become column headers)
            values: Optional list of specific values to pivot (if None, must be provided later or discovered)

        Returns:
            AsyncPivotedGroupedDataFrame that can be aggregated

        Example:
            >>> await df.group_by("category").pivot("status").agg("amount")
            >>> await df.group_by("category").pivot("status", values=["active", "inactive"]).agg("amount")
        """
        # Extract grouping keys from current plan
        from ..logical.plan import Aggregate

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
    """Represents an async DataFrame grouped by columns with a pivot operation applied.

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
                - Column expressions (e.g., sum(col("amount")))
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
            >>> from moltres.expressions.functions import sum
            >>> # Using string column name
            >>> await df.group_by("category").pivot("status").agg("amount")

            >>> # Using Column expression
            >>> await df.group_by("category").pivot("status").agg(sum(col("amount")))

            >>> # With specific pivot values
            >>> await df.group_by("category").pivot("status", values=["active", "inactive"]).agg("amount")
        """
        if not aggregations:
            raise ValueError("agg requires at least one aggregation expression")

        # Normalize all aggregations to Column expressions
        normalized_aggs = []
        for agg_expr in aggregations:
            if isinstance(agg_expr, str):
                # String column name - default to sum()
                from ..expressions.functions import sum as sum_func

                normalized_aggs.append(sum_func(col(agg_expr)))
            elif isinstance(agg_expr, dict):
                # Dictionary syntax: {"column": "function"}
                for col_name, func_name in agg_expr.items():
                    agg_col = self._create_aggregation_from_string(col_name, func_name)
                    normalized_aggs.append(agg_col)
            elif isinstance(agg_expr, Column):
                # Already a Column expression
                normalized_aggs.append(agg_expr)
            else:
                raise ValueError(
                    f"Invalid aggregation type: {type(agg_expr)}. "
                    "Expected Column, str, or Dict[str, str]"
                )

        # For pivoted grouped data, we can only aggregate one column at a time
        if len(normalized_aggs) > 1:
            raise ValueError(
                "Pivoted grouped aggregation supports only one aggregation expression. "
                "Multiple aggregations are not supported with pivot."
            )

        agg_expr = normalized_aggs[0]
        self._validate_aggregation(agg_expr)

        # Extract the value column from the aggregation
        value_column = self._extract_value_column(agg_expr)

        # Extract the aggregation function name
        agg_func = self._extract_agg_func(agg_expr)

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
            agg_expr: Aggregation Column expression (e.g., sum(col("amount")))

        Returns:
            Column name string (e.g., "amount")

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

    @staticmethod
    def _extract_agg_func(agg_expr: Column) -> str:
        """Extract the aggregation function name from an aggregation expression.

        Args:
            agg_expr: Aggregation Column expression (e.g., sum(col("amount")))

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

    @staticmethod
    def _create_aggregation_from_string(column_name: str, func_name: str) -> Column:
        """Create an aggregation Column from a column name and function name string.

        Args:
            column_name: Name of the column to aggregate
            func_name: Name of the aggregation function (e.g., "sum", "avg", "min", "max", "count")

        Returns:
            Column expression for the aggregation

        Raises:
            ValueError: If the function name is not recognized
        """
        from ..expressions.functions import (
            avg,
            count,
            max as max_func,
            min as min_func,
            sum as sum_func,
            count_distinct,
        )
        from typing import Callable

        func_map: Dict[str, Callable[[Column], Column]] = {
            "sum": sum_func,
            "avg": avg,
            "average": avg,  # Alias for avg
            "min": min_func,
            "max": max_func,
            "count": count,
            "count_distinct": count_distinct,
        }

        func_name_lower = func_name.lower()
        if func_name_lower not in func_map:
            raise ValueError(
                f"Unknown aggregation function: {func_name}. "
                f"Supported functions: {', '.join(func_map.keys())}"
            )

        agg_func = func_map[func_name_lower]
        result = agg_func(col(column_name))
        return result

    @staticmethod
    def _validate_aggregation(expr: Column) -> Column:
        """Validate that an expression is a valid aggregation.

        Args:
            expr: Column expression to validate

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
