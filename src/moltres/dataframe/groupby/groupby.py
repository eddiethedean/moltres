"""Grouped :class:`DataFrame` helper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Union

from ...expressions.column import Column, col
from ...logical import operators
from ...logical.plan import LogicalPlan
from ..core.dataframe import DataFrame


@dataclass(frozen=True)
class GroupedDataFrame:
    """Represents a :class:`DataFrame` grouped by one or more columns.

    This is returned by :class:`DataFrame`.group_by() and provides aggregation methods.
    """

    plan: LogicalPlan
    keys: tuple[Column, ...]
    parent: DataFrame

    def agg(self, *aggregations: Union[Column, str, Dict[str, str]]) -> DataFrame:
        """Apply aggregation functions to the grouped data.

        Args:
            *aggregations: One or more aggregation expressions. Can be:
                - :class:`Column` expressions (e.g., sum(col("amount")))
                - String column names (e.g., "amount" - defaults to sum())
                - Dictionary mapping column names to aggregation functions
                  (e.g., {"amount": "sum", "price": "avg"})

        Returns:
            :class:`DataFrame` with aggregated results

        Raises:
            ValueError: If no aggregations are provided or if invalid
                aggregation expressions are used

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("sales", [column("category", "TEXT"), column("amount", "REAL"), column("price", "REAL")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> :class:`Records`(_data=[{"category": "A", "amount": 100.0, "price": 10.0}, {"category": "A", "amount": 200.0, "price": 20.0}, {"category": "B", "amount": 150.0, "price": 15.0}], _database=db).insert_into("sales")
            >>> # Using :class:`Column` expressions
            >>> df = db.table("sales").select()
            >>> result = df.group_by("category").agg(F.sum(col("amount")).alias("total"), F.avg(col("price")).alias("avg_price"))
            >>> results = result.collect()
            >>> len(results)
            2
            >>> results[0]["total"]
            300.0
            >>> # Using string column names (defaults to sum)
            >>> result2 = df.group_by("category").agg("amount")
            >>> results2 = result2.collect()
            >>> results2[0]["amount"]
            300.0
            >>> # Using dictionary syntax
            >>> result3 = df.group_by("category").agg({"amount": "sum", "price": "avg"})
            >>> results3 = result3.collect()
            >>> results3[0]["amount"]
            300.0
            >>> db.close()
        """
        if not aggregations:
            raise ValueError("agg requires at least one aggregation expression")

        # Normalize all aggregations to Column expressions
        from ..helpers.groupby_helpers import normalize_aggregations, validate_aggregation

        # Allow empty aggregations for special cases like dropDuplicates
        normalized_aggs = normalize_aggregations(
            aggregations, alias_with_column_name=True, allow_empty=True
        )

        # If no aggregations, just return grouping columns (for dropDuplicates)
        if not normalized_aggs:
            # Select only grouping columns and apply distinct
            grouping_cols = list(self.keys)
            plan = operators.project(self.plan, tuple(grouping_cols))
            plan = operators.distinct(plan)  # type: ignore[assignment]
        else:
            normalized = tuple(validate_aggregation(expr) for expr in normalized_aggs)
            plan = operators.aggregate(self.plan, self.keys, normalized)  # type: ignore[assignment]
        return DataFrame(plan=plan, database=self.parent.database)

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
    ) -> "PivotedGroupedDataFrame":
        """Pivot the grouped data on a column.

        Args:
            pivot_col: :class:`Column` to pivot on (values become column headers)
            values: Optional list of specific values to pivot (if None, must be provided later or discovered)

        Returns:
            PivotedGroupedDataFrame that can be aggregated

        Example:
            >>> df.group_by("category").pivot("status").agg("amount")
            >>> df.group_by("category").pivot("status", values=["active", "inactive"]).agg("amount")
        """
        return PivotedGroupedDataFrame(
            plan=self.plan,
            keys=self.keys,
            pivot_column=pivot_col,
            pivot_values=tuple(values) if values else None,
            parent=self.parent,
        )

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
        if not expr.op.startswith("agg_"):
            raise ValueError(
                "Aggregation expressions must be created with moltres aggregate helpers "
                "(e.g., sum(), avg(), count(), min(), max())"
            )
        return expr


@dataclass(frozen=True)
class PivotedGroupedDataFrame:
    """Represents a :class:`DataFrame` grouped by columns with a pivot operation applied.

    This is returned by :class:`GroupedDataFrame`.pivot() and provides aggregation methods
    that will create pivoted columns.
    """

    plan: LogicalPlan
    keys: tuple[Column, ...]
    pivot_column: str
    pivot_values: Optional[tuple[str, ...]]
    parent: DataFrame

    def agg(self, *aggregations: Union[Column, str, Dict[str, str]]) -> DataFrame:
        """Apply aggregation functions to the pivoted grouped data.

        Args:
            *aggregations: One or more aggregation expressions. Can be:
                - :class:`Column` expressions (e.g., sum(col("amount")))
                - String column names (e.g., "amount" - defaults to sum())
                - Dictionary mapping column names to aggregation functions
                  (e.g., {"amount": "sum", "price": "avg"})

        Returns:
            :class:`DataFrame` with pivoted aggregated results

        Raises:
            ValueError: If no aggregations are provided or if invalid
                aggregation expressions are used

        Example:
            >>> from moltres import col
            >>> from moltres.expressions import functions as F
            >>> # Using string column name
            >>> df.group_by("category").pivot("status").agg("amount")

            >>> # Using :class:`Column` expression
            >>> df.group_by("category").pivot("status").agg(F.sum(col("amount")))

            >>> # With specific pivot values
            >>> df.group_by("category").pivot("status", values=["active", "inactive"]).agg("amount")
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
        # (PySpark behavior - pivot with multiple aggregations requires different syntax)
        if len(normalized_aggs) > 1:
            raise ValueError(
                "Pivoted grouped aggregation supports only one aggregation expression. "
                "Multiple aggregations are not supported with pivot."
            )

        agg_expr = normalized_aggs[0]
        validate_aggregation(agg_expr)

        # Extract the value column from the aggregation
        # For sum(col("amount")), we need "amount"
        value_column = extract_value_column(agg_expr)

        # Extract the aggregation function name
        agg_func = extract_agg_func(agg_expr)

        # If pivot_values is not provided, infer them from the data (PySpark behavior)
        pivot_values = self.pivot_values
        if pivot_values is None:
            # Query distinct values from the pivot column
            distinct_df = DataFrame(plan=self.plan, database=self.parent.database)
            distinct_df = distinct_df.select(col(self.pivot_column)).distinct()
            distinct_rows = distinct_df.collect()
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
        plan = operators.grouped_pivot(
            self.plan,
            grouping=self.keys,
            pivot_column=self.pivot_column,
            value_column=value_column,
            agg_func=agg_func,
            pivot_values=pivot_values,
        )
        return DataFrame(plan=plan, database=self.parent.database)

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
