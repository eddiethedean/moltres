"""Grouped DataFrame helper."""

from __future__ import annotations

from dataclasses import dataclass

from ..expressions.column import Column
from ..logical import operators
from ..logical.plan import LogicalPlan
from .dataframe import DataFrame


@dataclass(frozen=True)
class GroupedDataFrame:
    """Represents a DataFrame grouped by one or more columns.

    This is returned by DataFrame.group_by() and provides aggregation methods.
    """

    plan: LogicalPlan
    keys: tuple[Column, ...]
    parent: DataFrame

    def agg(self, *aggregations: Column) -> DataFrame:
        """Apply aggregation functions to the grouped data.

        Args:
            *aggregations: One or more aggregation expressions (e.g., sum(), avg(), count())

        Returns:
            DataFrame with aggregated results

        Raises:
            ValueError: If no aggregations are provided or if invalid
                aggregation expressions are used

        Example:
            >>> from moltres import col
            >>> from moltres.expressions.functions import sum, avg, count
            >>> df.group_by("category").agg(
            ...     sum(col("amount")).alias("total"),
            ...     avg(col("price")).alias("avg_price"),
            ...     count("*").alias("count")
            ... )
        """
        if not aggregations:
            raise ValueError("agg requires at least one aggregation expression")
        normalized = tuple(self._validate_aggregation(expr) for expr in aggregations)
        plan = operators.aggregate(self.plan, self.keys, normalized)
        return DataFrame(plan=plan, database=self.parent.database)

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
