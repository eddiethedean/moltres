"""DataFrame statistics operations.

This module handles statistics operations like count, nunique, describe, and summary.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Union

if TYPE_CHECKING:
    from ..core.dataframe import DataFrame


class StatisticsCalculator:
    """Handles statistics calculations for DataFrames."""

    def __init__(self, df: "DataFrame"):
        """Initialize statistics calculator with a DataFrame.

        Args:
            df: The DataFrame to calculate statistics for
        """
        self._df = df

    def count(self) -> int:
        """Return the number of rows in the DataFrame.

        Returns:
            Number of rows

        Note:
            This executes a COUNT(*) query against the database.
        """
        from ...expressions.functions import count as count_func
        from ...logical import operators

        # Create an aggregate with count(*)
        count_col = count_func("*").alias("count")
        result_df = self._df._with_plan(operators.aggregate(self._df.plan, (), (count_col,)))
        results = result_df.collect()
        if not isinstance(results, list):
            raise TypeError("count() requires collect() to return a list, not an iterator")
        if results:
            count_value = results[0].get("count", 0)
            return int(count_value) if isinstance(count_value, (int, float, str)) else 0
        return 0

    def nunique(self, column: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """Count distinct values in column(s).

        Args:
            column: Column name to count. If None, counts distinct values for all columns.

        Returns:
            If column is specified: integer count of distinct values.
            If column is None: dictionary mapping column names to distinct counts.
        """
        from ...expressions.column import col
        from ...expressions.functions import count_distinct

        if column is not None:
            # Count distinct values in the column
            count_df = self._df.select(count_distinct(col(column)).alias("count"))
            result = count_df.collect()
            if result and isinstance(result, list) and len(result) > 0:
                row = result[0]
                if isinstance(row, dict):
                    count_val = row.get("count", 0)
                    return int(count_val) if isinstance(count_val, (int, float)) else 0
            return 0
        else:
            # Count distinct for all columns
            from .schema import SchemaInspector

            inspector = SchemaInspector(self._df)
            counts: Dict[str, int] = {}
            for col_name in inspector.columns():
                count_df = self._df.select(count_distinct(col(col_name)).alias("count"))
                result = count_df.collect()
                if result and isinstance(result, list) and len(result) > 0:
                    row = result[0]
                    if isinstance(row, dict):
                        count_val = row.get("count", 0)
                        counts[col_name] = (
                            int(count_val) if isinstance(count_val, (int, float)) else 0
                        )
                    else:
                        counts[col_name] = 0
                else:
                    counts[col_name] = 0
            return counts

    def describe(self, *cols: str) -> "DataFrame":
        """Compute basic statistics for numeric columns.

        Args:
            *cols: Optional column names to describe. If not provided, describes all numeric columns.

        Returns:
            DataFrame with statistics: count, mean, stddev, min, max

        Note:
            This is a simplified implementation. A full implementation would
            automatically detect numeric columns if cols is not provided.
        """
        from ...expressions.column import col
        from ...expressions.functions import avg, count, max, min
        from ...logical import operators

        if not cols:
            # For now, return empty DataFrame if no columns specified
            # A full implementation would detect numeric columns
            return self._df.limit(0)

        # Build aggregations for each column
        aggregations = []
        for col_name in cols:
            col_expr = col(col_name)
            aggregations.extend(
                [
                    count(col_expr).alias(f"{col_name}_count"),
                    avg(col_expr).alias(f"{col_name}_mean"),
                    min(col_expr).alias(f"{col_name}_min"),
                    max(col_expr).alias(f"{col_name}_max"),
                ]
            )

        return self._df._with_plan(operators.aggregate(self._df.plan, (), tuple(aggregations)))

    def summary(self, *statistics: str) -> "DataFrame":
        """Compute summary statistics for numeric columns.

        Args:
            *statistics: Statistics to compute (e.g., "count", "mean", "stddev", "min", "max").
                        If not provided, computes common statistics.

        Returns:
            DataFrame with summary statistics

        Note:
            This is a simplified implementation. A full implementation would
            automatically detect numeric columns and compute all statistics.
        """
        if not statistics:
            statistics = ("count", "mean", "min", "max")

        # This is a placeholder - full implementation would compute statistics
        # For now, return empty DataFrame
        return self._df.limit(0)
