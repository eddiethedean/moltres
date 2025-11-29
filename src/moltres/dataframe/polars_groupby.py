"""Polars-style GroupBy interface for Moltres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Union

from ..expressions.column import Column, col
from .groupby import GroupedDataFrame

if TYPE_CHECKING:
    from .polars_dataframe import PolarsDataFrame


@dataclass(frozen=True)
class PolarsGroupBy:
    """Polars-style GroupBy wrapper around Moltres :class:`GroupedDataFrame`.

    Provides Polars-style groupby API with expression-based aggregations.
    """

    _grouped: GroupedDataFrame

    def agg(self, *exprs: Union[Column, Dict[str, str]]) -> "PolarsDataFrame":
        """Apply aggregations using Polars-style expressions.

        Args:
            *exprs: :class:`Column` expressions for aggregations, or dictionary mapping column names to function names

        Returns:
            :class:`PolarsDataFrame` with aggregated results

        Example:
            >>> df.group_by('country').agg(col('amount').sum(), col('price').mean())
            >>> df.group_by('country').agg({"amount": "sum", "price": "avg"})  # Dictionary syntax
        """
        from .polars_dataframe import PolarsDataFrame

        # Handle dictionary syntax
        normalized_exprs = []
        for expr in exprs:
            if isinstance(expr, dict):
                # Dictionary syntax: {"column": "function"}
                for col_name, func_name in expr.items():
                    agg_col = self._grouped._create_aggregation_from_string(col_name, func_name)
                    normalized_exprs.append(agg_col)
            else:
                # Column expression
                normalized_exprs.append(expr)

        result_df = self._grouped.agg(*normalized_exprs)
        return PolarsDataFrame.from_dataframe(result_df)

    def mean(self) -> "PolarsDataFrame":
        """Mean of all numeric columns in each group.

        Returns:
            :class:`PolarsDataFrame` with mean of all numeric columns for each group

        Note:
            This attempts to average all columns. For better control, use agg() with
            specific columns.
        """
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "mean() requires accessible columns - use agg() with explicit columns"
            )

        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.avg(col(col_name)).alias(f"{col_name}_mean"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No numeric columns found to average")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def sum(self) -> "PolarsDataFrame":
        """Sum all numeric columns in each group.

        Returns:
            :class:`PolarsDataFrame` with sum of all numeric columns for each group
        """
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "sum() requires accessible columns - use agg() with explicit columns"
            )

        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.sum(col(col_name)).alias(f"{col_name}_sum"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No numeric columns found to sum")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def min(self) -> "PolarsDataFrame":
        """Minimum value of all columns in each group."""
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "min() requires accessible columns - use agg() with explicit columns"
            )

        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.min(col(col_name)).alias(f"{col_name}_min"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No columns found for min()")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def max(self) -> "PolarsDataFrame":
        """Maximum value of all columns in each group."""
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "max() requires accessible columns - use agg() with explicit columns"
            )

        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.max(col(col_name)).alias(f"{col_name}_max"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No columns found for max()")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def count(self) -> "PolarsDataFrame":
        """Count rows in each group."""
        from ..expressions.functions import count
        from .polars_dataframe import PolarsDataFrame

        result_df = self._grouped.agg(count("*").alias("count"))
        return PolarsDataFrame.from_dataframe(result_df)

    def std(self) -> "PolarsDataFrame":
        """Standard deviation of all numeric columns in each group."""
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "std() requires accessible columns - use agg() with explicit columns"
            )

        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.stddev(col(col_name)).alias(f"{col_name}_std"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No numeric columns found for std()")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def var(self) -> "PolarsDataFrame":
        """Variance of all numeric columns in each group."""
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "var() requires accessible columns - use agg() with explicit columns"
            )

        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.variance(col(col_name)).alias(f"{col_name}_var"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No numeric columns found for var()")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def first(self) -> "PolarsDataFrame":
        """Get first value of each column in each group."""
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "first() requires accessible columns - use agg() with explicit columns"
            )

        # Use MIN as proxy for first (works if data is ordered)
        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.min(col(col_name)).alias(f"{col_name}_first"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No columns found for first()")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def last(self) -> "PolarsDataFrame":
        """Get last value of each column in each group."""
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "last() requires accessible columns - use agg() with explicit columns"
            )

        # Use MAX as proxy for last (works if data is ordered)
        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.max(col(col_name)).alias(f"{col_name}_last"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No columns found for last()")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def n_unique(self) -> "PolarsDataFrame":
        """Count distinct values for all columns in each group."""
        from .polars_dataframe import PolarsDataFrame
        from ..expressions import functions as F

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "n_unique() requires accessible columns - use agg() with explicit columns"
            )

        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.count_distinct(col(col_name)).alias(f"{col_name}_n_unique"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No columns found for n_unique()")

        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)
