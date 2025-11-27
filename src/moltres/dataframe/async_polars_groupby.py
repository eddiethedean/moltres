"""Async Polars-style GroupBy interface for Moltres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..expressions.column import Column, col
from .async_groupby import AsyncGroupedDataFrame

if TYPE_CHECKING:
    from .async_polars_dataframe import AsyncPolarsDataFrame


@dataclass(frozen=True)
class AsyncPolarsGroupBy:
    """Async Polars-style GroupBy wrapper around Moltres AsyncGroupedDataFrame.

    Provides Polars-style groupby API with expression-based aggregations.
    """

    _grouped: AsyncGroupedDataFrame

    def agg(self, *exprs: Column) -> "AsyncPolarsDataFrame":
        """Apply aggregations using Polars-style expressions.

        Args:
            *exprs: Column expressions for aggregations

        Returns:
            AsyncPolarsDataFrame with aggregated results

        Example:
            >>> await df.group_by('country').agg(col('amount').sum(), col('price').mean())
        """
        from .async_polars_dataframe import AsyncPolarsDataFrame

        result_df = self._grouped.agg(*exprs)
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def mean(self) -> "AsyncPolarsDataFrame":
        """Mean of all numeric columns in each group.

        Returns:
            AsyncPolarsDataFrame with mean of all numeric columns for each group

        Note:
            This attempts to average all columns. For better control, use agg() with
            specific columns.
        """
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "mean() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def sum(self) -> "AsyncPolarsDataFrame":
        """Sum all numeric columns in each group.

        Returns:
            AsyncPolarsDataFrame with sum of all numeric columns for each group
        """
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "sum() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def min(self) -> "AsyncPolarsDataFrame":
        """Minimum value of all columns in each group."""
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "min() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def max(self) -> "AsyncPolarsDataFrame":
        """Maximum value of all columns in each group."""
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "max() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def count(self) -> "AsyncPolarsDataFrame":
        """Count rows in each group."""
        from ..expressions.functions import count
        from .async_polars_dataframe import AsyncPolarsDataFrame

        result_df = self._grouped.agg(count("*").alias("count"))
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def std(self) -> "AsyncPolarsDataFrame":
        """Standard deviation of all numeric columns in each group."""
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "std() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def var(self) -> "AsyncPolarsDataFrame":
        """Variance of all numeric columns in each group."""
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "var() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def first(self) -> "AsyncPolarsDataFrame":
        """Get first value of each column in each group."""
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "first() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def last(self) -> "AsyncPolarsDataFrame":
        """Get last value of each column in each group."""
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "last() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)

    def n_unique(self) -> "AsyncPolarsDataFrame":
        """Count distinct values for all columns in each group."""
        from .async_polars_dataframe import AsyncPolarsDataFrame
        from .async_dataframe import AsyncDataFrame
        from ..expressions import functions as F

        # Extract columns from the plan's child (before aggregation)
        from ..logical.plan import Aggregate

        if isinstance(self._grouped.plan, Aggregate) and self._grouped.plan.child:
            child_plan = self._grouped.plan.child
            temp_df = AsyncDataFrame(plan=child_plan, database=self._grouped.database)
            try:
                columns = temp_df._extract_column_names(child_plan)
            except Exception:
                raise NotImplementedError(
                    "n_unique() requires accessible columns - use agg() with explicit columns"
                )
        else:
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
        return AsyncPolarsDataFrame.from_dataframe(result_df)
