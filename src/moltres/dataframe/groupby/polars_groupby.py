"""Polars-style GroupBy interface for Moltres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Union

from ...expressions.column import Column
from .groupby import GroupedDataFrame

if TYPE_CHECKING:
    from ..interfaces.polars_dataframe import PolarsDataFrame


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
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ..helpers.polars_groupby_helpers import build_polars_groupby_agg_with_dict_handling

        normalized_exprs, _ = build_polars_groupby_agg_with_dict_handling(self, *exprs)
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
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        agg_list = build_polars_groupby_column_aggregation(
            self,
            "mean",
            F.avg,
            "_mean",
            "No numeric columns found to average",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def sum(self) -> "PolarsDataFrame":
        """Sum all numeric columns in each group.

        Returns:
            :class:`PolarsDataFrame` with sum of all numeric columns for each group
        """
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        agg_list = build_polars_groupby_column_aggregation(
            self,
            "sum",
            F.sum,
            "_sum",
            "No numeric columns found to sum",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def min(self) -> "PolarsDataFrame":
        """Minimum value of all columns in each group."""
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        agg_list = build_polars_groupby_column_aggregation(
            self,
            "min",
            F.min,
            "_min",
            "No columns found for min()",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def max(self) -> "PolarsDataFrame":
        """Maximum value of all columns in each group."""
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        agg_list = build_polars_groupby_column_aggregation(
            self,
            "max",
            F.max,
            "_max",
            "No columns found for max()",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def count(self) -> "PolarsDataFrame":
        """Count rows in each group."""
        from ...expressions.functions import count
        from ..interfaces.polars_dataframe import PolarsDataFrame

        result_df = self._grouped.agg(count("*").alias("count"))
        return PolarsDataFrame.from_dataframe(result_df)

    def std(self) -> "PolarsDataFrame":
        """Standard deviation of all numeric columns in each group."""
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        agg_list = build_polars_groupby_column_aggregation(
            self,
            "std",
            F.stddev,
            "_std",
            "No numeric columns found for std()",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def var(self) -> "PolarsDataFrame":
        """Variance of all numeric columns in each group."""
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        agg_list = build_polars_groupby_column_aggregation(
            self,
            "var",
            F.variance,
            "_var",
            "No numeric columns found for var()",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def first(self) -> "PolarsDataFrame":
        """Get first value of each column in each group."""
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        # Use MIN as proxy for first (works if data is ordered)
        agg_list = build_polars_groupby_column_aggregation(
            self,
            "first",
            F.min,
            "_first",
            "No columns found for first()",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def last(self) -> "PolarsDataFrame":
        """Get last value of each column in each group."""
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        # Use MAX as proxy for last (works if data is ordered)
        agg_list = build_polars_groupby_column_aggregation(
            self,
            "last",
            F.max,
            "_last",
            "No columns found for last()",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)

    def n_unique(self) -> "PolarsDataFrame":
        """Count distinct values for all columns in each group."""
        from ..interfaces.polars_dataframe import PolarsDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        agg_list = build_polars_groupby_column_aggregation(
            self,
            "n_unique",
            F.count_distinct,
            "_n_unique",
            "No columns found for n_unique()",
        )
        result_df = self._grouped.agg(*agg_list)
        return PolarsDataFrame.from_dataframe(result_df)
