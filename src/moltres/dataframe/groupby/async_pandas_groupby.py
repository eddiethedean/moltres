"""Async Pandas-style GroupBy interface for Moltres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional, Union

from ...expressions.column import Column
from ..groupby.async_groupby import AsyncGroupedDataFrame

if TYPE_CHECKING:
    from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame


@dataclass(frozen=True)
class AsyncPandasGroupBy:
    """Async Pandas-style GroupBy wrapper around Moltres AsyncGroupedDataFrame.

    Provides pandas-style groupby API with dictionary aggregation support.
    """

    _grouped: AsyncGroupedDataFrame

    def agg(self, **aggregations: Union[str, Dict[str, str]]) -> "AsyncPandasDataFrame":
        """Apply aggregations using pandas-style dictionary syntax.

        Args:
            **aggregations: :class:`Column` names mapped to aggregation functions or dicts

        Returns:
            AsyncPandasDataFrame with aggregated results

        Example:
            >>> await df.groupby('country').agg(amount='sum', price='mean')
            >>> await df.groupby('country').agg({'amount': 'sum', 'price': ['mean', 'max']})
        """
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame

        # Convert pandas-style aggregations to Moltres format
        agg_list = []

        for col_name, func_spec in aggregations.items():
            if isinstance(func_spec, str):
                # Single function: {'amount': 'sum'}
                agg_expr = self._create_aggregation(col_name, func_spec)
                agg_list.append(agg_expr)
            elif isinstance(func_spec, dict):
                # Multiple functions: {'amount': {'sum': 'total', 'mean': 'avg'}}
                for func_name, alias in func_spec.items():
                    agg_expr = self._create_aggregation(col_name, func_name, alias=alias)
                    agg_list.append(agg_expr)
            elif isinstance(func_spec, (list, tuple)):
                # Multiple functions: {'amount': ['sum', 'mean']}
                for func_name in func_spec:
                    agg_expr = self._create_aggregation(col_name, func_name)
                    agg_list.append(agg_expr)
            else:
                raise ValueError(f"Invalid aggregation spec for column '{col_name}': {func_spec}")

        result_df = self._grouped.agg(*agg_list)
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def _create_aggregation(
        self, column_name: str, func_name: str, alias: Optional[str] = None
    ) -> Column:
        """Create an aggregation :class:`Column` from a column name and function name.

        Args:
            column_name: Name of the column to aggregate
            func_name: Name of the aggregation function (e.g., 'sum', 'mean', 'count')
            alias: Optional alias for the result column

        Returns:
            :class:`Column` expression for the aggregation
        """
        from ..helpers.groupby_helpers import create_aggregation_from_string

        # Use shared helper, but apply pandas-specific alias logic
        agg_expr = create_aggregation_from_string(column_name, func_name, alias=None)

        if alias:
            return agg_expr.alias(alias)
        elif func_name.lower() in ("mean", "avg"):
            return agg_expr.alias(f"{column_name}_mean")
        elif func_name.lower() == "count":
            return agg_expr.alias(f"{column_name}_count")
        else:
            return agg_expr.alias(f"{column_name}_{func_name.lower()}")

    def sum(self) -> "AsyncPandasDataFrame":
        """Sum all numeric columns in each group.

        Returns:
            AsyncPandasDataFrame with sum of all numeric columns for each group

        Note:
            This attempts to sum all columns. For better control, use agg() with
            specific columns.
        """
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame
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
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def mean(self) -> "AsyncPandasDataFrame":
        """Mean of all numeric columns in each group.

        Returns:
            AsyncPandasDataFrame with mean of all numeric columns for each group

        Note:
            This attempts to average all columns. For better control, use agg() with
            specific columns.
        """
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame
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
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def min(self) -> "AsyncPandasDataFrame":
        """Minimum value of all columns in each group."""
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame
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
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def max(self) -> "AsyncPandasDataFrame":
        """Maximum value of all columns in each group."""
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame
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
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def nunique(self) -> "AsyncPandasDataFrame":
        """Count distinct values for all columns in each group."""
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        agg_list = build_polars_groupby_column_aggregation(
            self,
            "nunique",
            F.count_distinct,
            "_nunique",
            "No columns found for nunique()",
        )
        result_df = self._grouped.agg(*agg_list)
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def first(self) -> "AsyncPandasDataFrame":
        """Get first value of each column in each group."""
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame
        from ...expressions import functions as F
        from ..helpers.polars_groupby_helpers import build_polars_groupby_column_aggregation

        # Use MIN as proxy for first (works if data is ordered)
        # For true first(), would need window functions or database-specific functions
        agg_list = build_polars_groupby_column_aggregation(
            self,
            "first",
            F.min,
            "_first",
            "No columns found for first()",
        )
        result_df = self._grouped.agg(*agg_list)
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def last(self) -> "AsyncPandasDataFrame":
        """Get last value of each column in each group."""
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame
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
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def count(self) -> "AsyncPandasDataFrame":
        """Count rows in each group."""
        from ...expressions.functions import count
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame

        result_df = self._grouped.agg(count("*").alias("count"))
        return AsyncPandasDataFrame.from_dataframe(result_df)

    def size(self) -> "AsyncPandasDataFrame":
        """Count rows in each group (alias for count)."""
        return self.count()
