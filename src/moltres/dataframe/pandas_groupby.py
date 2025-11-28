"""Pandas-style GroupBy interface for Moltres."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

from ..expressions.column import Column, col
from .groupby import GroupedDataFrame

if TYPE_CHECKING:
    from .pandas_dataframe import PandasDataFrame


@dataclass(frozen=True)
class PandasGroupBy:
    """Pandas-style GroupBy wrapper around Moltres GroupedDataFrame.

    Provides pandas-style groupby API with dictionary aggregation support.
    """

    _grouped: GroupedDataFrame

    def agg(self, **aggregations: Union[str, Dict[str, str]]) -> "PandasDataFrame":
        """Apply aggregations using pandas-style dictionary syntax.

        Args:
            **aggregations: Column names mapped to aggregation functions or dicts

        Returns:
            PandasDataFrame with aggregated results

        Example:
            >>> df.groupby('country').agg(amount='sum', price='mean')
            >>> df.groupby('country').agg({'amount': 'sum', 'price': ['mean', 'max']})
        """
        from .pandas_dataframe import PandasDataFrame

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
        return PandasDataFrame.from_dataframe(result_df)

    def _create_aggregation(
        self, column_name: str, func_name: str, alias: Optional[str] = None
    ) -> Column:
        """Create an aggregation Column from a column name and function name.

        Args:
            column_name: Name of the column to aggregate
            func_name: Name of the aggregation function (e.g., 'sum', 'mean', 'count')
            alias: Optional alias for the result column

        Returns:
            Column expression for the aggregation
        """
        from .groupby_helpers import create_aggregation_from_string

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

    def sum(self) -> "PandasDataFrame":
        """Sum all numeric columns in each group.

        Returns:
            PandasDataFrame with sum of all numeric columns for each group

        Note:
            This attempts to sum all columns. For better control, use agg() with
            specific columns.
        """
        from .pandas_dataframe import PandasDataFrame
        from ..expressions import functions as F
        from ..expressions.column import col

        # Get all columns from the grouped DataFrame
        # We need to access the parent DataFrame's columns
        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "sum() requires accessible columns - use agg() with explicit columns"
            )

        # Build aggregations - sum all columns
        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.sum(col(col_name)).alias(f"{col_name}_sum"))
            except Exception:
                # Skip columns that can't be summed (e.g., non-numeric)
                pass

        if not agg_list:
            raise ValueError("No numeric columns found to sum")

        result_df = self._grouped.agg(*agg_list)
        return PandasDataFrame.from_dataframe(result_df)

    def mean(self) -> "PandasDataFrame":
        """Mean of all numeric columns in each group.

        Returns:
            PandasDataFrame with mean of all numeric columns for each group

        Note:
            This attempts to average all columns. For better control, use agg() with
            specific columns.
        """
        from .pandas_dataframe import PandasDataFrame
        from ..expressions import functions as F
        from ..expressions.column import col

        # Get all columns from the grouped DataFrame
        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "mean() requires accessible columns - use agg() with explicit columns"
            )

        # Build aggregations - average all columns
        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.avg(col(col_name)).alias(f"{col_name}_mean"))
            except Exception:
                # Skip columns that can't be averaged
                pass

        if not agg_list:
            raise ValueError("No numeric columns found to average")

        result_df = self._grouped.agg(*agg_list)
        return PandasDataFrame.from_dataframe(result_df)

    def min(self) -> "PandasDataFrame":
        """Minimum value of all columns in each group."""
        from .pandas_dataframe import PandasDataFrame
        from ..expressions import functions as F
        from ..expressions.column import col

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
        return PandasDataFrame.from_dataframe(result_df)

    def max(self) -> "PandasDataFrame":
        """Maximum value of all columns in each group."""
        from .pandas_dataframe import PandasDataFrame
        from ..expressions import functions as F
        from ..expressions.column import col

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
        return PandasDataFrame.from_dataframe(result_df)

    def nunique(self) -> "PandasDataFrame":
        """Count distinct values for all columns in each group."""
        from .pandas_dataframe import PandasDataFrame
        from ..expressions import functions as F
        from ..expressions.column import col

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "nunique() requires accessible columns - use agg() with explicit columns"
            )

        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.count_distinct(col(col_name)).alias(f"{col_name}_nunique"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No columns found for nunique()")

        result_df = self._grouped.agg(*agg_list)
        return PandasDataFrame.from_dataframe(result_df)

    def first(self) -> "PandasDataFrame":
        """Get first value of each column in each group."""
        from .pandas_dataframe import PandasDataFrame
        from ..expressions import functions as F
        from ..expressions.column import col

        parent_df = self._grouped.parent
        try:
            columns = parent_df._extract_column_names(parent_df.plan)
        except Exception:
            raise NotImplementedError(
                "first() requires accessible columns - use agg() with explicit columns"
            )

        # Use MIN as proxy for first (works if data is ordered)
        # For true first(), would need window functions or database-specific functions
        agg_list = []
        for col_name in columns:
            try:
                agg_list.append(F.min(col(col_name)).alias(f"{col_name}_first"))
            except Exception:
                pass

        if not agg_list:
            raise ValueError("No columns found for first()")

        result_df = self._grouped.agg(*agg_list)
        return PandasDataFrame.from_dataframe(result_df)

    def last(self) -> "PandasDataFrame":
        """Get last value of each column in each group."""
        from .pandas_dataframe import PandasDataFrame
        from ..expressions import functions as F
        from ..expressions.column import col

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
        return PandasDataFrame.from_dataframe(result_df)

    def count(self) -> "PandasDataFrame":
        """Count rows in each group."""
        from ..expressions.functions import count
        from .pandas_dataframe import PandasDataFrame

        result_df = self._grouped.agg(count("*").alias("count"))
        return PandasDataFrame.from_dataframe(result_df)

    def size(self) -> "PandasDataFrame":
        """Count rows in each group (alias for count)."""
        return self.count()
