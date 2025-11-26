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
        from ..expressions.functions import (
            avg,
            count,
            max as max_func,
            min as min_func,
            sum as sum_func,
        )

        func_map: Dict[str, Callable[[Column], Column]] = {
            "sum": sum_func,
            "mean": avg,
            "avg": avg,
            "average": avg,
            "min": min_func,
            "max": max_func,
            "count": count,
        }

        func_name_lower = func_name.lower()
        if func_name_lower not in func_map:
            raise ValueError(
                f"Unknown aggregation function: {func_name}. "
                f"Supported: {', '.join(func_map.keys())}"
            )

        agg_func = func_map[func_name_lower]
        agg_expr: Column = agg_func(col(column_name))

        if alias:
            return agg_expr.alias(alias)
        elif func_name_lower == "mean" or func_name_lower == "avg":
            return agg_expr.alias(f"{column_name}_mean")
        elif func_name_lower == "count":
            return agg_expr.alias(f"{column_name}_count")
        else:
            return agg_expr.alias(f"{column_name}_{func_name_lower}")

    def sum(self) -> "PandasDataFrame":
        """Sum all numeric columns in each group."""
        # Get all numeric columns - this is a simplified version
        # In practice, we'd need to know the schema
        # For now, use a generic approach
        # We'll need to enhance this with schema inspection
        raise NotImplementedError(
            "sum() requires schema information - use agg() with explicit columns"
        )

    def mean(self) -> "PandasDataFrame":
        """Mean of all numeric columns in each group."""
        raise NotImplementedError(
            "mean() requires schema information - use agg() with explicit columns"
        )

    def count(self) -> "PandasDataFrame":
        """Count rows in each group."""
        from ..expressions.functions import count
        from .pandas_dataframe import PandasDataFrame

        result_df = self._grouped.agg(count("*").alias("count"))
        return PandasDataFrame.from_dataframe(result_df)

    def size(self) -> "PandasDataFrame":
        """Count rows in each group (alias for count)."""
        return self.count()
