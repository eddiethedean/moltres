"""Async grouped DataFrame operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from ..expressions.column import Column
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
        database: AsyncDatabase | None = None,
        _materialized_data: list[dict[str, object]] | None = None,
    ):
        self.plan = plan
        self.database = database
        self._materialized_data = _materialized_data

    def agg(self, *aggregates: Column) -> AsyncDataFrame:
        """Apply aggregate functions to the grouped data.

        Args:
            *aggregates: Column expressions representing aggregate functions

        Returns:
            AsyncDataFrame with aggregated results
        """
        # Extract grouping keys from current plan
        from ..logical.plan import Aggregate

        if not isinstance(self.plan, Aggregate) or not self.plan.grouping:
            raise ValueError("GroupedDataFrame must have grouping columns")

        grouping = self.plan.grouping
        new_plan = operators.aggregate(self.plan.child, keys=grouping, aggregates=aggregates)
        return AsyncDataFrame(
            plan=new_plan,
            database=self.database,
            _materialized_data=self._materialized_data,
        )
