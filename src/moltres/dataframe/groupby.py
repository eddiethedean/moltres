"""Grouped DataFrame helper."""
from __future__ import annotations

from dataclasses import dataclass

from ..expressions.column import Column
from ..logical import operators
from ..logical.plan import LogicalPlan
from .dataframe import DataFrame


@dataclass(frozen=True)
class GroupedDataFrame:
    plan: LogicalPlan
    keys: tuple[Column, ...]
    parent: DataFrame

    def agg(self, *aggregations: Column) -> DataFrame:
        if not aggregations:
            raise ValueError("agg requires at least one aggregation expression")
        normalized = tuple(self._validate_aggregation(expr) for expr in aggregations)
        plan = operators.aggregate(self.plan, self.keys, normalized)
        return DataFrame(plan=plan, database=self.parent.database)

    @staticmethod
    def _validate_aggregation(expr: Column) -> Column:
        if not expr.op.startswith("agg_"):
            raise ValueError("Aggregation expressions must be created with moltres aggregate helpers")
        return expr
