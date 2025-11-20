"""Factory helpers for logical plan nodes."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from ..expressions.column import Column
from .plan import (
    Aggregate,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Sort,
    SortOrder,
    TableScan,
)


def scan(table: str, alias: Optional[str] = None) -> TableScan:
    return TableScan(table=table, alias=alias)


def project(child: LogicalPlan, columns: Sequence[Column]) -> Project:
    return Project(child=child, projections=tuple(columns))


def filter(child: LogicalPlan, predicate: Column) -> Filter:
    return Filter(child=child, predicate=predicate)


def limit(child: LogicalPlan, count: int) -> Limit:
    return Limit(child=child, count=count)


def order_by(child: LogicalPlan, orders: Iterable[SortOrder]) -> Sort:
    return Sort(child=child, orders=tuple(orders))


def sort_order(expression: Column, descending: bool = False) -> SortOrder:
    return SortOrder(expression=expression, descending=descending)


def aggregate(
    child: LogicalPlan, keys: Sequence[Column], aggregates: Sequence[Column]
) -> Aggregate:
    return Aggregate(child=child, grouping=tuple(keys), aggregates=tuple(aggregates))


def join(
    left: LogicalPlan,
    right: LogicalPlan,
    *,
    how: str,
    on: Optional[Sequence[Tuple[str, str]]] = None,
    condition: Optional[Column] = None,
) -> Join:
    return Join(
        left=left, right=right, how=how, on=None if on is None else tuple(on), condition=condition
    )
