"""Logical plan node definitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..expressions.column import Column


@dataclass(frozen=True)
class WindowSpec:
    """Window specification for window functions."""

    partition_by: tuple[Column, ...] = ()
    order_by: tuple[Column, ...] = ()
    rows_between: Optional[tuple[Optional[int], Optional[int]]] = None
    range_between: Optional[tuple[Optional[int], Optional[int]]] = None


@dataclass(frozen=True)
class LogicalPlan:
    """Base class for logical operators."""

    def children(self) -> Sequence[LogicalPlan]:
        return ()


@dataclass(frozen=True)
class TableScan(LogicalPlan):
    table: str
    alias: str | None = None


@dataclass(frozen=True)
class Project(LogicalPlan):
    child: LogicalPlan
    projections: tuple[Column, ...]

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class Filter(LogicalPlan):
    child: LogicalPlan
    predicate: Column

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class Limit(LogicalPlan):
    child: LogicalPlan
    count: int
    offset: int = 0

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class SortOrder:
    expression: Column
    descending: bool = False


@dataclass(frozen=True)
class Sort(LogicalPlan):
    child: LogicalPlan
    orders: tuple[SortOrder, ...]

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class Aggregate(LogicalPlan):
    child: LogicalPlan
    grouping: tuple[Column, ...]
    aggregates: tuple[Column, ...]

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class Join(LogicalPlan):
    left: LogicalPlan
    right: LogicalPlan
    how: str
    on: tuple[tuple[str, str], ...] | None = None
    condition: Column | None = None

    def children(self) -> Sequence[LogicalPlan]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Distinct(LogicalPlan):
    child: LogicalPlan

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class Union(LogicalPlan):
    left: LogicalPlan
    right: LogicalPlan
    distinct: bool = True  # True for UNION, False for UNION ALL

    def children(self) -> Sequence[LogicalPlan]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Union(LogicalPlan):
    left: LogicalPlan
    right: LogicalPlan
    distinct: bool = True  # True for UNION, False for UNION ALL

    def children(self) -> Sequence[LogicalPlan]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Distinct(LogicalPlan):
    child: LogicalPlan

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)
