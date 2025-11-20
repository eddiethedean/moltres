"""Logical plan node definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from ..expressions.column import Column


@dataclass(frozen=True)
class LogicalPlan:
    """Base class for logical operators."""

    def children(self) -> Sequence["LogicalPlan"]:
        return ()


@dataclass(frozen=True)
class TableScan(LogicalPlan):
    table: str
    alias: Optional[str] = None


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
    on: Optional[Tuple[Tuple[str, str], ...]] = None
    condition: Optional[Column] = None

    def children(self) -> Sequence[LogicalPlan]:
        return (self.left, self.right)
