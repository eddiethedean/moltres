"""Logical plan node definitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, Optional

from ..expressions.column import Column

if TYPE_CHECKING:
    from ..table.schema import ColumnDef


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
class FileScan(LogicalPlan):
    """File scan operation for reading data from files.

    Args:
        path: Path to the file
        format: File format ("csv", "json", "jsonl", "parquet", "text")
        schema: Optional explicit schema for the file data
        options: Dictionary of format-specific read options
        column_name: :class:`Column` name for text files (default: "value")
    """

    path: str
    format: str  # "csv", "json", "jsonl", "parquet", "text"
    schema: Optional[Sequence["ColumnDef"]] = None
    options: Dict[str, object] = field(default_factory=dict)
    column_name: Optional[str] = None  # For text files


@dataclass(frozen=True)
class RawSQL(LogicalPlan):
    """Raw SQL query operation.

    Args:
        sql: SQL query string to execute
        params: Optional dictionary of named parameters for parameterized queries
    """

    sql: str
    params: Dict[str, object] | None = None


@dataclass(frozen=True)
class Project(LogicalPlan):
    child: LogicalPlan
    projections: tuple[Column, ...]
    for_update: bool = False
    for_share: bool = False
    for_update_nowait: bool = False
    for_update_skip_locked: bool = False

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
class Sample(LogicalPlan):
    """Sample rows from a :class:`DataFrame`.

    Args:
        child: The logical plan to sample from
        fraction: Fraction of rows to sample (0.0 to 1.0)
        seed: Optional random seed for reproducible sampling
    """

    child: LogicalPlan
    fraction: float
    seed: Optional[int] = None

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
    lateral: bool = False  # True for LATERAL join (PostgreSQL, MySQL 8.0+)
    hints: tuple[str, ...] | None = None  # Join hints (e.g., "USE_INDEX", "FORCE_INDEX")

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
class Intersect(LogicalPlan):
    left: LogicalPlan
    right: LogicalPlan
    distinct: bool = True  # True for INTERSECT, False for INTERSECT ALL

    def children(self) -> Sequence[LogicalPlan]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Except(LogicalPlan):
    left: LogicalPlan
    right: LogicalPlan
    distinct: bool = True  # True for EXCEPT, False for EXCEPT ALL

    def children(self) -> Sequence[LogicalPlan]:
        return (self.left, self.right)


@dataclass(frozen=True)
class CTE(LogicalPlan):
    """Common Table Expression (CTE) - a named subquery that can be referenced later."""

    name: str
    child: LogicalPlan

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class RecursiveCTE(LogicalPlan):
    """Recursive Common Table Expression (WITH RECURSIVE) - for recursive queries."""

    name: str
    initial: LogicalPlan  # Initial/seed query
    recursive: LogicalPlan  # Recursive part that references the CTE
    union_all: bool = False  # True for UNION ALL, False for UNION

    def children(self) -> Sequence[LogicalPlan]:
        return (self.initial, self.recursive)


@dataclass(frozen=True)
class SemiJoin(LogicalPlan):
    """Semi-join: returns rows from left where a matching row exists in right (EXISTS subquery)."""

    left: LogicalPlan
    right: LogicalPlan
    on: tuple[tuple[str, str], ...] | None = None
    condition: Column | None = None

    def children(self) -> Sequence[LogicalPlan]:
        return (self.left, self.right)


@dataclass(frozen=True)
class AntiJoin(LogicalPlan):
    """Anti-join: returns rows from left where no matching row exists in right (NOT EXISTS subquery)."""

    left: LogicalPlan
    right: LogicalPlan
    on: tuple[tuple[str, str], ...] | None = None
    condition: Column | None = None

    def children(self) -> Sequence[LogicalPlan]:
        return (self.left, self.right)


@dataclass(frozen=True)
class Pivot(LogicalPlan):
    """Pivot operation for data reshaping.

    Args:
        child: The logical plan to pivot
        pivot_column: :class:`Column` to pivot on (becomes column headers)
        value_column: :class:`Column` containing values to aggregate
        agg_func: Aggregation function name (e.g., "sum", "avg", "count")
        pivot_values: Optional list of specific values to pivot (if None, uses all distinct values)
    """

    child: LogicalPlan
    pivot_column: str
    value_column: str
    agg_func: str
    pivot_values: tuple[str, ...] | None = None

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class GroupedPivot(LogicalPlan):
    """Grouped pivot operation that combines GROUP BY with pivot.

    Args:
        child: The logical plan to pivot
        grouping: Columns to group by
        pivot_column: :class:`Column` to pivot on (becomes column headers)
        value_column: :class:`Column` containing values to aggregate
        agg_func: Aggregation function name (e.g., "sum", "avg", "count")
        pivot_values: Optional list of specific values to pivot (if None, must be provided or discovered)
    """

    child: LogicalPlan
    grouping: tuple[Column, ...]
    pivot_column: str
    value_column: str
    agg_func: str
    pivot_values: tuple[str, ...] | None = None

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)


@dataclass(frozen=True)
class Explode(LogicalPlan):
    """Explode: expands array/JSON column into multiple rows (one row per element)."""

    child: LogicalPlan
    column: Column
    alias: str = "value"  # Alias for the exploded value column

    def children(self) -> Sequence[LogicalPlan]:
        return (self.child,)
