"""Factory helpers for logical plan nodes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..expressions.column import Column
from .plan import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Sort,
    SortOrder,
    TableScan,
    Union,
)


def scan(table: str, alias: str | None = None) -> TableScan:
    """Create a TableScan logical plan node.

    Args:
        table: Name of the table to scan
        alias: Optional alias for the table

    Returns:
        TableScan logical plan node
    """
    return TableScan(table=table, alias=alias)


def project(child: LogicalPlan, columns: Sequence[Column]) -> Project:
    """Create a Project logical plan node.

    Args:
        child: Child logical plan
        columns: Sequence of column expressions to project

    Returns:
        Project logical plan node
    """
    return Project(child=child, projections=tuple(columns))


def filter(child: LogicalPlan, predicate: Column) -> Filter:
    """Create a Filter logical plan node.

    Args:
        child: Child logical plan
        predicate: Column expression for the filter condition

    Returns:
        Filter logical plan node
    """
    return Filter(child=child, predicate=predicate)


def limit(child: LogicalPlan, count: int, offset: int = 0) -> Limit:
    """Create a Limit logical plan node.

    Args:
        child: Child logical plan
        count: Maximum number of rows to return
        offset: Number of rows to skip before returning results

    Returns:
        Limit logical plan node
    """
    return Limit(child=child, count=count, offset=offset)


def distinct(child: LogicalPlan) -> Distinct:
    """Create a Distinct logical plan node.

    Args:
        child: Child logical plan

    Returns:
        Distinct logical plan node
    """
    return Distinct(child=child)


def union(left: LogicalPlan, right: LogicalPlan, distinct: bool = True) -> Union:
    """Create a Union logical plan node.

    Args:
        left: Left logical plan
        right: Right logical plan
        distinct: If True, use UNION (distinct), if False use UNION ALL

    Returns:
        Union logical plan node
    """
    return Union(left=left, right=right, distinct=distinct)


def order_by(child: LogicalPlan, orders: Iterable[SortOrder]) -> Sort:
    """Create a Sort logical plan node.

    Args:
        child: Child logical plan
        orders: Iterable of SortOrder objects defining sort criteria

    Returns:
        Sort logical plan node
    """
    return Sort(child=child, orders=tuple(orders))


def sort_order(expression: Column, descending: bool = False) -> SortOrder:
    """Create a SortOrder specification.

    Args:
        expression: Column expression to sort by
        descending: If True, sort in descending order (default: False)

    Returns:
        SortOrder specification
    """
    return SortOrder(expression=expression, descending=descending)


def aggregate(
    child: LogicalPlan, keys: Sequence[Column], aggregates: Sequence[Column]
) -> Aggregate:
    """Create an Aggregate logical plan node.

    Args:
        child: Child logical plan
        keys: Sequence of column expressions for grouping
        aggregates: Sequence of aggregate column expressions

    Returns:
        Aggregate logical plan node
    """
    return Aggregate(child=child, grouping=tuple(keys), aggregates=tuple(aggregates))


def join(
    left: LogicalPlan,
    right: LogicalPlan,
    *,
    how: str,
    on: Sequence[tuple[str, str]] | None = None,
    condition: Column | None = None,
) -> Join:
    """Create a Join logical plan node.

    Args:
        left: Left logical plan
        right: Right logical plan
        how: Join type ("inner", "left", "right", "full", "cross")
        on: Optional sequence of (left_column, right_column) tuples for equality joins
        condition: Optional column expression for custom join condition

    Returns:
        Join logical plan node

    Raises:
        CompilationError: If neither 'on' nor 'condition' is provided
    """
    return Join(
        left=left, right=right, how=how, on=None if on is None else tuple(on), condition=condition
    )


def union(left: LogicalPlan, right: LogicalPlan, distinct: bool = True) -> Union:
    """Create a Union logical plan node.

    Args:
        left: Left logical plan
        right: Right logical plan
        distinct: If True, use UNION (distinct), if False, use UNION ALL

    Returns:
        Union logical plan node
    """
    return Union(left=left, right=right, distinct=distinct)


def distinct(child: LogicalPlan) -> Distinct:
    """Create a Distinct logical plan node.

    Args:
        child: Child logical plan

    Returns:
        Distinct logical plan node
    """
    return Distinct(child=child)
