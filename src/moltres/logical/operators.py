"""Factory helpers for logical plan nodes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Dict, Optional

from ..expressions.column import Column

if TYPE_CHECKING:
    from ..table.schema import ColumnDef
from .plan import (
    Aggregate,
    AntiJoin,
    CTE,
    Distinct,
    Except,
    Explode,
    FileScan,
    Filter,
    GroupedPivot,
    Intersect,
    Join,
    Limit,
    LogicalPlan,
    Pivot,
    Project,
    RawSQL,
    RecursiveCTE,
    Sample,
    SemiJoin,
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


def file_scan(
    path: str,
    format: str,
    schema: Optional[Sequence["ColumnDef"]] = None,
    options: Optional[Dict[str, object]] = None,
    column_name: Optional[str] = None,
) -> FileScan:
    """Create a FileScan logical plan node.

    Args:
        path: Path to the file
        format: File format ("csv", "json", "jsonl", "parquet", "text")
        schema: Optional explicit schema for the file data
        options: Dictionary of format-specific read options
        column_name: :class:`Column` name for text files (default: "value")

    Returns:
        FileScan logical plan node
    """
    return FileScan(
        path=path,
        format=format,
        schema=schema,
        options=options or {},
        column_name=column_name,
    )


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
        predicate: :class:`Column` expression for the filter condition

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


def sample(child: LogicalPlan, fraction: float, seed: Optional[int] = None) -> Sample:
    """Create a Sample logical plan node.

    Args:
        child: Child logical plan
        fraction: Fraction of rows to sample (0.0 to 1.0)
        seed: Optional random seed for reproducible sampling

    Returns:
        Sample logical plan node
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Sample fraction must be between 0.0 and 1.0, got {fraction}")
    return Sample(child=child, fraction=fraction, seed=seed)


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


def intersect(left: LogicalPlan, right: LogicalPlan, distinct: bool = True) -> Intersect:
    """Create an Intersect logical plan node.

    Args:
        left: Left logical plan
        right: Right logical plan
        distinct: If True, use INTERSECT (distinct), if False use INTERSECT ALL

    Returns:
        Intersect logical plan node
    """
    return Intersect(left=left, right=right, distinct=distinct)


def except_(left: LogicalPlan, right: LogicalPlan, distinct: bool = True) -> Except:
    """Create an Except logical plan node.

    Args:
        left: Left logical plan
        right: Right logical plan
        distinct: If True, use EXCEPT (distinct), if False use EXCEPT ALL

    Returns:
        Except logical plan node
    """
    return Except(left=left, right=right, distinct=distinct)


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
        expression: :class:`Column` expression to sort by
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
    lateral: bool = False,
    hints: Sequence[str] | None = None,
) -> Join:
    """Create a Join logical plan node.

    Args:
        left: Left logical plan
        right: Right logical plan
        how: Join type ("inner", "left", "right", "full", "cross")
        on: Optional sequence of (left_column, right_column) tuples for equality joins
        condition: Optional column expression for custom join condition
        lateral: If True, create a LATERAL join (PostgreSQL, MySQL 8.0+)

    Returns:
        Join logical plan node

    Raises:
        CompilationError: If neither 'on' nor 'condition' is provided
    """
    hints_tuple = tuple(hints) if hints else None
    return Join(
        left=left,
        right=right,
        how=how,
        on=None if on is None else tuple(on),
        condition=condition,
        lateral=lateral,
        hints=hints_tuple,
    )


def cte(plan: LogicalPlan, name: str) -> CTE:
    """Create a CTE (Common Table Expression) logical plan node.

    Args:
        plan: Logical plan to wrap as a CTE
        name: Name for the CTE

    Returns:
        CTE logical plan node
    """
    return CTE(name=name, child=plan)


def recursive_cte(
    name: str, initial: LogicalPlan, recursive: LogicalPlan, union_all: bool = False
) -> RecursiveCTE:
    """Create a RecursiveCTE (WITH RECURSIVE) logical plan node.

    Args:
        name: Name for the recursive CTE
        initial: Initial/seed query (non-recursive part)
        recursive: Recursive part that references the CTE
        union_all: If True, use UNION ALL; if False, use UNION (distinct)

    Returns:
        RecursiveCTE logical plan node

    Example:
        >>> # Fibonacci sequence
            >>> from moltres.expressions import functions as F
            >>> initial = scan("seed").select(F.lit(1).alias("n"), F.lit(1).alias("fib"))
        >>> recursive = scan(name).select(...)  # References CTE name
        >>> fib_cte = recursive_cte("fib", initial, recursive)
    """
    return RecursiveCTE(name=name, initial=initial, recursive=recursive, union_all=union_all)


def semi_join(
    left: LogicalPlan,
    right: LogicalPlan,
    *,
    on: Sequence[tuple[str, str]] | None = None,
    condition: Column | None = None,
) -> SemiJoin:
    """Create a SemiJoin logical plan node (EXISTS subquery).

    Args:
        left: Left logical plan
        right: Right logical plan (subquery for EXISTS)
        on: Optional sequence of (left_column, right_column) tuples for equality joins
        condition: Optional column expression for custom join condition

    Returns:
        SemiJoin logical plan node
    """
    return SemiJoin(
        left=left, right=right, on=None if on is None else tuple(on), condition=condition
    )


def anti_join(
    left: LogicalPlan,
    right: LogicalPlan,
    *,
    on: Sequence[tuple[str, str]] | None = None,
    condition: Column | None = None,
) -> AntiJoin:
    """Create an AntiJoin logical plan node (NOT EXISTS subquery).

    Args:
        left: Left logical plan
        right: Right logical plan (subquery for NOT EXISTS)
        on: Optional sequence of (left_column, right_column) tuples for equality joins
        condition: Optional column expression for custom join condition

    Returns:
        AntiJoin logical plan node
    """
    return AntiJoin(
        left=left, right=right, on=None if on is None else tuple(on), condition=condition
    )


def pivot(
    child: LogicalPlan,
    pivot_column: str,
    value_column: str,
    agg_func: str = "sum",
    pivot_values: Sequence[str] | None = None,
) -> Pivot:
    """Create a Pivot logical plan node.

    Args:
        child: Child logical plan
        pivot_column: :class:`Column` to pivot on (becomes column headers)
        value_column: :class:`Column` containing values to aggregate
        agg_func: Aggregation function name (e.g., "sum", "avg", "count")
        pivot_values: Optional list of specific values to pivot (if None, uses all distinct values)

    Returns:
        Pivot logical plan node
    """
    pivot_values_tuple = tuple(pivot_values) if pivot_values else None
    return Pivot(
        child=child,
        pivot_column=pivot_column,
        value_column=value_column,
        agg_func=agg_func,
        pivot_values=pivot_values_tuple,
    )


def grouped_pivot(
    child: LogicalPlan,
    grouping: tuple[Column, ...],
    pivot_column: str,
    value_column: str,
    agg_func: str = "sum",
    pivot_values: Sequence[str] | None = None,
) -> GroupedPivot:
    """Create a GroupedPivot logical plan node.

    Args:
        child: Child logical plan
        grouping: Columns to group by
        pivot_column: :class:`Column` to pivot on (becomes column headers)
        value_column: :class:`Column` containing values to aggregate
        agg_func: Aggregation function name (e.g., "sum", "avg", "count")
        pivot_values: Optional list of specific values to pivot (if None, must be provided or discovered)

    Returns:
        GroupedPivot logical plan node
    """
    pivot_values_tuple = tuple(pivot_values) if pivot_values else None
    return GroupedPivot(
        child=child,
        grouping=grouping,
        pivot_column=pivot_column,
        value_column=value_column,
        agg_func=agg_func,
        pivot_values=pivot_values_tuple,
    )


def explode(child: LogicalPlan, column: Column, alias: str = "value") -> Explode:
    """Create an Explode logical plan node (expands array/JSON column into multiple rows).

    Args:
        child: Child logical plan
        column: :class:`Column` expression to explode (array or JSON column)
        alias: Alias for the exploded value column (default: "value")

    Returns:
        Explode logical plan node
    """
    return Explode(child=child, column=column, alias=alias)


def raw_sql(sql: str, params: Dict[str, object] | None = None) -> RawSQL:
    """Create a RawSQL logical plan node.

    Args:
        sql: SQL query string to execute
        params: Optional dictionary of named parameters for parameterized queries

    Returns:
        RawSQL logical plan node
    """
    return RawSQL(sql=sql, params=params)
