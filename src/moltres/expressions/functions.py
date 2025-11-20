"""Expression helper functions."""
from __future__ import annotations

from typing import Iterable, Union

from .column import Column, ColumnLike, ensure_column, literal

__all__ = [
    "lit",
    "sum",
    "avg",
    "min",
    "max",
    "count",
    "count_distinct",
    "coalesce",
    "concat",
    "upper",
    "lower",
    "greatest",
    "least",
]


def lit(value: object) -> Column:
    return literal(value)


def _aggregate(op: str, column: ColumnLike) -> Column:
    return Column(op=op, args=(ensure_column(column),))


def sum(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    return _aggregate("agg_sum", column)


def avg(column: ColumnLike) -> Column:
    return _aggregate("agg_avg", column)


def min(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    return _aggregate("agg_min", column)


def max(column: ColumnLike) -> Column:  # noqa: A001 - mirrored PySpark API
    return _aggregate("agg_max", column)


def count(column: Union[ColumnLike, str] = "*") -> Column:
    if isinstance(column, str) and column == "*":
        return Column(op="agg_count_star", args=())
    return _aggregate("agg_count", column)


def count_distinct(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("count_distinct requires at least one column")
    exprs = tuple(ensure_column(column) for column in columns)
    return Column(op="agg_count_distinct", args=exprs)


def coalesce(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("coalesce requires at least one column")
    return Column(op="coalesce", args=tuple(ensure_column(c) for c in columns))


def concat(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("concat requires at least one column")
    return Column(op="concat", args=tuple(ensure_column(c) for c in columns))


def upper(column: ColumnLike) -> Column:
    return Column(op="upper", args=(ensure_column(column),))


def lower(column: ColumnLike) -> Column:
    return Column(op="lower", args=(ensure_column(column),))


def greatest(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("greatest requires at least one column")
    return Column(op="greatest", args=tuple(ensure_column(c) for c in columns))


def least(*columns: ColumnLike) -> Column:
    if not columns:
        raise ValueError("least requires at least one column")
    return Column(op="least", args=tuple(ensure_column(c) for c in columns))
