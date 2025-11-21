"""Column helper similar to PySpark's ``Column``."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union
from typing_extensions import TypeAlias

from .expr import Expression

if TYPE_CHECKING:
    from ..dataframe.dataframe import DataFrame

LiteralValue = Union[bool, int, float, str, None]
ColumnLike: TypeAlias = Union["Column", LiteralValue]


@dataclass(frozen=True, eq=False)
class Column(Expression):
    """User-facing wrapper around expressions with rich operators."""

    source: Optional[str] = None

    # ------------------------------------------------------------------ helpers
    def alias(self, alias: str) -> "Column":
        return replace(self, _alias=alias)

    def cast(
        self, type_name: str, precision: Optional[int] = None, scale: Optional[int] = None
    ) -> "Column":
        """Cast a column to a different type.

        Args:
            type_name: SQL type name (e.g., "INTEGER", "DECIMAL", "TIMESTAMP", "DATE", "TIME", "VARCHAR")
            precision: Optional precision for DECIMAL/NUMERIC types
            scale: Optional scale for DECIMAL/NUMERIC types

        Returns:
            Column expression for the cast operation

        Example:
            >>> col("price").cast("DECIMAL", precision=10, scale=2)
            >>> col("date_str").cast("DATE")
            >>> col("timestamp_str").cast("TIMESTAMP")
        """
        args: tuple[Any, ...] = (self, type_name)
        if precision is not None or scale is not None:
            args = (self, type_name, precision, scale)
        return Column(op="cast", args=args)

    def is_null(self) -> "Column":
        return Column(op="is_null", args=(self,))

    def is_not_null(self) -> "Column":
        return Column(op="is_not_null", args=(self,))

    def like(self, pattern: str) -> "Column":
        return Column(op="like", args=(self, pattern))

    def ilike(self, pattern: str) -> "Column":
        return Column(op="ilike", args=(self, pattern))

    def between(self, lower: ColumnLike, upper: ColumnLike) -> "Column":
        return Column(
            op="between",
            args=(self, ensure_column(lower), ensure_column(upper)),
        )

    # ---------------------------------------------------------------- operators
    def _binary(self, op: str, other: ColumnLike) -> "Column":
        return Column(op=op, args=(self, ensure_column(other)))

    def _unary(self, op: str) -> "Column":
        return Column(op=op, args=(self,))

    def __add__(self, other: ColumnLike) -> "Column":
        return self._binary("add", other)

    def __sub__(self, other: ColumnLike) -> "Column":
        return self._binary("sub", other)

    def __mul__(self, other: ColumnLike) -> "Column":
        return self._binary("mul", other)

    def __truediv__(self, other: ColumnLike) -> "Column":
        return self._binary("div", other)

    def __floordiv__(self, other: ColumnLike) -> "Column":
        return self._binary("floor_div", other)

    def __mod__(self, other: ColumnLike) -> "Column":
        return self._binary("mod", other)

    def __pow__(self, power: ColumnLike, modulo: Optional[ColumnLike] = None) -> "Column":
        args: tuple[Any, ...]
        if modulo is None:
            args = (self, ensure_column(power))
        else:
            args = (self, ensure_column(power), ensure_column(modulo))
        return Column(op="pow", args=args)

    def __neg__(self) -> "Column":
        return self._unary("neg")

    def __pos__(self) -> "Column":
        return self

    def __eq__(self, other: object) -> "Column":  # type: ignore[override]
        return self._binary("eq", other)  # type: ignore[arg-type]

    def __ne__(self, other: object) -> "Column":  # type: ignore[override]
        return self._binary("ne", other)  # type: ignore[arg-type]

    def __lt__(self, other: ColumnLike) -> "Column":
        return self._binary("lt", other)

    def __le__(self, other: ColumnLike) -> "Column":
        return self._binary("le", other)

    def __gt__(self, other: ColumnLike) -> "Column":
        return self._binary("gt", other)

    def __ge__(self, other: ColumnLike) -> "Column":
        return self._binary("ge", other)

    def __and__(self, other: ColumnLike) -> "Column":
        return self._binary("and", other)

    def __or__(self, other: ColumnLike) -> "Column":
        return self._binary("or", other)

    def __invert__(self) -> "Column":
        return self._unary("not")

    def isin(self, values: Union[Iterable[ColumnLike], "DataFrame"]) -> "Column":
        """Check if column value is in a list of values or a subquery.

        Args:
            values: Either an iterable of values or a DataFrame (for subquery)

        Returns:
            Column expression for IN clause

        Example:
            >>> col("id").isin([1, 2, 3])  # IN (1, 2, 3)
            >>> col("id").isin(df.select("customer_id"))  # IN (SELECT customer_id FROM ...)
        """
        # Check if values is a DataFrame (subquery)
        if hasattr(values, "plan") and hasattr(values, "database"):
            # It's a DataFrame - store the plan for subquery compilation
            return Column(op="in_subquery", args=(self, values.plan))
        # Otherwise, it's an iterable of values
        expr_values = tuple(ensure_column(value) for value in values)
        return Column(op="in", args=(self, expr_values))

    def contains(self, substring: str) -> "Column":
        return Column(op="contains", args=(self, substring))

    def startswith(self, prefix: str) -> "Column":
        return Column(op="startswith", args=(self, prefix))

    def endswith(self, suffix: str) -> "Column":
        return Column(op="endswith", args=(self, suffix))

    def asc(self) -> "Column":
        return Column(op="sort_asc", args=(self,))

    def desc(self) -> "Column":
        return Column(op="sort_desc", args=(self,))

    def over(
        self,
        partition_by: Optional[Union["Column", Sequence["Column"]]] = None,
        order_by: Optional[Union["Column", Sequence["Column"]]] = None,
        rows_between: Optional[tuple[Optional[int], Optional[int]]] = None,
        range_between: Optional[tuple[Optional[int], Optional[int]]] = None,
    ) -> "Column":
        """Create a window function expression.

        Args:
            partition_by: Column(s) to partition by
            order_by: Column(s) to order by within partition
            rows_between: Tuple of (start, end) for ROWS BETWEEN clause
            range_between: Tuple of (start, end) for RANGE BETWEEN clause

        Returns:
            Column expression with window function applied
        """
        from ..logical.plan import WindowSpec

        # Normalize partition_by and order_by to sequences
        if partition_by is None:
            partition_by_cols: tuple[Column, ...] = ()
        elif isinstance(partition_by, Column):
            partition_by_cols = (partition_by,)
        else:
            partition_by_cols = tuple(partition_by)

        if order_by is None:
            order_by_cols: tuple[Column, ...] = ()
        elif isinstance(order_by, Column):
            order_by_cols = (order_by,)
        else:
            order_by_cols = tuple(order_by)

        window_spec = WindowSpec(
            partition_by=partition_by_cols,
            order_by=order_by_cols,
            rows_between=rows_between,
            range_between=range_between,
        )
        return Column(op="window", args=(self, window_spec))

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        raise TypeError("Column expressions cannot be used as booleans")


def literal(value: LiteralValue) -> Column:
    return Column(op="literal", args=(value,))


def ensure_column(value: ColumnLike) -> Column:
    if isinstance(value, Column):
        return value
    return literal(value)


def col(name: str) -> Column:
    return Column(op="column", args=(name,), source=name)
