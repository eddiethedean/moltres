"""Column helper similar to PySpark's ``Column``."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional, Union

from typing_extensions import TypeAlias

from .expr import Expression

if TYPE_CHECKING:
    from .window import WindowSpec

LiteralValue = Union[bool, int, float, str, None]
ColumnLike: TypeAlias = Union["Column", LiteralValue]


@dataclass(frozen=True, eq=False)
class Column(Expression):
    """User-facing wrapper around expressions with rich operators."""

    source: str | None = None
    _window: WindowSpec | None = None

    # ------------------------------------------------------------------ helpers
    def alias(self, alias: str) -> Column:
        return replace(self, _alias=alias)

    def cast(self, type_name: str) -> Column:
        return Column(op="cast", args=(self, type_name))

    def is_null(self) -> Column:
        return Column(op="is_null", args=(self,))

    def is_not_null(self) -> Column:
        return Column(op="is_not_null", args=(self,))

    def like(self, pattern: str) -> Column:
        return Column(op="like", args=(self, pattern))

    def ilike(self, pattern: str) -> Column:
        return Column(op="ilike", args=(self, pattern))

    def between(self, lower: ColumnLike, upper: ColumnLike) -> Column:
        return Column(
            op="between",
            args=(self, ensure_column(lower), ensure_column(upper)),
        )

    # ---------------------------------------------------------------- operators
    def _binary(self, op: str, other: ColumnLike) -> Column:
        return Column(op=op, args=(self, ensure_column(other)))

    def _unary(self, op: str) -> Column:
        return Column(op=op, args=(self,))

    def __add__(self, other: ColumnLike) -> Column:
        return self._binary("add", other)

    def __sub__(self, other: ColumnLike) -> Column:
        return self._binary("sub", other)

    def __mul__(self, other: ColumnLike) -> Column:
        return self._binary("mul", other)

    def __truediv__(self, other: ColumnLike) -> Column:
        return self._binary("div", other)

    def __floordiv__(self, other: ColumnLike) -> Column:
        return self._binary("floor_div", other)

    def __mod__(self, other: ColumnLike) -> Column:
        return self._binary("mod", other)

    def __pow__(self, power: ColumnLike, modulo: ColumnLike | None = None) -> Column:
        args: tuple[Any, ...]
        if modulo is None:
            args = (self, ensure_column(power))
        else:
            args = (self, ensure_column(power), ensure_column(modulo))
        return Column(op="pow", args=args)

    def __neg__(self) -> Column:
        return self._unary("neg")

    def __pos__(self) -> Column:
        return self

    def __eq__(self, other: object) -> Column:  # type: ignore[override]
        return self._binary("eq", other)  # type: ignore[arg-type]

    def __ne__(self, other: object) -> Column:  # type: ignore[override]
        return self._binary("ne", other)  # type: ignore[arg-type]

    def __lt__(self, other: ColumnLike) -> Column:
        return self._binary("lt", other)

    def __le__(self, other: ColumnLike) -> Column:
        return self._binary("le", other)

    def __gt__(self, other: ColumnLike) -> Column:
        return self._binary("gt", other)

    def __ge__(self, other: ColumnLike) -> Column:
        return self._binary("ge", other)

    def __and__(self, other: ColumnLike) -> Column:
        return self._binary("and", other)

    def __or__(self, other: ColumnLike) -> Column:
        return self._binary("or", other)

    def __invert__(self) -> Column:
        return self._unary("not")

    def isin(self, values: Iterable[ColumnLike]) -> Column:
        expr_values = tuple(ensure_column(value) for value in values)
        return Column(op="in", args=(self, expr_values))

    def contains(self, substring: str) -> Column:
        return Column(op="contains", args=(self, substring))

    def startswith(self, prefix: str) -> Column:
        return Column(op="startswith", args=(self, prefix))

    def endswith(self, suffix: str) -> Column:
        return Column(op="endswith", args=(self, suffix))

    def asc(self) -> Column:
        return Column(op="sort_asc", args=(self,))

    def desc(self) -> Column:
        return Column(op="sort_desc", args=(self,))

    def over(self, window: WindowSpec) -> Column:
        """Apply a window specification to this column expression.

        Args:
            window: WindowSpec defining the window

        Returns:
            Column expression with window applied
        """
        return replace(self, _window=window)

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        raise TypeError("Column expressions cannot be used as booleans")


def literal(value: LiteralValue) -> Column:
    return Column(op="literal", args=(value,))


def ensure_column(value: ColumnLike) -> Column:
    if isinstance(value, Column):
        return value
    return literal(value)  # type: ignore[arg-type]


def col(name: str) -> Column:
    return Column(op="column", args=(name,), source=name)
