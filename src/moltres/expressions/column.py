""":class:`Column` expressions for building SQL queries.

This module provides the :class:`Column` class, which represents a column or
expression in a SQL query. Columns support rich operators and can be used to
build complex expressions.

The :func:`col` function is the primary way to create column references.

Example:
    >>> from moltres import col
    >>> # Create column expressions
    >>> age_col = col("age")
    >>> name_col = col("users.name")
    >>> # Use in queries
    >>> df = db.table("users").select(age_col, name_col)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union
from typing_extensions import TypeAlias

from .expr import Expression

if TYPE_CHECKING:
    from ..dataframe.core.dataframe import DataFrame

LiteralValue = Union[bool, int, float, str, None]
ColumnLike: TypeAlias = Union["Column", LiteralValue]


@dataclass(frozen=True, eq=False, repr=False)
class Column(Expression):
    """User-facing wrapper around expressions with rich operators.

    A :class:`Column` represents a column or expression in a SQL query.
    Columns support arithmetic, comparison, and logical operators, and can be
    used to build complex expressions.

    Columns are typically created using the :func:`col` function.

    Example:
        >>> from moltres import col
        >>> age = col("age")
        >>> # Use in expressions
        >>> df = db.table("users").select(age, (age * 2).alias("double_age"))
    """

    source: Optional[str] = None

    def __repr__(self) -> str:
        """Return a user-friendly string representation of the column expression."""
        # Handle simple column reference
        if self.op == "column":
            col_name = self.args[0] if self.args else "?"
            result = f"col('{col_name}')"
        # Handle literal values
        elif self.op == "literal":
            value = self.args[0] if self.args else None
            if isinstance(value, str):
                result = f"lit('{value}')"
            else:
                result = f"lit({value!r})"
        # Handle binary arithmetic operations
        elif self.op in ("add", "sub", "mul", "div", "floor_div", "mod", "pow"):
            op_symbols = {
                "add": "+",
                "sub": "-",
                "mul": "*",
                "div": "/",
                "floor_div": "//",
                "mod": "%",
                "pow": "**",
            }
            op_symbol = op_symbols.get(self.op, self.op)
            left = self.args[0] if len(self.args) > 0 else "?"
            right = self.args[1] if len(self.args) > 1 else "?"
            # Add parentheses for complex expressions (but not for simple columns or literals)
            left_str = (
                f"({left!r})"
                if isinstance(left, Column) and left.op not in ("column", "literal")
                else f"{left!r}"
            )
            right_str = (
                f"({right!r})"
                if isinstance(right, Column) and right.op not in ("column", "literal")
                else f"{right!r}"
            )
            result = f"{left_str} {op_symbol} {right_str}"
        # Handle comparison operations
        elif self.op in ("gt", "lt", "ge", "le", "eq", "ne"):
            op_symbols = {
                "gt": ">",
                "lt": "<",
                "ge": ">=",
                "le": "<=",
                "eq": "==",
                "ne": "!=",
            }
            op_symbol = op_symbols.get(self.op, self.op)
            left = self.args[0] if len(self.args) > 0 else "?"
            right = self.args[1] if len(self.args) > 1 else "?"
            left_str = (
                f"({left!r})"
                if isinstance(left, Column) and left.op not in ("column", "literal")
                else f"{left!r}"
            )
            right_str = (
                f"({right!r})"
                if isinstance(right, Column) and right.op not in ("column", "literal")
                else f"{right!r}"
            )
            result = f"{left_str} {op_symbol} {right_str}"
        # Handle logical operations
        elif self.op in ("and", "or"):
            op_symbol = " & " if self.op == "and" else " | "
            left = self.args[0] if len(self.args) > 0 else "?"
            right = self.args[1] if len(self.args) > 1 else "?"
            left_str = (
                f"({left!r})"
                if isinstance(left, Column) and left.op not in ("column", "literal")
                else f"{left!r}"
            )
            right_str = (
                f"({right!r})"
                if isinstance(right, Column) and right.op not in ("column", "literal")
                else f"{right!r}"
            )
            result = f"{left_str}{op_symbol}{right_str}"
        # Handle unary operations
        elif self.op == "neg":
            expr = self.args[0] if self.args else "?"
            expr_str = (
                f"({expr!r})" if isinstance(expr, Column) and expr.op != "column" else f"{expr!r}"
            )
            result = f"-{expr_str}"
        elif self.op == "not":
            expr = self.args[0] if self.args else "?"
            expr_str = (
                f"({expr!r})" if isinstance(expr, Column) and expr.op != "column" else f"{expr!r}"
            )
            result = f"~{expr_str}"
        # Handle cast
        elif self.op == "cast":
            expr = self.args[0] if len(self.args) > 0 else "?"
            type_name = self.args[1] if len(self.args) > 1 else "?"
            if len(self.args) > 3:
                precision = self.args[2]
                scale = self.args[3]
                result = f"{expr!r}.cast('{type_name}', precision={precision}, scale={scale})"
            elif len(self.args) > 2:
                precision = self.args[2]
                result = f"{expr!r}.cast('{type_name}', precision={precision})"
            else:
                result = f"{expr!r}.cast('{type_name}')"
        # Handle null checks
        elif self.op == "is_null":
            expr = self.args[0] if self.args else "?"
            result = f"{expr!r}.is_null()"
        elif self.op == "is_not_null":
            expr = self.args[0] if self.args else "?"
            result = f"{expr!r}.is_not_null()"
        # Handle string operations
        elif self.op in ("like", "ilike", "contains", "startswith", "endswith"):
            expr = self.args[0] if len(self.args) > 0 else "?"
            pattern = self.args[1] if len(self.args) > 1 else "?"
            if self.op == "like":
                result = f"{expr!r}.like('{pattern}')"
            elif self.op == "ilike":
                result = f"{expr!r}.ilike('{pattern}')"
            elif self.op == "contains":
                result = f"{expr!r}.contains('{pattern}')"
            elif self.op == "startswith":
                result = f"{expr!r}.startswith('{pattern}')"
            elif self.op == "endswith":
                result = f"{expr!r}.endswith('{pattern}')"
            else:
                result = f"{expr!r}.{self.op}('{pattern}')"
        # Handle between
        elif self.op == "between":
            expr = self.args[0] if len(self.args) > 0 else "?"
            lower = self.args[1] if len(self.args) > 1 else "?"
            upper = self.args[2] if len(self.args) > 2 else "?"
            result = f"{expr!r}.between({lower!r}, {upper!r})"
        # Handle in
        elif self.op == "in":
            expr = self.args[0] if len(self.args) > 0 else "?"
            values = self.args[1] if len(self.args) > 1 else ()
            if isinstance(values, tuple):
                values_str = (
                    "["
                    + ", ".join(repr(v) for v in values[:5])
                    + ("..." if len(values) > 5 else "")
                    + "]"
                )
            else:
                values_str = repr(values)
            result = f"{expr!r}.isin({values_str})"
        # Handle window functions
        elif self.op == "window":
            expr = self.args[0] if len(self.args) > 0 else "?"
            result = f"{expr!r}.over(...)"
        # Handle sort operations
        elif self.op == "sort_asc":
            expr = self.args[0] if self.args else "?"
            result = f"{expr!r}.asc()"
        elif self.op == "sort_desc":
            expr = self.args[0] if self.args else "?"
            result = f"{expr!r}.desc()"
        # Handle function calls (aggregations, etc.)
        elif self.op.startswith("agg_") or self.op == "function":
            # For function calls, show function name and args
            if self.op.startswith("agg_"):
                func_name = self.op[4:]  # Remove 'agg_' prefix
                # For aggregations, all args are the function arguments
                args = self.args
            else:
                func_name = str(self.args[0]) if self.args else "?"
                args = self.args[1:]
            if args:
                args_str = ", ".join(repr(arg) for arg in args)
                result = f"{func_name}({args_str})"
            else:
                result = f"{func_name}()"
        # Fallback for unknown operations
        else:
            args_str = ", ".join(repr(arg) for arg in self.args[:3])
            if len(self.args) > 3:
                args_str += "..."
            result = f"Column.{self.op}({args_str})"

        # Add alias if present
        if self._alias:
            result += f".alias('{self._alias}')"

        return result

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
            :class:`Column`: Column expression for the cast operation

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> _ = db.create_table("products", [column("id", "INTEGER"), column("price", "REAL"), column("date_str", "TEXT")]).collect()  # doctest: +ELLIPSIS
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "price": 10.5, "date_str": "2024-01-01"}], _database=db).insert_into("products")
            >>> # Cast price to DECIMAL
            >>> df = db.table("products").select(col("price").cast("DECIMAL", precision=10, scale=2).alias("price_decimal"))
            >>> results = df.collect()
            >>> float(results[0]["price_decimal"])
            10.5
            >>> db.close()
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
            values: Either an iterable of values or a :class:`DataFrame` (for subquery)

        Returns:
            :class:`Column` expression for IN clause

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> _ = db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}], _database=db).insert_into("users")
            >>> # Check if id is in a list of values
            >>> df = db.table("users").select().where(col("id").isin([1, 3]))
            >>> results = df.collect()
            >>> len(results)
            2
            >>> results[0]["name"]
            'Alice'
            >>> db.close()
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
            partition_by: :class:`Column`(s) to partition by
            order_by: :class:`Column`(s) to order by within partition
            rows_between: Tuple of (start, end) for ROWS BETWEEN clause
            range_between: Tuple of (start, end) for RANGE BETWEEN clause

        Returns:
            :class:`Column` expression with window function applied
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

    def filter(self, condition: ColumnLike) -> "Column":
        """Apply a FILTER clause to an aggregation expression.

        The FILTER clause allows conditional aggregation without subqueries.
        This is supported by PostgreSQL 9.4+, MySQL 8.0+, SQL Server, Oracle.
        For unsupported dialects (e.g., SQLite), it will be compiled as a
        CASE WHEN expression.

        Args:
            condition: :class:`Column` expression representing the filter condition

        Returns:
            :class:`Column` expression with FILTER clause attached

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> _ = db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL"), column("status", "TEXT")]).collect()  # doctest: +ELLIPSIS
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "amount": 100.0, "status": "active"}, {"id": 2, "amount": 200.0, "status": "completed"}], _database=db).insert_into("orders")
            >>> # Conditional aggregation with FILTER clause
            >>> df = db.table("orders").select(F.sum(col("amount")).filter(col("status") == "active").alias("active_total"))
            >>> results = df.collect()
            >>> results[0]["active_total"]
            100.0
            >>> db.close()
        """
        if not self.op.startswith("agg_"):
            raise ValueError(
                f"Filter clause can only be applied to aggregate functions, not {self.op!r}"
            )
        return replace(self, _filter=ensure_column(condition))

    def __bool__(self) -> bool:  # pragma: no cover - defensive
        raise TypeError("Column expressions cannot be used as booleans")


def literal(value: LiteralValue) -> Column:
    return Column(op="literal", args=(value,))


def ensure_column(value: ColumnLike) -> Column:
    if isinstance(value, Column):
        return value
    return literal(value)


def col(name: str) -> Column:
    """Create a :class:`Column` expression from a column name.

    This is the primary way to reference columns in Moltres queries.
    :class:`Column` names can be simple (e.g., "age") or qualified (e.g., "users.age").

    Args:
        name: :class:`Column` name as a string. Can be a simple name or qualified
              with table name (e.g., "table.column").

    Returns:
        :class:`Column`: Column expression that can be used in :class:`DataFrame` operations

    Example:
        >>> from moltres import connect, col
        >>> db = connect("sqlite:///:memory:")
        >>> from moltres.table.schema import column
        >>> _ = db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("users")
        >>> df = db.table("users").select(col("name"))
        >>> results = df.collect()
        >>> results[0]["name"]
        'Alice'
        >>> # Use in expressions
        >>> df2 = db.table("users").select((col("id") * 2).alias("double_id"))
        >>> results2 = df2.collect()
        >>> results2[0]["double_id"]
        2
        >>> db.close()
    """
    return Column(op="column", args=(name,), source=name)
