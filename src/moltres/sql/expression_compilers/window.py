"""Compilation of window function expressions.

This module handles compilation of window functions like row_number, rank, lag, lead, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import func
from sqlalchemy.sql import ColumnElement

if TYPE_CHECKING:
    from ..expression_compiler import ExpressionCompiler
    from ...logical.plan import WindowSpec


def compile_window_operation(
    compiler: "ExpressionCompiler", op: str, expression: Any
) -> "ColumnElement[Any] | None":
    """Compile a window function operation expression.

    Args:
        compiler: The ExpressionCompiler instance
        op: Operation name (e.g., "window_row_number", "window_rank", "window")
        expression: :class:`Column` expression to compile

    Returns:
        Compiled SQLAlchemy column element, or None if not handled
    """
    result: "ColumnElement[Any]"

    if op == "window_row_number":
        result = func.row_number()
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_rank":
        result = func.rank()
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_dense_rank":
        result = func.dense_rank()
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_lag":
        column = compiler._compile(expression.args[0])
        offset = expression.args[1] if len(expression.args) > 1 else 1
        if len(expression.args) > 2:
            default = compiler._compile(expression.args[2])
            result = func.lag(column, offset, default)
        else:
            result = func.lag(column, offset)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_lead":
        column = compiler._compile(expression.args[0])
        offset = expression.args[1] if len(expression.args) > 1 else 1
        if len(expression.args) > 2:
            default = compiler._compile(expression.args[2])
            result = func.lead(column, offset, default)
        else:
            result = func.lead(column, offset)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_first_value":
        col_expr = compiler._compile(expression.args[0])
        result = func.first_value(col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_last_value":
        col_expr = compiler._compile(expression.args[0])
        result = func.last_value(col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_percent_rank":
        result = func.percent_rank()
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_cume_dist":
        result = func.cume_dist()
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_nth_value":
        column = compiler._compile(expression.args[0])
        n = expression.args[1]
        result = func.nth_value(column, n)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window_ntile":
        n = expression.args[0]
        result = func.ntile(n)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "window":
        # Window function: args[0] is the function, args[1] is WindowSpec
        func_expr = compiler._compile(expression.args[0])
        window_spec_arg = expression.args[1]
        from ...logical.plan import WindowSpec

        if not isinstance(window_spec_arg, WindowSpec):
            raise TypeError(
                f"Expected WindowSpec for window function, got {type(window_spec_arg).__name__}"
            )
        window_spec: WindowSpec = window_spec_arg

        # Build SQLAlchemy window using .over() method on the function

        # Create partition by clauses
        partition_by: Optional[list[ColumnElement[Any]]] = None
        if window_spec.partition_by:
            partition_by = [compiler._compile(col) for col in window_spec.partition_by]

        # Create order by clauses
        order_by: Optional[list[ColumnElement[Any]]] = None
        if window_spec.order_by:
            order_by = []
            for col_expr in window_spec.order_by:  # type: ignore[assignment]
                # col_expr is a Column from window_spec.order_by: tuple[Column, ...]
                # _compile returns ColumnElement, but mypy may infer Column due to type complexity
                sa_order_col = compiler._compile(col_expr)
                # Check if it has desc/asc already applied
                from ...expressions.column import Column

                if isinstance(col_expr, Column) and col_expr.op == "sort_desc":
                    sa_order_col = sa_order_col.desc()
                elif isinstance(col_expr, Column) and col_expr.op == "sort_asc":
                    sa_order_col = sa_order_col.asc()
                order_by.append(sa_order_col)

        # Handle ROWS BETWEEN or RANGE BETWEEN
        # SQLAlchemy's .over() method accepts rows and range_ parameters directly
        rows_param = None
        range_param = None
        if window_spec.rows_between:
            rows_param = window_spec.rows_between
        elif window_spec.range_between:
            range_param = window_spec.range_between

        # Build window using .over() method with frame specification
        if partition_by and order_by:
            result = func_expr.over(
                partition_by=partition_by,
                order_by=order_by,
                rows=rows_param,
                range_=range_param,
            )
        elif partition_by:
            result = func_expr.over(partition_by=partition_by, rows=rows_param, range_=range_param)
        elif order_by:
            result = func_expr.over(order_by=order_by, rows=rows_param, range_=range_param)
        else:
            result = func_expr.over(rows=rows_param, range_=range_param)

        if expression._alias:
            result = result.label(expression._alias)
        return result

    return None  # Not handled by this module
