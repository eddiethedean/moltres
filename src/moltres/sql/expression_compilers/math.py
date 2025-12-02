"""Compilation of math expression operations.

This module handles compilation of basic math operations like add, sub, mul, div, and math functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func, case as sa_case, literal

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement
    from ..expression_compiler import ExpressionCompiler


def compile_math_operation(
    compiler: "ExpressionCompiler", op: str, expression: Any
) -> "ColumnElement[Any] | None":
    """Compile a math operation expression.

    Args:
        compiler: The ExpressionCompiler instance
        op: Operation name (e.g., "add", "sub", "mul", "div", "abs", "sqrt", etc.)
        expression: :class:`Column` expression to compile

    Returns:
        Compiled SQLAlchemy column element, or None if not handled
    """
    result: "ColumnElement[Any]"

    # Basic arithmetic operations
    if op == "add":
        left, right = expression.args
        result = compiler._compile(left) + compiler._compile(right)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "sub":
        left, right = expression.args
        result = compiler._compile(left) - compiler._compile(right)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "mul":
        left, right = expression.args
        result = compiler._compile(left) * compiler._compile(right)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "div":
        left, right = expression.args
        result = compiler._compile(left) / compiler._compile(right)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "floor_div":
        left, right = expression.args
        result = func.floor(compiler._compile(left) / compiler._compile(right))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "round":
        col_expr = compiler._compile(expression.args[0])
        scale = expression.args[1] if len(expression.args) > 1 else 0
        result = func.round(col_expr, scale)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "floor":
        result = func.floor(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "ceil":
        col_expr = compiler._compile(expression.args[0])
        # SQLite doesn't have ceil() function, use workaround
        if compiler.dialect.name == "sqlite":
            from sqlalchemy import cast, types as sa_types

            # SQLite ceil workaround:
            # CASE WHEN x > CAST(x AS INTEGER) THEN CAST(x AS INTEGER) + 1 ELSE CAST(x AS INTEGER) END
            int_part = cast(col_expr, sa_types.Integer)
            result = sa_case(
                (col_expr > int_part, int_part + literal(1)),
                else_=int_part,
            )
        else:
            result = func.ceil(col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "abs":
        result = func.abs(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "sqrt":
        col_expr = compiler._compile(expression.args[0])
        # SQLite doesn't have sqrt() function natively
        # Some SQLite builds may have it via extensions
        # If not available, execution will fail and test should handle it
        result = func.sqrt(col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "exp":
        result = func.exp(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "log":
        col_expr = compiler._compile(expression.args[0])
        # SQLite doesn't have ln() or log() function natively
        # Some SQLite builds may have these via extensions
        if compiler.dialect.name == "sqlite":
            # Try func.ln first (SQLAlchemy may handle it if SQLite has extension)
            # If that doesn't work, the test should catch the exception
            result = func.ln(col_expr)
        else:
            result = func.ln(col_expr)  # Natural log
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "log10":
        col_expr = compiler._compile(expression.args[0])
        # SQLite doesn't have log() function with base parameter natively
        # Some SQLite builds may have log10 via extensions
        # If not available, execution will fail and test should handle it
        result = func.log(10, col_expr)  # Base-10 log
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "log2":
        col_expr = compiler._compile(expression.args[0])
        # Use dialect-specific log2
        if compiler.dialect.name == "postgresql":
            result = func.log(2, col_expr)
        elif compiler.dialect.name == "mysql":
            result = func.log(2, col_expr)
        else:
            # SQLite: log(2, x)
            result = func.log(2, col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "sin":
        result = func.sin(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "cos":
        result = func.cos(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "tan":
        result = func.tan(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "asin":
        result = func.asin(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "acos":
        result = func.acos(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "atan":
        result = func.atan(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "atan2":
        y, x = expression.args
        result = func.atan2(compiler._compile(y), compiler._compile(x))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "signum" or op == "sign":
        col_expr = compiler._compile(expression.args[0])
        # Use dialect-specific SIGN function
        if compiler.dialect.name in ("postgresql", "mysql", "duckdb"):
            result = func.sign(col_expr)
        else:
            # SQLite doesn't have SIGN, use CASE WHEN
            result = sa_case(
                (col_expr > 0, literal(1)),
                (col_expr < 0, literal(-1)),
                else_=literal(0),
            )
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "hypot":
        x, y = expression.args
        x_expr = compiler._compile(x)
        y_expr = compiler._compile(y)
        # Use dialect-specific hypot
        if compiler.dialect.name == "postgresql":
            result = func.hypot(x_expr, y_expr)
        else:
            # MySQL/SQLite: manual calculation sqrt(x² + y²)
            result = func.sqrt(x_expr * x_expr + y_expr * y_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    return None  # Not handled by this module
