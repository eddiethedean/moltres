"""Compilation of logical expression operations.

This module handles compilation of logical operations like and, or, not, between, in, case_when, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import and_, or_, not_, case as sa_case

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement
    from ..expression_compiler import ExpressionCompiler


def compile_logical_operation(
    compiler: "ExpressionCompiler", op: str, expression: Any
) -> "ColumnElement[Any] | None":
    """Compile a logical operation expression.

    Args:
        compiler: The ExpressionCompiler instance
        op: Operation name (e.g., "and", "or", "not", "between", "in", "case_when")
        expression: :class:`Column` expression to compile

    Returns:
        Compiled SQLAlchemy column element, or None if not handled
    """
    result: "ColumnElement[Any]"

    if op == "and":
        left, right = expression.args
        result = and_(compiler._compile(left), compiler._compile(right))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "or":
        left, right = expression.args
        result = or_(compiler._compile(left), compiler._compile(right))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "not":
        result = not_(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "between":
        value, lower, upper = expression.args
        result = compiler._compile(value).between(
            compiler._compile(lower), compiler._compile(upper)
        )
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "in":
        value, options = expression.args
        if not isinstance(options, (list, tuple)):
            raise TypeError(f"Expected iterable for 'in' options, got {type(options).__name__}")
        option_values = [compiler._compile(opt) for opt in options]
        result = compiler._compile(value).in_(option_values)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "case_when":
        # CASE WHEN expression: args[0] is tuple of (condition, value) pairs, args[1] is else value
        conditions = expression.args[0]
        if not isinstance(conditions, (list, tuple)):
            raise TypeError(
                f"Expected iterable for CASE conditions, got {type(conditions).__name__}"
            )
        else_value = compiler._compile(expression.args[1])

        # Build CASE statement
        # Start with empty when clauses, add them one by one
        when_clauses: list[tuple[ColumnElement[Any], Any]] = []
        for condition, value in conditions:
            when_clauses.append((compiler._compile(condition), compiler._compile(value)))
        case_stmt = sa_case(*when_clauses, else_=else_value)

        result = case_stmt
        if expression._alias:
            result = result.label(expression._alias)
        return result

    return None  # Not handled by this module
