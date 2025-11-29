"""Compilation of string expressions.

This module handles compilation of string manipulation functions like concat, upper, lower, substring, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement
    from ..expression_compiler import ExpressionCompiler


def compile_string_operation(
    compiler: "ExpressionCompiler", op: str, expression: Any
) -> "ColumnElement[Any] | None":
    """Compile a string operation expression.

    Args:
        compiler: The ExpressionCompiler instance
        op: Operation name (e.g., "concat", "upper", "lower")
        expression: :class:`Column` expression to compile

    Returns:
        Compiled SQLAlchemy column element, or None if not handled
    """
    result: "ColumnElement[Any]"
    if op == "coalesce":
        args = [compiler._compile(arg) for arg in expression.args]
        result = func.coalesce(*args)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "concat":
        args = [compiler._compile(arg) for arg in expression.args]
        # SQLite doesn't have concat() function, uses || operator instead
        if compiler.dialect.name == "sqlite":
            # Build concatenation using || operator
            result = args[0]
            for arg in args[1:]:
                result = result.op("||")(arg)
        else:
            # PostgreSQL and MySQL support concat() function
            result = func.concat(*args)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "upper":
        result = func.upper(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "lower":
        result = func.lower(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "substring":
        col_expr = compiler._compile(expression.args[0])
        pos = expression.args[1]
        if len(expression.args) > 2:
            length = expression.args[2]
            result = func.substring(col_expr, pos, length)
        else:
            result = func.substring(col_expr, pos)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "trim":
        result = func.trim(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "ltrim":
        result = func.ltrim(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "rtrim":
        result = func.rtrim(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "initcap":
        col_expr = compiler._compile(expression.args[0])
        # Use dialect-specific initcap
        if compiler.dialect.name == "postgresql":
            result = func.initcap(col_expr)
        elif compiler.dialect.name == "duckdb":
            from ...utils.exceptions import CompilationError

            raise CompilationError(
                f"initcap() is not supported for {compiler.dialect.name} dialect. "
                "Supported dialects: PostgreSQL"
            )
        else:
            # MySQL/SQLite: not directly supported, use literal_column for workaround
            from sqlalchemy import literal_column

            result = literal_column(f"INITCAP({col_expr})")
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "instr":
        col_expr = compiler._compile(expression.args[0])
        substr_expr = compiler._compile(expression.args[1])
        # Use dialect-specific instr
        if compiler.dialect.name == "postgresql":
            result = func.strpos(col_expr, substr_expr)
        elif compiler.dialect.name == "mysql":
            result = func.locate(substr_expr, col_expr)
        else:
            # SQLite: instr
            result = func.instr(col_expr, substr_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "locate":
        substr_expr = compiler._compile(expression.args[0])
        col_expr = compiler._compile(expression.args[1])
        pos = expression.args[2] if len(expression.args) > 2 else 1
        from sqlalchemy import literal

        # Use dialect-specific locate
        if compiler.dialect.name == "postgresql":
            # PostgreSQL: strpos doesn't support start position, use substring
            if pos > 1:
                from sqlalchemy import literal_column

                result = func.strpos(func.substring(col_expr, pos), substr_expr) + literal(pos - 1)
            else:
                result = func.strpos(col_expr, substr_expr)
        elif compiler.dialect.name == "mysql":
            result = func.locate(substr_expr, col_expr, pos)
        else:
            # SQLite: instr with offset
            if pos > 1:
                from sqlalchemy import literal_column

                result = func.instr(func.substring(col_expr, pos), substr_expr) + literal(pos - 1)
            else:
                result = func.instr(col_expr, substr_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "translate":
        col_expr = compiler._compile(expression.args[0])
        from_chars = expression.args[1]
        to_chars = expression.args[2]
        # Use dialect-specific translate
        if compiler.dialect.name == "postgresql":
            result = func.translate(col_expr, from_chars, to_chars)
        elif compiler.dialect.name == "duckdb":
            from ...utils.exceptions import CompilationError

            raise CompilationError(
                f"translate() is not supported for {compiler.dialect.name} dialect. "
                "Supported dialects: PostgreSQL"
            )
        else:
            # MySQL/SQLite: requires workaround (not directly supported)
            from ...utils.exceptions import CompilationError

            raise CompilationError(
                f"translate() is not supported for {compiler.dialect.name} dialect. "
                "PostgreSQL supports this function natively."
            )
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "regexp_extract":
        # SQLAlchemy doesn't have a direct regexp_extract, use dialect-specific function
        col_expr = compiler._compile(expression.args[0])
        pattern = expression.args[1]
        group_idx = expression.args[2] if len(expression.args) > 2 else 0
        # Use func for dialect-specific regex functions
        result = func.regexp_extract(col_expr, pattern, group_idx)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "regexp_replace":
        col_expr = compiler._compile(expression.args[0])
        pattern = expression.args[1]
        replacement = expression.args[2]
        result = func.regexp_replace(col_expr, pattern, replacement)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "split":
        # SQLAlchemy doesn't have split, use string_to_array or similar
        col_expr = compiler._compile(expression.args[0])
        delimiter = expression.args[1]
        result = func.string_to_array(col_expr, delimiter)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "replace":
        col_expr = compiler._compile(expression.args[0])
        search = expression.args[1]
        replacement = expression.args[2]
        result = func.replace(col_expr, search, replacement)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "length":
        result = func.length(compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "lpad":
        col_expr = compiler._compile(expression.args[0])
        length = expression.args[1]
        pad = expression.args[2] if len(expression.args) > 2 else " "
        result = func.lpad(col_expr, length, pad)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "rpad":
        col_expr = compiler._compile(expression.args[0])
        length = expression.args[1]
        pad = expression.args[2] if len(expression.args) > 2 else " "
        result = func.rpad(col_expr, length, pad)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "greatest":
        args = [compiler._compile(arg) for arg in expression.args]
        result = func.greatest(*args)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "least":
        args = [compiler._compile(arg) for arg in expression.args]
        result = func.least(*args)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    return None  # Not handled by this module
