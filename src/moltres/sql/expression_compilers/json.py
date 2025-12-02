"""Compilation of JSON expression operations.

This module handles compilation of JSON functions like json_extract, json_tuple, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func, literal

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement
    from ..expression_compiler import ExpressionCompiler


def compile_json_operation(
    compiler: "ExpressionCompiler", op: str, expression: Any
) -> "ColumnElement[Any] | None":
    """Compile a JSON operation expression.

    Args:
        compiler: The ExpressionCompiler instance
        op: Operation name (e.g., "json_extract", "json_tuple")
        expression: :class:`Column` expression to compile

    Returns:
        Compiled SQLAlchemy column element, or None if not handled
    """
    result: "ColumnElement[Any]"

    if op == "json_extract":
        col_expr = compiler._compile(expression.args[0])
        path_arg = expression.args[1]
        if not isinstance(path_arg, str):
            raise TypeError(f"Expected string for JSON path, got {type(path_arg).__name__}")
        path: str = path_arg
        # Use dialect-specific JSON extraction
        # PostgreSQL: -> operator or json_extract_path_text
        # SQLite: json_extract (JSON1 extension)
        # MySQL: JSON_EXTRACT or -> operator
        # Generic: Use func.json_extract which SQLAlchemy may handle
        if compiler.dialect.name == "postgresql":
            # PostgreSQL uses -> or ->> operators for JSONB
            # Convert $.key to 'key' and use -> operator
            # For paths like $.key.nested, convert to ['key', 'nested']
            if path.startswith("$."):
                # Remove $. prefix and split by . for nested paths
                path_parts = path[2:].split(".")
                # Use -> operator for JSONB (returns JSONB) or ->> for text
                # For now, use ->> to get text result
                result = col_expr
                for part in path_parts:
                    result = result.op("->>")(literal(part))
            else:
                # Use json_extract_path_text with path elements
                path_parts = path.split(".") if "." in path else [path]
                result = func.json_extract_path_text(col_expr, *[literal(p) for p in path_parts])
        elif compiler.dialect.name == "sqlite":
            # SQLite JSON1 extension
            result = func.json_extract(col_expr, path)
        else:
            # Generic fallback - try JSON_EXTRACT
            result = func.json_extract(col_expr, path)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "json_tuple":
        col_expr = compiler._compile(expression.args[0])
        # json_tuple extracts multiple keys from JSON
        # args[1:] are the keys to extract
        keys = expression.args[1:]
        if not keys:
            from ...utils.exceptions import CompilationError

            raise CompilationError("json_tuple requires at least one key argument")

        # For PostgreSQL, use json_extract_path_text for each key
        if compiler.dialect.name == "postgresql":
            from sqlalchemy import literal_column

            # Build multiple json_extract_path_text calls
            # This is a simplified implementation - full json_tuple would return multiple columns
            # For now, return the first key's value
            first_key = keys[0]
            if isinstance(first_key, str):
                result = func.json_extract_path_text(col_expr, literal(first_key))
            else:
                result = func.json_extract_path_text(col_expr, compiler._compile(first_key))
        elif compiler.dialect.name == "sqlite":
            # SQLite: json_extract for each key
            first_key = keys[0]
            if isinstance(first_key, str):
                result = func.json_extract(col_expr, f"$.{first_key}")
            else:
                result = func.json_extract(col_expr, compiler._compile(first_key))
        else:
            # Generic fallback
            first_key = keys[0]
            if isinstance(first_key, str):
                result = func.json_extract(col_expr, f"$.{first_key}")
            else:
                result = func.json_extract(col_expr, compiler._compile(first_key))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "json_array_length":
        col_expr = compiler._compile(expression.args[0])
        # Use dialect-specific array length
        if compiler.dialect.name == "postgresql":
            result = func.jsonb_array_length(col_expr)
        elif compiler.dialect.name == "mysql":
            result = func.json_length(col_expr)
        else:
            result = func.json_array_length(col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "json_tuple":
        col_expr = compiler._compile(expression.args[0])
        paths = expression.args[1:]
        if compiler.dialect.name == "postgresql":
            from sqlalchemy import literal_column

            path_list = ", ".join(f"'{p}'" for p in paths)
            result = literal_column(
                f"ARRAY(SELECT jsonb_path_query({col_expr}, p) FROM unnest(ARRAY[{path_list}]) AS p)"
            )
        else:
            results = [func.json_extract(col_expr, path) for path in paths]
            result = func.json_array(*results)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "from_json":
        col_expr = compiler._compile(expression.args[0])
        from sqlalchemy import types as sa_types

        if len(expression.args) > 1:
            # schema = expression.args[1]  # Not used in current implementation
            result = func.cast(col_expr, sa_types.JSON())
        else:
            result = func.cast(col_expr, sa_types.JSON())
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "to_json":
        col_expr = compiler._compile(expression.args[0])
        if compiler.dialect.name == "postgresql":
            result = func.to_jsonb(col_expr)
        elif compiler.dialect.name == "mysql":
            result = func.json_quote(col_expr)
        else:
            result = func.json(col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    return None  # Not handled by this module
