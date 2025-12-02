"""Compilation of type casting expressions.

This module handles compilation of type casting operations like cast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import cast as sqlalchemy_cast, types as sa_types

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement
    from ..expression_compiler import ExpressionCompiler


def compile_type_casting_operation(
    compiler: "ExpressionCompiler", op: str, expression: Any
) -> "ColumnElement[Any] | None":
    """Compile a type casting operation expression.

    Args:
        compiler: The ExpressionCompiler instance
        op: Operation name (e.g., "cast")
        expression: :class:`Column` expression to compile

    Returns:
        Compiled SQLAlchemy column element, or None if not handled
    """
    result: "ColumnElement[Any]"

    if op == "cast":
        column = expression.args[0]
        type_name = expression.args[1]
        if not isinstance(type_name, str):
            raise TypeError(f"Expected string for type_name, got {type(type_name).__name__}")
        precision = expression.args[2] if len(expression.args) > 2 else None
        scale = expression.args[3] if len(expression.args) > 3 else None
        if precision is not None and not isinstance(precision, (int, type(None))):
            raise TypeError(f"Expected int or None for precision, got {type(precision).__name__}")
        if scale is not None and not isinstance(scale, (int, type(None))):
            raise TypeError(f"Expected int or None for scale, got {type(scale).__name__}")

        # Map type names to SQLAlchemy types
        type_name_upper = type_name.upper()
        # Type can be either a TypeEngine instance or a TypeEngine class
        sa_type: sa_types.TypeEngine[Any]
        if type_name_upper == "DECIMAL" or type_name_upper == "NUMERIC":
            if precision is not None and scale is not None:
                sa_type = sa_types.Numeric(precision=precision, scale=scale)
            elif precision is not None:
                sa_type = sa_types.Numeric(precision=precision)
            else:
                sa_type = sa_types.Numeric()
        elif type_name_upper == "TIMESTAMP":
            sa_type = sa_types.TIMESTAMP()
        elif type_name_upper == "DATE":
            sa_type = sa_types.DATE()
        elif type_name_upper == "TIME":
            sa_type = sa_types.TIME()
        elif type_name_upper == "INTERVAL":
            sa_type = sa_types.Interval()
        elif type_name_upper == "UUID":
            # Handle UUID type with dialect-specific implementations
            if compiler.dialect.name == "postgresql":
                sa_type = sa_types.UUID()
            elif compiler.dialect.name == "mysql":
                sa_type = sa_types.CHAR(36)
            else:
                # SQLite and others: use String
                sa_type = sa_types.String()
        elif type_name_upper == "BOOLEAN" or type_name_upper == "BOOL":
            sa_type = sa_types.Boolean()
        elif type_name_upper == "BIGINT":
            sa_type = sa_types.BigInteger()
        elif type_name_upper == "SMALLINT":
            sa_type = sa_types.SmallInteger()
        elif type_name_upper == "TINYINT":
            sa_type = sa_types.SmallInteger()  # SQLite doesn't have TINYINT, use SMALLINT
        elif type_name_upper == "DOUBLE" or type_name_upper == "DOUBLE_PRECISION":
            sa_type = sa_types.Float()
        elif type_name_upper == "REAL":
            sa_type = sa_types.REAL()
        elif type_name_upper == "JSON" or type_name_upper == "JSONB":
            # Handle JSON/JSONB type with dialect-specific implementations
            if compiler.dialect.name == "postgresql":
                sa_type = sa_types.JSON()
                # Note: SQLAlchemy doesn't distinguish JSONB from JSON in type system
                # The actual SQL will use JSONB if specified in DDL
            elif compiler.dialect.name == "mysql":
                sa_type = sa_types.JSON()
            else:
                # SQLite and others: use String
                sa_type = sa_types.String()
        elif type_name_upper == "INTEGER" or type_name_upper == "INT":
            sa_type = sa_types.Integer()
        elif type_name_upper == "TEXT":
            sa_type = sa_types.Text()
        elif type_name_upper == "REAL" or type_name_upper == "FLOAT" or type_name_upper == "DOUBLE":
            sa_type = sa_types.Float()
        elif type_name_upper == "VARCHAR" or type_name_upper == "STRING":
            if precision is not None:
                if not isinstance(precision, int):
                    raise TypeError(
                        f"Expected int for VARCHAR length, got {type(precision).__name__}"
                    )
                sa_type = sa_types.String(length=precision)
            else:
                sa_type = sa_types.String()
        elif type_name_upper == "CHAR":
            if precision is not None:
                if not isinstance(precision, int):
                    raise TypeError(f"Expected int for CHAR length, got {type(precision).__name__}")
                sa_type = sa_types.CHAR(length=precision)
            else:
                sa_type = sa_types.CHAR()
        elif type_name_upper == "BINARY" or type_name_upper == "VARBINARY":
            sa_type = sa_types.LargeBinary()
        elif "[]" in type_name_upper:
            # PostgreSQL array types like INTEGER[], TEXT[], etc.
            # Extract base type before []
            base_type = type_name_upper.split("[")[0]
            if compiler.dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
                import sqlalchemy.types as sa_types_array

                # Map base types to SQLAlchemy types
                if base_type == "INTEGER" or base_type == "INT":
                    sa_type = PG_ARRAY(sa_types_array.Integer)
                elif base_type == "TEXT" or base_type == "VARCHAR" or base_type == "STRING":
                    sa_type = PG_ARRAY(sa_types_array.Text)
                elif base_type == "REAL" or base_type == "FLOAT":
                    sa_type = PG_ARRAY(sa_types_array.Float)
                elif base_type == "BOOLEAN" or base_type == "BOOL":
                    sa_type = PG_ARRAY(sa_types_array.Boolean)
                else:
                    # Fallback to TEXT array
                    sa_type = PG_ARRAY(sa_types_array.Text)
            else:
                # For non-PostgreSQL dialects, fallback to String
                sa_type = sa_types.String()
        elif type_name_upper == "ARRAY":
            # Array type - PostgreSQL specific
            if compiler.dialect.name == "postgresql":
                from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY
                import sqlalchemy.types as sa_types_array

                # Try to infer element type from context or default to Text
                # For now, default to Text array
                if len(expression.args) > 4:
                    element_type_name = expression.args[4]
                    if isinstance(element_type_name, str):
                        element_type_upper = element_type_name.upper()
                        if element_type_upper == "INTEGER" or element_type_upper == "INT":
                            sa_type = PG_ARRAY(sa_types_array.Integer)
                        elif element_type_upper == "TEXT" or element_type_upper == "VARCHAR":
                            sa_type = PG_ARRAY(sa_types_array.Text)
                        elif element_type_upper == "BOOLEAN" or element_type_upper == "BOOL":
                            sa_type = PG_ARRAY(sa_types_array.Boolean)
                        else:
                            # Fallback to TEXT array
                            sa_type = PG_ARRAY(sa_types_array.Text)
                    else:
                        sa_type = PG_ARRAY(sa_types_array.Text)
                else:
                    # Fallback to TEXT array
                    sa_type = PG_ARRAY(sa_types_array.Text)
            else:
                # For non-PostgreSQL dialects, fallback to String
                sa_type = sa_types.String()
        else:
            # Fallback to String for unknown types
            sa_type = sa_types.String()

        result = sqlalchemy_cast(compiler._compile(column), sa_type)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    return None  # Not handled by this module
