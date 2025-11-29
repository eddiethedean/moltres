"""Compilation of datetime expressions.

This module handles compilation of date/time functions like year, month, day, date_format, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func, literal
from sqlalchemy import types as sa_types

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement
    from ..expression_compiler import ExpressionCompiler


def compile_datetime_operation(
    compiler: "ExpressionCompiler", op: str, expression: Any
) -> "ColumnElement[Any] | None":
    """Compile a datetime operation expression.

    Args:
        compiler: The ExpressionCompiler instance
        op: Operation name (e.g., "year", "month", "date_format")
        expression: :class:`Column` expression to compile

    Returns:
        Compiled SQLAlchemy column element, or None if not handled
    """
    result: "ColumnElement[Any]"
    if op == "year":
        result = func.extract("year", compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "month":
        result = func.extract("month", compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "day":
        result = func.extract("day", compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "dayofweek":
        result = func.extract("dow", compiler._compile(expression.args[0]))  # Day of week
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "hour":
        result = func.extract("hour", compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "minute":
        result = func.extract("minute", compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "second":
        result = func.extract("second", compiler._compile(expression.args[0]))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "date_format":
        col_expr = compiler._compile(expression.args[0])
        format_str = expression.args[1]
        # Use to_char for PostgreSQL, DATE_FORMAT for MySQL, strftime for SQLite
        result = func.to_char(col_expr, format_str)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "to_date":
        col_expr = compiler._compile(expression.args[0])
        if len(expression.args) > 1:
            format_str = expression.args[1]
            result = func.to_date(col_expr, format_str)
        else:
            result = func.to_date(col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "current_date":
        result = func.current_date()
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "current_timestamp":
        result = func.now()
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "datediff":
        end = compiler._compile(expression.args[0])
        start = compiler._compile(expression.args[1])
        result = end - start  # Simplified - actual datediff varies by dialect
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "date_add":
        col_expr = compiler._compile(expression.args[0])
        interval_str = expression.args[1]  # e.g., "1 DAY", "2 MONTH"
        from sqlalchemy import literal_column

        # Parse interval string (format: "N UNIT" where N is number and UNIT is DAY, MONTH, YEAR, HOUR, etc.)
        parts = interval_str.split()
        if len(parts) != 2:
            from ...utils.exceptions import CompilationError

            raise CompilationError(
                f"Invalid interval format: {interval_str}. Expected format: 'N UNIT' (e.g., '1 DAY')"
            )
        num, unit = parts
        unit_upper = unit.upper()

        # For PostgreSQL/DuckDB, use INTERVAL literal
        if compiler.dialect.name in ("postgresql", "duckdb"):
            interval_col: ColumnElement[Any] = literal_column(f"INTERVAL '{interval_str}'")
            result = col_expr + interval_col
        elif compiler.dialect.name == "mysql":
            # MySQL uses DATE_ADD with INTERVAL
            result = func.date_add(col_expr, literal_column(f"INTERVAL {num} {unit_upper}"))
        else:
            # SQLite: use datetime() function with modifier
            modifier = f"+{num} {unit_upper.lower()}"
            result = func.datetime(col_expr, modifier)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "date_sub":
        col_expr = compiler._compile(expression.args[0])
        interval_str = expression.args[1]  # e.g., "1 DAY", "2 MONTH"
        from sqlalchemy import literal_column

        # Parse interval string
        parts = interval_str.split()
        if len(parts) != 2:
            from ...utils.exceptions import CompilationError

            raise CompilationError(
                f"Invalid interval format: {interval_str}. Expected format: 'N UNIT' (e.g., '1 DAY')"
            )
        num, unit = parts
        unit_upper = unit.upper()

        # For PostgreSQL/DuckDB, use INTERVAL literal
        if compiler.dialect.name in ("postgresql", "duckdb"):
            interval_expr: "ColumnElement[Any]" = literal_column(f"INTERVAL '{interval_str}'")
            result = col_expr - interval_expr
        elif compiler.dialect.name == "mysql":
            # MySQL uses DATE_SUB with INTERVAL
            result = func.date_sub(col_expr, literal_column(f"INTERVAL {num} {unit_upper}"))
        else:
            # SQLite: use datetime() function with modifier
            modifier = f"-{num} {unit_upper.lower()}"
            result = func.datetime(col_expr, modifier)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "add_months":
        col_expr = compiler._compile(expression.args[0])
        num_months = expression.args[1]
        # Use SQLAlchemy's interval handling
        try:
            interval_months: "ColumnElement[Any]" = func.make_interval(months=abs(num_months))
            if num_months >= 0:
                result = col_expr + interval_months
            else:
                result = col_expr - interval_months
        except (NotImplementedError, AttributeError, TypeError) as e:
            # Fallback: use date_add function (MySQL/SQLite compatible)
            import logging

            logger = logging.getLogger(__name__)
            logger.debug("make_interval not available for dialect, using date_add fallback: %s", e)
            result = func.date_add(col_expr, literal(num_months))
        except Exception as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Unexpected error using make_interval, falling back to date_add: %s", e)
            result = func.date_add(col_expr, literal(num_months))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "to_timestamp":
        col_expr = compiler._compile(expression.args[0])
        if len(expression.args) > 1:
            format_str = expression.args[1]
            if compiler.dialect.name == "duckdb":
                # DuckDB uses strptime for parsing with format (uses %Y format, not yyyy)
                from sqlalchemy import literal_column

                # Convert PySpark format to strptime format
                duckdb_format = (
                    format_str.replace("yyyy", "%Y")
                    .replace("MM", "%m")
                    .replace("dd", "%d")
                    .replace("HH", "%H")
                    .replace("mm", "%M")
                    .replace("ss", "%S")
                )
                result = literal_column(f"strptime({col_expr}, '{duckdb_format}')")
            else:
                result = func.to_timestamp(col_expr, format_str)
        else:
            if compiler.dialect.name == "duckdb":
                # DuckDB's to_timestamp expects a numeric unix timestamp
                result = func.to_timestamp(col_expr)
            else:
                result = func.to_timestamp(col_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "unix_timestamp":
        from sqlalchemy import literal_column

        if len(expression.args) == 0:
            # Current Unix timestamp
            if compiler.dialect.name == "postgresql":
                result = func.extract("epoch", func.now())
            elif compiler.dialect.name == "mysql":
                result = func.unix_timestamp()
            elif compiler.dialect.name == "duckdb":
                # DuckDB: extract(epoch from now())
                result = func.extract("epoch", func.now())
            else:
                # SQLite: strftime('%s', 'now')
                result = func.strftime("%s", "now")
        elif len(expression.args) == 1:
            col_expr = compiler._compile(expression.args[0])
            if compiler.dialect.name == "postgresql":
                result = func.extract("epoch", col_expr)
            elif compiler.dialect.name == "mysql":
                result = func.unix_timestamp(col_expr)
            elif compiler.dialect.name == "duckdb":
                result = func.extract("epoch", col_expr)
            else:
                # SQLite: strftime('%s', col)
                result = func.strftime("%s", col_expr)
        else:
            # With format string
            col_expr = compiler._compile(expression.args[0])
            format_str = expression.args[1]
            if compiler.dialect.name == "postgresql":
                # Parse with to_timestamp then extract epoch
                parsed: "ColumnElement[Any]" = func.to_timestamp(col_expr, format_str)
                result = func.extract("epoch", parsed)
            elif compiler.dialect.name == "mysql":
                result = func.unix_timestamp(col_expr, format_str)
            elif compiler.dialect.name == "duckdb":
                # DuckDB: strptime then extract epoch
                duckdb_format = (
                    format_str.replace("yyyy", "%Y")
                    .replace("MM", "%m")
                    .replace("dd", "%d")
                    .replace("HH", "%H")
                    .replace("mm", "%M")
                    .replace("ss", "%S")
                )
                parsed = literal_column(f"strptime({col_expr}, '{duckdb_format}')")
                result = func.extract("epoch", parsed)
            else:
                # SQLite: strftime('%s', strftime(format, col))
                parsed = func.strftime(format_str, col_expr)
                result = func.strftime("%s", parsed)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "from_unixtime":
        col_expr = compiler._compile(expression.args[0])
        if len(expression.args) > 1:
            format_str = expression.args[1]
            if compiler.dialect.name == "postgresql":
                # Use to_char with to_timestamp
                timestamp: "ColumnElement[Any]" = func.to_timestamp(col_expr)
                result = func.to_char(timestamp, format_str)
            elif compiler.dialect.name == "mysql":
                result = func.from_unixtime(col_expr, format_str)
            elif compiler.dialect.name == "duckdb":
                # DuckDB: to_char with to_timestamp
                timestamp = func.to_timestamp(col_expr)
                result = func.to_char(timestamp, format_str)
            else:
                # SQLite: strftime with datetime
                timestamp = func.datetime(col_expr, "unixepoch")
                result = func.strftime(format_str, timestamp)
        else:
            if compiler.dialect.name == "postgresql":
                result = func.to_timestamp(col_expr)
            elif compiler.dialect.name == "mysql":
                result = func.from_unixtime(col_expr)
            elif compiler.dialect.name == "duckdb":
                result = func.to_timestamp(col_expr)
            else:
                # SQLite: datetime with unixepoch
                result = func.datetime(col_expr, "unixepoch")
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "date_trunc":
        col_expr = compiler._compile(expression.args[0])
        unit = expression.args[1]
        if compiler.dialect.name == "postgresql":
            result = func.date_trunc(unit, col_expr)
        elif compiler.dialect.name == "duckdb":
            result = func.date_trunc(unit, col_expr)
        else:
            # MySQL/SQLite: use workaround
            from sqlalchemy import literal_column

            result = literal_column(f"DATE_TRUNC('{unit}', {col_expr})")
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "quarter":
        col_expr = compiler._compile(expression.args[0])
        # Extract quarter from date
        if compiler.dialect.name == "postgresql":
            result = func.extract("quarter", col_expr)
        elif compiler.dialect.name == "mysql":
            result = func.quarter(col_expr)
        elif compiler.dialect.name == "duckdb":
            result = func.extract("quarter", col_expr)
        else:
            # SQLite: use strftime
            result = func.cast(func.strftime("%m", col_expr), sa_types.Integer)
            # Calculate quarter: (month - 1) // 3 + 1
            result = (result - 1) // 3 + 1
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "weekofyear" or op == "week":
        col_expr = compiler._compile(expression.args[0])
        if compiler.dialect.name == "postgresql":
            result = func.extract("week", col_expr)
        elif compiler.dialect.name == "mysql":
            result = func.week(col_expr)
        elif compiler.dialect.name == "duckdb":
            result = func.extract("week", col_expr)
        else:
            # SQLite: use strftime
            result = func.cast(func.strftime("%W", col_expr), sa_types.Integer)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "dayofyear":
        col_expr = compiler._compile(expression.args[0])
        if compiler.dialect.name == "postgresql":
            result = func.extract("doy", col_expr)
        elif compiler.dialect.name == "mysql":
            result = func.dayofyear(col_expr)
        elif compiler.dialect.name == "duckdb":
            result = func.extract("doy", col_expr)
        else:
            # SQLite: use strftime
            result = func.cast(func.strftime("%j", col_expr), sa_types.Integer)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "last_day":
        col_expr = compiler._compile(expression.args[0])
        if compiler.dialect.name == "postgresql":
            result = func.date_trunc("month", col_expr) + literal_column("INTERVAL '1 month'")
            result = result - literal_column("INTERVAL '1 day'")
        elif compiler.dialect.name == "mysql":
            result = func.last_day(col_expr)
        elif compiler.dialect.name == "duckdb":
            result = func.last_day(col_expr)
        else:
            # SQLite: use workaround
            from sqlalchemy import literal_column

            result = literal_column(f"date({col_expr}, 'start of month', '+1 month', '-1 day')")
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "months_between":
        date1 = compiler._compile(expression.args[0])
        date2 = compiler._compile(expression.args[1])
        if compiler.dialect.name == "postgresql":
            # Use age() function and extract months
            age_result = func.age(date1, date2)
            result = func.extract("year", age_result) * 12 + func.extract("month", age_result)
        elif compiler.dialect.name == "mysql":
            # MySQL: TIMESTAMPDIFF(MONTH, date2, date1)
            result = func.timestampdiff("MONTH", date2, date1)
        elif compiler.dialect.name == "duckdb":
            # DuckDB: similar to PostgreSQL
            age_result = func.age(date1, date2)
            result = func.extract("year", age_result) * 12 + func.extract("month", age_result)
        else:
            # SQLite: use workaround
            from sqlalchemy import literal_column

            result = literal_column(f"(julianday({date1}) - julianday({date2})) / 30.44")
        if expression._alias:
            result = result.label(expression._alias)
        return result

    return None  # Not handled by this module
