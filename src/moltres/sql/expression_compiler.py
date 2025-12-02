"""Compile expression trees into SQLAlchemy column expressions."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union
from typing import cast as typing_cast

from sqlalchemy import (
    func,
    case as sa_case,
    null,
    literal,
    select,
)
from sqlalchemy.sql import Select, ColumnElement

from ..engine.dialects import DialectSpec
from ..expressions.column import Column
from ..utils.exceptions import CompilationError, ValidationError
from .builders import quote_identifier

if TYPE_CHECKING:
    from sqlalchemy.sql import Subquery, TableClause
    from .plan_compiler import SQLCompiler

logger = logging.getLogger(__name__)


class ExpressionCompiler:
    """Compile expression trees into SQLAlchemy column expressions."""

    def __init__(self, dialect: DialectSpec, plan_compiler: Optional[SQLCompiler] = None):
        self.dialect = dialect
        self._table_cache: dict[str, Any] = {}
        self._current_subq: Subquery | None = None
        self._join_info: Optional[dict[str, str]] = None
        self._plan_compiler = plan_compiler

    def compile_expr(self, expression: Column) -> ColumnElement:
        """Compile a :class:`Column` expression to a SQLAlchemy column expression."""
        return self._compile(expression)

    def emit(self, expression: Column) -> str:
        """Compile a :class:`Column` expression to a SQL string.

        Args:
            expression: :class:`Column` expression to compile

        Returns:
            SQL string representation of the expression
        """
        compiled = self.compile_expr(expression)
        # Convert SQLAlchemy column element to SQL string
        return str(compiled.compile(compile_kwargs={"literal_binds": True}))

    def _compile(self, expression: Union[Column, str, Any]) -> ColumnElement:
        """Compile a :class:`Column` expression to a SQLAlchemy column expression."""
        # Import here to ensure it's available throughout the method
        from sqlalchemy import column as sa_column, literal_column
        from sqlalchemy.sql import ColumnElement

        # Handle string literals directly
        if isinstance(expression, str):
            return literal(expression)

        # Must be a Column expression at this point
        if not isinstance(expression, Column):
            raise CompilationError(f"Expected Column expression, got {type(expression)}")

        op = expression.op

        if op == "star":
            # "*" means select all columns - use literal_column("*")
            star_result: ColumnElement[Any] = literal_column("*")
            if expression._alias:
                star_result = star_result.label(expression._alias)
            return star_result

        if op == "column":
            col_name = expression.args[0]
            # Type narrowing: column op always has a string as first arg
            if not isinstance(col_name, str):
                raise TypeError(f"Expected string for column name, got {type(col_name).__name__}")
            # Validate column name to prevent SQL injection
            try:
                # This will raise ValidationError if column name is invalid
                quote_identifier(col_name, self.dialect.quote_char)
            except ValidationError:
                # Re-raise with clearer context
                raise ValidationError(
                    f"Invalid column name: {col_name!r}. "
                    "Column names may only contain letters, digits, underscores, and dots."
                ) from None

            # Handle qualified column names (table.column)

            sa_col: ColumnElement[Any]
            if "." in col_name:
                parts = col_name.split(".", 1)
                table_name = parts[0]
                col_name = parts[1]  # Extract column name
                # If we have a current subquery (from join), try to access column from it
                # After a join with SELECT *, columns lose table qualification
                if self._current_subq is not None:
                    try:
                        # Try to access column from subquery's column collection
                        # The column name in the subquery is just the column name, not table.column
                        sa_col = typing_cast(ColumnElement[Any], self._current_subq.c[col_name])
                    except (KeyError, AttributeError, TypeError):
                        # Column not found in subquery, try using table-qualified literal
                        # This might work if the subquery preserves table info
                        quote = self.dialect.quote_char
                        sa_col = typing_cast(
                            ColumnElement[Any],
                            literal_column(f"{quote}{table_name}{quote}.{quote}{col_name}{quote}"),
                        )
                else:
                    # No subquery context, use qualified literal column
                    quote = self.dialect.quote_char
                    sa_col = typing_cast(
                        ColumnElement[Any],
                        literal_column(f"{quote}{table_name}{quote}.{quote}{col_name}{quote}"),
                    )
            else:
                # Unqualified column - will be resolved in context
                sa_col = typing_cast(ColumnElement[Any], sa_column(col_name))
            if expression._alias:
                # label() returns a Label which is a ColumnElement, but mypy sees it as Any
                sa_col = typing_cast(ColumnElement[Any], sa_col.label(expression._alias))
            return sa_col

        if op == "literal":
            value = expression.args[0]
            sa_lit: ColumnElement[Any] = literal(value)
            if expression._alias:
                sa_lit = sa_lit.label(expression._alias)
            return sa_lit

        # Try math compiler for basic arithmetic and math functions
        if op in (
            "add",
            "sub",
            "mul",
            "div",
            "floor_div",
            "round",
            "floor",
            "ceil",
            "abs",
            "sqrt",
            "exp",
            "log",
            "log10",
            "log2",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "signum",
            "sign",
            "hypot",
        ):
            from .expression_compilers.math import compile_math_operation

            result = compile_math_operation(self, op, expression)
            if result is not None:
                return result
        if op == "eq":
            left, right = expression.args
            result = self._compile(left) == self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "ne":
            left, right = expression.args
            result = self._compile(left) != self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "lt":
            left, right = expression.args
            result = self._compile(left) < self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "le":
            left, right = expression.args
            result = self._compile(left) <= self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "gt":
            left, right = expression.args
            result = self._compile(left) > self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "ge":
            left, right = expression.args
            result = self._compile(left) >= self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result

        # Date/time functions
        if op == "year":
            result = func.extract("year", self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "month":
            result = func.extract("month", self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "day":
            result = func.extract("day", self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "dayofweek":
            result = func.extract("dow", self._compile(expression.args[0]))  # Day of week
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "hour":
            result = func.extract("hour", self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "minute":
            result = func.extract("minute", self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "second":
            result = func.extract("second", self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "date_format":
            col_expr = self._compile(expression.args[0])
            format_str = expression.args[1]
            # Use to_char for PostgreSQL, DATE_FORMAT for MySQL, strftime for SQLite
            result = func.to_char(col_expr, format_str)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "to_date":
            col_expr = self._compile(expression.args[0])
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
            end = self._compile(expression.args[0])
            start = self._compile(expression.args[1])
            result = end - start  # Simplified - actual datediff varies by dialect
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "date_add":
            col_expr = self._compile(expression.args[0])
            interval_str = expression.args[1]  # e.g., "1 DAY", "2 MONTH"
            if not isinstance(interval_str, str):
                raise TypeError(f"Expected string for interval, got {type(interval_str).__name__}")
            from sqlalchemy import literal_column

            # Parse interval string (format: "N UNIT" where N is number and UNIT is DAY, MONTH, YEAR, HOUR, etc.)
            parts = interval_str.split()
            if len(parts) != 2:
                raise CompilationError(
                    f"Invalid interval format: {interval_str}. Expected format: 'N UNIT' (e.g., '1 DAY')"
                )
            num, unit = parts
            unit_upper = unit.upper()

            # For PostgreSQL/DuckDB, use INTERVAL literal
            if self.dialect.name in ("postgresql", "duckdb"):
                interval_col: ColumnElement[Any] = literal_column(f"INTERVAL '{interval_str}'")
                result = col_expr + interval_col
            elif self.dialect.name == "mysql":
                # MySQL uses DATE_ADD with INTERVAL
                result = func.date_add(col_expr, literal_column(f"INTERVAL {num} {unit_upper}"))
            else:
                # SQLite: use datetime() function with modifier
                # SQLite format: datetime(col, '+1 day'), datetime(col, '-1 hour')
                modifier = f"+{num} {unit_upper.lower()}"
                result = func.datetime(col_expr, modifier)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "date_sub":
            col_expr = self._compile(expression.args[0])
            interval_str = expression.args[1]  # e.g., "1 DAY", "2 MONTH"
            if not isinstance(interval_str, str):
                raise TypeError(f"Expected string for interval, got {type(interval_str).__name__}")
            from sqlalchemy import literal_column

            # Parse interval string
            parts = interval_str.split()
            if len(parts) != 2:
                raise CompilationError(
                    f"Invalid interval format: {interval_str}. Expected format: 'N UNIT' (e.g., '1 DAY')"
                )
            num, unit = parts
            unit_upper = unit.upper()

            # For PostgreSQL/DuckDB, use INTERVAL literal
            if self.dialect.name in ("postgresql", "duckdb"):
                interval_expr: ColumnElement[Any] = literal_column(f"INTERVAL '{interval_str}'")
                result = col_expr - interval_expr
            elif self.dialect.name == "mysql":
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
            col_expr = self._compile(expression.args[0])
            num_months = expression.args[1]
            if not isinstance(num_months, (int, float)):
                raise TypeError(f"Expected number for months, got {type(num_months).__name__}")
            # Use SQLAlchemy's interval handling
            # Try to use make_interval if available (PostgreSQL), otherwise use date_add
            try:
                interval_months: ColumnElement[Any] = func.make_interval(
                    months=abs(int(num_months))
                )
                if num_months >= 0:
                    result = col_expr + interval_months
                else:
                    result = col_expr - interval_months
            except (NotImplementedError, AttributeError, TypeError) as e:
                # Fallback: use date_add function (MySQL/SQLite compatible)
                # This is a simplified fallback
                logger.debug(
                    "make_interval not available for dialect, using date_add fallback: %s", e
                )
                result = func.date_add(col_expr, literal(num_months))
            except Exception as e:
                # Catch any other unexpected errors and log them before falling back
                logger.warning(
                    "Unexpected error using make_interval, falling back to date_add: %s", e
                )
                result = func.date_add(col_expr, literal(num_months))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "to_timestamp":
            col_expr = self._compile(expression.args[0])
            if len(expression.args) > 1:
                format_str = expression.args[1]
                if not isinstance(format_str, str):
                    raise TypeError(f"Expected string for format, got {type(format_str).__name__}")
                if self.dialect.name == "duckdb":
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
                if self.dialect.name == "duckdb":
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
                if self.dialect.name == "postgresql":
                    result = func.extract("epoch", func.now())
                elif self.dialect.name == "mysql":
                    result = func.unix_timestamp()
                elif self.dialect.name == "duckdb":
                    # DuckDB: extract(epoch from now())
                    result = func.extract("epoch", func.now())
                else:
                    # SQLite: strftime('%s', 'now')
                    result = func.strftime("%s", "now")
            elif len(expression.args) == 1:
                col_expr = self._compile(expression.args[0])
                if self.dialect.name == "postgresql":
                    result = func.extract("epoch", col_expr)
                elif self.dialect.name == "mysql":
                    result = func.unix_timestamp(col_expr)
                elif self.dialect.name == "duckdb":
                    # DuckDB: extract(epoch from col)
                    result = func.extract("epoch", col_expr)
                else:
                    # SQLite: strftime('%s', col)
                    result = func.strftime("%s", col_expr)
            else:
                col_expr = self._compile(expression.args[0])
                format_str = expression.args[1]
                if self.dialect.name == "postgresql":
                    # Parse with format then extract epoch
                    parsed = func.to_timestamp(col_expr, format_str)
                    result = func.extract("epoch", parsed)
                elif self.dialect.name == "mysql":
                    result = func.unix_timestamp(col_expr, format_str)
                elif self.dialect.name == "duckdb":
                    # DuckDB: parse with format then extract epoch
                    parsed = func.to_timestamp(col_expr, format_str)
                    result = func.extract("epoch", parsed)
                else:
                    # SQLite: requires parsing first
                    result = func.strftime("%s", func.datetime(col_expr, format_str))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "from_unixtime":
            col_expr = self._compile(expression.args[0])
            if len(expression.args) > 1:
                format_str = expression.args[1]
                if not isinstance(format_str, str):
                    raise TypeError(f"Expected string for format, got {type(format_str).__name__}")
                if self.dialect.name == "postgresql":
                    result = func.to_char(func.to_timestamp(col_expr), format_str)
                elif self.dialect.name == "mysql":
                    result = func.from_unixtime(col_expr, format_str)
                elif self.dialect.name == "duckdb":
                    # DuckDB: to_timestamp(unix_time) then format with strftime
                    # DuckDB's strftime is strftime(timestamp, format)
                    from sqlalchemy import literal_column

                    duckdb_format = (
                        format_str.replace("yyyy", "%Y")
                        .replace("MM", "%m")
                        .replace("dd", "%d")
                        .replace("HH", "%H")
                        .replace("mm", "%M")
                        .replace("ss", "%S")
                    )
                    result = literal_column(
                        f"strftime(to_timestamp({col_expr}), '{duckdb_format}')"
                    )
                else:
                    # SQLite: datetime(unix_time, 'unixepoch')
                    result = func.strftime(format_str, func.datetime(col_expr, "unixepoch"))
            else:
                if self.dialect.name in ("postgresql", "duckdb"):
                    result = func.to_timestamp(col_expr)
                elif self.dialect.name == "mysql":
                    result = func.from_unixtime(col_expr)
                else:
                    # SQLite: datetime(unix_time, 'unixepoch')
                    result = func.datetime(col_expr, "unixepoch")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "date_trunc":
            trunc_unit_arg = expression.args[0]
            if not isinstance(trunc_unit_arg, str):
                raise TypeError(
                    f"Expected string for date_trunc unit, got {type(trunc_unit_arg).__name__}"
                )
            trunc_unit: str = trunc_unit_arg
            col_expr = self._compile(expression.args[1])
            if self.dialect.name == "postgresql":
                result = func.date_trunc(trunc_unit, col_expr)
            elif self.dialect.name == "mysql":
                # MySQL: use DATE_FORMAT with truncation
                from sqlalchemy import literal_column

                unit_map = {
                    "year": "%Y-01-01",
                    "month": "%Y-%m-01",
                    "day": "%Y-%m-%d",
                    "hour": "%Y-%m-%d %H:00:00",
                    "minute": "%Y-%m-%d %H:%i:00",
                    "second": "%Y-%m-%d %H:%i:%s",
                }
                format_str = unit_map.get(unit.lower(), "%Y-%m-%d")
                result = func.date_format(col_expr, format_str)
            else:
                # SQLite: use strftime
                unit_map = {
                    "year": "%Y-01-01",
                    "month": "%Y-%m-01",
                    "day": "%Y-%m-%d",
                    "hour": "%Y-%m-%d %H:00:00",
                    "minute": "%Y-%m-%d %H:%M:00",
                    "second": "%Y-%m-%d %H:%M:%S",
                }
                format_str = unit_map.get(unit.lower(), "%Y-%m-%d")
                result = func.strftime(format_str, col_expr)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "quarter":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.extract("quarter", col_expr)
            elif self.dialect.name == "mysql":
                result = func.quarter(col_expr)
            else:
                # SQLite: strftime('%m') then calculate quarter
                from sqlalchemy import literal_column, types as sa_types

                month = func.cast(func.strftime("%m", col_expr), sa_types.Integer)
                result = sa_case(
                    (month <= 3, literal(1)),
                    (month <= 6, literal(2)),
                    (month <= 9, literal(3)),
                    else_=literal(4),
                )
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "weekofyear" or op == "week":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.extract("week", col_expr)
            elif self.dialect.name == "mysql":
                result = func.week(col_expr)
            else:
                # SQLite: strftime('%W')
                from sqlalchemy import types as sa_types

                result = func.cast(func.strftime("%W", col_expr), sa_types.Integer) + 1
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "dayofyear":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.extract("doy", col_expr)
            elif self.dialect.name == "mysql":
                result = func.dayofyear(col_expr)
            else:
                # SQLite: strftime('%j')
                from sqlalchemy import types as sa_types

                result = func.cast(func.strftime("%j", col_expr), sa_types.Integer)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "last_day":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                # PostgreSQL: date_trunc('month', date) + interval '1 month' - interval '1 day'
                from sqlalchemy import literal_column

                result = (
                    func.date_trunc("month", col_expr)
                    + literal_column("INTERVAL '1 month'")
                    - literal_column("INTERVAL '1 day'")
                )
            elif self.dialect.name in ("mysql", "duckdb"):
                result = func.last_day(col_expr)
            else:
                # SQLite: requires workaround
                from sqlalchemy import literal_column

                # Use date() with month+1, day=0 trick
                result = func.date(col_expr, "+1 month", "-1 day")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "months_between":
            date1 = self._compile(expression.args[0])
            date2 = self._compile(expression.args[1])
            if self.dialect.name == "postgresql":
                # PostgreSQL: EXTRACT(EPOCH FROM (date1 - date2)) / (30.0 * 86400)
                from sqlalchemy import literal_column

                result = func.extract("epoch", date1 - date2) / literal(30.0 * 86400)
            elif self.dialect.name == "mysql":
                result = func.timestampdiff("month", date2, date1)
            elif self.dialect.name == "duckdb":
                # DuckDB: Calculate months using date difference
                from sqlalchemy import literal_column

                # Use datediff with month unit
                result = func.datediff("month", date2, date1)
            else:
                # SQLite: requires calculation
                from sqlalchemy import literal_column

                # Approximate: (julianday(date1) - julianday(date2)) / 30.0
                result = (func.julianday(date1) - func.julianday(date2)) / literal(30.0)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "mod":
            left, right = expression.args
            return func.mod(self._compile(left), self._compile(right))
        if op == "pow":
            base, exp = expression.args[:2]
            result = func.power(self._compile(base), self._compile(exp))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "neg":
            return -self._compile(expression.args[0])
        # Try logical compiler for logical operations
        if op in ("and", "or", "not", "between", "in", "case_when"):
            from .expression_compilers.logical import compile_logical_operation

            result = compile_logical_operation(self, op, expression)
            if result is not None:
                return result
        if op == "in_subquery":
            # IN with subquery: col("id").isin(df.select("id"))
            value, subquery_plan = expression.args
            from ..logical.plan import LogicalPlan

            if not isinstance(subquery_plan, LogicalPlan):
                raise TypeError(
                    f"Expected LogicalPlan for subquery, got {type(subquery_plan).__name__}"
                )
            if self._plan_compiler is None:
                raise CompilationError("Subquery compilation requires plan compiler")
            subquery_stmt = self._plan_compiler._compile_plan(subquery_plan)
            return self._compile(value).in_(subquery_stmt)
        if op == "scalar_subquery":
            # Scalar subquery in SELECT: scalar_subquery(df.select())
            (subquery_plan,) = expression.args
            from ..logical.plan import LogicalPlan

            if not isinstance(subquery_plan, LogicalPlan):
                raise TypeError(
                    f"Expected LogicalPlan for subquery, got {type(subquery_plan).__name__}"
                )
            if self._plan_compiler is None:
                raise CompilationError("Subquery compilation requires plan compiler")
            subquery_stmt = self._plan_compiler._compile_plan(subquery_plan)
            # SQLAlchemy scalar_subquery() - must return single row/column
            # Use select().scalar_subquery() method
            from sqlalchemy.sql import ColumnElement as ColumnElementType

            if isinstance(subquery_stmt, Select):
                scalar_result: ColumnElementType[Any] = subquery_stmt.scalar_subquery()
            else:
                # Fallback: wrap in select
                scalar_result = select(subquery_stmt).scalar_subquery()
            if expression._alias:
                scalar_result = scalar_result.label(expression._alias)
            return scalar_result
        if op == "exists":
            # EXISTS subquery: exists(df.select())
            (subquery_plan,) = expression.args
            from ..logical.plan import LogicalPlan

            if not isinstance(subquery_plan, LogicalPlan):
                raise TypeError(
                    f"Expected LogicalPlan for subquery, got {type(subquery_plan).__name__}"
                )
            if self._plan_compiler is None:
                raise CompilationError("Subquery compilation requires plan compiler")
            subquery_stmt = self._plan_compiler._compile_plan(subquery_plan)
            # SQLAlchemy's exists() function
            from sqlalchemy import exists as sa_exists

            result = sa_exists(subquery_stmt)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "not_exists":
            # NOT EXISTS subquery: not_exists(df.select())
            (subquery_plan,) = expression.args
            from ..logical.plan import LogicalPlan

            if not isinstance(subquery_plan, LogicalPlan):
                raise TypeError(
                    f"Expected LogicalPlan for subquery, got {type(subquery_plan).__name__}"
                )
            if self._plan_compiler is None:
                raise CompilationError("Subquery compilation requires plan compiler")
            subquery_stmt = self._plan_compiler._compile_plan(subquery_plan)
            # SQLAlchemy's exists() function with negation
            from sqlalchemy import exists as sa_exists

            result = ~sa_exists(subquery_stmt)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "like":
            left, pattern = expression.args
            # Pattern is stored as a string - SQLAlchemy's like() accepts string directly
            compiled_left = self._compile(left)
            result = compiled_left.like(pattern)  # pattern is a string
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "ilike":
            left, pattern = expression.args
            # Pattern is stored as a string - SQLAlchemy's ilike() accepts string directly
            compiled_left = self._compile(left)
            result = compiled_left.ilike(pattern)  # pattern is a string
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "contains":
            column, substring = expression.args
            # substring might be a Column or a string
            if isinstance(substring, Column):
                pattern = func.concat(literal("%"), self._compile(substring), literal("%"))
            else:
                pattern = f"%{substring}%"
            return self._compile(column).like(pattern)
        if op == "startswith":
            column, prefix = expression.args
            if isinstance(prefix, Column):
                pattern = func.concat(self._compile(prefix), literal("%"))
            else:
                pattern = f"{prefix}%"
            return self._compile(column).like(pattern)
        if op == "endswith":
            column, suffix = expression.args
            if isinstance(suffix, Column):
                pattern = func.concat(literal("%"), self._compile(suffix))
            else:
                pattern = f"%{suffix}"
            return self._compile(column).like(pattern)
        # Try type casting compiler
        if op == "cast":
            from .expression_compilers.type_casting import compile_type_casting_operation

            result = compile_type_casting_operation(self, op, expression)
            if result is not None:
                return result
        if op == "is_null":
            return self._compile(expression.args[0]).is_(null())
        if op == "is_not_null":
            return self._compile(expression.args[0]).isnot(null())
        if op == "isnan":
            # NaN check - SQL doesn't have direct isnan, use IS NULL or comparison
            # This is a simplified implementation
            col_expr = self._compile(expression.args[0])
            result = col_expr.is_(null())  # Simplified - actual NaN check varies by dialect
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "isinf":
            # Infinity check - SQL doesn't have direct isinf
            # This is a simplified implementation
            col_expr = self._compile(expression.args[0])
            # Use a comparison that would never be true for finite numbers
            # This is dialect-specific and simplified
            result = (col_expr == literal(float("inf"))) | (col_expr == literal(float("-inf")))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        # Try category-specific compilers
        if op.startswith("agg_"):
            from .expression_compilers.aggregation import compile_aggregation

            result = compile_aggregation(self, op, expression)
            if result is not None:
                return result

        if op in (
            "coalesce",
            "concat",
            "upper",
            "lower",
            "substring",
            "trim",
            "ltrim",
            "rtrim",
            "initcap",
            "instr",
            "locate",
            "translate",
            "regexp_extract",
            "regexp_replace",
            "split",
            "replace",
            "length",
            "lpad",
            "rpad",
            "greatest",
            "least",
        ):
            from .expression_compilers.string import compile_string_operation

            result = compile_string_operation(self, op, expression)
            if result is not None:
                return result

        if op in (
            "year",
            "month",
            "day",
            "dayofweek",
            "hour",
            "minute",
            "second",
            "date_format",
            "to_date",
            "current_date",
            "current_timestamp",
            "datediff",
            "date_add",
            "date_sub",
            "add_months",
            "to_timestamp",
            "unix_timestamp",
            "from_unixtime",
            "date_trunc",
            "quarter",
            "weekofyear",
            "week",
            "dayofyear",
            "last_day",
            "months_between",
        ):
            from .expression_compilers.datetime import compile_datetime_operation

            result = compile_datetime_operation(self, op, expression)
            if result is not None:
                return result

        # Try category-specific compilers
        if op.startswith("agg_"):
            from .expression_compilers.aggregation import compile_aggregation

            result = compile_aggregation(self, op, expression)
            if result is not None:
                return result

        if op in (
            "coalesce",
            "concat",
            "upper",
            "lower",
            "substring",
            "trim",
            "ltrim",
            "rtrim",
            "initcap",
            "instr",
            "locate",
            "translate",
            "regexp_extract",
            "regexp_replace",
            "split",
            "replace",
            "length",
            "lpad",
            "rpad",
            "greatest",
            "least",
        ):
            from .expression_compilers.string import compile_string_operation

            result = compile_string_operation(self, op, expression)
            if result is not None:
                return result

        if op in (
            "year",
            "month",
            "day",
            "dayofweek",
            "hour",
            "minute",
            "second",
            "date_format",
            "to_date",
            "current_date",
            "current_timestamp",
            "datediff",
            "date_add",
            "date_sub",
            "add_months",
            "to_timestamp",
            "unix_timestamp",
            "from_unixtime",
            "date_trunc",
            "quarter",
            "weekofyear",
            "week",
            "dayofyear",
            "last_day",
            "months_between",
        ):
            from .expression_compilers.datetime import compile_datetime_operation

            result = compile_datetime_operation(self, op, expression)
            if result is not None:
                return result

        # Try JSON compiler
        if op in ("json_extract", "json_tuple", "json_array_length", "from_json", "to_json"):
            from .expression_compilers.json import compile_json_operation

            result = compile_json_operation(self, op, expression)
            if result is not None:
                return result

        # Array functions
        if op == "array":
            args = [self._compile(arg) for arg in expression.args]
            # Use dialect-specific array construction
            # PostgreSQL: ARRAY[...]
            # SQLite: json_array(...) or string_to_array
            # MySQL: JSON_ARRAY(...)
            if self.dialect.name == "postgresql":
                # PostgreSQL uses ARRAY[...] syntax
                # Use literal_column to generate ARRAY[arg1, arg2, ...] directly
                from sqlalchemy import literal_column

                # Build ARRAY literal by compiling each argument
                array_elements = []
                for arg in args:
                    if hasattr(arg, "compile"):
                        # Compile with literal_binds to get the actual value
                        compiled = arg.compile(compile_kwargs={"literal_binds": True})
                        array_elements.append(str(compiled))
                    else:
                        array_elements.append(str(arg))
                result = literal_column(f"ARRAY[{', '.join(array_elements)}]")
            elif self.dialect.name == "sqlite":
                # SQLite doesn't have native arrays, use JSON array
                result = func.json_array(*args)
            elif self.dialect.name == "mysql":
                # MySQL: Use JSON_ARRAY() with literal values
                # MySQL's JSON_ARRAY doesn't work well with bound parameters
                # Build JSON_ARRAY literal by compiling arguments with literal_binds
                from sqlalchemy import literal_column

                array_elements = []
                for arg in args:
                    if hasattr(arg, "compile"):
                        # Compile with literal_binds to get the actual value
                        compiled = arg.compile(compile_kwargs={"literal_binds": True})
                        array_elements.append(str(compiled))
                    else:
                        array_elements.append(str(arg))
                result = literal_column(f"JSON_ARRAY({', '.join(array_elements)})")
            else:
                # Generic fallback
                result = func.json_array(*args)
            if expression._alias:
                result = result.label(expression._alias)
            return result

        if op == "array_length":
            col_expr = self._compile(expression.args[0])
            # Use dialect-specific array length
            # PostgreSQL: array_length(array, 1)
            # DuckDB: array_length(array) - no dimension argument
            # SQLite: json_array_length(json_array)
            # MySQL: JSON_LENGTH(json_array)
            if self.dialect.name == "postgresql":
                result = func.array_length(col_expr, 1)
            elif self.dialect.name == "duckdb":
                result = func.array_length(col_expr)
            elif self.dialect.name == "sqlite":
                result = func.json_array_length(col_expr)
            else:
                result = func.json_length(col_expr)
            if expression._alias:
                result = result.label(expression._alias)
            return result

        if op == "array_contains":
            col_expr = self._compile(expression.args[0])
            value_expr = self._compile(expression.args[1])
            # Use dialect-specific array contains
            # PostgreSQL: value = ANY(array)
            # SQLite: JSON_CONTAINS (if available) or json_each
            # MySQL: JSON_CONTAINS(json_array, value)
            if self.dialect.name in ("postgresql", "duckdb"):
                # PostgreSQL/DuckDB uses = ANY(array)
                from sqlalchemy import any_

                result = value_expr == any_(col_expr)
            elif self.dialect.name == "mysql":
                # MySQL: JSON_CONTAINS(json_doc, val[, path])
                # The value needs to be a JSON value
                # Use CAST(value AS JSON) to convert the value to JSON type
                from sqlalchemy import cast
                from sqlalchemy.dialects.mysql import JSON as MySQLJSON

                json_value = cast(value_expr, MySQLJSON)
                result = func.json_contains(col_expr, json_value)
            else:
                # SQLite and others - use JSON_CONTAINS if available
                # For SQLite, we'll use a workaround with json_array_length
                try:
                    result = func.json_contains(col_expr, value_expr)
                except (NotImplementedError, AttributeError, TypeError) as e:
                    # Fallback for SQLite - check if removing the value changes length
                    logger.debug(
                        "json_contains not available for dialect, using json_array_length fallback: %s",
                        e,
                    )
                    result = func.json_array_length(col_expr) > func.coalesce(
                        func.json_array_length(
                            func.json_remove(col_expr, func.json_quote(value_expr))
                        ),
                        0,
                    )
                except Exception as e:
                    # Catch any other unexpected errors and log them before falling back
                    logger.warning(
                        "Unexpected error using json_contains, falling back to json_array_length: %s",
                        e,
                    )
                    result = func.json_array_length(col_expr) > func.coalesce(
                        func.json_array_length(
                            func.json_remove(col_expr, func.json_quote(value_expr))
                        ),
                        0,
                    )
            if expression._alias:
                result = result.label(expression._alias)
            return result

        if op == "array_position":
            col_expr = self._compile(expression.args[0])
            value_expr = self._compile(expression.args[1])
            # Use dialect-specific array position
            # PostgreSQL: array_position(array, value)
            # SQLite: Use json_each with rowid - requires subquery (complex)
            # MySQL: JSON_SEARCH(json_array, 'one', value) - returns path, extract index
            if self.dialect.name in ("postgresql", "duckdb"):
                result = func.array_position(col_expr, value_expr)
            elif self.dialect.name == "sqlite":
                # SQLite: Use json_each to find position
                # We need to use a subquery approach, but for simplicity in compilation,
                # we'll use a CASE expression with json_array_length and iteration
                # This is a simplified approach - a full implementation would use json_each in a subquery
                from sqlalchemy import literal_column, cast

                # For SQLite, we'll use a workaround with json_extract and iteration
                # This is not perfect but works for most cases
                # Full implementation would require a correlated subquery with json_each
                result = literal_column(
                    "NULL"
                )  # Simplified - full implementation requires subquery
            elif self.dialect.name == "mysql":
                # MySQL: JSON_SEARCH returns a path like "$[0]", need to extract index
                # Extract index from JSON path: "$[0]" -> 0
                # Use: CAST(SUBSTRING_INDEX(SUBSTRING_INDEX(JSON_UNQUOTE(JSON_SEARCH(...)), '[', -1), ']', 1) AS UNSIGNED)
                from sqlalchemy import literal_column, cast
                from sqlalchemy.dialects.mysql import INTEGER

                # JSON_SEARCH returns the path, e.g., "$[0]" for index 0
                json_search_result = func.json_search(col_expr, literal("one"), value_expr)

                # Extract index from path: "$[0]" -> "0" -> 0
                # Step 1: JSON_UNQUOTE to remove quotes: "$[0]" -> $[0]
                unquoted = func.json_unquote(json_search_result)
                # Step 2: SUBSTRING_INDEX with '[', -1 to get everything after '[': $[0] -> 0]
                after_bracket = func.substring_index(unquoted, literal("["), literal(-1))
                # Step 3: SUBSTRING_INDEX with ']', 1 to get everything before ']': 0] -> 0
                index_str = func.substring_index(after_bracket, literal("]"), literal(1))
                # Step 4: CAST to UNSIGNED to get integer, subtract 1 for 0-based, add 1 for 1-based result
                # Actually, MySQL array indices are 0-based, but array_position should return 1-based
                # So we add 1: CAST(...) + 1
                result = sa_case(
                    (json_search_result.is_(None), None), else_=cast(index_str, INTEGER) + 1
                )
            else:
                # Generic fallback - use JSON_SEARCH
                result = func.json_search(col_expr, literal("one"), value_expr)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "array_append":
            col_expr = self._compile(expression.args[0])
            elem_expr = self._compile(expression.args[1])
            if self.dialect.name in ("postgresql", "duckdb"):
                result = func.array_append(col_expr, elem_expr)
            else:
                result = func.json_array_append(col_expr, "$", elem_expr)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "array_prepend":
            col_expr = self._compile(expression.args[0])
            elem_expr = self._compile(expression.args[1])
            if self.dialect.name in ("postgresql", "duckdb"):
                result = func.array_prepend(elem_expr, col_expr)
            else:
                result = func.json_array_insert(col_expr, "$[0]", elem_expr)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "array_remove":
            col_expr = self._compile(expression.args[0])
            elem_arg = expression.args[1]
            # For DuckDB, we need the literal value to inline in lambda
            # Check if it's a literal Column or a raw value
            from ..expressions.column import Column as MoltresColumn
            from .builders import format_literal

            if isinstance(elem_arg, MoltresColumn) and elem_arg.op == "literal":
                # It's a literal, get the value
                elem_value = elem_arg.args[0] if elem_arg.args else None
                elem_str = format_literal(elem_value)
            else:
                # Compile it normally and hope it works
                elem_expr = self._compile(elem_arg)
                elem_str = str(elem_expr)

            if self.dialect.name == "postgresql":
                elem_expr = self._compile(elem_arg)
                result = func.array_remove(col_expr, elem_expr)
            elif self.dialect.name == "duckdb":
                # DuckDB uses list_filter to remove elements
                from sqlalchemy import literal_column

                result = literal_column(f"list_filter({col_expr}, x -> x != {elem_str})")
            else:
                from sqlalchemy import literal_column

                elem_expr = self._compile(elem_arg)
                result = literal_column(
                    f"JSON_REMOVE({col_expr}, JSON_SEARCH({col_expr}, 'one', {elem_expr}))"
                )
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "array_distinct":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                from sqlalchemy import literal_column

                result = literal_column(f"ARRAY(SELECT DISTINCT unnest({col_expr}))")
            elif self.dialect.name == "duckdb":
                # DuckDB uses list_distinct
                from sqlalchemy import literal_column

                result = literal_column(f"list_distinct({col_expr})")
            else:
                from sqlalchemy import literal_column

                result = literal_column(f"JSON_ARRAY_DISTINCT({col_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "array_sort":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.array_sort(col_expr)
            elif self.dialect.name == "duckdb":
                # DuckDB uses list_sort
                from sqlalchemy import literal_column

                result = literal_column(f"list_sort({col_expr})")
            else:
                from sqlalchemy import literal_column

                result = literal_column(f"JSON_ARRAY_SORT({col_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "array_max":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.array_max(col_expr)
            elif self.dialect.name == "duckdb":
                # DuckDB uses list_max instead of array_max
                from sqlalchemy import literal_column

                result = literal_column(f"list_max({col_expr})")
            else:
                # SQLite/MySQL: Use json_each to find max value
                # This requires a subquery - simplified implementation
                from sqlalchemy import literal_column

                # For SQLite, use a workaround with json_each
                # SELECT MAX(value) FROM json_each(array_col)
                result = literal_column(f"(SELECT MAX(value) FROM json_each({col_expr}))")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "array_min":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.array_min(col_expr)
            elif self.dialect.name == "duckdb":
                # DuckDB uses list_min instead of array_min
                from sqlalchemy import literal_column

                result = literal_column(f"list_min({col_expr})")
            else:
                # SQLite/MySQL: Use json_each to find min value
                from sqlalchemy import literal_column

                # For SQLite, use a workaround with json_each
                result = literal_column(f"(SELECT MIN(value) FROM json_each({col_expr}))")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "array_sum":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                from sqlalchemy import literal_column

                result = literal_column(f"(SELECT sum(x) FROM unnest({col_expr}) AS x)")
            elif self.dialect.name == "duckdb":
                # DuckDB uses list_sum instead of array_sum
                from sqlalchemy import literal_column

                result = literal_column(f"list_sum({col_expr})")
            else:
                from sqlalchemy import literal_column

                result = literal_column(f"JSON_ARRAY_SUM({col_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "rand":
            if len(expression.args) == 0:
                if self.dialect.name == "postgresql":
                    result = func.random()
                elif self.dialect.name == "mysql":
                    result = func.rand()
                else:
                    result = func.random()
            else:
                # seed = expression.args[0]  # Not used in current implementation
                if self.dialect.name == "postgresql":
                    result = func.random()
                elif self.dialect.name == "mysql":
                    result = func.rand()
                else:
                    result = func.random()
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "randn":
            from sqlalchemy import literal_column

            if self.dialect.name == "postgresql":
                result = literal_column("random_normal()")
            elif self.dialect.name == "duckdb":
                # DuckDB doesn't have random_normal, use random() as fallback
                # Note: This is not a true normal distribution
                result = func.random()
            else:
                raise CompilationError(
                    f"randn() is not supported for {self.dialect.name} dialect. "
                    "PostgreSQL may support this with extensions."
                )
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "hash":
            cols = [self._compile(c) for c in expression.args]
            if self.dialect.name == "postgresql":
                if len(cols) == 1:
                    result = func.hashtext(cols[0])
                else:
                    from sqlalchemy import literal_column

                    concat_expr = func.concat(*cols)
                    result = func.hashtext(concat_expr)
            elif self.dialect.name == "mysql":
                if len(cols) == 1:
                    result = func.md5(cols[0])
                else:
                    concat_expr = func.concat(*cols)
                    result = func.md5(concat_expr)
            else:
                from sqlalchemy import literal_column

                if len(cols) == 1:
                    result = literal_column(f"HASH({cols[0]})")
                else:
                    concat_expr = func.concat(*cols)
                    result = literal_column(f"HASH({concat_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "md5":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.md5(col_expr)
            elif self.dialect.name == "mysql":
                result = func.md5(col_expr)
            else:
                # SQLite: md5 is not available by default, requires extension
                # For now, raise an error to indicate limitation
                from sqlalchemy import literal_column

                # Try to use md5 if available, otherwise raise error
                result = literal_column(f"md5({col_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "sha1":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.digest(col_expr, "sha1")
            elif self.dialect.name == "mysql":
                result = func.sha1(col_expr)
            else:
                from sqlalchemy import literal_column

                result = literal_column(f"SHA1({col_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "sha2":
            col_expr = self._compile(expression.args[0])
            num_bits_arg = expression.args[1]
            if not isinstance(num_bits_arg, int):
                raise TypeError(
                    f"Expected int for sha2 num_bits, got {type(num_bits_arg).__name__}"
                )
            num_bits: int = num_bits_arg
            if self.dialect.name == "postgresql":
                algo_map = {224: "sha224", 256: "sha256", 384: "sha384", 512: "sha512"}
                algo = algo_map.get(num_bits, "sha256")
                result = func.digest(col_expr, algo)
            elif self.dialect.name == "mysql":
                result = func.sha2(col_expr, num_bits)
            elif self.dialect.name == "duckdb":
                # DuckDB uses sha256, sha384, sha512 functions
                from sqlalchemy import literal_column

                if num_bits == 256:
                    result = func.sha256(col_expr)
                elif num_bits == 384:
                    result = func.sha384(col_expr)
                elif num_bits == 512:
                    result = func.sha512(col_expr)
                else:
                    # Default to sha256
                    result = func.sha256(col_expr)
            else:
                from sqlalchemy import literal_column

                result = literal_column(f"SHA2({col_expr}, {num_bits})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "base64":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.encode(col_expr, "base64")
            elif self.dialect.name == "mysql":
                result = func.to_base64(col_expr)
            elif self.dialect.name == "duckdb":
                # DuckDB requires BLOB type
                from sqlalchemy import cast
                from sqlalchemy import types as sa_types

                blob_expr = cast(col_expr, sa_types.BLOB)
                result = func.base64(blob_expr)
            else:
                from sqlalchemy import literal_column

                result = literal_column(f"BASE64({col_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "monotonically_increasing_id":
            # monotonically_increasing_id is implemented as row_number()
            # It always needs an OVER clause, even if empty
            # If used with .over(), the window handling will apply the .over() call
            # Otherwise, we use row_number() OVER () for SQLite compatibility
            result = func.row_number().over()
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "crc32":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "mysql":
                result = func.crc32(col_expr)
            else:
                from sqlalchemy import literal_column

                result = literal_column(f"CRC32({col_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "soundex":
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.soundex(col_expr)
            elif self.dialect.name == "mysql":
                result = func.soundex(col_expr)
            elif self.dialect.name == "duckdb":
                # DuckDB doesn't support soundex
                raise CompilationError(
                    f"soundex() is not supported for {self.dialect.name} dialect. "
                    "Supported dialects: PostgreSQL, MySQL"
                )
            else:
                from sqlalchemy import literal_column

                result = literal_column(f"SOUNDEX({col_expr})")
            if expression._alias:
                result = result.label(expression._alias)
            return result

        # Try window compiler
        if op.startswith("window_") or op == "window":
            from .expression_compilers.window import compile_window_operation

            result = compile_window_operation(self, op, expression)
            if result is not None:
                return result

        raise CompilationError(
            f"Unsupported expression operation '{op}'. "
            "This may indicate a missing function implementation or an invalid expression.",
            context={
                "operation": op,
                "expression_args": [
                    str(arg) for arg in expression.args[:3]
                ],  # Limit to first 3 args
                "dialect": self.dialect.name,
                "expression_alias": expression._alias,
            },
        )

    def _get_table(self, table_name: str) -> TableClause:
        """Get or create a SQLAlchemy table object for the given table name."""
        from sqlalchemy import table as sa_table

        if table_name not in self._table_cache:
            self._table_cache[table_name] = sa_table(table_name)
        result = self._table_cache[table_name]
        return result  # type: ignore[no-any-return]
