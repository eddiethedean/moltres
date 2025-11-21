"""Compile logical plans into SQL."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..engine.dialects import DialectSpec, get_dialect
from ..expressions.column import Column

if TYPE_CHECKING:
    from ..expressions.window import WindowSpec
from ..logical.plan import (
    Aggregate,
    Distinct,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Sort,
    SortOrder,
    TableScan,
)
from ..logical.plan import Union as UnionPlan
from ..utils.exceptions import CompilationError
from .builders import comma_separated, format_literal, quote_identifier


def compile_plan(plan: LogicalPlan, dialect: str | DialectSpec = "ansi") -> str:
    spec = get_dialect(dialect) if isinstance(dialect, str) else dialect
    compiler = SQLCompiler(spec)
    return compiler.compile(plan)


@dataclass(frozen=True)
class CompilationState:
    source: str
    alias: str
    select: tuple[Column, ...] | None = None
    predicate: Column | None = None
    group_by: tuple[Column, ...] = ()
    orders: tuple[SortOrder, ...] = ()
    limit: int | None = None
    offset: int = 0
    distinct: bool = False


class SQLCompiler:
    """Main entry point for compiling logical plans to SQL strings."""

    def __init__(self, dialect: DialectSpec):
        self.dialect = dialect
        self._expr = ExpressionCompiler(dialect)

    def compile(self, plan: LogicalPlan) -> str:
        state = self._analyze(plan)
        return self._state_to_sql(state)

    # ----------------------------------------------------------------- analyzers
    def _analyze(self, plan: LogicalPlan) -> CompilationState:
        if isinstance(plan, TableScan):
            table_sql = quote_identifier(plan.table, self.dialect.quote_char)
            if plan.alias:
                alias_sql = quote_identifier(plan.alias, self.dialect.quote_char)
                table_sql = f"{table_sql} AS {alias_sql}"
            alias = plan.alias or plan.table
            return CompilationState(source=table_sql, alias=alias)

        if isinstance(plan, Project):
            child_state = self._analyze(plan.child)
            return replace(child_state, select=plan.projections)

        if isinstance(plan, Filter):
            child_state = self._analyze(plan.child)
            predicate = (
                plan.predicate
                if child_state.predicate is None
                else Column(op="and", args=(child_state.predicate, plan.predicate))
            )
            return replace(child_state, predicate=predicate)

        if isinstance(plan, Limit):
            child_state = self._analyze(plan.child)
            new_limit = (
                plan.count if child_state.limit is None else min(child_state.limit, plan.count)
            )
            new_offset = plan.offset if plan.offset > 0 else child_state.offset
            return replace(child_state, limit=new_limit, offset=new_offset)

        if isinstance(plan, Distinct):
            child_state = self._analyze(plan.child)
            return replace(child_state, distinct=True)

        if isinstance(plan, UnionPlan):
            left_sql = self._state_to_sql(self._analyze(plan.left))
            right_sql = self._state_to_sql(self._analyze(plan.right))
            union_op = "UNION" if plan.distinct else "UNION ALL"
            source = f"({left_sql}) {union_op} ({right_sql})"
            alias = "union"
            return CompilationState(source=source, alias=alias)

        if isinstance(plan, Sort):
            child_state = self._analyze(plan.child)
            return replace(child_state, orders=plan.orders)

        if isinstance(plan, Aggregate):
            child_state = self._analyze(plan.child)
            projections = plan.grouping + plan.aggregates
            return replace(child_state, select=projections, group_by=plan.grouping)

        if isinstance(plan, Join):
            left_state = self._analyze(plan.left)
            right_state = self._analyze(plan.right)
            left_source = self._as_subquery(left_state)
            right_source = self._as_subquery(right_state)
            condition = self._compile_join_condition(plan, left_state.alias, right_state.alias)
            source = f"{left_source} {plan.how.upper()} JOIN {right_source} ON {condition}"
            alias = f"{left_state.alias}_{right_state.alias}"
            return CompilationState(source=source, alias=alias)

        raise CompilationError(
            f"Unsupported logical plan node: {type(plan).__name__}. "
            "Supported nodes: TableScan, Project, Filter, Limit, Sort, "
            "Aggregate, Join, Distinct, Union"
        )

    # ---------------------------------------------------------------- formatters
    def _format_projection(self, expr: Column) -> str:
        sql = self._expr.emit(expr)
        # Handle window functions
        if hasattr(expr, "_window") and expr._window is not None:  # pylint: disable=protected-access
            window_sql = self._compile_window(expr._window)  # pylint: disable=protected-access
            sql = f"{sql} OVER ({window_sql})"
        if expr._alias:  # pylint: disable=protected-access
            alias_sql = quote_identifier(expr._alias, self.dialect.quote_char)
            return f"{sql} AS {alias_sql}"
        return sql

    def _compile_window(self, window: WindowSpec) -> str:
        """Compile a window specification to SQL."""
        parts = []
        if window.partition_by:
            part_cols = comma_separated(self._expr.emit(col) for col in window.partition_by)
            parts.append(f"PARTITION BY {part_cols}")
        if window.order_by:
            order_cols = comma_separated(self._expr.emit(col) for col in window.order_by)
            parts.append(f"ORDER BY {order_cols}")
        if window.rows_between:
            start, end = window.rows_between
            frame = self._compile_frame(start, end, "ROWS")
            if frame:
                parts.append(frame)
        elif window.range_between:
            start, end = window.range_between
            frame = self._compile_frame(start, end, "RANGE")
            if frame:
                parts.append(frame)
        return " ".join(parts)

    def _compile_frame(self, start: int | None, end: int | None, frame_type: str) -> str:
        """Compile a window frame specification."""
        if start is None and end is None:
            return ""
        start_sql = self._frame_bound(start, "PRECEDING")
        end_sql = self._frame_bound(end, "FOLLOWING")
        return f"{frame_type} BETWEEN {start_sql} AND {end_sql}"

    def _frame_bound(self, bound: int | None, direction: str) -> str:
        """Compile a window frame bound."""
        if bound is None:
            return f"UNBOUNDED {direction}"
        if bound == 0:
            return "CURRENT ROW"
        if direction == "PRECEDING":
            if bound < 0:
                return f"{abs(bound)} {direction}"
            return f"{bound} {direction}"
        # FOLLOWING
        if bound > 0:
            return f"{bound} {direction}"
        return f"{abs(bound)} {direction}"

    def _format_order(self, order: SortOrder) -> str:
        direction = "DESC" if order.descending else "ASC"
        return f"{self._expr.emit(order.expression)} {direction}"

    def _state_to_sql(self, state: CompilationState) -> str:
        select_sql = "*"
        if state.select is not None:
            select_sql = comma_separated(self._format_projection(expr) for expr in state.select)

        distinct_clause = "DISTINCT " if state.distinct else ""
        sql = f"SELECT {distinct_clause}{select_sql} FROM {state.source}"

        if state.predicate is not None:
            sql += f" WHERE {self._expr.emit(state.predicate)}"

        if state.group_by:
            group_sql = comma_separated(self._expr.emit(expr) for expr in state.group_by)
            sql += f" GROUP BY {group_sql}"

        if state.orders:
            order_sql = comma_separated(self._format_order(order) for order in state.orders)
            sql += f" ORDER BY {order_sql}"

        if state.limit is not None:
            sql += f" LIMIT {state.limit}"
            if state.offset > 0:
                sql += f" OFFSET {state.offset}"

        return sql

    def _as_subquery(self, state: CompilationState) -> str:
        sql = self._state_to_sql(state)
        alias_sql = quote_identifier(state.alias, self.dialect.quote_char)
        return f"({sql}) AS {alias_sql}"

    def _compile_join_condition(self, plan: Join, left_alias: str, right_alias: str) -> str:
        if plan.on:
            clauses = []
            for left_col, right_col in plan.on:
                left_ref = quote_identifier(f"{left_alias}.{left_col}", self.dialect.quote_char)
                right_ref = quote_identifier(f"{right_alias}.{right_col}", self.dialect.quote_char)
                clauses.append(f"({left_ref} = {right_ref})")
            return " AND ".join(clauses)
        if plan.condition is not None:
            return self._expr.emit(plan.condition)
        raise CompilationError(
            "Join requires either equality keys (via 'on' parameter) or an explicit condition. "
            f"Join type: {plan.how}, left alias: {left_alias}, right alias: {right_alias}"
        )


class ExpressionCompiler:
    """Compile expression trees into SQL snippets."""

    BINARY_OPERATORS = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "div": "/",
        "eq": "=",
        "ne": "<>",
        "lt": "<",
        "le": "<=",
        "gt": ">",
        "ge": ">=",
    }

    def __init__(self, dialect: DialectSpec):
        self.dialect = dialect

    def emit(self, expression: Column) -> str:
        op = expression.op

        if op == "column":
            return quote_identifier(expression.args[0], self.dialect.quote_char)
        if op == "literal":
            return format_literal(expression.args[0])
        if op in self.BINARY_OPERATORS:
            left, right = expression.args
            operator = self.BINARY_OPERATORS[op]
            return f"({self.emit(left)} {operator} {self.emit(right)})"
        if op == "floor_div":
            left, right = expression.args
            return f"FLOOR({self.emit(left)} / {self.emit(right)})"
        if op == "mod":
            left, right = expression.args
            return f"MOD({self.emit(left)}, {self.emit(right)})"
        if op == "pow":
            args = ", ".join(self.emit(arg) for arg in expression.args)
            return f"POWER({args})"
        if op == "neg":
            return f"(-{self.emit(expression.args[0])})"
        if op == "and":
            left, right = expression.args
            return f"({self.emit(left)} AND {self.emit(right)})"
        if op == "or":
            left, right = expression.args
            return f"({self.emit(left)} OR {self.emit(right)})"
        if op == "not":
            return f"(NOT {self.emit(expression.args[0])})"
        if op == "between":
            value, lower, upper = expression.args
            return f"({self.emit(value)} BETWEEN {self.emit(lower)} AND {self.emit(upper)})"
        if op == "in":
            value, options = expression.args
            options_sql = comma_separated(self.emit(option) for option in options)
            return f"({self.emit(value)} IN ({options_sql}))"
        if op == "like":
            left, pattern = expression.args
            return f"({self.emit(left)} LIKE {self.emit(pattern)})"
        if op == "ilike":
            left, pattern = expression.args
            return f"({self.emit(left)} ILIKE {self.emit(pattern)})"
        if op == "contains":
            column, substring = expression.args
            return f"({self.emit(column)} LIKE '%' || {self.emit(substring)} || '%')"
        if op == "startswith":
            column, prefix = expression.args
            return f"({self.emit(column)} LIKE {self.emit(prefix)} || '%')"
        if op == "endswith":
            column, suffix = expression.args
            return f"({self.emit(column)} LIKE '%' || {self.emit(suffix)})"
        if op == "cast":
            column, type_name = expression.args
            return f"CAST({self.emit(column)} AS {type_name})"
        if op == "is_null":
            return f"({self.emit(expression.args[0])} IS NULL)"
        if op == "is_not_null":
            return f"({self.emit(expression.args[0])} IS NOT NULL)"
        if op == "coalesce":
            args = comma_separated(self.emit(arg) for arg in expression.args)
            return f"COALESCE({args})"
        if op == "concat":
            args = comma_separated(self.emit(arg) for arg in expression.args)
            return f"CONCAT({args})"
        if op == "upper":
            return f"UPPER({self.emit(expression.args[0])})"
        if op == "lower":
            return f"LOWER({self.emit(expression.args[0])})"
        if op == "greatest":
            args = comma_separated(self.emit(arg) for arg in expression.args)
            return f"GREATEST({args})"
        if op == "least":
            args = comma_separated(self.emit(arg) for arg in expression.args)
            return f"LEAST({args})"
        if op == "substring":
            if len(expression.args) == 2:
                col_expr, start = expression.args
                return f"SUBSTRING({self.emit(col_expr)}, {start})"
            if len(expression.args) == 3:
                col_expr, start, length = expression.args
                return f"SUBSTRING({self.emit(col_expr)}, {start}, {length})"
        if op == "trim":
            return f"TRIM({self.emit(expression.args[0])})"
        if op == "ltrim":
            return f"LTRIM({self.emit(expression.args[0])})"
        if op == "rtrim":
            return f"RTRIM({self.emit(expression.args[0])})"
        if op == "replace":
            col_expr, old, new = expression.args
            return f"REPLACE({self.emit(col_expr)}, {format_literal(old)}, {format_literal(new)})"
        if op == "length":
            return f"LENGTH({self.emit(expression.args[0])})"
        if op == "abs":
            return f"ABS({self.emit(expression.args[0])})"
        if op == "round":
            if len(expression.args) == 1:
                return f"ROUND({self.emit(expression.args[0])})"
            if len(expression.args) == 2:
                col_expr, decimals = expression.args
                return f"ROUND({self.emit(col_expr)}, {decimals})"
        if op == "floor":
            return f"FLOOR({self.emit(expression.args[0])})"
        if op == "ceil":
            return f"CEIL({self.emit(expression.args[0])})"
        if op == "ceiling":  # Some databases use CEILING
            return f"CEILING({self.emit(expression.args[0])})"
        if op == "trunc":
            return f"TRUNC({self.emit(expression.args[0])})"
        if op == "sqrt":
            return f"SQRT({self.emit(expression.args[0])})"
        if op == "exp":
            return f"EXP({self.emit(expression.args[0])})"
        if op == "log":
            return f"LN({self.emit(expression.args[0])})"  # Natural log
        if op == "log10":
            return f"LOG10({self.emit(expression.args[0])})"
        if op == "current_date":
            return "CURRENT_DATE"
        if op == "current_timestamp":
            return "CURRENT_TIMESTAMP"
        if op == "date_add":
            col_expr, days = expression.args
            return f"DATE_ADD({self.emit(col_expr)}, INTERVAL {days} DAY)"
        if op == "date_sub":
            col_expr, days = expression.args
            return f"DATE_SUB({self.emit(col_expr)}, INTERVAL {days} DAY)"
        if op == "datediff":
            end, start = expression.args
            return f"DATEDIFF({self.emit(end)}, {self.emit(start)})"
        if op == "year":
            return f"YEAR({self.emit(expression.args[0])})"
        if op == "month":
            return f"MONTH({self.emit(expression.args[0])})"
        if op == "day":
            return f"DAY({self.emit(expression.args[0])})"
        if op == "hour":
            return f"HOUR({self.emit(expression.args[0])})"
        if op == "minute":
            return f"MINUTE({self.emit(expression.args[0])})"
        if op == "second":
            return f"SECOND({self.emit(expression.args[0])})"
        if op == "case_when":
            # CASE WHEN cond1 THEN val1 WHEN cond2 THEN val2 ... ELSE default END
            sql = "CASE"
            i = 0
            while i < len(expression.args) - 1:
                cond = expression.args[i]
                val = expression.args[i + 1]
                sql += f" WHEN {self.emit(cond)} THEN {self.emit(val)}"
                i += 2
            if i < len(expression.args):
                # There's an ELSE clause
                sql += f" ELSE {self.emit(expression.args[i])}"
            sql += " END"
            return sql
        if op == "agg_sum":
            return f"SUM({self.emit(expression.args[0])})"
        if op == "agg_avg":
            return f"AVG({self.emit(expression.args[0])})"
        if op == "agg_min":
            return f"MIN({self.emit(expression.args[0])})"
        if op == "agg_max":
            return f"MAX({self.emit(expression.args[0])})"
        if op == "agg_count":
            return f"COUNT({self.emit(expression.args[0])})"
        if op == "agg_count_star":
            return "COUNT(*)"
        if op == "agg_count_distinct":
            args = comma_separated(self.emit(arg) for arg in expression.args)
            return f"COUNT(DISTINCT {args})"
        # Window functions
        if op == "window_row_number":
            return "ROW_NUMBER()"
        if op == "window_rank":
            return "RANK()"
        if op == "window_dense_rank":
            return "DENSE_RANK()"
        if op == "window_lag":
            if len(expression.args) == 2:
                col_expr, offset = expression.args
                return f"LAG({self.emit(col_expr)}, {offset})"
            if len(expression.args) == 3:
                col_expr, offset, default = expression.args
                return f"LAG({self.emit(col_expr)}, {offset}, {self.emit(default)})"
        if op == "window_lead":
            if len(expression.args) == 2:
                col_expr, offset = expression.args
                return f"LEAD({self.emit(col_expr)}, {offset})"
            if len(expression.args) == 3:
                col_expr, offset, default = expression.args
                return f"LEAD({self.emit(col_expr)}, {offset}, {self.emit(default)})"
        if op == "window_first_value":
            return f"FIRST_VALUE({self.emit(expression.args[0])})"
        if op == "window_last_value":
            return f"LAST_VALUE({self.emit(expression.args[0])})"
        # Window aggregate functions (sum, avg, etc. with OVER)
        # These are handled by checking for _window attribute in _format_projection

        raise CompilationError(
            f"Unsupported expression operation '{op}'. "
            "This may indicate a missing function implementation or an invalid expression."
        )
