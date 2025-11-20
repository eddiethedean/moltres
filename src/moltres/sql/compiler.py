"""Compile logical plans into SQL."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Tuple, Union

from ..engine.dialects import DialectSpec, get_dialect
from ..expressions.column import Column
from ..logical.plan import (
    Aggregate,
    Filter,
    Join,
    Limit,
    LogicalPlan,
    Project,
    Sort,
    SortOrder,
    TableScan,
)
from ..utils.exceptions import CompilationError
from .builders import comma_separated, format_literal, quote_identifier


def compile_plan(plan: LogicalPlan, dialect: Union[str, DialectSpec] = "ansi") -> str:
    spec = get_dialect(dialect) if isinstance(dialect, str) else dialect
    compiler = SQLCompiler(spec)
    return compiler.compile(plan)


@dataclass(frozen=True)
class CompilationState:
    source: str
    alias: str
    select: Optional[tuple[Column, ...]] = None
    predicate: Optional[Column] = None
    group_by: tuple[Column, ...] = ()
    orders: tuple[SortOrder, ...] = ()
    limit: Optional[int] = None


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
            new_limit = plan.count if child_state.limit is None else min(child_state.limit, plan.count)
            return replace(child_state, limit=new_limit)

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

        raise CompilationError(f"Unsupported logical plan node: {type(plan).__name__}")

    # ---------------------------------------------------------------- formatters
    def _format_projection(self, expr: Column) -> str:
        sql = self._expr.emit(expr)
        if expr._alias:  # pylint: disable=protected-access
            alias_sql = quote_identifier(expr._alias, self.dialect.quote_char)
            return f"{sql} AS {alias_sql}"
        return sql

    def _format_order(self, order: SortOrder) -> str:
        direction = "DESC" if order.descending else "ASC"
        return f"{self._expr.emit(order.expression)} {direction}"

    def _state_to_sql(self, state: CompilationState) -> str:
        select_sql = "*"
        if state.select is not None:
            select_sql = comma_separated(self._format_projection(expr) for expr in state.select)

        sql = f"SELECT {select_sql} FROM {state.source}"

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
        raise CompilationError("Join requires either equality keys or an explicit condition")


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

        raise CompilationError(f"Unsupported expression operation '{op}'")
