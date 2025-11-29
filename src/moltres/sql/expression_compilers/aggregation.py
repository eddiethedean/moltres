"""Compilation of aggregation expressions.

This module handles compilation of aggregation functions like sum, avg, min, max, count, etc.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import func, case as sa_case, literal

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement
    from ..expression_compiler import ExpressionCompiler


def compile_aggregation(
    compiler: "ExpressionCompiler", op: str, expression: Any
) -> "ColumnElement[Any]":
    """Compile an aggregation expression.

    Args:
        compiler: The ExpressionCompiler instance
        op: Operation name (e.g., "agg_sum", "agg_avg")
        expression: :class:`Column` expression to compile

    Returns:
        Compiled SQLAlchemy column element
    """

    # Helper to apply filter clause with fallback
    def apply_filter(
        result: "ColumnElement[Any]", col_expr: "ColumnElement[Any]"
    ) -> "ColumnElement[Any]":
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                # Fallback to CASE WHEN for unsupported dialects
                # Recompile the aggregation with the case expression
                # This requires re-evaluating the operation, so we return a modified result
                # For now, we'll handle this in each specific aggregation
                _ = sa_case((filter_condition, col_expr), else_=None)
                pass
        return result

    if op == "agg_sum":
        col_expr = compiler._compile(expression.args[0])
        result: "ColumnElement[Any]" = func.sum(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                result = func.sum(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_avg":
        col_expr = compiler._compile(expression.args[0])
        result = func.avg(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                result = func.avg(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_min":
        col_expr = compiler._compile(expression.args[0])
        result = func.min(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                result = func.min(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_max":
        col_expr = compiler._compile(expression.args[0])
        result = func.max(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                result = func.max(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_count":
        col_expr = compiler._compile(expression.args[0])
        result = func.count(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                result = func.count(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_stddev":
        col_expr = compiler._compile(expression.args[0])
        result = func.stddev(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                result = func.stddev(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_variance":
        col_expr = compiler._compile(expression.args[0])
        result = func.variance(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                result = func.variance(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_corr":
        col1, col2 = expression.args
        col1_expr = compiler._compile(col1)
        col2_expr = compiler._compile(col2)
        result = func.corr(col1_expr, col2_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_col1 = sa_case((filter_condition, col1_expr), else_=None)
                case_col2 = sa_case((filter_condition, col2_expr), else_=None)
                result = func.corr(case_col1, case_col2)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_covar":
        col1, col2 = expression.args
        col1_expr = compiler._compile(col1)
        col2_expr = compiler._compile(col2)
        result = func.covar_pop(col1_expr, col2_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_col1 = sa_case((filter_condition, col1_expr), else_=None)
                case_col2 = sa_case((filter_condition, col2_expr), else_=None)
                result = func.covar_pop(case_col1, case_col2)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_count_star":
        result = func.count()
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, literal(1)), else_=None)
                result = func.count(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_count_distinct":
        args = [compiler._compile(arg) for arg in expression.args]
        result = func.count(func.distinct(*args))
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_args = [sa_case((filter_condition, arg), else_=None) for arg in args]
                result = func.count(func.distinct(*case_args))
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_collect_list":
        col_expr = compiler._compile(expression.args[0])
        if compiler.dialect.name == "postgresql" or compiler.dialect.name == "duckdb":
            result = func.array_agg(col_expr)
        elif compiler.dialect.name == "sqlite":
            result = func.json_group_array(col_expr)
        else:
            result = func.json_arrayagg(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                if compiler.dialect.name == "postgresql" or compiler.dialect.name == "duckdb":
                    result = func.array_agg(case_expr)
                elif compiler.dialect.name == "sqlite":
                    result = func.json_group_array(case_expr)
                else:
                    result = func.json_arrayagg(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_collect_set":
        col_expr = compiler._compile(expression.args[0])
        if compiler.dialect.name == "postgresql":
            result = func.array_agg(func.distinct(col_expr))
        elif compiler.dialect.name == "duckdb":
            result = func.array_agg(func.distinct(col_expr))
        elif compiler.dialect.name == "sqlite":
            result = func.json_group_array(func.distinct(col_expr))
        else:
            result = func.json_arrayagg(col_expr)
        if expression._filter is not None:
            filter_condition = compiler._compile(expression._filter)
            if compiler.dialect.supports_filter_clause:
                result = result.filter(filter_condition)
            else:
                case_expr = sa_case((filter_condition, col_expr), else_=None)
                if compiler.dialect.name == "postgresql":
                    result = func.array_agg(func.distinct(case_expr))
                elif compiler.dialect.name == "sqlite":
                    result = func.json_group_array(func.distinct(case_expr))
                else:
                    result = func.json_arrayagg(case_expr)
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_percentile_cont":
        col_expr = compiler._compile(expression.args[0])
        fraction = expression.args[1]
        if compiler.dialect.name in ("postgresql", "mssql", "oracle"):
            result = func.percentile_cont(fraction).within_group(col_expr)
        else:
            if compiler.dialect.name == "duckdb":
                from sqlalchemy import literal_column

                result = literal_column(
                    f"percentile_cont({fraction}) WITHIN GROUP (ORDER BY {col_expr})"
                )
            else:
                from ...utils.exceptions import CompilationError

                raise CompilationError(
                    f"percentile_cont() is not supported for {compiler.dialect.name} dialect. "
                    "Supported dialects: PostgreSQL, SQL Server, Oracle, DuckDB"
                )
        if expression._alias:
            result = result.label(expression._alias)
        return result

    if op == "agg_percentile_disc":
        col_expr = compiler._compile(expression.args[0])
        fraction = expression.args[1]
        if compiler.dialect.name in ("postgresql", "mssql", "oracle"):
            result = func.percentile_disc(fraction).within_group(col_expr)
        else:
            if compiler.dialect.name == "duckdb":
                from sqlalchemy import literal_column

                result = literal_column(
                    f"percentile_disc({fraction}) WITHIN GROUP (ORDER BY {col_expr})"
                )
            else:
                from ...utils.exceptions import CompilationError

                raise CompilationError(
                    f"percentile_disc() is not supported for {compiler.dialect.name} dialect. "
                    "Supported dialects: PostgreSQL, SQL Server, Oracle, DuckDB"
                )
        if expression._alias:
            result = result.label(expression._alias)
        return result

    return None  # type: ignore[return-value]  # Not handled by this module
