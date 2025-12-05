"""Compile logical plans into SQLAlchemy Select statements."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import (
    select,
    table,
    func,
    case as sa_case,
    and_,
    text,
)
from sqlalchemy.sql import Select, ColumnElement

if TYPE_CHECKING:
    from sqlalchemy.sql.selectable import FromClause

from ..engine.dialects import DialectSpec
from ..logical.plan import (
    Aggregate,
    AntiJoin,
    CTE,
    Distinct,
    Except,
    Explode,
    FileScan,
    Filter,
    GroupedPivot,
    Intersect,
    Join,
    Limit,
    LogicalPlan,
    Pivot,
    Project,
    RawSQL,
    RecursiveCTE,
    Sample,
    SemiJoin,
    Sort,
    TableScan,
    Union as LogicalUnion,
)
from ..utils.exceptions import CompilationError
from .expression_compiler import ExpressionCompiler

logger = logging.getLogger(__name__)


class SQLCompiler:
    """Main entry point for compiling logical plans to SQLAlchemy Select statements."""

    def __init__(self, dialect: DialectSpec):
        self.dialect = dialect
        self._expr = ExpressionCompiler(dialect, plan_compiler=self)

    def compile(self, plan: LogicalPlan) -> Select:
        """Compile a logical plan to a SQLAlchemy Select statement."""
        return self._compile_plan(plan)

    def _extract_table_name(self, plan: LogicalPlan) -> Optional[str]:
        """Extract table name from a plan (for TableScan) or None."""
        if isinstance(plan, TableScan):
            return plan.alias or plan.table
        # For other plan types, try to extract from child
        if hasattr(plan, "child"):
            return self._extract_table_name(plan.child)
        if hasattr(plan, "left"):
            return self._extract_table_name(plan.left)
        return None

    def _compile_plan(self, plan: LogicalPlan) -> Select:
        """Compile a logical plan to a SQLAlchemy Select statement."""
        if isinstance(plan, FileScan):
            # FileScan should be materialized before compilation
            # This should not happen if materialization is done correctly
            raise CompilationError(
                "FileScan cannot be compiled directly to SQL. "
                "FileScan nodes must be materialized into temporary tables before compilation. "
                "This is typically handled automatically by DataFrame.collect()."
            )

        if isinstance(plan, CTE):
            # Compile the child plan and convert it to a CTE
            child_stmt = self._compile_plan(plan.child)
            cte_obj = child_stmt.cte(plan.name)
            # Return a select from the CTE
            from sqlalchemy import literal, literal_column

            return select(literal_column("*")).select_from(cte_obj)

        if isinstance(plan, RecursiveCTE):
            # WITH RECURSIVE: compile initial and recursive parts
            initial_stmt = self._compile_plan(plan.initial)
            recursive_stmt = self._compile_plan(plan.recursive)

            # Create recursive CTE using SQLAlchemy's recursive CTE support
            # SQLAlchemy uses .cte(recursive=True) for recursive CTEs
            from sqlalchemy import literal, literal_column

            # For recursive CTEs, we need to combine initial and recursive with UNION/UNION ALL
            if plan.union_all:
                combined = initial_stmt.union_all(recursive_stmt)
            else:
                combined = initial_stmt.union(recursive_stmt)

            # Create recursive CTE
            recursive_cte_obj = combined.cte(name=plan.name, recursive=True)

            # Return a select from the recursive CTE
            return select(literal_column("*")).select_from(recursive_cte_obj)

        if isinstance(plan, RawSQL):
            # Wrap raw SQL in a subquery so it can be used in SELECT statements
            # and support chaining operations
            from sqlalchemy import literal_column

            # Create a text() statement from the SQL string
            sql_text = text(plan.sql)

            # If parameters are provided, bind them
            if plan.params:
                sql_text = sql_text.bindparams(**plan.params)

            # Use text().columns() to define it as a subquery with all columns
            # This allows SQLAlchemy to properly wrap it in parentheses
            # We use literal_column("*") to represent all columns
            sql_text_with_cols = sql_text.columns(literal_column("*"))

            # Create a subquery from the text with columns
            sql_subq = sql_text_with_cols.subquery()

            # Return a SELECT * from the subquery
            return select(literal_column("*")).select_from(sql_subq)

        if isinstance(plan, TableScan):
            sa_table = table(plan.table)
            if plan.alias:
                sa_table = sa_table.alias(plan.alias)  # type: ignore[assignment]
            # Use select() with explicit * to select all columns from the table
            # table() objects don't have column metadata, so we need to use *
            from sqlalchemy import literal_column

            stmt: Select[Any] = select(literal_column("*")).select_from(sa_table)
            return stmt

        if isinstance(plan, Project):
            child_stmt = self._compile_plan(plan.child)
            # Convert child to subquery if it's a Select statement
            if isinstance(child_stmt, Select):
                child_subq = child_stmt.subquery()
            else:
                child_subq = child_stmt

            # If child is a Join, store join subquery info for qualified column resolution
            join_info = None
            if isinstance(plan.child, Join):
                # Extract table names from join sides
                left_table_name = self._extract_table_name(plan.child.left)
                right_table_name = self._extract_table_name(plan.child.right)

                if left_table_name and right_table_name:
                    join_info = {
                        "left_table": left_table_name,
                        "right_table": right_table_name,
                    }

            # Compile column expressions with subquery context for qualified names
            old_subq = self._expr._current_subq
            old_join_info = self._expr._join_info
            self._expr._current_subq = child_subq
            if join_info:
                self._expr._join_info = join_info
            try:
                if plan.projections:
                    columns = [self._expr.compile_expr(col) for col in plan.projections]
                else:
                    # Empty projections means SELECT *
                    from sqlalchemy import literal_column

                    columns = [literal_column("*")]
            finally:
                self._expr._current_subq = old_subq
                self._expr._join_info = old_join_info
            # Create new select with these columns
            stmt = select(*columns).select_from(child_subq)

            # Add row-level locking clauses if specified
            if plan.for_update or plan.for_share:
                if not self.dialect.supports_row_locking:
                    raise CompilationError(
                        f"Dialect '{self.dialect.name}' does not support row-level locking"
                    )

                if plan.for_update:
                    if plan.for_update_nowait:
                        if not self.dialect.supports_for_update_nowait:
                            raise CompilationError(
                                f"Dialect '{self.dialect.name}' does not support FOR UPDATE NOWAIT"
                            )
                        stmt = stmt.with_for_update(nowait=True)
                    elif plan.for_update_skip_locked:
                        if not self.dialect.supports_for_update_skip_locked:
                            raise CompilationError(
                                f"Dialect '{self.dialect.name}' does not support FOR UPDATE SKIP LOCKED"
                            )
                        # SQLAlchemy uses skip_locked=True for skip locked
                        stmt = stmt.with_for_update(skip_locked=True)
                    else:
                        stmt = stmt.with_for_update()
                elif plan.for_share:
                    if plan.for_update_nowait:
                        if not self.dialect.supports_for_update_nowait:
                            raise CompilationError(
                                f"Dialect '{self.dialect.name}' does not support FOR UPDATE NOWAIT"
                            )
                        stmt = stmt.with_for_update(read=True, nowait=True)
                    elif plan.for_update_skip_locked:
                        if not self.dialect.supports_for_update_skip_locked:
                            raise CompilationError(
                                f"Dialect '{self.dialect.name}' does not support FOR UPDATE SKIP LOCKED"
                            )
                        stmt = stmt.with_for_update(read=True, skip_locked=True)
                    else:
                        stmt = stmt.with_for_update(read=True)

            return stmt

        if isinstance(plan, Filter):
            child_stmt = self._compile_plan(plan.child)
            predicate = self._expr.compile_expr(plan.predicate)
            return child_stmt.where(predicate)

        if isinstance(plan, Limit):
            child_stmt = self._compile_plan(plan.child)
            stmt = child_stmt.limit(plan.count)
            if plan.offset:
                stmt = stmt.offset(plan.offset)
            return stmt

        if isinstance(plan, Sample):
            child_stmt = self._compile_plan(plan.child)
            # Degenerate cases: fraction <= 0 -> empty, fraction >= 1 -> passthrough
            if plan.fraction <= 0:
                return child_stmt.limit(0)
            if plan.fraction >= 1:
                return child_stmt

            from sqlalchemy import func as sa_func, literal, literal_column

            if isinstance(child_stmt, Select):
                child_subq = child_stmt.subquery()
            else:
                child_subq = child_stmt

            fraction_literal = literal(plan.fraction)

            # Type annotation needed because different branches return different SQLAlchemy types
            rand_expr: ColumnElement[Any]
            if self.dialect.name == "mysql":
                rand_expr = sa_func.rand(plan.seed) if plan.seed is not None else sa_func.rand()
            elif self.dialect.name == "sqlite":
                # SQLite random() returns signed 64-bit integer; normalize to [0,1)
                rand_expr = sa_func.abs(sa_func.random()) / literal(9223372036854775808.0)
            else:
                rand_expr = sa_func.random()

            predicate = rand_expr <= fraction_literal
            stmt = select(literal_column("*")).select_from(child_subq).where(predicate)
            return stmt

        if isinstance(plan, Sort):
            child_stmt = self._compile_plan(plan.child)
            order_by_clauses = []
            for order in plan.orders:
                expr = self._expr.compile_expr(order.expression)
                if order.descending:
                    expr = expr.desc()
                else:
                    expr = expr.asc()
                order_by_clauses.append(expr)
            return child_stmt.order_by(*order_by_clauses)

        if isinstance(plan, Aggregate):
            child_stmt = self._compile_plan(plan.child)
            # Convert child to subquery if it's a Select statement
            if isinstance(child_stmt, Select):
                child_subq = child_stmt.subquery()
            else:
                child_subq = child_stmt
            group_by_cols = [self._expr.compile_expr(col) for col in plan.grouping]
            agg_cols = [self._expr.compile_expr(col) for col in plan.aggregates]
            stmt = select(*group_by_cols, *agg_cols).select_from(child_subq)
            if group_by_cols:
                stmt = stmt.group_by(*group_by_cols)
            return stmt

        if isinstance(plan, Join):
            left_stmt = self._compile_plan(plan.left)
            right_stmt = self._compile_plan(plan.right)

            # Extract table names for aliasing subqueries (helps with qualified column resolution)
            left_table_name = self._extract_table_name(plan.left)
            right_table_name = self._extract_table_name(plan.right)

            left_subq, _ = self._normalize_join_side(plan.left, left_stmt, left_table_name)
            right_subq, right_hint_target = self._normalize_join_side(
                plan.right, right_stmt, right_table_name
            )

            # Build join condition
            if plan.on:
                conditions = []
                from sqlalchemy import column as sa_column, literal_column

                for left_col, right_col in plan.on:
                    # Use table-qualified column names in join condition to avoid ambiguity
                    # Reference columns from the aliased subqueries
                    if left_table_name:
                        try:
                            left_expr = left_subq.c[left_col]
                        except (KeyError, AttributeError, TypeError):
                            # Fallback to literal with table qualification using dialect quote char
                            quote = self.dialect.quote_char
                            left_expr = literal_column(
                                f"{quote}{left_table_name}{quote}.{quote}{left_col}{quote}"
                            )
                    else:
                        left_expr = sa_column(left_col)

                    if right_table_name:
                        try:
                            right_expr = right_subq.c[right_col]
                        except (KeyError, AttributeError, TypeError):
                            # Fallback to literal with table qualification using dialect quote char
                            quote = self.dialect.quote_char
                            right_expr = literal_column(
                                f"{quote}{right_table_name}{quote}.{quote}{right_col}{quote}"
                            )
                    else:
                        right_expr = sa_column(right_col)

                    conditions.append(left_expr == right_expr)
                join_condition = and_(*conditions) if len(conditions) > 1 else conditions[0]
            elif plan.condition is not None:
                join_condition = self._expr.compile_expr(plan.condition)
            elif plan.how == "cross":
                # Cross joins don't require a condition - handled separately
                join_condition = None
            else:
                raise CompilationError(
                    "Join requires either 'on' keys or a 'condition' (except for cross joins)",
                    suggestion=(
                        "Provide join conditions using either:\n"
                        "  - on=[('left_col', 'right_col')] for equality joins\n"
                        "  - condition=col('left_col') == col('right_col') for custom conditions\n"
                        "  - how='cross' for cross joins (no condition needed)"
                    ),
                )

            # Build join - use SELECT * to get all columns from both sides
            from sqlalchemy import literal_column

            # Handle LATERAL joins (PostgreSQL, MySQL 8.0+)
            if plan.lateral:
                if self.dialect.name not in ("postgresql", "mysql"):
                    raise CompilationError(
                        f"LATERAL joins are not supported for {self.dialect.name} dialect. "
                        "Supported dialects: PostgreSQL, MySQL 8.0+"
                    )
                from sqlalchemy import lateral

                # Wrap right side in lateral()
                right_lateral = lateral(right_subq)
                if plan.how == "cross":
                    stmt = select(literal_column("*")).select_from(left_subq, right_lateral)
                elif plan.how == "inner":
                    stmt = select(literal_column("*")).select_from(
                        left_subq.join(right_lateral, join_condition)
                    )
                elif plan.how == "left":
                    stmt = select(literal_column("*")).select_from(
                        left_subq.join(right_lateral, join_condition, isouter=True)
                    )
                else:
                    raise CompilationError(
                        f"LATERAL join with '{plan.how}' join type is not supported. "
                        "Supported types: inner, left, cross"
                    )
            elif plan.how == "cross":
                # Cross join doesn't need a condition
                stmt = select(literal_column("*")).select_from(left_subq, right_subq)
            elif plan.how == "inner":
                stmt = select(literal_column("*")).select_from(
                    left_subq.join(right_subq, join_condition)
                )
            elif plan.how == "left":
                stmt = select(literal_column("*")).select_from(
                    left_subq.join(right_subq, join_condition, isouter=True)
                )
            elif plan.how == "right":
                stmt = select(literal_column("*")).select_from(
                    right_subq.join(left_subq, join_condition, isouter=True)
                )
            elif plan.how in ("outer", "full"):
                stmt = select(literal_column("*")).select_from(
                    left_subq.join(right_subq, join_condition, full=True)
                )
            else:
                raise CompilationError(
                    f"Unsupported join type: {plan.how}",
                    suggestion=(
                        f"Supported join types are: 'inner', 'left', 'right', 'full', 'cross'. "
                        f"Received: {plan.how!r}"
                    ),
                )

            if plan.hints:
                hint_target = right_hint_target if right_hint_target is not None else right_subq
                stmt = self._apply_join_hints(stmt, target=hint_target, hints=plan.hints)

            return stmt

        if isinstance(plan, SemiJoin):
            # Semi-join: equivalent to INNER JOIN with DISTINCT
            # SELECT DISTINCT left.* FROM left INNER JOIN right ON condition
            left_stmt = self._compile_plan(plan.left)
            right_stmt = self._compile_plan(plan.right)

            # Extract table names for aliasing
            from sqlalchemy import literal_column

            left_table_name = self._extract_table_name(plan.left)
            right_table_name = self._extract_table_name(plan.right)

            # Convert to subqueries
            if isinstance(left_stmt, Select):
                left_subq = (
                    left_stmt.subquery(name=left_table_name)
                    if left_table_name
                    else left_stmt.subquery()
                )
            else:
                left_subq = left_stmt
            if isinstance(right_stmt, Select):
                right_subq = (
                    right_stmt.subquery(name=right_table_name)
                    if right_table_name
                    else right_stmt.subquery()
                )
            else:
                right_subq = right_stmt

            # Build join condition
            if plan.on:
                conditions = []
                from sqlalchemy import column as sa_column, literal_column

                for left_col, right_col in plan.on:
                    # Use table-qualified column names in join condition to avoid ambiguity
                    if left_table_name:
                        try:
                            left_expr = left_subq.c[left_col]
                        except (KeyError, AttributeError, TypeError):
                            # Fallback to literal with table qualification
                            left_expr = literal_column(f'"{left_table_name}"."{left_col}"')
                    else:
                        left_expr = sa_column(left_col)
                    if right_table_name:
                        try:
                            right_expr = right_subq.c[right_col]
                        except (KeyError, AttributeError, TypeError):
                            # Fallback to literal with table qualification
                            right_expr = literal_column(f'"{right_table_name}"."{right_col}"')
                    else:
                        right_expr = sa_column(right_col)
                    conditions.append(left_expr == right_expr)
                join_condition = and_(*conditions) if len(conditions) > 1 else conditions[0]
            elif plan.condition is not None:
                join_condition = self._expr.compile_expr(plan.condition)
            else:
                raise CompilationError(
                    "SemiJoin requires either 'on' keys or a 'condition'",
                    suggestion=(
                        "Provide join conditions using either:\n"
                        "  - on=[('left_col', 'right_col')] for equality joins\n"
                        "  - condition=col('left_col') == col('right_col') for custom conditions"
                    ),
                )

            # Build INNER JOIN with DISTINCT (equivalent to semi-join)
            # Select only columns from left table to avoid ambiguity
            # Get column names from left_subq
            if hasattr(left_subq, "c"):
                left_cols = [left_subq.c[col] for col in left_subq.c.keys()]
                stmt = (
                    select(*left_cols)
                    .select_from(left_subq.join(right_subq, join_condition))
                    .distinct()
                )
            else:
                # Fallback: use * but this may cause ambiguity
                stmt = (
                    select(literal_column("*"))
                    .select_from(left_subq.join(right_subq, join_condition))
                    .distinct()
                )
            return stmt

        if isinstance(plan, Pivot):
            # Pivot operation - use CASE WHEN with GROUP BY for cross-dialect compatibility
            child_stmt = self._compile_plan(plan.child)
            from sqlalchemy import literal, literal_column

            if not plan.pivot_values:
                raise CompilationError(
                    "PIVOT without pivot_values requires querying distinct values first. "
                    "Please provide pivot_values explicitly.",
                    suggestion="Specify pivot_values parameter: df.pivot(..., pivot_values=['value1', 'value2'])",
                )

            # Use CASE WHEN with aggregation for cross-dialect compatibility
            projections = []
            from typing import Callable as CallableType
            from sqlalchemy.sql import ColumnElement as ColumnElementType

            agg_func_map: dict[str, CallableType[..., ColumnElementType[Any]]] = {
                "sum": func.sum,
                "avg": func.avg,
                "count": func.count,
                "min": func.min,
                "max": func.max,
            }
            agg = agg_func_map.get(plan.agg_func.lower(), func.sum)
            assert agg is not None  # Always has default

            for pivot_value in plan.pivot_values:
                # Create aggregation with CASE WHEN
                case_expr = agg(
                    sa_case(
                        (
                            literal_column(plan.pivot_column) == literal(pivot_value),
                            literal_column(plan.value_column),
                        ),
                        else_=None,
                    )
                ).label(pivot_value)
                projections.append(case_expr)

            stmt = select(*projections).select_from(child_stmt.subquery())
            return stmt

        if isinstance(plan, GroupedPivot):
            # Grouped pivot operation - combines GROUP BY with pivot
            child_stmt = self._compile_plan(plan.child)
            from sqlalchemy import literal, literal_column

            # pivot_values should always be provided at this point (inferred in agg() if not provided)
            if not plan.pivot_values:
                raise CompilationError(
                    f"GROUPED_PIVOT without pivot_values. "
                    f"Plan structure: {type(plan).__name__} with pivot_column='{plan.pivot_column}', "
                    f"value_column='{plan.value_column}', agg_func='{plan.agg_func}'. "
                    f"pivot_values should be inferred in agg() if not provided. "
                    f"If you're creating a GroupedPivot manually, ensure pivot_values is set. "
                    f"Otherwise, this may indicate a bug in the pivot value inference logic."
                )

            # Convert child to subquery
            if isinstance(child_stmt, Select):
                child_subq = child_stmt.subquery()
            else:
                child_subq = child_stmt

            # Compile grouping columns
            group_by_cols = [self._expr.compile_expr(col) for col in plan.grouping]

            # Use CASE WHEN with aggregation for cross-dialect compatibility
            projections = list(group_by_cols)  # Start with grouping columns
            # Reuse agg_func_map pattern from Pivot (defined above)
            from typing import Callable as CallableType
            from sqlalchemy.sql import ColumnElement as ColumnElementType

            grouped_agg_func_map: dict[str, CallableType[..., ColumnElementType[Any]]] = {
                "sum": func.sum,
                "avg": func.avg,
                "count": func.count,
                "min": func.min,
                "max": func.max,
            }
            agg = grouped_agg_func_map.get(plan.agg_func.lower(), func.sum)
            assert agg is not None  # Always has default

            for pivot_value in plan.pivot_values:
                # Create aggregation with CASE WHEN
                # Reference columns from the child subquery using literal_column
                # SQLAlchemy will resolve these from the subquery context

                pivot_col_ref: ColumnElement[Any] = literal_column(plan.pivot_column)
                value_col_ref: ColumnElement[Any] = literal_column(plan.value_column)
                case_expr = agg(
                    sa_case(
                        (
                            pivot_col_ref == literal(pivot_value),
                            value_col_ref,
                        ),
                        else_=None,
                    )
                ).label(pivot_value)
                projections.append(case_expr)

            stmt = select(*projections).select_from(child_subq)
            if group_by_cols:
                stmt = stmt.group_by(*group_by_cols)
            return stmt

        if isinstance(plan, AntiJoin):
            # Anti-join: equivalent to LEFT JOIN with IS NULL filter
            # SELECT left.* FROM left LEFT JOIN right ON condition WHERE right.key IS NULL
            left_stmt = self._compile_plan(plan.left)
            right_stmt = self._compile_plan(plan.right)

            # Extract table names for aliasing
            from sqlalchemy import literal_column, null

            left_table_name = self._extract_table_name(plan.left)
            right_table_name = self._extract_table_name(plan.right)

            # Convert to subqueries
            if isinstance(left_stmt, Select):
                left_subq = (
                    left_stmt.subquery(name=left_table_name)
                    if left_table_name
                    else left_stmt.subquery()
                )
            else:
                left_subq = left_stmt
            if isinstance(right_stmt, Select):
                right_subq = (
                    right_stmt.subquery(name=right_table_name)
                    if right_table_name
                    else right_stmt.subquery()
                )
            else:
                right_subq = right_stmt

            # Build join condition
            if plan.on:
                conditions = []
                from sqlalchemy import column as sa_column, literal_column

                for left_col, right_col in plan.on:
                    # Use table-qualified column names in join condition to avoid ambiguity
                    if left_table_name:
                        try:
                            left_expr = left_subq.c[left_col]
                        except (KeyError, AttributeError, TypeError):
                            # Fallback to literal with table qualification
                            left_expr = literal_column(f'"{left_table_name}"."{left_col}"')
                    else:
                        left_expr = sa_column(left_col)
                    if right_table_name:
                        try:
                            right_expr = right_subq.c[right_col]
                        except (KeyError, AttributeError, TypeError):
                            # Fallback to literal with table qualification
                            right_expr = literal_column(f'"{right_table_name}"."{right_col}"')
                    else:
                        right_expr = sa_column(right_col)
                    conditions.append(left_expr == right_expr)
                join_condition = and_(*conditions) if len(conditions) > 1 else conditions[0]
            elif plan.condition is not None:
                join_condition = self._expr.compile_expr(plan.condition)
            else:
                raise CompilationError(
                    "AntiJoin requires either 'on' keys or a 'condition'",
                    suggestion=(
                        "Provide join conditions using either:\n"
                        "  - on=[('left_col', 'right_col')] for equality joins\n"
                        "  - condition=col('left_col') == col('right_col') for custom conditions"
                    ),
                )

            # Build LEFT JOIN with IS NULL filter (equivalent to anti-join)
            # We need to check that the right side's join key is NULL
            # Use the first right column from the join condition
            null_check_col: ColumnElement[Any]
            if plan.on:
                first_right_col = plan.on[0][1]
                if right_table_name:
                    try:
                        null_check_col = right_subq.c[first_right_col]
                    except (KeyError, AttributeError, TypeError):
                        # Fallback to literal with table qualification
                        from sqlalchemy import literal_column

                        null_check_col = literal_column(f'"{right_table_name}"."{first_right_col}"')
                else:
                    from sqlalchemy import column as sa_column

                    null_check_col = sa_column(first_right_col)
            else:
                # Fallback: use a column from right_subq if available
                try:
                    if hasattr(right_subq, "c"):
                        null_check_col = list(right_subq.c)[0]
                    else:
                        null_check_col = null()
                except (IndexError, AttributeError, TypeError):
                    null_check_col = null()

            # Select only columns from left table to avoid ambiguity
            if hasattr(left_subq, "c"):
                left_cols = [left_subq.c[col] for col in left_subq.c.keys()]
                stmt = (
                    select(*left_cols)
                    .select_from(left_subq.join(right_subq, join_condition, isouter=True))
                    .where(null_check_col.is_(null()))
                )
            else:
                # Fallback: use * but this may cause ambiguity
                stmt = (
                    select(literal_column("*"))
                    .select_from(left_subq.join(right_subq, join_condition, isouter=True))
                    .where(null_check_col.is_(null()))
                )
            return stmt

        if isinstance(plan, Explode):
            # Explode: expand array/JSON column into multiple rows
            # This requires table-valued functions which are dialect-specific
            child_stmt = self._compile_plan(plan.child)
            column_expr = self._expr.compile_expr(plan.column)

            # Create subquery from child
            child_subq = child_stmt.subquery()

            if self.dialect.name == "sqlite":
                # SQLite: Use json_each() table-valued function
                # json_each returns a table with 'key' and 'value' columns
                from sqlalchemy import func as sa_func

                # Create table-valued function with explicit column names
                json_each_func = sa_func.json_each(column_expr).table_valued("key", "value")
                json_each_alias = json_each_func.alias("json_each_result")

                # Select all columns from child plus the exploded value
                # Get all columns from child subquery
                child_columns = list(child_subq.c.values())
                exploded_value = json_each_alias.c.value.label(plan.alias)

                # Create CROSS JOIN by selecting from both tables
                result = select(*child_columns, exploded_value).select_from(
                    child_subq, json_each_alias
                )
                return result

            elif self.dialect.name == "postgresql":
                # PostgreSQL: Use jsonb_array_elements() for JSON arrays
                from sqlalchemy import func as sa_func

                # jsonb_array_elements returns a table with a single 'value' column
                json_elements_func = sa_func.jsonb_array_elements(column_expr).table_valued("value")
                json_elements_alias = json_elements_func.alias("json_elements_result")

                # Select all columns from child plus the exploded value
                child_columns = list(child_subq.c.values())
                exploded_value = json_elements_alias.c.value.label(plan.alias)

                # Create CROSS JOIN
                result = select(*child_columns, exploded_value).select_from(
                    child_subq, json_elements_alias
                )
                return result
            elif self.dialect.name == "duckdb":
                # DuckDB: Use unnest() for arrays
                from sqlalchemy import func as sa_func, literal_column

                # unnest returns a table with a single column
                # Use table_valued function
                unnest_func = sa_func.unnest(column_expr).table_valued()
                unnest_alias = unnest_func.alias("unnest_result")

                # Select all columns from child plus the exploded value
                child_columns = list(child_subq.c.values())
                # Get the column from the unnest result - it should have one column
                unnest_cols = list(unnest_alias.c.values())
                if unnest_cols:
                    exploded_value = unnest_cols[0].label(plan.alias)
                else:
                    # Fallback: use literal_column if no columns found
                    exploded_value = literal_column("unnest_result").label(plan.alias)

                # Create CROSS JOIN
                result = select(*child_columns, exploded_value).select_from(
                    child_subq, unnest_alias
                )
                return result

            else:
                # For other dialects, provide a helpful error message
                raise CompilationError(
                    f"explode() is not yet implemented for {self.dialect.name} dialect. "
                    "Currently supported dialects: sqlite, postgresql, duckdb. "
                    "This feature requires table-valued function support which is dialect-specific."
                )

        if isinstance(plan, LogicalUnion):
            left_stmt = self._compile_plan(plan.left)
            right_stmt = self._compile_plan(plan.right)

            from sqlalchemy import literal_column

            if plan.distinct:
                # UNION (distinct)
                stmt = select(literal_column("*")).select_from(
                    left_stmt.union(right_stmt).subquery()
                )
            else:
                # UNION ALL
                stmt = select(literal_column("*")).select_from(
                    left_stmt.union_all(right_stmt).subquery()
                )

            return stmt

        if isinstance(plan, Intersect):
            left_stmt = self._compile_plan(plan.left)
            right_stmt = self._compile_plan(plan.right)

            from sqlalchemy import literal_column

            if plan.distinct:
                # INTERSECT (distinct)
                stmt = select(literal_column("*")).select_from(
                    left_stmt.intersect(right_stmt).subquery()
                )
            else:
                # INTERSECT ALL
                stmt = select(literal_column("*")).select_from(
                    left_stmt.intersect_all(right_stmt).subquery()
                )

            return stmt

        if isinstance(plan, Except):
            left_stmt = self._compile_plan(plan.left)
            right_stmt = self._compile_plan(plan.right)

            from sqlalchemy import literal_column

            if plan.distinct:
                # EXCEPT (distinct)
                stmt = select(literal_column("*")).select_from(
                    left_stmt.except_(right_stmt).subquery()
                )
            else:
                # EXCEPT ALL
                stmt = select(literal_column("*")).select_from(
                    left_stmt.except_all(right_stmt).subquery()
                )

            return stmt

        if isinstance(plan, Distinct):
            child_stmt = self._compile_plan(plan.child)
            # Apply DISTINCT to the select statement
            return child_stmt.distinct()

        raise CompilationError(
            f"Unsupported logical plan node: {type(plan).__name__}. "
            f"Supported nodes: TableScan, Project, Filter, Limit, Sample, Sort, Aggregate, Join, SemiJoin, AntiJoin, Pivot, Explode, LogicalUnion, Intersect, Except, CTE, Distinct",
            context={
                "plan_type": type(plan).__name__,
                "plan_attributes": {
                    k: str(v)[:100]  # Limit string length
                    for k, v in plan.__dict__.items()
                    if not k.startswith("_") and len(str(v)) < 200
                },
                "dialect": self.dialect.name,
            },
        )

    def _apply_join_hints(
        self,
        stmt: Select[Any],
        target: "FromClause | None",
        hints: tuple[str, ...],
    ) -> Select[Any]:
        """Attach join hints to the compiled select statement."""
        if not hints:
            return stmt

        # MySQL recognizes USE/FORCE/IGNORE INDEX hints placed next to the table reference.
        if self.dialect.name == "mysql" and target is not None:
            for hint in hints:
                stmt = stmt.with_hint(target, hint, dialect_name="mysql")
            return stmt

        # Fallback: use statement-level hints (e.g., /*+ ... */) for dialects that support them.
        for hint in hints:
            stripped = hint.strip()
            formatted = stripped if stripped.startswith("/*") else f"/*+ {stripped} */"
            stmt = stmt.with_statement_hint(formatted, dialect_name=self.dialect.name)
        return stmt

    def _normalize_join_side(
        self,
        plan_side: LogicalPlan,
        compiled_stmt: Select[Any],
        table_name: Optional[str],
    ) -> tuple["FromClause", "FromClause | None"]:
        """Convert a compiled child into a joinable FROM clause and hint target."""
        from sqlalchemy import table as sa_table

        if isinstance(plan_side, TableScan):
            sa_tbl = sa_table(plan_side.table)
            if plan_side.alias:
                aliased_tbl = sa_tbl.alias(plan_side.alias)
                return aliased_tbl, aliased_tbl
            return sa_tbl, sa_tbl

        if isinstance(compiled_stmt, Select):
            subq = (
                compiled_stmt.subquery(name=table_name) if table_name else compiled_stmt.subquery()
            )
            return subq, None

        return compiled_stmt, None
