"""Compile logical plans into SQL using SQLAlchemy Core API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

from typing import cast as typing_cast

from sqlalchemy import select, table, func, case, null, and_, or_, not_, literal, cast as sqlalchemy_cast
from sqlalchemy.sql import Select, ColumnElement

from ..engine.dialects import DialectSpec, get_dialect
from ..expressions.column import Column
from ..logical.plan import (
    Aggregate,
    AntiJoin,
    CTE,
    Distinct,
    Except,
    Explode,
    Filter,
    Intersect,
    Join,
    Limit,
    LogicalPlan,
    Pivot,
    Project,
    RecursiveCTE,
    Sample,
    SemiJoin,
    Sort,
    SortOrder,
    TableScan,
    Union as LogicalUnion,
    WindowSpec,
)
from ..utils.exceptions import CompilationError, ValidationError
from .builders import quote_identifier


def compile_plan(plan: LogicalPlan, dialect: Union[str, DialectSpec] = "ansi") -> Select:
    """Compile a logical plan to a SQLAlchemy Select statement.

    Args:
        plan: Logical plan to compile
        dialect: SQL dialect specification

    Returns:
        SQLAlchemy Select statement
    """
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
        if isinstance(plan, CTE):
            # Compile the child plan and convert it to a CTE
            child_stmt = self._compile_plan(plan.child)
            cte_obj = child_stmt.cte(plan.name)
            # Return a select from the CTE
            from sqlalchemy import literal_column

            return select(literal_column("*")).select_from(cte_obj)

        if isinstance(plan, RecursiveCTE):
            # WITH RECURSIVE: compile initial and recursive parts
            initial_stmt = self._compile_plan(plan.initial)
            recursive_stmt = self._compile_plan(plan.recursive)

            # Create recursive CTE using SQLAlchemy's recursive CTE support
            # SQLAlchemy uses .cte(recursive=True) for recursive CTEs
            from sqlalchemy import literal_column

            # For recursive CTEs, we need to combine initial and recursive with UNION/UNION ALL
            if plan.union_all:
                combined = initial_stmt.union_all(recursive_stmt)
            else:
                combined = initial_stmt.union(recursive_stmt)

            # Create recursive CTE
            recursive_cte_obj = combined.cte(name=plan.name, recursive=True)

            # Return a select from the recursive CTE
            return select(literal_column("*")).select_from(recursive_cte_obj)

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
                columns = [self._expr.compile_expr(col) for col in plan.projections]
            finally:
                self._expr._current_subq = old_subq
                self._expr._join_info = old_join_info
            # Create new select with these columns
            stmt = select(*columns).select_from(child_subq)
            return stmt

        if isinstance(plan, Filter):
            child_stmt = self._compile_plan(plan.child)
            predicate = self._expr.compile_expr(plan.predicate)
            return child_stmt.where(predicate)

        if isinstance(plan, Limit):
            child_stmt = self._compile_plan(plan.child)
            return child_stmt.limit(plan.count)

        if isinstance(plan, Sample):
            child_stmt = self._compile_plan(plan.child)
            # Use ORDER BY RANDOM() with LIMIT based on fraction
            # This works across most SQL dialects (SQLite, PostgreSQL, MySQL)
            from sqlalchemy import func as sa_func

            # Calculate approximate limit - use a reasonable default
            # In practice, for exact fraction sampling, we'd need to know row count first
            # For now, use a large limit as approximation
            limit_count = max(1, int(plan.fraction * 10000))  # Reasonable default for large tables
            result = child_stmt.order_by(sa_func.random())
            if plan.seed is not None:
                # Most databases don't support seed in RANDOM(), but we accept it for API consistency
                # PostgreSQL supports setseed() but it's not easily composable here
                pass
            return result.limit(limit_count)

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

            # Convert to subqueries if they're Select statements
            if isinstance(left_stmt, Select):
                # Use table name as alias if available, otherwise let SQLAlchemy generate one
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
                    # Reference columns from the aliased subqueries
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
            elif plan.condition:
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
            elif plan.condition:
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
            from sqlalchemy import literal_column

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
                    case(
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
            elif plan.condition:
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
            # This is a complex operation that requires table-valued functions
            # For now, we'll raise a CompilationError indicating this needs dialect-specific support
            # TODO: Implement proper table-valued function support for explode()
            raise CompilationError(
                f"explode() is not yet fully implemented for {self.dialect.name} dialect. "
                "This feature requires table-valued function support which is being developed."
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
            f"Supported nodes: TableScan, Project, Filter, Limit, Sample, Sort, Aggregate, Join, SemiJoin, AntiJoin, Pivot, Explode, LogicalUnion, Intersect, Except, CTE, Distinct"
        )


class ExpressionCompiler:
    """Compile expression trees into SQLAlchemy column expressions."""

    def __init__(self, dialect: DialectSpec, plan_compiler: Optional["SQLCompiler"] = None):
        self.dialect = dialect
        self._table_cache: dict[str, Any] = {}
        self._current_subq: Any = None
        self._join_info: Optional[dict[str, str]] = None
        self._plan_compiler = plan_compiler

    def compile_expr(self, expression: Column) -> ColumnElement:
        """Compile a Column expression to a SQLAlchemy column expression."""
        return self._compile(expression)

    def emit(self, expression: Column) -> str:
        """Compile a Column expression to a SQL string.

        Args:
            expression: Column expression to compile

        Returns:
            SQL string representation of the expression
        """
        compiled = self.compile_expr(expression)
        # Convert SQLAlchemy column element to SQL string
        return str(compiled.compile(compile_kwargs={"literal_binds": True}))

    def _compile(self, expression: Column) -> ColumnElement:
        """Compile a Column expression to a SQLAlchemy column expression."""
        op = expression.op

        if op == "column":
            col_name = expression.args[0]
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
            from sqlalchemy import column as sa_column

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
                        from sqlalchemy import literal_column

                        quote = self.dialect.quote_char
                        sa_col = typing_cast(
                            ColumnElement[Any],
                            literal_column(f"{quote}{table_name}{quote}.{quote}{col_name}{quote}"),
                        )
                else:
                    # No subquery context, use qualified literal column
                    from sqlalchemy import literal_column

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

        if op == "add":
            left, right = expression.args
            result = self._compile(left) + self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "sub":
            left, right = expression.args
            result = self._compile(left) - self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "mul":
            left, right = expression.args
            result = self._compile(left) * self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "div":
            left, right = expression.args
            result = self._compile(left) / self._compile(right)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "eq":
            left, right = expression.args
            return self._compile(left) == self._compile(right)
        if op == "ne":
            left, right = expression.args
            return self._compile(left) != self._compile(right)
        if op == "lt":
            left, right = expression.args
            return self._compile(left) < self._compile(right)
        if op == "le":
            left, right = expression.args
            return self._compile(left) <= self._compile(right)
        if op == "gt":
            left, right = expression.args
            return self._compile(left) > self._compile(right)
        if op == "ge":
            left, right = expression.args
            return self._compile(left) >= self._compile(right)

        if op == "floor_div":
            left, right = expression.args
            return func.floor(self._compile(left) / self._compile(right))
        if op == "round":
            col_expr = self._compile(expression.args[0])
            scale = expression.args[1] if len(expression.args) > 1 else 0
            result = func.round(col_expr, scale)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "floor":
            result = func.floor(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "ceil":
            result = func.ceil(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "abs":
            result = func.abs(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "sqrt":
            result = func.sqrt(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "exp":
            result = func.exp(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "log":
            result = func.ln(self._compile(expression.args[0]))  # Natural log
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "log10":
            result = func.log(10, self._compile(expression.args[0]))  # Base-10 log
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "sin":
            result = func.sin(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "cos":
            result = func.cos(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "tan":
            result = func.tan(self._compile(expression.args[0]))
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
            from sqlalchemy import literal_column

            # Parse interval string (format: "N UNIT" where N is number and UNIT is DAY, MONTH, YEAR, HOUR, etc.)
            parts = interval_str.split()
            if len(parts) != 2:
                raise CompilationError(
                    f"Invalid interval format: {interval_str}. Expected format: 'N UNIT' (e.g., '1 DAY')"
                )
            num, unit = parts
            unit_upper = unit.upper()

            # For PostgreSQL, use INTERVAL literal
            if self.dialect.name == "postgresql":
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
            from sqlalchemy import literal_column

            # Parse interval string
            parts = interval_str.split()
            if len(parts) != 2:
                raise CompilationError(
                    f"Invalid interval format: {interval_str}. Expected format: 'N UNIT' (e.g., '1 DAY')"
                )
            num, unit = parts
            unit_upper = unit.upper()

            # For PostgreSQL, use INTERVAL literal
            if self.dialect.name == "postgresql":
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
            # Use SQLAlchemy's interval handling
            # Try to use make_interval if available (PostgreSQL), otherwise use date_add
            try:
                interval_months: ColumnElement[Any] = func.make_interval(months=abs(num_months))
                if num_months >= 0:
                    result = col_expr + interval_months
                else:
                    result = col_expr - interval_months
            except Exception:
                # Fallback: use date_add function (MySQL/SQLite compatible)
                # This is a simplified fallback
                result = func.date_add(col_expr, literal(num_months))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "mod":
            left, right = expression.args
            return func.mod(self._compile(left), self._compile(right))
        if op == "pow":
            base, exp = expression.args[:2]
            return func.power(self._compile(base), self._compile(exp))
        if op == "neg":
            return -self._compile(expression.args[0])
        if op == "and":
            left, right = expression.args
            return and_(self._compile(left), self._compile(right))
        if op == "or":
            left, right = expression.args
            return or_(self._compile(left), self._compile(right))
        if op == "not":
            return not_(self._compile(expression.args[0]))
        if op == "between":
            value, lower, upper = expression.args
            return self._compile(value).between(self._compile(lower), self._compile(upper))
        if op == "in":
            value, options = expression.args
            option_values = [self._compile(opt) for opt in options]
            return self._compile(value).in_(option_values)
        if op == "in_subquery":
            # IN with subquery: col("id").isin(df.select("id"))
            value, subquery_plan = expression.args
            if self._plan_compiler is None:
                raise CompilationError("Subquery compilation requires plan compiler")
            subquery_stmt = self._plan_compiler._compile_plan(subquery_plan)
            return self._compile(value).in_(subquery_stmt)
        if op == "scalar_subquery":
            # Scalar subquery in SELECT: scalar_subquery(df.select())
            (subquery_plan,) = expression.args
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
            return self._compile(left).like(self._compile(pattern))
        if op == "ilike":
            left, pattern = expression.args
            return self._compile(left).ilike(self._compile(pattern))
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
        if op == "cast":
            column = expression.args[0]
            type_name = expression.args[1]
            precision = expression.args[2] if len(expression.args) > 2 else None
            scale = expression.args[3] if len(expression.args) > 3 else None

            from sqlalchemy import types as sa_types

            # Map type names to SQLAlchemy types
            type_name_upper = type_name.upper()
            # Type can be either a TypeEngine instance or a TypeEngine class
            sa_type: Any
            if type_name_upper == "DECIMAL" or type_name_upper == "NUMERIC":
                if precision is not None and scale is not None:
                    sa_type = sa_types.Numeric(precision=precision, scale=scale)
                elif precision is not None:
                    sa_type = sa_types.Numeric(precision=precision)
                else:
                    sa_type = sa_types.Numeric()
            elif type_name_upper == "TIMESTAMP":
                sa_type = sa_types.TIMESTAMP
            elif type_name_upper == "DATE":
                sa_type = sa_types.DATE
            elif type_name_upper == "TIME":
                sa_type = sa_types.TIME
            elif type_name_upper == "INTERVAL":
                sa_type = sa_types.Interval
            elif type_name_upper == "UUID":
                # Handle UUID type with dialect-specific implementations
                if self.dialect.name == "postgresql":
                    sa_type = sa_types.UUID()
                elif self.dialect.name == "mysql":
                    sa_type = sa_types.CHAR(36)
                else:
                    # SQLite and others: use String
                    sa_type = sa_types.String()
            elif type_name_upper == "JSON" or type_name_upper == "JSONB":
                # Handle JSON/JSONB type with dialect-specific implementations
                if self.dialect.name == "postgresql":
                    sa_type = sa_types.JSON()
                    # Note: SQLAlchemy doesn't distinguish JSONB from JSON in type system
                    # The actual SQL will use JSONB if specified in DDL
                elif self.dialect.name == "mysql":
                    sa_type = sa_types.JSON()
                else:
                    # SQLite and others: use String
                    sa_type = sa_types.String()
            elif type_name_upper == "INTEGER" or type_name_upper == "INT":
                sa_type = sa_types.Integer
            elif type_name_upper == "TEXT":
                sa_type = sa_types.Text
            elif (
                type_name_upper == "REAL"
                or type_name_upper == "FLOAT"
                or type_name_upper == "DOUBLE"
            ):
                sa_type = sa_types.Float
            elif type_name_upper == "VARCHAR" or type_name_upper == "STRING":
                if precision is not None:
                    sa_type = sa_types.String(length=precision)
                else:
                    sa_type = sa_types.String
            elif type_name_upper == "CHAR":
                if precision is not None:
                    sa_type = sa_types.CHAR(length=precision)
                else:
                    sa_type = sa_types.CHAR
            elif type_name_upper == "BOOLEAN" or type_name_upper == "BOOL":
                sa_type = sa_types.Boolean
            else:
                # Fallback to String for unknown types
                sa_type = sa_types.String

            result = sqlalchemy_cast(self._compile(column), sa_type)
            if expression._alias:
                result = result.label(expression._alias)
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
        if op == "case_when":
            # CASE WHEN expression: args[0] is tuple of (condition, value) pairs, args[1] is else value
            conditions = expression.args[0]
            else_value = self._compile(expression.args[1])

            # Build CASE statement
            # Start with empty when clauses, add them one by one
            when_clauses: list[tuple[ColumnElement[Any], Any]] = []
            for condition, value in conditions:
                when_clauses.append((self._compile(condition), self._compile(value)))
            case_stmt = case(*when_clauses, else_=else_value)

            result = case_stmt
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "coalesce":
            args = [self._compile(arg) for arg in expression.args]
            result = func.coalesce(*args)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "concat":
            args = [self._compile(arg) for arg in expression.args]
            result = func.concat(*args)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "upper":
            result = func.upper(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "lower":
            result = func.lower(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "substring":
            col_expr = self._compile(expression.args[0])
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
            result = func.trim(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "ltrim":
            result = func.ltrim(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "rtrim":
            result = func.rtrim(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "regexp_extract":
            # SQLAlchemy doesn't have a direct regexp_extract, use dialect-specific function
            col_expr = self._compile(expression.args[0])
            pattern = expression.args[1]
            group_idx = expression.args[2] if len(expression.args) > 2 else 0
            # Use func for dialect-specific regex functions
            # PostgreSQL uses regexp_match, SQLite uses regexp, etc.
            result = func.regexp_extract(col_expr, pattern, group_idx)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "regexp_replace":
            col_expr = self._compile(expression.args[0])
            pattern = expression.args[1]
            replacement = expression.args[2]
            result = func.regexp_replace(col_expr, pattern, replacement)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "split":
            # SQLAlchemy doesn't have split, use string_to_array or similar
            col_expr = self._compile(expression.args[0])
            delimiter = expression.args[1]
            result = func.string_to_array(col_expr, delimiter)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "replace":
            col_expr = self._compile(expression.args[0])
            search = expression.args[1]
            replacement = expression.args[2]
            result = func.replace(col_expr, search, replacement)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "length":
            result = func.length(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "lpad":
            col_expr = self._compile(expression.args[0])
            length = expression.args[1]
            pad = expression.args[2] if len(expression.args) > 2 else " "
            result = func.lpad(col_expr, length, pad)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "rpad":
            col_expr = self._compile(expression.args[0])
            length = expression.args[1]
            pad = expression.args[2] if len(expression.args) > 2 else " "
            result = func.rpad(col_expr, length, pad)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "greatest":
            args = [self._compile(arg) for arg in expression.args]
            result = func.greatest(*args)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "least":
            args = [self._compile(arg) for arg in expression.args]
            result = func.least(*args)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_sum":
            result = func.sum(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_avg":
            result = func.avg(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_min":
            result = func.min(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_max":
            result = func.max(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_count":
            result = func.count(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_stddev":
            # Use stddev_pop or stddev_samp - SQLAlchemy uses stddev_samp by default
            result = func.stddev(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_variance":
            # Use var_pop or var_samp - SQLAlchemy uses var_samp by default
            result = func.variance(self._compile(expression.args[0]))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_corr":
            # Correlation between two columns
            col1, col2 = expression.args
            result = func.corr(self._compile(col1), self._compile(col2))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_covar":
            # Covariance between two columns
            col1, col2 = expression.args
            result = func.covar_pop(self._compile(col1), self._compile(col2))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_count_star":
            result = func.count()
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_count_distinct":
            args = [self._compile(arg) for arg in expression.args]
            result = func.count(func.distinct(*args))
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_collect_list":
            # Collect values into an array
            # PostgreSQL: array_agg(column)
            # SQLite: json_group_array(column) - JSON1 extension
            # MySQL: JSON_ARRAYAGG(column)
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.array_agg(col_expr)
            elif self.dialect.name == "sqlite":
                result = func.json_group_array(col_expr)
            else:
                # MySQL and others - use JSON_ARRAYAGG
                result = func.json_arrayagg(col_expr)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "agg_collect_set":
            # Collect distinct values into an array
            # PostgreSQL: array_agg(DISTINCT column)
            # SQLite: json_group_array(DISTINCT column) - JSON1 extension
            # MySQL: JSON_ARRAYAGG(DISTINCT column)
            col_expr = self._compile(expression.args[0])
            if self.dialect.name == "postgresql":
                result = func.array_agg(func.distinct(col_expr))
            elif self.dialect.name == "sqlite":
                # SQLite doesn't support DISTINCT in json_group_array directly
                # We'll use a workaround or note the limitation
                result = func.json_group_array(func.distinct(col_expr))
            else:
                # MySQL: JSON_ARRAYAGG doesn't support DISTINCT directly
                # Use a subquery with DISTINCT first, then aggregate
                # For now, use GROUP_CONCAT(DISTINCT ...) wrapped in JSON_ARRAY
                # Actually, MySQL 8.0+ supports JSON_ARRAYAGG but not with DISTINCT
                # We'll use a workaround: select distinct values in a subquery
                # For simplicity, use GROUP_CONCAT(DISTINCT) and wrap in JSON_ARRAY
                from sqlalchemy import literal_column

                # Use CAST(GROUP_CONCAT(DISTINCT ...) AS JSON) as a workaround
                # Or use JSON_ARRAYAGG on a subquery with DISTINCT
                # For now, just use json_arrayagg without distinct (limitation)
                result = func.json_arrayagg(col_expr)
            if expression._alias:
                result = result.label(expression._alias)
            return result

        if op == "agg_percentile_cont":
            # Continuous percentile (interpolated)
            # PostgreSQL: percentile_cont(fraction) WITHIN GROUP (ORDER BY column)
            # SQL Server: PERCENTILE_CONT(fraction) WITHIN GROUP (ORDER BY column)
            # Oracle: PERCENTILE_CONT(fraction) WITHIN GROUP (ORDER BY column)
            # SQLite/MySQL: Not natively supported
            col_expr = self._compile(expression.args[0])
            fraction = expression.args[1]
            if self.dialect.name in ("postgresql", "mssql", "oracle"):
                # Use percentile_cont with WITHIN GROUP
                # SQLAlchemy's within_group is a method on the function
                result = func.percentile_cont(fraction).within_group(col_expr)
            else:
                # For unsupported dialects, raise an error or use a workaround
                # For now, we'll raise an error to indicate lack of support
                raise CompilationError(
                    f"percentile_cont() is not supported for {self.dialect.name} dialect. "
                    "Supported dialects: PostgreSQL, SQL Server, Oracle"
                )
            if expression._alias:
                result = result.label(expression._alias)
            return result

        if op == "agg_percentile_disc":
            # Discrete percentile (actual value)
            # PostgreSQL: percentile_disc(fraction) WITHIN GROUP (ORDER BY column)
            # SQL Server: PERCENTILE_DISC(fraction) WITHIN GROUP (ORDER BY column)
            # Oracle: PERCENTILE_DISC(fraction) WITHIN GROUP (ORDER BY column)
            # SQLite/MySQL: Not natively supported
            col_expr = self._compile(expression.args[0])
            fraction = expression.args[1]
            if self.dialect.name in ("postgresql", "mssql", "oracle"):
                # Use percentile_disc with WITHIN GROUP
                # SQLAlchemy's within_group is a method on the function
                result = func.percentile_disc(fraction).within_group(col_expr)
            else:
                # For unsupported dialects, raise an error
                raise CompilationError(
                    f"percentile_disc() is not supported for {self.dialect.name} dialect. "
                    "Supported dialects: PostgreSQL, SQL Server, Oracle"
                )
            if expression._alias:
                result = result.label(expression._alias)
            return result

        # JSON functions
        if op == "json_extract":
            col_expr = self._compile(expression.args[0])
            path = expression.args[1]
            # Use dialect-specific JSON extraction
            # PostgreSQL: -> operator or json_extract_path_text
            # SQLite: json_extract (JSON1 extension)
            # MySQL: JSON_EXTRACT or -> operator
            # Generic: Use func.json_extract which SQLAlchemy may handle
            if self.dialect.name == "postgresql":
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
                    result = func.json_extract_path_text(
                        col_expr, *[literal(p) for p in path_parts]
                    )
            elif self.dialect.name == "sqlite":
                # SQLite JSON1 extension
                result = func.json_extract(col_expr, path)
            else:
                # Generic fallback - try JSON_EXTRACT
                result = func.json_extract(col_expr, path)
            if expression._alias:
                result = result.label(expression._alias)
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
            # SQLite: json_array_length(json_array)
            # MySQL: JSON_LENGTH(json_array)
            if self.dialect.name == "postgresql":
                result = func.array_length(col_expr, 1)
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
            if self.dialect.name == "postgresql":
                # PostgreSQL uses = ANY(array)
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
                except Exception:
                    # Fallback for SQLite - check if removing the value changes length
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
            if self.dialect.name == "postgresql":
                result = func.array_position(col_expr, value_expr)
            elif self.dialect.name == "sqlite":
                # SQLite: Use json_each to find position
                # This is a simplified version - full implementation would use a subquery
                # For now, we'll use a workaround that may not be perfect
                from sqlalchemy import literal_column

                # Note: This is a limitation - full implementation requires subquery
                # For SQLite, array_position with JSON arrays is complex
                result = literal_column("NULL")  # Placeholder - requires subquery with json_each
            elif self.dialect.name == "mysql":
                # MySQL: JSON_SEARCH returns a path like "$[0]", need to extract index
                # This is complex - for now, return NULL as array_position is not fully supported
                # A full implementation would need to parse the JSON path string "$[0]" -> 0
                from sqlalchemy import literal_column

                # TODO: Implement proper index extraction from JSON path
                # For now, return NULL to indicate limitation
                result = literal_column("NULL")
            else:
                # Generic fallback - use JSON_SEARCH
                result = func.json_search(col_expr, literal("one"), value_expr)
            if expression._alias:
                result = result.label(expression._alias)
            return result

        # Window-specific functions
        if op == "window_row_number":
            result = func.row_number()
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "window_rank":
            result = func.rank()
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "window_dense_rank":
            result = func.dense_rank()
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "window_lag":
            column = self._compile(expression.args[0])
            offset = expression.args[1] if len(expression.args) > 1 else 1
            if len(expression.args) > 2:
                default = self._compile(expression.args[2])
                result = func.lag(column, offset, default)
            else:
                result = func.lag(column, offset)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "window_lead":
            column = self._compile(expression.args[0])
            offset = expression.args[1] if len(expression.args) > 1 else 1
            if len(expression.args) > 2:
                default = self._compile(expression.args[2])
                result = func.lead(column, offset, default)
            else:
                result = func.lead(column, offset)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "window_percent_rank":
            result = func.percent_rank()
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "window_cume_dist":
            result = func.cume_dist()
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "window_nth_value":
            column = self._compile(expression.args[0])
            n = expression.args[1]
            result = func.nth_value(column, n)
            if expression._alias:
                result = result.label(expression._alias)
            return result
        if op == "window_ntile":
            n = expression.args[0]
            result = func.ntile(n)
            if expression._alias:
                result = result.label(expression._alias)
            return result

        if op == "window":
            # Window function: args[0] is the function, args[1] is WindowSpec
            func_expr = self._compile(expression.args[0])
            window_spec: WindowSpec = expression.args[1]

            # Build SQLAlchemy window using .over() method on the function

            # Create partition by clauses
            partition_by = None
            if window_spec.partition_by:
                partition_by = [self._compile(col) for col in window_spec.partition_by]

            # Create order by clauses
            order_by: Optional[list[ColumnElement[Any]]] = None
            if window_spec.order_by:
                order_by = []
                for col_expr in window_spec.order_by:  # type: ignore[assignment]
                    # col_expr is a Column from window_spec.order_by: tuple[Column, ...]
                    # _compile returns ColumnElement, but mypy may infer Column due to type complexity
                    sa_order_col = self._compile(col_expr)  # type: ignore[arg-type]
                    # Check if it has desc/asc already applied
                    if isinstance(col_expr, Column) and col_expr.op == "sort_desc":
                        sa_order_col = sa_order_col.desc()
                    elif isinstance(col_expr, Column) and col_expr.op == "sort_asc":
                        sa_order_col = sa_order_col.asc()
                    order_by.append(sa_order_col)

            # Build window using .over() method
            if partition_by and order_by:
                result = func_expr.over(partition_by=partition_by, order_by=order_by)
            elif partition_by:
                result = func_expr.over(partition_by=partition_by)
            elif order_by:
                result = func_expr.over(order_by=order_by)
            else:
                result = func_expr.over()

            # Handle ROWS BETWEEN or RANGE BETWEEN
            # Note: SQLAlchemy's window API is complex for BETWEEN clauses
            # We'll use a simpler approach for now
            if window_spec.rows_between or window_spec.range_between:
                # For now, we'll compile this as a text expression
                # A more complete implementation would use SQLAlchemy's window range API
                pass  # TODO: Implement ROWS/RANGE BETWEEN properly

            if expression._alias:
                result = result.label(expression._alias)
            return result  # type: ignore[no-any-return]

        raise CompilationError(
            f"Unsupported expression operation '{op}'. "
            "This may indicate a missing function implementation or an invalid expression."
        )

    def _get_table(self, table_name: str) -> Any:
        """Get or create a SQLAlchemy table object for the given table name."""
        if table_name not in self._table_cache:
            self._table_cache[table_name] = table(table_name)
        return self._table_cache[table_name]
