"""Shared lazy operation builders for DataFrame and AsyncDataFrame.

This module contains shared logic for building logical plans from DataFrame operations.
These operations are lazy (don't execute queries) and can be shared between sync and async
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, TypeVar, Union, cast

from ...expressions.column import Column
from ...logical import operators
from ...logical.plan import LogicalPlan

if TYPE_CHECKING:
    from ..helpers.dataframe_helpers import DataFrameHelpersProtocol

    DataFrameProtocol = DataFrameHelpersProtocol
else:
    DataFrameProtocol = Any

# Type variable for generic DataFrame operations
T = TypeVar("T", bound=Any)


@dataclass
class SelectOperationResult:
    """Result from building a select operation."""

    plan: LogicalPlan
    """The resulting logical plan after the select operation."""

    should_apply: bool
    """Whether the plan should be applied (False if no-op like empty select or "*")."""


def build_select_operation(
    self: Any,  # DataFrameProtocol - using Any for mypy compatibility
    columns: Sequence[Union[Column, str]],
) -> SelectOperationResult:
    """Build a select operation logical plan.

    This handles column selection, "*" processing, and explode logic that is
    shared between DataFrame and AsyncDataFrame.

    Args:
        self: DataFrame instance (must have plan attribute and _normalize_projection method)
        columns: Columns to select

    Returns:
        SelectOperationResult with the new plan and whether it should be applied
    """
    if not columns:
        return SelectOperationResult(plan=self.plan, should_apply=False)

    # Handle "*" as special case
    if len(columns) == 1 and isinstance(columns[0], str) and columns[0] == "*":
        return SelectOperationResult(plan=self.plan, should_apply=False)

    # Check if "*" is in the columns (only check string elements, not Column objects)
    has_star = any(isinstance(col, str) and col == "*" for col in columns)

    # Normalize all columns first and check for explode
    normalized_columns: List[Column] = []
    explode_column: Optional[Column] = None

    for col_expr in columns:
        if isinstance(col_expr, str) and col_expr == "*":
            # Handle "*" separately - add star column
            star_col = Column(op="star", args=(), _alias=None)
            normalized_columns.append(star_col)
            continue

        normalized = self._normalize_projection(col_expr)

        # Check if this is an explode() column
        if isinstance(normalized, Column) and normalized.op == "explode":
            if explode_column is not None:
                raise ValueError(
                    "Multiple explode() columns are not supported. "
                    "Only one explode() can be used per select() operation."
                )
            explode_column = normalized
        else:
            normalized_columns.append(normalized)

    # If we have an explode column, we need to handle it specially
    if explode_column is not None:
        # Extract the column being exploded and the alias
        exploded_column = explode_column.args[0] if explode_column.args else None
        if not isinstance(exploded_column, Column):
            raise ValueError("explode() requires a Column expression")

        alias = explode_column._alias or "value"

        # Create Explode logical plan
        exploded_plan = operators.explode(self.plan, exploded_column, alias=alias)

        # Create Project on top of Explode
        # If we have "*", we want all columns from the exploded result
        # Otherwise, we want the exploded column (with alias) plus any other specified columns
        project_columns: List[Column] = []

        if has_star:
            # Select all columns from exploded result (including the exploded column)
            star_col = Column(op="star", args=(), _alias=None)
            project_columns.append(star_col)
            # Also add any other explicitly specified columns
            for col in normalized_columns:
                if col.op != "star":
                    project_columns.append(col)
        else:
            # Add the exploded column with its alias first
            exploded_result_col = Column(op="column", args=(alias,), _alias=None)
            project_columns.append(exploded_result_col)
            # Add any other columns
            project_columns.extend(normalized_columns)

        return SelectOperationResult(
            plan=operators.project(exploded_plan, tuple(project_columns)),
            should_apply=True,
        )

    # No explode columns, normal projection
    if has_star and not normalized_columns:
        return SelectOperationResult(
            plan=self.plan, should_apply=False
        )  # Only "*", same as empty select

    return SelectOperationResult(
        plan=operators.project(self.plan, tuple(normalized_columns)),
        should_apply=True,
    )


def build_where_operation(
    self: Any,  # DataFrameProtocol - using Any for mypy compatibility
    predicate: Union[Column, str],
) -> LogicalPlan:
    """Build a where/filter operation logical plan.

    Args:
        self: DataFrame instance (must have plan attribute)
        predicate: Filter condition as Column or SQL string

    Returns:
        New logical plan with filter applied
    """
    # If predicate is a string, parse it into a Column expression
    if isinstance(predicate, str):
        from ...expressions.sql_parser import parse_sql_expr

        # Get available column names from the DataFrame for context
        available_columns: Optional[set[str]] = None
        try:
            # Try to extract column names from the current plan
            if hasattr(self.plan, "projections"):
                available_columns = set()
                for proj in self.plan.projections:
                    if isinstance(proj, Column) and proj.op == "column" and proj.args:
                        available_columns.add(str(proj.args[0]))
        except Exception:
            # If we can't extract columns, that's okay - parser will still work
            pass

        predicate = parse_sql_expr(predicate, available_columns)

    return operators.filter(self.plan, predicate)


def build_limit_operation(
    plan: LogicalPlan,
    count: int,
) -> LogicalPlan:
    """Build a limit operation logical plan.

    Args:
        plan: Current logical plan
        count: Maximum number of rows to return

    Returns:
        New logical plan with limit applied

    Raises:
        ValueError: If count is negative
    """
    if count < 0:
        raise ValueError("limit() count must be non-negative")
    return operators.limit(plan, count)


def build_order_by_operation(
    self: Any,  # DataFrameProtocol - using Any for mypy compatibility
    columns: Sequence[Union[Column, str]],
) -> LogicalPlan:
    """Build an order_by operation logical plan.

    Args:
        self: DataFrame instance (must have plan attribute and _normalize_sort_expression method)
        columns: Columns to sort by

    Returns:
        New logical plan with sort applied
    """
    if not columns:
        return cast("LogicalPlan", self.plan)
    orders = tuple(
        self._normalize_sort_expression(self._normalize_projection(col)) for col in columns
    )
    return operators.order_by(self.plan, orders)


def build_sample_operation(
    plan: LogicalPlan,
    fraction: float,
    seed: Optional[int],
) -> LogicalPlan:
    """Build a sample operation logical plan.

    Args:
        plan: Current logical plan
        fraction: Fraction of rows to sample (0.0 to 1.0)
        seed: Optional random seed

    Returns:
        New logical plan with sample applied

    Raises:
        ValueError: If fraction is not between 0 and 1
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("sample() fraction must be between 0.0 and 1.0")
    return operators.sample(plan, fraction, seed)


def join_dataframes(
    left: T,
    right: T,
    *,
    on: Optional[
        Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
    ] = None,
    how: str = "inner",
    lateral: bool = False,
    hints: Optional[Sequence[str]] = None,
) -> T:
    """Join two DataFrames together.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Join condition
        how: Join type ("inner", "left", "right", "full", "cross")
        lateral: Whether to create a LATERAL join
        hints: Optional join hints

    Returns:
        New DataFrame with the join result

    Raises:
        RuntimeError: If DataFrames are not bound to the same database
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before joining")
    if left.database is not right.database:
        raise ValueError("Cannot join DataFrames from different Database instances")

    # Cross joins don't require an 'on' clause
    if how.lower() == "cross":
        normalized_on = None
        condition = None
    else:
        normalized_condition = left._normalize_join_condition(on)
        if isinstance(normalized_condition, Column):
            # PySpark-style Column expression
            normalized_on = None
            condition = normalized_condition
        else:
            # Tuple-based join (backward compatible)
            normalized_on = normalized_condition
            condition = None

    plan = operators.join(
        left.plan,
        right.plan,
        how=how.lower(),
        on=normalized_on,
        condition=condition,
        lateral=lateral,
    )
    return cast(T, left._with_plan(plan))


def union_dataframes(
    left: T,
    right: T,
    *,
    distinct: bool = True,
) -> T:
    """Union two DataFrames together.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        distinct: Whether to return distinct rows only

    Returns:
        New DataFrame with the union result

    Raises:
        RuntimeError: If DataFrames are not bound to the same database
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before union")
    if left.database is not right.database:
        raise ValueError("Cannot union DataFrames from different Database instances")

    plan = operators.union(left.plan, right.plan, distinct=distinct)
    return cast(T, left._with_plan(plan))


def intersect_dataframes(
    left: T,
    right: T,
    *,
    distinct: bool = True,
) -> T:
    """Intersect two DataFrames together.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        distinct: Whether to return distinct rows only

    Returns:
        New DataFrame with the intersection result

    Raises:
        RuntimeError: If DataFrames are not bound to the same database
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before intersect")
    if left.database is not right.database:
        raise ValueError("Cannot intersect DataFrames from different Database instances")

    plan = operators.intersect(left.plan, right.plan, distinct=distinct)
    return cast(T, left._with_plan(plan))


def except_dataframes(
    left: T,
    right: T,
    *,
    distinct: bool = True,
) -> T:
    """Return rows in left DataFrame that are not in right DataFrame.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        distinct: Whether to return distinct rows only

    Returns:
        New DataFrame with rows in left but not in right

    Raises:
        RuntimeError: If DataFrames are not bound to the same database
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before except")
    if left.database is not right.database:
        raise ValueError("Cannot except DataFrames from different Database instances")

    plan = operators.except_(left.plan, right.plan, distinct=distinct)
    return cast(T, left._with_plan(plan))


def semi_join_dataframes(
    left: T,
    right: T,
    *,
    on: Optional[
        Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
    ] = None,
) -> T:
    """Perform a semi-join: return rows from left where a matching row exists in right.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Join condition

    Returns:
        New DataFrame with rows from left that have matches in right

    Raises:
        RuntimeError: If DataFrames are not bound to the same database
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before semi_join")
    if left.database is not right.database:
        raise ValueError("Cannot semi_join DataFrames from different Database instances")

    normalized_condition = left._normalize_join_condition(on)
    if isinstance(normalized_condition, Column):
        plan = operators.semi_join(left.plan, right.plan, condition=normalized_condition, on=None)
    else:
        plan = operators.semi_join(left.plan, right.plan, on=normalized_condition, condition=None)
    return cast(T, left._with_plan(plan))


def anti_join_dataframes(
    left: T,
    right: T,
    *,
    on: Optional[
        Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
    ] = None,
) -> T:
    """Perform an anti-join: return rows from left where no matching row exists in right.

    Args:
        left: Left DataFrame
        right: Right DataFrame
        on: Join condition

    Returns:
        New DataFrame with rows from left that have no matches in right

    Raises:
        RuntimeError: If DataFrames are not bound to the same database
    """
    if left.database is None or right.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before anti_join")
    if left.database is not right.database:
        raise ValueError("Cannot anti_join DataFrames from different Database instances")

    normalized_condition = left._normalize_join_condition(on)
    if isinstance(normalized_condition, Column):
        plan = operators.anti_join(left.plan, right.plan, condition=normalized_condition, on=None)
    else:
        plan = operators.anti_join(left.plan, right.plan, on=normalized_condition, condition=None)
    return cast(T, left._with_plan(plan))


def cte_dataframe(
    df: T,
    name: str,
) -> T:
    """Create a Common Table Expression (CTE) from a DataFrame.

    Args:
        df: DataFrame to convert to CTE
        name: Name for the CTE

    Returns:
        New DataFrame representing the CTE
    """
    plan = operators.cte(df.plan, name)
    return cast(T, df._with_plan(plan))


def recursive_cte_dataframe(
    initial: T,
    name: str,
    recursive: T,
    union_all: bool = False,
) -> T:
    """Create a Recursive Common Table Expression (WITH RECURSIVE) from DataFrames.

    Args:
        initial: Initial DataFrame (base case)
        name: Name for the recursive CTE
        recursive: Recursive DataFrame (references the CTE)
        union_all: Whether to use UNION ALL (True) or UNION (False)

    Returns:
        New DataFrame representing the recursive CTE

    Raises:
        RuntimeError: If DataFrames are not bound to the same database
    """
    if initial.database is None or recursive.database is None:
        raise RuntimeError("Both DataFrames must be bound to a Database before recursive_cte")
    if initial.database is not recursive.database:
        raise ValueError(
            "Cannot create recursive CTE with DataFrames from different Database instances"
        )

    plan = operators.recursive_cte(name, initial.plan, recursive.plan, union_all=union_all)
    return cast(T, initial._with_plan(plan))


def explode_dataframe(
    df: T,
    column: Union[Column, str],
    alias: str = "value",
) -> T:
    """Explode an array/JSON column into multiple rows.

    Args:
        df: DataFrame to explode
        column: Column to explode (must be array or JSON)
        alias: Alias for the exploded value column

    Returns:
        New DataFrame with exploded rows
    """
    normalized_column = df._normalize_projection(column)
    if not isinstance(normalized_column, Column):
        raise ValueError("explode() requires a Column expression")

    plan = operators.explode(df.plan, normalized_column, alias=alias)
    return cast(T, df._with_plan(plan))


def pivot_dataframe(
    df: T,
    pivot_column: str,
    value_column: str,
    agg_func: str = "sum",
    pivot_values: Optional[Sequence[str]] = None,
) -> T:
    """Pivot a DataFrame to reshape data from long to wide format.

    Args:
        df: DataFrame to pivot
        pivot_column: Column to pivot on (values become column headers)
        value_column: Column containing values to aggregate
        agg_func: Aggregation function to apply (default: "sum")
        pivot_values: Optional list of specific values to pivot

    Returns:
        New DataFrame with pivoted data
    """
    plan = operators.pivot(
        df.plan, pivot_column, value_column, agg_func=agg_func, pivot_values=pivot_values
    )
    return cast(T, df._with_plan(plan))
