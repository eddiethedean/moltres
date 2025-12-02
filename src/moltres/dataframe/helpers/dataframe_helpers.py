"""Shared helper methods for :class:`DataFrame` and AsyncDataFrame classes.

This module contains common utility methods that are duplicated between
the sync and async :class:`DataFrame` implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Sequence, Tuple, Union

from ...expressions.column import Column, col
from ...logical import operators
from ...logical.plan import LogicalPlan, SortOrder

if TYPE_CHECKING:
    from ...utils.inspector import ColumnInfo
    from ...table.async_table import AsyncDatabase
    from ...table.table import Database
else:
    Database = Any
    AsyncDatabase = Any


class DataFrameHelpersProtocol(Protocol):
    """Protocol defining the interface that classes using DataFrameHelpersMixin must implement."""

    database: Optional[Union[Database, AsyncDatabase]]
    plan: LogicalPlan

    def _normalize_projection(self, expr: Union[Column, str]) -> Column:
        """Normalize a projection expression to a Column."""
        ...

    def _normalize_sort_expression(self, expr: Column) -> SortOrder:
        """Normalize a sort expression to a SortOrder."""
        ...

    def _normalize_join_condition(
        self,
        on: Optional[
            Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
        ],
    ) -> Union[Sequence[Tuple[str, str]], Column]:
        """Normalize join condition."""
        ...

    def _with_plan(self, plan: LogicalPlan) -> Any:
        """Create a new DataFrame with the given plan."""
        ...


class DataFrameHelpersMixin:
    """Mixin class providing shared helper methods for :class:`DataFrame` implementations.

    This mixin can be used by both :class:`DataFrame` and AsyncDataFrame to eliminate
    code duplication in helper methods.
    """

    # Subclasses must provide these attributes:
    # - database: Optional[Union[Database, AsyncDatabase]]
    # - plan: LogicalPlan
    database: Optional[Union[Database, AsyncDatabase]]

    def _normalize_projection(self, expr: Union[Column, str]) -> Column:
        """Normalize a projection expression to a :class:`Column`.

        Args:
            expr: :class:`Column` expression or string column name

        Returns:
            :class:`Column` expression
        """
        if isinstance(expr, Column):
            return expr
        return col(expr)

    def _is_window_function(self, col_expr: Column) -> bool:
        """Check if a :class:`Column` expression is a window function.

        Args:
            col_expr: :class:`Column` expression to check

        Returns:
            True if the expression is a window function, False otherwise
        """
        if not isinstance(col_expr, Column):
            return False

        # Check if it's wrapped in a window (after .over())
        if col_expr.op == "window":
            return True

        # Check if it's a window function operation
        window_ops = {
            "window_row_number",
            "window_rank",
            "window_dense_rank",
            "window_percent_rank",
            "window_cume_dist",
            "window_nth_value",
            "window_ntile",
            "window_lag",
            "window_lead",
            "window_first_value",
            "window_last_value",
        }
        if col_expr.op in window_ops:
            return True

        # Recursively check args for nested window functions
        for arg in col_expr.args:
            if isinstance(arg, Column) and self._is_window_function(arg):
                return True

        return False

    def _extract_column_name(self, col_expr: Column) -> Optional[str]:
        """Extract column name from a :class:`Column` expression.

        Args:
            col_expr: :class:`Column` expression to extract name from

        Returns:
            :class:`Column` name string, or None if cannot be determined
        """
        # If column has an alias, use that
        if col_expr._alias:
            return col_expr._alias

        # For simple column references
        if col_expr.op == "column" and col_expr.args:
            return str(col_expr.args[0])

        # For star columns, return None (will need to query schema)
        if col_expr.op == "star":
            return None

        # For other expressions, try to infer name from expression
        # This is a best-effort approach
        if col_expr.source:
            return col_expr.source

        # If we can't determine, return None
        return None

    def _find_base_plan(self, plan: LogicalPlan) -> LogicalPlan:
        """Find the base plan (TableScan, FileScan, or Project) by traversing down.

        Args:
            plan: Logical plan to traverse

        Returns:
            Base plan (TableScan, FileScan, or Project with no Project child)
        """
        from ...logical.plan import (
            Aggregate,
            Distinct,
            Explode,
            Filter,
            Join,
            Limit,
            Sample,
            Sort,
            TableScan,
            FileScan,
            Project,
        )

        # If this is a base plan type, return it
        if isinstance(plan, (TableScan, FileScan)):
            return plan

        # If this is a Project, return it (it's the final projection)
        # Even if the child is also a Project, we want the outermost one
        if isinstance(plan, Project):
            return plan

        # For operations that have a single child, traverse down
        if isinstance(plan, (Filter, Limit, Sample, Sort, Distinct, Explode)):
            return self._find_base_plan(plan.child)

        # For Aggregate, the schema comes from aggregates, not child
        if isinstance(plan, Aggregate):
            return plan

        # For Join, we need to handle both sides - return the plan itself
        # as we'll need to combine schemas
        if isinstance(plan, Join):
            return plan

        # For other plan types, return as-is
        return plan

    def _extract_column_names(self, plan: LogicalPlan) -> List[str]:
        """Extract column names from a logical plan.

        Args:
            plan: Logical plan to extract column names from

        Returns:
            List of column name strings

        Raises:
            RuntimeError: If column names cannot be determined (e.g., RawSQL)
        """
        from ...logical.plan import (
            Aggregate,
            Explode,
            FileScan,
            Join,
            Project,
            RawSQL,
            TableScan,
        )

        base_plan = self._find_base_plan(plan)

        # Handle RawSQL - cannot determine schema without execution
        if isinstance(base_plan, RawSQL):
            raise RuntimeError(
                "Cannot determine column names from RawSQL without executing the query. "
                "Use df.collect() first or specify columns explicitly."
            )

        # Handle Project - extract from projections
        if isinstance(base_plan, Project):
            column_names: List[str] = []
            has_star = False
            explicit_columns = []

            # First pass: collect explicit columns and check for star
            for proj in base_plan.projections:
                if proj.op == "star":
                    has_star = True
                else:
                    col_name = self._extract_column_name(proj)
                    if col_name:
                        explicit_columns.append(col_name)

            # If we have a star, get all columns from child
            if has_star:
                child_names = self._extract_column_names(base_plan.child)
                column_names.extend(child_names)

            # Add explicit columns, avoiding duplicates
            for col_name in explicit_columns:
                if col_name not in column_names:
                    column_names.append(col_name)
                else:
                    # If column already exists (from star), it means it was redefined
                    # Keep the explicit one (it will replace the one from star)
                    # Actually, we want to keep both in order, so we should track positions
                    pass

            # If no star and no explicit columns, return empty (shouldn't happen)
            if not has_star and not explicit_columns:
                return []

            return column_names

        # Handle Aggregate - extract from aggregates
        if isinstance(base_plan, Aggregate):
            column_names = []
            # Add grouping columns
            for group_col in base_plan.grouping:
                col_name = self._extract_column_name(group_col)
                if col_name:
                    column_names.append(col_name)
            # Add aggregate columns
            for agg_col in base_plan.aggregates:
                col_name = self._extract_column_name(agg_col)
                if col_name:
                    column_names.append(col_name)
            return column_names

        # Handle Join - combine columns from both sides
        if isinstance(base_plan, Join):
            left_names = self._extract_column_names(base_plan.left)
            right_names = self._extract_column_names(base_plan.right)
            return left_names + right_names

        # Handle Explode - add exploded column alias
        if isinstance(base_plan, Explode):
            child_names = self._extract_column_names(base_plan.child)
            # Add the exploded column alias
            alias = base_plan.alias or "value"
            if alias not in child_names:
                child_names.append(alias)
            return child_names

        # Handle TableScan - query database metadata
        if isinstance(base_plan, TableScan):
            if self.database is None:
                raise RuntimeError(
                    "Cannot determine column names: DataFrame has no database connection"
                )
            from ...utils.inspector import get_table_columns

            table_name = base_plan.alias or base_plan.table
            columns = get_table_columns(self.database, table_name)
            return [col_info.name for col_info in columns]

        # Handle FileScan - use schema if available
        if isinstance(base_plan, FileScan):
            if base_plan.schema:
                return [col_def.name for col_def in base_plan.schema]
            # If no schema, try to infer from column_name (for text files)
            if base_plan.column_name:
                return [base_plan.column_name]
            # Cannot determine without schema
            raise RuntimeError(
                f"Cannot determine column names from FileScan without schema. "
                f"File: {base_plan.path}, Format: {base_plan.format}"
            )

        # For other plan types, try to get from child
        children = base_plan.children()
        if children:
            return self._extract_column_names(children[0])

        # If we can't determine, raise error
        raise RuntimeError(
            f"Cannot determine column names from plan type: {type(base_plan).__name__}"
        )

    def _extract_schema_from_plan(self, plan: LogicalPlan) -> List["ColumnInfo"]:
        """Extract schema information from a logical plan.

        Args:
            plan: Logical plan to extract schema from

        Returns:
            List of ColumnInfo objects with column names and types

        Raises:
            RuntimeError: If schema cannot be determined
        """
        from ...logical.plan import (
            Aggregate,
            Explode,
            FileScan,
            Join,
            Project,
            RawSQL,
            TableScan,
        )
        from ...utils.inspector import ColumnInfo

        base_plan = self._find_base_plan(plan)

        # Handle RawSQL - cannot determine schema without execution
        if isinstance(base_plan, RawSQL):
            raise RuntimeError(
                "Cannot determine schema from RawSQL without executing the query. "
                "Use df.collect() first or specify schema explicitly."
            )

        # Handle Project - extract from projections
        if isinstance(base_plan, Project):
            schema: List[ColumnInfo] = []
            child_schema = self._extract_schema_from_plan(base_plan.child)

            for proj in base_plan.projections:
                col_name = self._extract_column_name(proj)
                if col_name:
                    # Try to find type from child schema
                    col_type = "UNKNOWN"
                    for child_col in child_schema:
                        if child_col.name == col_name or (
                            proj.op == "column"
                            and proj.args
                            and str(proj.args[0]) == child_col.name
                        ):
                            col_type = child_col.type_name
                            break
                    schema.append(ColumnInfo(name=col_name, type_name=col_type))
                elif proj.op == "star":
                    # For "*", include all columns from child
                    schema.extend(child_schema)
            return schema

        # Handle Aggregate - extract from aggregates
        if isinstance(base_plan, Aggregate):
            schema = []
            child_schema = self._extract_schema_from_plan(base_plan.child)

            # Add grouping columns
            for group_col in base_plan.grouping:
                col_name = self._extract_column_name(group_col)
                if col_name:
                    col_type = "UNKNOWN"
                    for child_col in child_schema:
                        if child_col.name == col_name:
                            col_type = child_col.type_name
                            break
                    schema.append(ColumnInfo(name=col_name, type_name=col_type))

            # Add aggregate columns (typically numeric types)
            for agg_col in base_plan.aggregates:
                col_name = self._extract_column_name(agg_col)
                if col_name:
                    # Aggregates are typically numeric, but we can't be sure
                    # Use a generic type or try to infer from expression
                    schema.append(ColumnInfo(name=col_name, type_name="NUMERIC"))
            return schema

        # Handle Join - combine schemas from both sides
        if isinstance(base_plan, Join):
            left_schema = self._extract_schema_from_plan(base_plan.left)
            right_schema = self._extract_schema_from_plan(base_plan.right)
            return left_schema + right_schema

        # Handle Explode - add exploded column
        if isinstance(base_plan, Explode):
            child_schema = self._extract_schema_from_plan(base_plan.child)
            alias = base_plan.alias or "value"
            # Add exploded column (typically array/JSON element, so use TEXT)
            child_schema.append(ColumnInfo(name=alias, type_name="TEXT"))
            return child_schema

        # Handle TableScan - query database metadata
        if isinstance(base_plan, TableScan):
            if self.database is None:
                raise RuntimeError("Cannot determine schema: DataFrame has no database connection")
            from ...utils.inspector import get_table_columns

            table_name = base_plan.alias or base_plan.table
            return get_table_columns(self.database, table_name)

        # Handle FileScan - use schema if available
        if isinstance(base_plan, FileScan):
            if base_plan.schema:
                return [
                    ColumnInfo(name=col_def.name, type_name=col_def.type_name)
                    for col_def in base_plan.schema
                ]
            # If no schema, try to infer from column_name (for text files)
            if base_plan.column_name:
                return [ColumnInfo(name=base_plan.column_name, type_name="TEXT")]
            # Cannot determine without schema
            raise RuntimeError(
                f"Cannot determine schema from FileScan without explicit schema. "
                f"File: {base_plan.path}, Format: {base_plan.format}"
            )

        # For other plan types, try to get from child
        children = base_plan.children()
        if children:
            return self._extract_schema_from_plan(children[0])

        # If we can't determine, raise error
        raise RuntimeError(f"Cannot determine schema from plan type: {type(base_plan).__name__}")

    def _normalize_sort_expression(self, expr: Column) -> SortOrder:
        """Normalize a sort expression to a SortOrder.

        Args:
            expr: :class:`Column` expression to normalize

        Returns:
            SortOrder object
        """
        if expr.op == "sort_desc":
            col_arg = expr.args[0]
            if not isinstance(col_arg, Column):
                raise TypeError(
                    f"Expected Column for sort expression, got {type(col_arg).__name__}"
                )
            return operators.sort_order(col_arg, descending=True)
        if expr.op == "sort_asc":
            col_arg = expr.args[0]
            if not isinstance(col_arg, Column):
                raise TypeError(
                    f"Expected Column for sort expression, got {type(col_arg).__name__}"
                )
            return operators.sort_order(col_arg, descending=False)
        return operators.sort_order(expr, descending=False)

    def _normalize_join_condition(
        self,
        on: Optional[
            Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
        ],
    ) -> Union[Sequence[Tuple[str, str]], Column]:
        """Normalize join condition to either tuple pairs or a :class:`Column` expression.

        Args:
            on: Join condition - can be string, sequence of strings/tuples, or :class:`Column` expression(s)

        Returns:
            - Sequence[Tuple[str, str]]: For tuple/string-based joins (backward compatible)
            - :class:`Column`: For PySpark-style :class:`Column` expression joins

        Raises:
            ValueError: If join condition is invalid
        """
        if on is None:
            raise ValueError("join requires an `on` argument for equality joins")

        # Handle Column expressions (PySpark-style)
        if isinstance(on, Column):
            return on
        if isinstance(on, Sequence) and len(on) > 0 and isinstance(on[0], Column):
            # Multiple Column expressions - combine with AND
            conditions: List[Column] = []
            for entry in on:
                if not isinstance(entry, Column):
                    raise ValueError(
                        "All elements in join condition must be Column expressions when using PySpark-style syntax"
                    )
                conditions.append(entry)
            # Combine all conditions with AND
            result = conditions[0]
            for cond in conditions[1:]:
                result = result & cond
            return result

        # Handle tuple/string-based joins (backward compatible)
        if isinstance(on, str):
            return [(on, on)]
        normalized: List[Tuple[str, str]] = []
        for entry in on:
            if isinstance(entry, tuple):
                if len(entry) != 2:
                    raise ValueError("join tuples must specify (left, right) column names")
                normalized.append((entry[0], entry[1]))
            else:
                # At this point, entry must be a string (not a Column, as we've already checked)
                assert isinstance(entry, str), "entry must be a string at this point"
                normalized.append((entry, entry))
        return normalized
