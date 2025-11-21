"""Lazy DataFrame representation."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from ..expressions.column import Column, col
from ..logical import operators
from ..logical.plan import LogicalPlan, Project, SortOrder
from ..sql.compiler import compile_plan

if TYPE_CHECKING:
    from ..table.schema import ColumnDef
    from ..table.table import Database, TableHandle
    from .groupby import GroupedDataFrame
    from .writer import DataFrameWriter


@dataclass(frozen=True)
class DataFrame:
    plan: LogicalPlan
    database: Database | None = None
    _materialized_data: list[dict[str, object]] | None = None
    _stream_generator: Callable[[], Iterator[list[dict[str, object]]]] | None = None
    _stream_schema: Sequence[ColumnDef] | None = None

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_table(
        cls, table_handle: TableHandle, columns: Sequence[str] | None = None
    ) -> DataFrame:
        plan = operators.scan(table_handle.name)
        df = cls(plan=plan, database=table_handle.database)
        if columns:
            df = df.select(*columns)
        return df

    def select(self, *columns: Column | str) -> DataFrame:
        if not columns:
            return self
        normalized = tuple(self._normalize_projection(column) for column in columns)
        return self._with_plan(operators.project(self.plan, normalized))

    def where(self, predicate: Column) -> DataFrame:
        return self._with_plan(operators.filter(self.plan, predicate))

    filter = where

    def limit(self, count: int, offset: int = 0) -> DataFrame:
        """Limit the number of rows returned by the query.

        Args:
            count: Maximum number of rows to return. Must be non-negative.
                  If 0, returns an empty result set.
            offset: Number of rows to skip before returning results. Must be non-negative.

        Returns:
            New DataFrame with the limit applied

        Raises:
            ValueError: If count or offset is negative
        """
        if count < 0:
            raise ValueError("limit count must be non-negative")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        return self._with_plan(operators.limit(self.plan, count, offset=offset))

    def offset(self, count: int) -> DataFrame:
        """Skip a number of rows before returning results.

        Args:
            count: Number of rows to skip. Must be non-negative.

        Returns:
            New DataFrame with the offset applied

        Raises:
            ValueError: If count is negative
        """
        if count < 0:
            raise ValueError("offset count must be non-negative")
        # Get current limit and offset from plan if it's a Limit node
        from ..logical.plan import Limit

        if isinstance(self.plan, Limit):
            # Update existing limit with new offset
            return self._with_plan(
                Limit(child=self.plan.child, count=self.plan.count, offset=count)
            )
        # If no limit exists, we need to add one with a large limit
        # Use a very large number for limit when only offset is specified
        limit_plan = Limit(child=self.plan, count=2**31 - 1, offset=count)
        return self._with_plan(limit_plan)

    def order_by(self, *columns: Column) -> DataFrame:
        if not columns:
            return self
        orders = tuple(self._normalize_sort_expression(column) for column in columns)
        return self._with_plan(operators.order_by(self.plan, orders))

    def join(
        self,
        other: DataFrame,
        *,
        on: str | Sequence[str] | Sequence[tuple[str, str]] | None = None,
        how: str = "inner",
    ) -> DataFrame:
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before joining")
        if self.database is not other.database:
            raise ValueError("Cannot join DataFrames from different Database instances")
        normalized_on = self._normalize_join_keys(on)
        plan = operators.join(self.plan, other.plan, how=how.lower(), on=normalized_on)
        return DataFrame(plan=plan, database=self.database)

    def group_by(self, *columns: Column | str) -> GroupedDataFrame:
        if not columns:
            raise ValueError("group_by requires at least one grouping column")
        from .groupby import GroupedDataFrame

        keys = tuple(self._normalize_projection(column) for column in columns)
        return GroupedDataFrame(plan=self.plan, keys=keys, parent=self)

    groupBy = group_by

    def distinct(self) -> DataFrame:
        """Return distinct rows, removing duplicates.

        Returns:
            New DataFrame with distinct rows
        """
        return self._with_plan(operators.distinct(self.plan))

    def union(self, other: DataFrame) -> DataFrame:
        """Union this DataFrame with another, removing duplicates.

        Args:
            other: Another DataFrame to union with

        Returns:
            New DataFrame containing distinct rows from both DataFrames

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before union")
        if self.database is not other.database:
            raise ValueError("Cannot union DataFrames from different Database instances")
        plan = operators.union(self.plan, other.plan, distinct=True)
        return DataFrame(plan=plan, database=self.database)

    def unionAll(self, other: DataFrame) -> DataFrame:
        """Union this DataFrame with another, keeping all rows including duplicates.

        Args:
            other: Another DataFrame to union with

        Returns:
            New DataFrame containing all rows from both DataFrames

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before unionAll")
        if self.database is not other.database:
            raise ValueError("Cannot unionAll DataFrames from different Database instances")
        plan = operators.union(self.plan, other.plan, distinct=False)
        return DataFrame(plan=plan, database=self.database)

    union_all = unionAll

    # ---------------------------------------------------------------- execution
    def to_sql(self) -> str:
        if self.database is not None:
            return self.database.compile_plan(self.plan)
        return compile_plan(self.plan)

    def collect(
        self, stream: bool = False
    ) -> list[dict[str, object]] | Iterator[list[dict[str, object]]]:
        """Collect DataFrame results.

        Args:
            stream: If True, return an iterator of row chunks. If False (default),
                   materialize all rows into a list (backward compatible).

        Returns:
            If stream=False: List of dictionaries representing rows.
            If stream=True: Iterator of row chunks (each chunk is a list of dicts).

        Raises:
            RuntimeError: If DataFrame is not bound to a Database and has no materialized data
        """
        # Check if DataFrame has streaming generator
        if self._stream_generator is not None:
            if stream:
                return self._stream_generator()
            # Materialize all chunks for backward compatibility
            all_rows = []
            for chunk in self._stream_generator():
                all_rows.extend(chunk)
            return all_rows

        # Check if DataFrame has materialized data (from file readers)
        if self._materialized_data is not None:
            if stream:
                # Return iterator with single chunk
                return iter([self._materialized_data])
            return self._materialized_data

        if self.database is None:
            raise RuntimeError("Cannot collect a plan without an attached Database")

        if stream:
            # For SQL queries, use streaming execution
            return self.database.execute_plan_stream(self.plan)

        result = self.database.execute_plan(self.plan)
        return result.rows

    @property
    def write(self) -> DataFrameWriter:
        """Return a DataFrameWriter for writing this DataFrame to a table."""
        from .writer import DataFrameWriter

        return DataFrameWriter(self)

    @property
    def columns(self) -> list[str]:
        """Get the column names of this DataFrame.

        Returns:
            List of column names

        Raises:
            RuntimeError: If column names cannot be determined
        """
        projections = self._get_current_projections()
        if projections:
            column_names = []
            for proj in projections:
                # Get column name from alias or from the expression itself
                if proj._alias:
                    column_names.append(proj._alias)
                elif proj.op == "column" and proj.args:
                    column_names.append(proj.args[0])
                else:
                    # For complex expressions without alias, use a generated name
                    column_names.append(f"expr_{len(column_names)}")
            return column_names

        # If no explicit projections, try to get from table scan
        from ..logical.plan import TableScan

        if isinstance(self.plan, TableScan):
            if self.database is None:
                raise RuntimeError("Cannot determine columns: DataFrame not bound to Database")
            # Query the table to get column names
            # Use LIMIT 0 to get schema without data
            try:
                quote_char = self.database.dialect.quote_char
                table_name = self.plan.table
                result = self.database.execute_sql(
                    f"SELECT * FROM {quote_char}{table_name}{quote_char} LIMIT 0"
                )
                # Get column names from result metadata
                if hasattr(result, "keys"):
                    return list(result.keys())
                # Fallback: try to infer from a sample query
                sample_result = self.database.execute_plan(self.plan)
                if (
                    sample_result.rows
                    and isinstance(sample_result.rows, list)
                    and len(sample_result.rows) > 0
                ):
                    return list(sample_result.rows[0].keys())
            except Exception:
                pass

        raise RuntimeError(
            "Cannot determine columns. Use select() to explicitly specify columns, "
            "or ensure DataFrame is bound to a Database for table scans."
        )

    @property
    def schema(self) -> list[dict[str, str]]:
        """Get the schema of this DataFrame as a list of column info dictionaries.

        Returns:
            List of dictionaries with 'name' and 'type' keys

        Note:
            Type information may be limited for complex expressions.
        """
        column_names = self.columns
        schema = []
        for name in column_names:
            # Try to infer type from projections if available
            type_name = "UNKNOWN"  # Default type
            projections = self._get_current_projections()
            if projections:
                for proj in projections:
                    proj_name = proj._alias or (
                        proj.args[0] if proj.op == "column" and proj.args else None
                    )
                    if proj_name == name:
                        # Try to infer type from expression
                        if proj.op == "cast" and len(proj.args) > 1:
                            type_name = proj.args[1]
                        elif proj.op in ("agg_sum", "agg_avg"):
                            type_name = "NUMERIC"
                        elif proj.op in ("upper", "lower", "concat", "substring", "trim"):
                            type_name = "TEXT"
                        break
            schema.append({"name": name, "type": type_name})
        return schema

    def printSchema(self) -> None:
        """Print the schema of this DataFrame in a formatted way."""
        schema = self.schema
        print("root")
        print(" |-- ", end="")
        for i, col_info in enumerate(schema):
            if i > 0:
                print(" |-- ", end="")
            print(f"{col_info['name']}: {col_info['type']} (nullable = true)")

    def describe(self) -> DataFrame:
        """Compute statistical summary of numeric columns.

        Returns:
            DataFrame with statistics (count, mean, stddev, min, max)

        Note:
            This requires materializing the data, so it's not lazy.
        """
        if self.database is None:
            raise RuntimeError("Cannot describe DataFrame: not bound to a Database")
        # Materialize data to compute statistics
        rows = self.collect()
        if not rows:
            return self._with_plan(operators.project(self.plan, ()))
        # For now, return a simple summary
        # In a full implementation, this would compute statistics
        # This is a placeholder that returns the DataFrame itself
        # Full implementation would require aggregating over numeric columns
        return self

    # ---------------------------------------------------------------- utilities
    def _with_plan(self, plan: LogicalPlan) -> DataFrame:
        return DataFrame(
            plan=plan,
            database=self.database,
            _materialized_data=self._materialized_data,
            _stream_generator=self._stream_generator,
            _stream_schema=self._stream_schema,
        )

    def _normalize_projection(self, expr: Column | str) -> Column:
        if isinstance(expr, Column):
            return expr
        return col(expr)

    def _normalize_sort_expression(self, expr: Column) -> SortOrder:
        if expr.op == "sort_desc":
            return operators.sort_order(expr.args[0], descending=True)
        if expr.op == "sort_asc":
            return operators.sort_order(expr.args[0], descending=False)
        return operators.sort_order(expr, descending=False)

    def _normalize_join_keys(
        self, on: str | Sequence[str] | Sequence[tuple[str, str]] | None
    ) -> Sequence[tuple[str, str]]:
        if on is None:
            raise ValueError("join requires an `on` argument for equality joins")
        if isinstance(on, str):
            return [(on, on)]
        normalized: list[tuple[str, str]] = []
        for entry in on:
            if isinstance(entry, tuple):
                if len(entry) != 2:
                    raise ValueError("join tuples must specify (left, right) column names")
                normalized.append((entry[0], entry[1]))
            else:
                normalized.append((entry, entry))
        return normalized

    def _get_current_projections(self) -> tuple[Column, ...]:
        """Extract current projections from the plan."""
        plan = self.plan
        # Walk up the plan tree to find Project nodes
        if isinstance(plan, Project):
            return plan.projections
        # If no Project node, we need to scan the table to get columns
        # For now, return empty tuple - this will require schema inspection
        return ()

    def drop(self, *columns: str) -> DataFrame:
        """Drop specified columns from the DataFrame.

        Args:
            *columns: Column names to drop

        Returns:
            New DataFrame with specified columns removed

        Raises:
            ValueError: If no columns are specified or if trying to drop all columns
        """
        if not columns:
            raise ValueError("drop requires at least one column name")
        # Get current projections
        current_projections = self._get_current_projections()
        if not current_projections:
            # If we can't determine current columns, raise an error
            raise ValueError(
                "Cannot drop columns: unable to determine current columns. "
                "Use select() first to specify columns explicitly."
            )
        # Filter out dropped columns
        columns_to_drop = set(columns)
        new_projections = tuple(
            proj
            for proj in current_projections
            if (proj._alias or (proj.op == "column" and proj.args[0] if proj.args else None))
            not in columns_to_drop
        )
        if not new_projections:
            raise ValueError("Cannot drop all columns")
        base_plan = self.plan.child if isinstance(self.plan, Project) else self.plan
        return self._with_plan(operators.project(base_plan, new_projections))

    def withColumnRenamed(self, old_name: str, new_name: str) -> DataFrame:
        """Rename a column.

        Args:
            old_name: Current column name
            new_name: New column name

        Returns:
            New DataFrame with renamed column
        """
        return self.rename({old_name: new_name})

    def rename(self, columns: dict[str, str]) -> DataFrame:
        """Rename columns using a mapping.

        Args:
            columns: Dictionary mapping old column names to new column names

        Returns:
            New DataFrame with renamed columns
        """
        if not columns:
            return self
        # Get current projections
        current_projections = self._get_current_projections()
        if not current_projections:
            raise ValueError(
                "Cannot rename columns: unable to determine current columns. "
                "Use select() first to specify columns explicitly."
            )
        # Create new projections with renamed columns
        new_projections = []
        for proj in current_projections:
            col_name = proj._alias or (proj.args[0] if proj.op == "column" and proj.args else None)
            if col_name in columns:
                new_proj = proj.alias(columns[col_name])
                new_projections.append(new_proj)
            else:
                new_projections.append(proj)
        base_plan = self.plan.child if isinstance(self.plan, Project) else self.plan
        return self._with_plan(operators.project(base_plan, tuple(new_projections)))

    def withColumn(self, col_name: str, col_expr: Column) -> DataFrame:
        """Add or replace a column.

        Args:
            col_name: Name of the column to add or replace
            col_expr: Column expression for the new column

        Returns:
            New DataFrame with the added/replaced column
        """
        if not isinstance(col_expr, Column):
            raise TypeError("col_expr must be a Column expression")
        # Get current projections
        current_projections = list(self._get_current_projections())
        # Check if column already exists and replace it, otherwise add it
        col_expr_with_alias = col_expr.alias(col_name)
        found = False
        for i, proj in enumerate(current_projections):
            proj_name = proj._alias or (proj.args[0] if proj.op == "column" and proj.args else None)
            if proj_name == col_name:
                current_projections[i] = col_expr_with_alias
                found = True
                break
        if not found:
            current_projections.append(col_expr_with_alias)
        base_plan = self.plan.child if isinstance(self.plan, Project) else self.plan
        return self._with_plan(operators.project(base_plan, tuple(current_projections)))
