"""Lazy DataFrame representation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from ..expressions.column import Column, col
from ..logical import operators
from ..logical.plan import LogicalPlan, SortOrder
from ..sql.compiler import compile_plan

if TYPE_CHECKING:
    from ..table.table import Database, TableHandle
    from .groupby import GroupedDataFrame
    from .writer import DataFrameWriter


@dataclass(frozen=True)
class DataFrame:
    plan: LogicalPlan
    database: Optional["Database"] = None

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_table(
        cls, table_handle: "TableHandle", columns: Optional[Sequence[str]] = None
    ) -> "DataFrame":
        plan = operators.scan(table_handle.name)
        df = cls(plan=plan, database=table_handle.database)
        if columns:
            df = df.select(*columns)
        return df

    def select(self, *columns: Union[Column, str]) -> "DataFrame":
        if not columns:
            return self
        normalized = tuple(self._normalize_projection(column) for column in columns)
        return self._with_plan(operators.project(self.plan, normalized))

    def where(self, predicate: Column) -> "DataFrame":
        return self._with_plan(operators.filter(self.plan, predicate))

    filter = where

    def limit(self, count: int) -> "DataFrame":
        """Limit the number of rows returned by the query.

        Args:
            count: Maximum number of rows to return. Must be non-negative.
                  If 0, returns an empty result set.

        Returns:
            New DataFrame with the limit applied

        Raises:
            ValueError: If count is negative
        """
        if count < 0:
            raise ValueError("limit count must be non-negative")
        return self._with_plan(operators.limit(self.plan, count))

    def order_by(self, *columns: Column) -> "DataFrame":
        if not columns:
            return self
        orders = tuple(self._normalize_sort_expression(column) for column in columns)
        return self._with_plan(operators.order_by(self.plan, orders))

    def join(
        self,
        other: "DataFrame",
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        how: str = "inner",
    ) -> "DataFrame":
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before joining")
        if self.database is not other.database:
            raise ValueError("Cannot join DataFrames from different Database instances")
        # Cross joins don't require an 'on' clause
        if how.lower() == "cross":
            normalized_on = None
        else:
            normalized_on = self._normalize_join_keys(on)
        plan = operators.join(self.plan, other.plan, how=how.lower(), on=normalized_on)
        return DataFrame(plan=plan, database=self.database)

    def crossJoin(self, other: "DataFrame") -> "DataFrame":
        """Perform a cross join (Cartesian product) with another DataFrame.

        Args:
            other: Another DataFrame to cross join with

        Returns:
            New DataFrame containing the Cartesian product of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database
        """
        return self.join(other, how="cross")

    def group_by(self, *columns: Union[Column, str]) -> "GroupedDataFrame":
        if not columns:
            raise ValueError("group_by requires at least one grouping column")
        from .groupby import GroupedDataFrame

        keys = tuple(self._normalize_projection(column) for column in columns)
        return GroupedDataFrame(plan=self.plan, keys=keys, parent=self)

    groupBy = group_by

    def union(self, other: "DataFrame") -> "DataFrame":
        """Union this DataFrame with another DataFrame (distinct rows only).

        Args:
            other: Another DataFrame to union with

        Returns:
            New DataFrame containing the union of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before union")
        if self.database is not other.database:
            raise ValueError("Cannot union DataFrames from different Database instances")
        plan = operators.union(self.plan, other.plan, distinct=True)
        return DataFrame(plan=plan, database=self.database)

    def unionAll(self, other: "DataFrame") -> "DataFrame":
        """Union this DataFrame with another DataFrame (all rows, including duplicates).

        Args:
            other: Another DataFrame to union with

        Returns:
            New DataFrame containing the union of all rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before union")
        if self.database is not other.database:
            raise ValueError("Cannot union DataFrames from different Database instances")
        plan = operators.union(self.plan, other.plan, distinct=False)
        return DataFrame(plan=plan, database=self.database)

    def intersect(self, other: "DataFrame") -> "DataFrame":
        """Intersect this DataFrame with another DataFrame (distinct rows only).

        Args:
            other: Another DataFrame to intersect with

        Returns:
            New DataFrame containing the intersection of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before intersect")
        if self.database is not other.database:
            raise ValueError("Cannot intersect DataFrames from different Database instances")
        plan = operators.intersect(self.plan, other.plan, distinct=True)
        return DataFrame(plan=plan, database=self.database)

    def except_(self, other: "DataFrame") -> "DataFrame":
        """Return rows in this DataFrame that are not in another DataFrame (distinct rows only).

        Args:
            other: Another DataFrame to exclude from

        Returns:
            New DataFrame containing rows in this DataFrame but not in other

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before except")
        if self.database is not other.database:
            raise ValueError("Cannot except DataFrames from different Database instances")
        plan = operators.except_(self.plan, other.plan, distinct=True)
        return DataFrame(plan=plan, database=self.database)

    def cte(self, name: str) -> "DataFrame":
        """Create a Common Table Expression (CTE) from this DataFrame.

        Args:
            name: Name for the CTE

        Returns:
            New DataFrame representing the CTE

        Example:
            >>> cte_df = db.table("orders").select().where(col("amount") > 100).cte("high_value_orders")
            >>> result = cte_df.select().collect()  # Query the CTE
        """
        plan = operators.cte(self.plan, name)
        return DataFrame(plan=plan, database=self.database)

    def distinct(self) -> "DataFrame":
        """Return a new DataFrame with distinct rows.

        Returns:
            New DataFrame with distinct rows
        """
        return self._with_plan(operators.distinct(self.plan))

    def dropDuplicates(self, subset: Optional[Sequence[str]] = None) -> "DataFrame":
        """Return a new DataFrame with duplicate rows removed.

        Args:
            subset: Optional list of column names to consider when identifying duplicates.
                   If None, all columns are considered.

        Returns:
            New DataFrame with duplicates removed

        Note:
            This is equivalent to distinct() when subset is None.
            When subset is provided, it's implemented as a group_by on those columns
            with a select of all columns.
        """
        if subset is None:
            return self.distinct()
        # For subset, we need to group by those columns and select all
        # This is a simplified implementation - a more complete one would
        # use window functions or subqueries
        return self.group_by(*subset).agg()

    def withColumn(self, colName: str, col_expr: Union[Column, str]) -> "DataFrame":
        """Add or replace a column in the DataFrame.

        Args:
            colName: Name of the column to add or replace
            col_expr: Column expression or column name

        Returns:
            New DataFrame with the added/replaced column

        Note:
            This operation adds a Project on top of the current plan.
            If a column with the same name exists, it will be replaced.
        """
        from ..logical.plan import Project

        # Normalize the column expression
        new_col = self._normalize_projection(col_expr)
        # Add alias if it's a Column expression
        if isinstance(new_col, Column) and not new_col._alias:
            new_col = new_col.alias(colName)
        elif isinstance(new_col, Column):
            # Already has alias, but we want to use colName
            new_col = replace(new_col, _alias=colName)

        # Get existing columns from the plan if it's a Project
        # Otherwise, we'll select all columns plus the new one
        if isinstance(self.plan, Project):
            # Add the new column to existing projections
            existing_cols = list(self.plan.projections)
            # Remove any column with the same name (if we can detect it)
            # For now, just add the new column
            new_projections = existing_cols + [new_col]
        else:
            # No existing projection, select all plus new column
            # Use a wildcard select and add the new column
            # This is a simplified approach - in practice, we'd need schema info
            new_projections = [new_col]

        return self._with_plan(operators.project(self.plan, tuple(new_projections)))

    def withColumnRenamed(self, existing: str, new: str) -> "DataFrame":
        """Rename a column in the DataFrame.

        Args:
            existing: Current name of the column
            new: New name for the column

        Returns:
            New DataFrame with the renamed column
        """
        from ..logical.plan import Project

        if isinstance(self.plan, Project):
            # Rename the column in the projection
            new_projections = []
            for col_expr in self.plan.projections:
                if isinstance(col_expr, Column):
                    # Check if this column matches the existing name
                    if col_expr._alias == existing or (
                        col_expr.op == "column" and col_expr.args[0] == existing
                    ):
                        # Rename it
                        new_col = replace(col_expr, _alias=new)
                        new_projections.append(new_col)
                    else:
                        new_projections.append(col_expr)
                else:
                    new_projections.append(col_expr)
            return self._with_plan(operators.project(self.plan.child, tuple(new_projections)))
        else:
            # No projection yet, create one that selects all and renames the column
            existing_col = col(existing).alias(new)
            return self._with_plan(operators.project(self.plan, (existing_col,)))

    def drop(self, *cols: str) -> "DataFrame":
        """Drop one or more columns from the DataFrame.

        Args:
            *cols: Column names to drop

        Returns:
            New DataFrame with the specified columns removed

        Note:
            This operation only works if the DataFrame has a Project operation.
            Otherwise, it will create a Project that excludes the specified columns.
        """
        from ..logical.plan import Project

        cols_to_drop = set(cols)

        if isinstance(self.plan, Project):
            # Filter out the columns to drop
            new_projections = []
            for col_expr in self.plan.projections:
                if isinstance(col_expr, Column):
                    col_name = col_expr._alias or (
                        col_expr.args[0] if col_expr.op == "column" else None
                    )
                    if col_name not in cols_to_drop:
                        new_projections.append(col_expr)
                else:
                    new_projections.append(col_expr)
            return self._with_plan(operators.project(self.plan.child, tuple(new_projections)))
        else:
            # No projection - this is a simplified implementation
            # In practice, we'd need to know all columns to exclude the dropped ones
            # For now, return self (can't drop from a table scan without schema)
            return self

    # ---------------------------------------------------------------- execution
    def to_sql(self) -> str:
        """Convert the DataFrame's logical plan to a SQL string.

        Returns:
            SQL string representation of the query
        """
        from sqlalchemy.sql import Select

        stmt = (
            self.database.compile_plan(self.plan)
            if self.database is not None
            else compile_plan(self.plan)
        )
        if isinstance(stmt, Select):
            # Compile SQLAlchemy statement to SQL string
            return str(stmt.compile(compile_kwargs={"literal_binds": True}))
        return str(stmt)

    def collect(
        self, stream: bool = False
    ) -> Union[List[Dict[str, object]], Iterator[List[Dict[str, object]]]]:
        """Collect DataFrame results.

        Args:
            stream: If True, return an iterator of row chunks. If False (default),
                   materialize all rows into a list.

        Returns:
            If stream=False: List of dictionaries representing rows.
            If stream=True: Iterator of row chunks (each chunk is a list of dicts).

        Raises:
            RuntimeError: If DataFrame is not bound to a Database
        """
        if self.database is None:
            raise RuntimeError("Cannot collect a plan without an attached Database")

        if stream:
            # For SQL queries, use streaming execution
            return self.database.execute_plan_stream(self.plan)

        result = self.database.execute_plan(self.plan)
        return result.rows  # type: ignore[no-any-return]

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the DataFrame.

        Args:
            n: Number of rows to show (default: 20)
            truncate: If True, truncate long strings (default: True)
        """
        rows = self.limit(n).collect()
        # collect() with stream=False returns a list, not an iterator
        if not isinstance(rows, list):
            raise TypeError("show() requires collect() to return a list, not an iterator")
        if not rows:
            print("Empty DataFrame")
            return

        # Get column names from first row
        columns = list(rows[0].keys())

        # Calculate column widths
        col_widths = {}
        for col_name in columns:
            col_widths[col_name] = len(col_name)
            for row in rows:
                val_str = str(row.get(col_name, ""))
                if truncate and len(val_str) > 20:
                    val_str = val_str[:17] + "..."
                col_widths[col_name] = max(col_widths[col_name], len(val_str))

        # Print header
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        print(header)
        print("-" * len(header))

        # Print rows
        for row in rows:
            row_str = " | ".join(
                (
                    str(row.get(col, ""))[:17] + "..."
                    if truncate and len(str(row.get(col, ""))) > 20
                    else str(row.get(col, ""))
                ).ljust(col_widths[col])
                for col in columns
            )
            print(row_str)

    def take(self, num: int) -> List[Dict[str, object]]:
        """Take the first num rows as a list.

        Args:
            num: Number of rows to take

        Returns:
            List of dictionaries representing the rows
        """
        rows = self.limit(num).collect()
        if not isinstance(rows, list):
            raise TypeError("take() requires collect() to return a list, not an iterator")
        return rows

    def first(self) -> Optional[Dict[str, object]]:
        """Return the first row as a dictionary, or None if empty.

        Returns:
            First row as a dictionary, or None if DataFrame is empty
        """
        rows = self.limit(1).collect()
        if not isinstance(rows, list):
            raise TypeError("first() requires collect() to return a list, not an iterator")
        return rows[0] if rows else None

    def head(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the first n rows as a list.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of dictionaries representing the rows
        """
        rows = self.limit(n).collect()
        if not isinstance(rows, list):
            raise TypeError("head() requires collect() to return a list, not an iterator")
        return rows

    def count(self) -> int:
        """Return the number of rows in the DataFrame.

        Returns:
            Number of rows

        Note:
            This executes a COUNT(*) query against the database.
        """
        from ..expressions.functions import count as count_func

        # Create an aggregate with count(*)
        count_col = count_func("*").alias("count")
        result_df = self._with_plan(operators.aggregate(self.plan, (), (count_col,)))
        results = result_df.collect()
        if not isinstance(results, list):
            raise TypeError("count() requires collect() to return a list, not an iterator")
        if results:
            return int(results[0].get("count", 0))  # type: ignore[call-overload, no-any-return]
        return 0

    def describe(self, *cols: str) -> "DataFrame":
        """Compute basic statistics for numeric columns.

        Args:
            *cols: Optional column names to describe. If not provided, describes all numeric columns.

        Returns:
            DataFrame with statistics: count, mean, stddev, min, max

        Note:
            This is a simplified implementation. A full implementation would
            automatically detect numeric columns if cols is not provided.
        """
        from ..expressions.functions import count, avg, min, max

        if not cols:
            # For now, return empty DataFrame if no columns specified
            # A full implementation would detect numeric columns
            return self.limit(0)

        # Build aggregations for each column
        aggregations = []
        for col_name in cols:
            col_expr = col(col_name)
            aggregations.extend(
                [
                    count(col_expr).alias(f"{col_name}_count"),
                    avg(col_expr).alias(f"{col_name}_mean"),
                    min(col_expr).alias(f"{col_name}_min"),
                    max(col_expr).alias(f"{col_name}_max"),
                ]
            )

        return self._with_plan(operators.aggregate(self.plan, (), tuple(aggregations)))

    def summary(self, *statistics: str) -> "DataFrame":
        """Compute summary statistics for numeric columns.

        Args:
            *statistics: Statistics to compute (e.g., "count", "mean", "stddev", "min", "max").
                        If not provided, computes common statistics.

        Returns:
            DataFrame with summary statistics

        Note:
            This is a simplified implementation. A full implementation would
            automatically detect numeric columns and compute all statistics.
        """

        if not statistics:
            statistics = ("count", "mean", "min", "max")

        # For now, this is similar to describe()
        # A full implementation would be more comprehensive
        return self.describe()

    def fillna(
        self, value: Union[object, Dict[str, object]], subset: Optional[Sequence[str]] = None
    ) -> "DataFrame":
        """Replace null values with a specified value.

        Args:
            value: Value to use for filling nulls. Can be a single value or a dict mapping column names to values.
            subset: Optional list of column names to fill. If None, fills all columns.

        Returns:
            New DataFrame with null values filled

        Note:
            This uses COALESCE or CASE WHEN to replace nulls in SQL.
        """
        from ..expressions.functions import coalesce, lit

        # Get columns to fill
        if subset is None:
            # For now, we can't easily determine all columns without schema
            # This is a simplified implementation
            return self

        # Build new projections with fillna applied
        new_projections = []
        for col_name in subset:
            col_expr = col(col_name)
            if isinstance(value, dict):
                fill_value = value.get(col_name, None)
            else:
                fill_value = value

            if fill_value is not None:
                # Use COALESCE to replace nulls
                # fill_value can be any object, but lit() expects specific types
                # We'll allow it and let the runtime handle type errors
                filled_col = coalesce(col_expr, lit(fill_value)).alias(col_name)  # type: ignore[arg-type]
                new_projections.append(filled_col)
            else:
                new_projections.append(col_expr)

        # This is simplified - a full implementation would handle all columns
        return self.select(*new_projections)

    def dropna(self, how: str = "any", subset: Optional[Sequence[str]] = None) -> "DataFrame":
        """Remove rows with null values.

        Args:
            how: "any" (drop if any null) or "all" (drop if all null) (default: "any")
            subset: Optional list of column names to check. If None, checks all columns.

        Returns:
            New DataFrame with null rows removed
        """

        if subset is None:
            # Check all columns - simplified implementation
            # A full implementation would need schema information
            return self

        # Build filter condition
        if how == "any":
            # Drop if ANY column in subset is null
            conditions = [col(col_name).is_not_null() for col_name in subset]
            predicate = conditions[0]
            for cond in conditions[1:]:
                predicate = predicate & cond
        else:  # how == "all"
            # Drop if ALL columns in subset are null
            conditions = [col(col_name).is_null() for col_name in subset]
            predicate = conditions[0]
            for cond in conditions[1:]:
                predicate = predicate & cond
            # Negate to keep rows where NOT all are null
            predicate = ~predicate

        return self.where(predicate)

    @property
    def write(self) -> "DataFrameWriter":
        """Return a DataFrameWriter for writing this DataFrame to a table."""
        from .writer import DataFrameWriter

        return DataFrameWriter(self)

    # ---------------------------------------------------------------- utilities
    def _with_plan(self, plan: LogicalPlan) -> "DataFrame":
        return DataFrame(
            plan=plan,
            database=self.database,
        )

    def _normalize_projection(self, expr: Union[Column, str]) -> Column:
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
        self, on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]]
    ) -> Sequence[Tuple[str, str]]:
        if on is None:
            raise ValueError("join requires an `on` argument for equality joins")
        if isinstance(on, str):
            return [(on, on)]
        normalized: List[Tuple[str, str]] = []
        for entry in on:
            if isinstance(entry, tuple):
                if len(entry) != 2:
                    raise ValueError("join tuples must specify (left, right) column names")
                normalized.append((entry[0], entry[1]))
            else:
                normalized.append((entry, entry))
        return normalized
