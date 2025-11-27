"""Async lazy DataFrame representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    overload,
)

from ..expressions.column import Column, col
from ..logical import operators
from ..logical.plan import FileScan, LogicalPlan, RawSQL, SortOrder
from ..sql.compiler import compile_plan

if TYPE_CHECKING:
    from ..io.records import AsyncRecords
    from ..table.async_table import AsyncDatabase, AsyncTableHandle
    from ..utils.inspector import ColumnInfo
    from .async_groupby import AsyncGroupedDataFrame
    from .async_pandas_dataframe import AsyncPandasDataFrame
    from .async_polars_dataframe import AsyncPolarsDataFrame
    from .async_writer import AsyncDataFrameWriter


@dataclass(frozen=True)
class AsyncDataFrame:
    """Async lazy DataFrame representation."""

    plan: LogicalPlan
    database: Optional["AsyncDatabase"] = None

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_table(
        cls, table_handle: "AsyncTableHandle", columns: Optional[Sequence[str]] = None
    ) -> "AsyncDataFrame":
        """Create an AsyncDataFrame from a table handle."""
        plan = operators.scan(table_handle.name)
        df = cls(plan=plan, database=table_handle.database)
        if columns:
            df = df.select(*columns)
        return df

    def select(self, *columns: Union[Column, str]) -> "AsyncDataFrame":
        """Select columns from the DataFrame.

        Args:
            *columns: Column names or Column expressions to select.
                     Use "*" to select all columns (same as empty select).
                     Can combine "*" with other columns: select("*", col("new_col"))

        Returns:
            New AsyncDataFrame with selected columns

        Example:
            >>> import asyncio
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("email", "TEXT")]).collect()
            ...     from moltres.io.records import AsyncRecords
            ...     records = AsyncRecords(_data=[{"id": 1, "name": "Alice", "email": "alice@example.com"}], _database=db)
            ...     await records.insert_into("users")
            ...     # Select specific columns
            ...     table_handle = await db.table("users")
            ...     df = table_handle.select("id", "name")
            ...     results = await df.collect()
            ...     results[0]["name"]
            ...     'Alice'
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
        """
        if not columns:
            return self

        # Handle "*" as special case
        if len(columns) == 1 and isinstance(columns[0], str) and columns[0] == "*":
            return self

        # Check if "*" is in the columns (only check string elements, not Column objects)
        has_star = any(isinstance(col, str) and col == "*" for col in columns)

        # Import Column at the top of the method
        from ..expressions.column import Column

        # Normalize all columns first and check for explode
        normalized_columns = []
        explode_column = None

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
            project_columns = []

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

            return self._with_plan(operators.project(exploded_plan, tuple(project_columns)))

        # No explode columns, normal projection
        if has_star and not normalized_columns:
            return self  # Only "*", same as empty select

        return self._with_plan(operators.project(self.plan, tuple(normalized_columns)))

    def selectExpr(self, *exprs: str) -> "AsyncDataFrame":
        """Select columns using SQL expressions (async version).

        This method allows you to write SQL expressions directly instead of
        building Column objects manually, similar to PySpark's selectExpr().

        Args:
            *exprs: SQL expression strings (e.g., "amount * 1.1 as with_tax")

        Returns:
            New AsyncDataFrame with selected expressions

        Example:
            >>> import asyncio
            >>> from moltres import async_connect
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL"), column("name", "TEXT")]).collect()
            ...     from moltres.io.records import AsyncRecords
            ...     records = AsyncRecords(_data=[{"id": 1, "amount": 100.0, "name": "Alice"}], _database=db)
            ...     await records.insert_into("orders")
            ...     # With expressions and aliases
            ...     table_handle = await db.table("orders")
            ...     df = table_handle.select()
            ...     df2 = df.selectExpr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")
            ...     results = await df2.collect()
            ...     results[0]["with_tax"]
            ...     110.0
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
        """
        from ..expressions.sql_parser import parse_sql_expr

        if not exprs:
            return self

        # Get available column names from the DataFrame for context
        available_columns: Optional[Set[str]] = None
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

        # Parse each SQL expression into a Column expression
        parsed_columns = []
        for expr_str in exprs:
            parsed_col = parse_sql_expr(expr_str, available_columns)
            parsed_columns.append(parsed_col)

        # Use the existing select() method with parsed columns
        return self.select(*parsed_columns)

    def where(self, predicate: Union[Column, str]) -> "AsyncDataFrame":
        """Filter rows based on a predicate.

        Args:
            predicate: Column expression or SQL string representing the filter condition.
                      Can be a Column object or a SQL string like "age > 18".

        Returns:
            New AsyncDataFrame with filtered rows

        Example:
            >>> import asyncio
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            ...     from moltres.io.records import AsyncRecords
            ...     records = AsyncRecords(_data=[{"id": 1, "name": "Alice", "age": 25}, {"id": 2, "name": "Bob", "age": 17}], _database=db)
            ...     await records.insert_into("users")
            ...     # Filter by condition using Column
            ...     table_handle = await db.table("users")
            ...     df = table_handle.select().where(col("age") >= 18)
            ...     results = await df.collect()
            ...     len(results)
            ...     1
            ...     results[0]["name"]
            ...     'Alice'
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
        """
        # If predicate is a string, parse it into a Column expression
        if isinstance(predicate, str):
            from ..expressions.sql_parser import parse_sql_expr

            # Get available column names from the DataFrame for context
            available_columns: Optional[Set[str]] = None
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

        return self._with_plan(operators.filter(self.plan, predicate))

    filter = where

    def join(
        self,
        other: "AsyncDataFrame",
        on: Optional[
            Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
        ] = None,
        how: str = "inner",
    ) -> "AsyncDataFrame":
        """Join with another DataFrame.

        Args:
            other: Another DataFrame to join with
            on: Join condition - can be:
                - A single column name (assumes same name in both DataFrames): ``on="order_id"``
                - A sequence of column names (assumes same names in both): ``on=["col1", "col2"]``
                - A sequence of (left_column, right_column) tuples: ``on=[("id", "customer_id")]``
                - A Column expression (PySpark-style): ``on=[col("left_col") == col("right_col")]``
                - A single Column expression: ``on=col("left_col") == col("right_col")``
            how: Join type ("inner", "left", "right", "full", "cross")

        Returns:
            New AsyncDataFrame containing the join result

        Example:
            >>> import asyncio
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("customers", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            ...     await db.create_table("orders", [column("id", "INTEGER"), column("customer_id", "INTEGER"), column("amount", "REAL")]).collect()
            ...     from moltres.io.records import AsyncRecords
            ...     records1 = AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)
            ...     await records1.insert_into("customers")
            ...     records2 = AsyncRecords(_data=[{"id": 1, "customer_id": 1, "amount": 100.0}], _database=db)
            ...     await records2.insert_into("orders")
            ...     # PySpark-style join
            ...     customers_table = await db.table("customers")
            ...     orders_table = await db.table("orders")
            ...     customers_df = customers_table.select()
            ...     orders_df = orders_table.select()
            ...     df = customers_df.join(orders_df, on=[col("customers.id") == col("orders.customer_id")], how="inner")
            ...     results = await df.collect()
            ...     len(results)
            ...     1
            ...     results[0]["name"]
            ...     'Alice'
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to an AsyncDatabase before joining")
        if self.database is not other.database:
            raise ValueError("Cannot join DataFrames from different AsyncDatabase instances")

        # Cross joins don't require an 'on' clause
        if how.lower() == "cross":
            normalized_on = None
            condition = None
        else:
            normalized_condition = self._normalize_join_condition(on)
            if isinstance(normalized_condition, Column):
                # PySpark-style Column expression
                normalized_on = None
                condition = normalized_condition
            else:
                # Tuple-based join (backward compatible)
                normalized_on = normalized_condition
                condition = None
        return self._with_plan(
            operators.join(
                self.plan, other.plan, how=how.lower(), on=normalized_on, condition=condition
            )
        )

    def crossJoin(self, other: "AsyncDataFrame") -> "AsyncDataFrame":
        """Perform a cross join (Cartesian product) with another DataFrame.

        Args:
            other: Another DataFrame to cross join with

        Returns:
            New DataFrame containing the Cartesian product of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same AsyncDatabase
        """
        return self.join(other, how="cross")

    def semi_join(
        self,
        other: "AsyncDataFrame",
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
    ) -> "AsyncDataFrame":
        """Perform a semi-join: return rows from this DataFrame where a matching row exists in other.

        This is equivalent to filtering with EXISTS subquery.

        Args:
            other: Another DataFrame to semi-join with (used as EXISTS subquery)
            on: Join condition - can be:
                - A single column name (assumes same name in both DataFrames)
                - A sequence of column names (assumes same names in both)
                - A sequence of (left_column, right_column) tuples

        Returns:
            New DataFrame containing rows from this DataFrame that have matches in other

        Raises:
            RuntimeError: If DataFrames are not bound to the same AsyncDatabase
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to an AsyncDatabase before semi_join")
        if self.database is not other.database:
            raise ValueError("Cannot semi_join DataFrames from different AsyncDatabase instances")
        normalized_condition = self._normalize_join_condition(on)
        if isinstance(normalized_condition, Column):
            raise ValueError("semi_join does not support Column expressions, use tuple syntax")
        normalized_on = normalized_condition
        plan = operators.semi_join(self.plan, other.plan, on=normalized_on)
        return AsyncDataFrame(plan=plan, database=self.database)

    def anti_join(
        self,
        other: "AsyncDataFrame",
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
    ) -> "AsyncDataFrame":
        """Perform an anti-join: return rows from this DataFrame where no matching row exists in other.

        This is equivalent to filtering with NOT EXISTS subquery.

        Args:
            other: Another DataFrame to anti-join with (used as NOT EXISTS subquery)
            on: Join condition - can be:
                - A single column name (assumes same name in both DataFrames)
                - A sequence of column names (assumes same names in both)
                - A sequence of (left_column, right_column) tuples

        Returns:
            New DataFrame containing rows from this DataFrame that have no matches in other

        Raises:
            RuntimeError: If DataFrames are not bound to the same AsyncDatabase
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to an AsyncDatabase before anti_join")
        if self.database is not other.database:
            raise ValueError("Cannot anti_join DataFrames from different AsyncDatabase instances")
        normalized_condition = self._normalize_join_condition(on)
        if isinstance(normalized_condition, Column):
            raise ValueError("anti_join does not support Column expressions, use tuple syntax")
        normalized_on = normalized_condition
        plan = operators.anti_join(self.plan, other.plan, on=normalized_on)
        return AsyncDataFrame(plan=plan, database=self.database)

    def group_by(self, *columns: Union[Column, str]) -> "AsyncGroupedDataFrame":
        """Group by the specified columns.

        Example:
            >>> import asyncio
            >>> from moltres import async_connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("sales", [column("category", "TEXT"), column("amount", "REAL")]).collect()
            ...     from moltres.io.records import AsyncRecords
            ...     records = AsyncRecords(_data=[{"category": "A", "amount": 100.0}, {"category": "A", "amount": 200.0}, {"category": "B", "amount": 150.0}], _database=db)
            ...     await records.insert_into("sales")
            ...     table_handle = await db.table("sales")
            ...     df = table_handle.select()
            ...     grouped = df.group_by("category")
            ...     result = grouped.agg(F.sum(col("amount")).alias("total"))
            ...     results = await result.collect()
            ...     len(results)
            ...     2
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
        """
        from .async_groupby import AsyncGroupedDataFrame

        normalized = tuple(self._normalize_projection(col) for col in columns)
        grouped_plan = operators.aggregate(self.plan, keys=normalized, aggregates=())
        return AsyncGroupedDataFrame(plan=grouped_plan, database=self.database)

    groupBy = group_by

    def order_by(self, *columns: Union[Column, str]) -> "AsyncDataFrame":
        """Sort rows by one or more columns.

        Args:
            *columns: Column expressions or column names to sort by. Use .asc() or .desc() for sort order.
                     Can be strings (column names) or Column objects.

        Returns:
            New AsyncDataFrame with sorted rows

        Example:
            >>> from moltres import col
            >>> # Sort ascending with string column name
            >>> df = await db.table("users").select().order_by("name")
            >>> # SQL: SELECT * FROM users ORDER BY name

            >>> # Sort ascending with Column object
            >>> df = await db.table("users").select().order_by(col("name"))
            >>> # SQL: SELECT * FROM users ORDER BY name

            >>> # Sort descending
            >>> df = await db.table("orders").select().order_by(col("amount").desc())
            >>> # SQL: SELECT * FROM orders ORDER BY amount DESC

            >>> # Multiple sort columns (mixed string and Column)
            >>> df = (
            ...     await db.table("sales")
            ...     .select()
            ...     .order_by("region", col("amount").desc())
            ... )
            >>> # SQL: SELECT * FROM sales ORDER BY region, amount DESC
        """
        sort_orders = tuple(
            self._normalize_sort_expression(self._normalize_projection(col)) for col in columns
        )
        return self._with_plan(operators.order_by(self.plan, sort_orders))

    orderBy = order_by
    sort = order_by  # PySpark-style alias

    def limit(self, count: int) -> "AsyncDataFrame":
        """Limit the number of rows returned."""
        if count < 0:
            raise ValueError("limit() requires a non-negative integer")
        return self._with_plan(operators.limit(self.plan, count))

    def sample(self, fraction: float, seed: Optional[int] = None) -> "AsyncDataFrame":
        """Sample a fraction of rows from the DataFrame.

        Args:
            fraction: Fraction of rows to sample (0.0 to 1.0)
            seed: Optional random seed for reproducible sampling

        Returns:
            New AsyncDataFrame with sampled rows

        Example:
            >>> df = await db.table("users").select().sample(0.1)  # Sample 10% of rows
            >>> # SQL (PostgreSQL): SELECT * FROM users TABLESAMPLE BERNOULLI(10)
            >>> # SQL (SQLite): SELECT * FROM users ORDER BY RANDOM() LIMIT (COUNT(*) * 0.1)
        """
        return self._with_plan(operators.sample(self.plan, fraction, seed))

    def union(self, other: "AsyncDataFrame") -> "AsyncDataFrame":
        """Union this DataFrame with another DataFrame (distinct rows only)."""
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to an AsyncDatabase before union")
        if self.database is not other.database:
            raise ValueError("Cannot union DataFrames from different AsyncDatabase instances")
        plan = operators.union(self.plan, other.plan, distinct=True)
        return AsyncDataFrame(plan=plan, database=self.database)

    def unionAll(self, other: "AsyncDataFrame") -> "AsyncDataFrame":
        """Union this DataFrame with another DataFrame (all rows, including duplicates)."""
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to an AsyncDatabase before union")
        if self.database is not other.database:
            raise ValueError("Cannot union DataFrames from different AsyncDatabase instances")
        plan = operators.union(self.plan, other.plan, distinct=False)
        return AsyncDataFrame(plan=plan, database=self.database)

    def intersect(self, other: "AsyncDataFrame") -> "AsyncDataFrame":
        """Intersect this DataFrame with another DataFrame (distinct rows only)."""
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to an AsyncDatabase before intersect")
        if self.database is not other.database:
            raise ValueError("Cannot intersect DataFrames from different AsyncDatabase instances")
        plan = operators.intersect(self.plan, other.plan, distinct=True)
        return AsyncDataFrame(plan=plan, database=self.database)

    def except_(self, other: "AsyncDataFrame") -> "AsyncDataFrame":
        """Return rows in this DataFrame that are not in another DataFrame (distinct rows only)."""
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to an AsyncDatabase before except")
        if self.database is not other.database:
            raise ValueError("Cannot except DataFrames from different AsyncDatabase instances")
        plan = operators.except_(self.plan, other.plan, distinct=True)
        return AsyncDataFrame(plan=plan, database=self.database)

    def cte(self, name: str) -> "AsyncDataFrame":
        """Create a Common Table Expression (CTE) from this DataFrame.

        Args:
            name: Name for the CTE

        Returns:
            New AsyncDataFrame representing the CTE
        """
        plan = operators.cte(self.plan, name)
        return AsyncDataFrame(plan=plan, database=self.database)

    def distinct(self) -> "AsyncDataFrame":
        """Return a new DataFrame with distinct rows."""
        return self._with_plan(operators.distinct(self.plan))

    def dropDuplicates(self, subset: Optional[Sequence[str]] = None) -> "AsyncDataFrame":
        """Return a new DataFrame with duplicate rows removed."""
        if subset is None:
            return self.distinct()
        # Simplified implementation
        return self.group_by(*subset).agg()

    def withColumn(self, colName: str, col_expr: Union[Column, str]) -> "AsyncDataFrame":
        """Add or replace a column in the DataFrame.

        Args:
            colName: Name of the column to add or replace
            col_expr: Column expression or column name

        Returns:
            New AsyncDataFrame with the added/replaced column

        Note:
            This operation adds a Project on top of the current plan.
            If a column with the same name exists, it will be replaced.
            Window functions are supported and will ensure all columns are available.

        Example:
            >>> from moltres.expressions import functions as F
            >>> from moltres.expressions.window import Window
            >>> window = Window.partition_by("category").order_by("amount")
            >>> await df.withColumn("row_num", F.row_number().over(window)).collect()
        """
        from ..expressions.column import Column
        from ..logical.plan import Project
        from dataclasses import replace as dataclass_replace

        new_col = self._normalize_projection(col_expr)
        if isinstance(new_col, Column) and not new_col._alias:
            new_col = new_col.alias(colName)
        elif isinstance(new_col, Column):
            new_col = dataclass_replace(new_col, _alias=colName)

        # Check if this is a window function
        is_window_func = isinstance(new_col, Column) and self._is_window_function(new_col)

        if isinstance(self.plan, Project):
            # Add the new column to existing projections
            # Remove any column with the same name (replace behavior)
            existing_cols = []
            has_star = False
            for col_expr in self.plan.projections:
                if isinstance(col_expr, Column):
                    # Check if this is a star column
                    if col_expr.op == "star":
                        has_star = True
                        # For window functions, keep the star to ensure all columns are available
                        if is_window_func:
                            existing_cols.append(col_expr)
                        continue
                    # Check if this column matches the colName (by alias or column name)
                    col_alias = col_expr._alias
                    col_name = (
                        col_expr.args[0] if col_expr.op == "column" and col_expr.args else None
                    )
                    if col_alias == colName or col_name == colName:
                        # Skip this column - it will be replaced by new_col
                        continue
                existing_cols.append(col_expr)

            # For window functions, ensure we have a star column if we don't already have one
            # This ensures all columns are available for the window function
            if is_window_func and not has_star:
                star_col = Column(op="star", args=(), _alias=None)
                new_projections = [star_col] + existing_cols + [new_col]
            else:
                # Add the new column at the end
                new_projections = existing_cols + [new_col]
        else:
            # No existing projection, select all plus new column
            # Use a wildcard select and add the new column
            from ..expressions.column import Column

            star_col = Column(op="star", args=(), _alias=None)
            new_projections = [star_col, new_col]

        return self._with_plan(operators.project(self.plan, tuple(new_projections)))

    def withColumnRenamed(self, existing: str, new: str) -> "AsyncDataFrame":
        """Rename a column in the DataFrame."""
        from ..logical.plan import Project
        from dataclasses import replace as dataclass_replace

        if isinstance(self.plan, Project):
            new_projections = []
            for col_expr in self.plan.projections:
                if isinstance(col_expr, Column):
                    if col_expr._alias == existing or (
                        col_expr.op == "column" and col_expr.args[0] == existing
                    ):
                        new_col = dataclass_replace(col_expr, _alias=new)
                        new_projections.append(new_col)
                    else:
                        new_projections.append(col_expr)
                else:
                    new_projections.append(col_expr)
            return self._with_plan(operators.project(self.plan.child, tuple(new_projections)))
        else:
            existing_col = col(existing).alias(new)
            return self._with_plan(operators.project(self.plan, (existing_col,)))

    def drop(self, *cols: Union[str, Column]) -> "AsyncDataFrame":
        """Drop one or more columns from the DataFrame.

        Args:
            *cols: Column names or Column objects to drop

        Returns:
            New AsyncDataFrame with the specified columns removed

        Example:
            >>> # Drop by string column name
            >>> await df.drop("col1", "col2").collect()
            >>> # Drop by Column object
            >>> await df.drop(col("col1"), col("col2")).collect()
            >>> # Mixed usage
            >>> await df.drop("col1", col("col2")).collect()
        """
        from ..logical.plan import Project

        # Extract column names from both strings and Column objects
        cols_to_drop = set()
        for col_expr in cols:
            if isinstance(col_expr, str):
                cols_to_drop.add(col_expr)
            elif isinstance(col_expr, Column):
                col_name = self._extract_column_name(col_expr)
                if col_name:
                    cols_to_drop.add(col_name)

        if isinstance(self.plan, Project):
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
            return self

    async def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the DataFrame."""
        rows = await self.limit(n).collect()
        # collect() with stream=False returns a list, not an iterator
        if not isinstance(rows, list):
            raise TypeError("show() requires collect() to return a list, not an iterator")
        if not rows:
            print("Empty DataFrame")
            return

        columns = list(rows[0].keys())
        col_widths = {}
        for col_name in columns:
            col_widths[col_name] = len(col_name)
            for row in rows:
                val_str = str(row.get(col_name, ""))
                if truncate and len(val_str) > 20:
                    val_str = val_str[:17] + "..."
                col_widths[col_name] = max(col_widths[col_name], len(val_str))

        header = " | ".join(col_name.ljust(col_widths[col_name]) for col_name in columns)
        print(header)
        print("-" * len(header))

        for row in rows:
            row_str = " | ".join(
                (
                    str(row.get(col_name, ""))[:17] + "..."
                    if truncate and len(str(row.get(col_name, ""))) > 20
                    else str(row.get(col_name, ""))
                ).ljust(col_widths[col_name])
                for col_name in columns
            )
            print(row_str)

    async def take(self, num: int) -> List[Dict[str, object]]:
        """Take the first num rows as a list."""
        rows = await self.limit(num).collect()
        if not isinstance(rows, list):
            raise TypeError("take() requires collect() to return a list, not an iterator")
        return rows

    async def first(self) -> Optional[Dict[str, object]]:
        """Return the first row as a dictionary, or None if empty."""
        rows = await self.limit(1).collect()
        if not isinstance(rows, list):
            raise TypeError("first() requires collect() to return a list, not an iterator")
        return rows[0] if rows else None

    async def head(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the first n rows as a list."""
        rows = await self.limit(n).collect()
        if not isinstance(rows, list):
            raise TypeError("head() requires collect() to return a list, not an iterator")
        return rows

    async def count(self) -> int:
        """Return the number of rows in the DataFrame."""
        from ..expressions.functions import count as count_func

        count_col = count_func("*").alias("count")
        result_df = self._with_plan(operators.aggregate(self.plan, (), (count_col,)))
        results = await result_df.collect()
        if not isinstance(results, list):
            raise TypeError("count() requires collect() to return a list, not an iterator")
        if results:
            return int(results[0].get("count", 0))  # type: ignore[call-overload, no-any-return]
        return 0

    async def describe(self, *cols: str) -> "AsyncDataFrame":
        """Compute basic statistics for numeric columns."""
        from ..expressions.functions import count, avg, min, max

        if not cols:
            return self.limit(0)

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

    async def summary(self, *statistics: str) -> "AsyncDataFrame":
        """Compute summary statistics for numeric columns."""
        return await self.describe()

    def fillna(
        self, value: Union[object, Dict[str, object]], subset: Optional[Sequence[str]] = None
    ) -> "AsyncDataFrame":
        """Replace null values with a specified value."""
        from ..expressions.functions import coalesce, lit

        if subset is None:
            return self

        new_projections = []
        for col_name in subset:
            col_expr = col(col_name)
            if isinstance(value, dict):
                fill_value = value.get(col_name, None)
            else:
                fill_value = value

            if fill_value is not None:
                # fill_value can be any object, but lit() expects specific types
                # We'll allow it and let the runtime handle type errors
                filled_col = coalesce(col_expr, lit(fill_value)).alias(col_name)  # type: ignore[arg-type]
                new_projections.append(filled_col)
            else:
                new_projections.append(col_expr)

        return self.select(*new_projections)

    def dropna(self, how: str = "any", subset: Optional[Sequence[str]] = None) -> "AsyncDataFrame":
        """Remove rows with null values."""
        if subset is None:
            return self

        if how == "any":
            conditions = [col(col_name).is_not_null() for col_name in subset]
            predicate = conditions[0]
            for cond in conditions[1:]:
                predicate = predicate & cond
        else:
            conditions = [col(col_name).is_null() for col_name in subset]
            predicate = conditions[0]
            for cond in conditions[1:]:
                predicate = predicate & cond
            predicate = ~predicate

        return self.where(predicate)

    # ---------------------------------------------------------------- execution
    def to_sql(self) -> str:
        """Compile the DataFrame plan to SQL."""
        from sqlalchemy.sql import Select

        stmt = (
            self.database.compile_plan(self.plan)
            if self.database is not None
            else compile_plan(self.plan)
        )
        if isinstance(stmt, Select):
            return str(stmt.compile(compile_kwargs={"literal_binds": True}))
        return str(stmt)

    @overload
    async def collect(self, stream: Literal[False] = False) -> List[Dict[str, object]]: ...

    @overload
    async def collect(self, stream: Literal[True]) -> AsyncIterator[List[Dict[str, object]]]: ...

    async def collect(
        self, stream: bool = False
    ) -> Union[List[Dict[str, object]], AsyncIterator[List[Dict[str, object]]]]:
        """Collect DataFrame results asynchronously.

        Args:
            stream: If True, return an async iterator of row chunks. If False (default),
                   materialize all rows into a list.

        Returns:
            If stream=False: List of dictionaries representing rows.
            If stream=True: AsyncIterator of row chunks (each chunk is a list of dicts).

        Raises:
            RuntimeError: If DataFrame is not bound to an AsyncDatabase

        Example:
            >>> import asyncio
            >>> from moltres import async_connect
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            ...     from moltres.io.records import AsyncRecords
            ...     records = AsyncRecords(_data=[{"id": 1, "name": "Alice"}], _database=db)
            ...     await records.insert_into("users")
            ...     table_handle = await db.table("users")
            ...     df = table_handle.select()
            ...     # Collect results (non-streaming)
            ...     results = await df.collect()
            ...     len(results)
            ...     1
            ...     results[0]["name"]
            ...     'Alice'
            ...     # Collect results (streaming)
            ...     async for chunk in await df.collect(stream=True):
            ...         pass  # Process chunks
            ...     await db.close()
            ...     # asyncio.run(example())  # doctest: +SKIP
        """
        if self.database is None:
            raise RuntimeError("Cannot collect a plan without an attached AsyncDatabase")

        # Handle RawSQL at root level - execute directly for efficiency
        if isinstance(self.plan, RawSQL):
            if stream:
                # For streaming, we need to use execute_plan_stream which expects a compiled plan
                # So we'll compile the RawSQL plan
                plan = await self._materialize_filescan(self.plan)

                async def stream_gen() -> AsyncIterator[List[Dict[str, object]]]:
                    async for chunk in self.database.execute_plan_stream(plan):  # type: ignore[union-attr]
                        yield chunk

                return stream_gen()
            else:
                # Execute RawSQL directly
                result = await self.database.execute_sql(self.plan.sql, params=self.plan.params)
                if result.rows is None:
                    return []
                # Convert to list if it's a DataFrame
                if hasattr(result.rows, "to_dict"):
                    records = result.rows.to_dict("records")  # type: ignore[call-overload]
                    # Convert Hashable keys to str keys
                    return [{str(k): v for k, v in row.items()} for row in records]
                if hasattr(result.rows, "to_dicts"):
                    records = list(result.rows.to_dicts())
                    # Convert Hashable keys to str keys
                    return [{str(k): v for k, v in row.items()} for row in records]
                return result.rows

        # Handle FileScan by materializing file data into a temporary table
        plan = await self._materialize_filescan(self.plan)

        if stream:
            # For SQL queries, use streaming execution
            if self.database is None:
                raise RuntimeError("Cannot collect a plan without an attached AsyncDatabase")

            async def stream_gen() -> AsyncIterator[List[Dict[str, object]]]:
                async for chunk in self.database.execute_plan_stream(plan):  # type: ignore[union-attr]
                    yield chunk

            return stream_gen()

        result = await self.database.execute_plan(plan)
        if result.rows is None:
            return []
        # Convert to list if it's a DataFrame
        if hasattr(result.rows, "to_dict"):
            records = result.rows.to_dict("records")  # type: ignore[call-overload]
            # Convert Hashable keys to str keys
            return [{str(k): v for k, v in row.items()} for row in records]
        if hasattr(result.rows, "to_dicts"):
            records = list(result.rows.to_dicts())
            # Convert Hashable keys to str keys
            return [{str(k): v for k, v in row.items()} for row in records]
        return result.rows

    async def _materialize_filescan(self, plan: LogicalPlan) -> LogicalPlan:
        """Materialize FileScan nodes by reading files and creating temporary tables.

        When a FileScan is encountered, the file is read, materialized into a temporary
        table, and the FileScan is replaced with a TableScan.

        By default, files are read in chunks (streaming mode) to safely handle large files
        without loading everything into memory. Set stream=False in options to use
        in-memory reading for small files.

        Args:
            plan: Logical plan that may contain FileScan nodes

        Returns:
            Logical plan with FileScan nodes replaced by TableScan nodes
        """
        from dataclasses import replace

        if self.database is None:
            raise RuntimeError("Cannot materialize FileScan without an attached AsyncDatabase")

        if isinstance(plan, FileScan):
            # Check if streaming is disabled (opt-out mechanism)
            # Default is True (streaming/chunked reading) for safety with large files
            stream_enabled = plan.options.get("stream", True)
            if isinstance(stream_enabled, bool) and not stream_enabled:
                # Non-streaming mode: load entire file into memory (current behavior)
                rows = await self._read_file(plan)

                # Materialize into temporary table using createDataFrame
                # This enables SQL pushdown for subsequent operations
                # Use auto_pk to create an auto-incrementing primary key for temporary tables
                temp_df = await self.database.createDataFrame(
                    rows, schema=plan.schema, auto_pk="__moltres_rowid__"
                )

                # createDataFrame returns an AsyncDataFrame with a TableScan plan
                # Return the TableScan plan to replace the FileScan
                return temp_df.plan
            else:
                # Streaming mode (default): read file in chunks and insert incrementally
                from ..dataframe.create_dataframe import create_temp_table_from_streaming_async
                from ..logical.operators import scan

                # Read file using streaming readers
                records = await self._read_file_streaming(plan)

                # Create temp table from streaming records (chunked insertion)
                table_name, final_schema = await create_temp_table_from_streaming_async(
                    self.database,
                    records,
                    schema=plan.schema,
                    auto_pk="__moltres_rowid__",
                )

                # Return TableScan plan to replace the FileScan
                return scan(table_name)

        # Recursively handle children
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
            Pivot,
            Project,
            RecursiveCTE,
            Sample,
            SemiJoin,
            Sort,
            Union,
        )

        # RawSQL doesn't need materialization - it's handled directly in collect()
        if isinstance(plan, RawSQL):
            return plan

        if isinstance(
            plan, (Project, Filter, Limit, Sample, Sort, Distinct, Aggregate, Explode, Pivot)
        ):
            child = await self._materialize_filescan(plan.child)
            return replace(plan, child=child)
        elif isinstance(plan, (Join, Union, Intersect, Except, SemiJoin, AntiJoin)):
            left = await self._materialize_filescan(plan.left)
            right = await self._materialize_filescan(plan.right)
            return replace(plan, left=left, right=right)
        elif isinstance(plan, (CTE, RecursiveCTE)):
            # For CTEs, we need to handle the child
            if isinstance(plan, CTE):
                child = await self._materialize_filescan(plan.child)
                return replace(plan, child=child)
            else:  # RecursiveCTE
                initial = await self._materialize_filescan(plan.initial)
                recursive = await self._materialize_filescan(plan.recursive)
                return replace(plan, initial=initial, recursive=recursive)

        # For other plan types, return as-is
        return plan

    async def _read_file(self, filescan: FileScan) -> List[Dict[str, object]]:
        """Read a file based on FileScan configuration (non-streaming, loads all into memory).

        Args:
            filescan: FileScan logical plan node

        Returns:
            List of dictionaries representing the file data

        Note:
            This method loads the entire file into memory. For large files, use
            _read_file_streaming() instead.
        """
        if self.database is None:
            raise RuntimeError("Cannot read file without an attached AsyncDatabase")

        from ..dataframe.readers.async_csv_reader import read_csv
        from ..dataframe.readers.async_json_reader import read_json, read_jsonl
        from ..dataframe.readers.async_text_reader import read_text

        if filescan.format == "csv":
            records = await read_csv(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "json":
            records = await read_json(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "jsonl":
            records = await read_jsonl(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "parquet":
            # Lazy import for parquet
            try:
                from ..dataframe.readers.async_parquet_reader import read_parquet
            except ImportError:
                raise ImportError(
                    "Parquet support requires pyarrow. Install with: pip install pyarrow"
                )
            records = await read_parquet(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "text":
            records = await read_text(
                filescan.path,
                self.database,
                filescan.schema,
                filescan.options,
                filescan.column_name or "value",
            )
        else:
            raise ValueError(f"Unsupported file format: {filescan.format}")

        # AsyncRecords.rows() returns a coroutine, so we need to await it
        return await records.rows()

    async def _read_file_streaming(self, filescan: FileScan) -> "AsyncRecords":
        """Read a file in streaming mode (chunked, safe for large files).

        Args:
            filescan: FileScan logical plan node

        Returns:
            AsyncRecords object with _generator set (streaming mode)

        Note:
            This method returns AsyncRecords with a generator, allowing chunked processing
            without loading the entire file into memory. Use this for large files.
        """
        if self.database is None:
            raise RuntimeError("Cannot read file without an attached AsyncDatabase")

        from ..dataframe.readers.async_csv_reader import read_csv_stream
        from ..dataframe.readers.async_json_reader import read_json_stream, read_jsonl_stream
        from ..dataframe.readers.async_text_reader import read_text_stream

        if filescan.format == "csv":
            records = await read_csv_stream(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "json":
            records = await read_json_stream(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "jsonl":
            records = await read_jsonl_stream(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "parquet":
            # Lazy import for parquet
            try:
                from ..dataframe.readers.async_parquet_reader import read_parquet_stream
            except ImportError:
                raise ImportError(
                    "Parquet support requires pyarrow. Install with: pip install pyarrow"
                )
            records = await read_parquet_stream(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "text":
            records = await read_text_stream(
                filescan.path,
                self.database,
                filescan.schema,
                filescan.options,
                filescan.column_name or "value",
            )
        else:
            raise ValueError(f"Unsupported file format: {filescan.format}")

        return records

    @property
    def na(self) -> "AsyncNullHandling":
        """Access null handling methods via the `na` property.

        Returns:
            AsyncNullHandling helper object with drop() and fill() methods

        Example:
            >>> await df.na.drop().collect()  # Drop rows with nulls
            >>> await df.na.fill(0).collect()  # Fill nulls with 0
        """
        return AsyncNullHandling(self)

    @property
    def write(self) -> "AsyncDataFrameWriter":
        """Return an AsyncDataFrameWriter for writing this DataFrame to a table."""
        from .async_writer import AsyncDataFrameWriter

        return AsyncDataFrameWriter(self)

    @property
    def columns(self) -> List[str]:
        """Return a list of column names in this DataFrame.

        Similar to PySpark's DataFrame.columns property, this extracts column
        names from the logical plan without requiring query execution.

        Returns:
            List of column name strings

        Raises:
            RuntimeError: If column names cannot be determined (e.g., RawSQL without execution)

        Example:
            >>> df = await db.table("users").select()
            >>> print(df.columns)  # ['id', 'name', 'email', ...]
            >>> df2 = df.select("id", "name")
            >>> print(df2.columns)  # ['id', 'name']
        """
        return self._extract_column_names(self.plan)

    @property
    def schema(self) -> List["ColumnInfo"]:
        """Return the schema of this DataFrame as a list of ColumnInfo objects.

        Similar to PySpark's DataFrame.schema property, this extracts column
        names and types from the logical plan without requiring query execution.

        Returns:
            List of ColumnInfo objects with column names and types

        Raises:
            RuntimeError: If schema cannot be determined (e.g., RawSQL without execution)

        Example:
            >>> df = await db.table("users").select()
            >>> schema = df.schema
            >>> for col_info in schema:
            ...     print(f"{col_info.name}: {col_info.type_name}")
            # id: INTEGER
            # name: VARCHAR(255)
            # email: VARCHAR(255)
        """
        return self._extract_schema_from_plan(self.plan)

    @property
    def dtypes(self) -> List[Tuple[str, str]]:
        """Return a list of tuples containing column names and their data types.

        Similar to PySpark's DataFrame.dtypes property, this returns a list
        of (column_name, type_name) tuples.

        Returns:
            List of tuples (column_name, type_name)

        Raises:
            RuntimeError: If schema cannot be determined (e.g., RawSQL without execution)

        Example:
            >>> df = await db.table("users").select()
            >>> print(df.dtypes)
            # [('id', 'INTEGER'), ('name', 'VARCHAR(255)'), ('email', 'VARCHAR(255)')]
        """
        schema = self.schema
        return [(col_info.name, col_info.type_name) for col_info in schema]

    def printSchema(self) -> None:
        """Print the schema of this DataFrame in a tree format.

        Similar to PySpark's DataFrame.printSchema() method, this prints
        a formatted representation of the DataFrame's schema.

        Example:
            >>> df = await db.table("users").select()
            >>> df.printSchema()
            # root
            #  |-- id: INTEGER (nullable = true)
            #  |-- name: VARCHAR(255) (nullable = true)
            #  |-- email: VARCHAR(255) (nullable = true)

        Note:
            Currently, nullable information is not available from the schema,
            so it's always shown as `nullable = true`.
        """
        schema = self.schema
        print("root")
        for col_info in schema:
            # Format similar to PySpark: |-- column_name: type_name (nullable = true)
            print(f" |-- {col_info.name}: {col_info.type_name} (nullable = true)")

    def __getattr__(self, name: str) -> Column:
        """Enable dot notation column access (e.g., df.id, df.name).

        This method is called when attribute lookup fails. It allows accessing
        columns via dot notation, similar to PySpark's API.

        Args:
            name: Column name to access

        Returns:
            Column object for the specified column name

        Raises:
            AttributeError: If the attribute doesn't exist and isn't a valid column name

        Example:
            >>> df = await db.table("users").select()
            >>> df.select(df.id, df.name)  # Dot notation
            >>> df.where(df.age > 18)  # In filter expressions
        """
        # Check if it's a dataclass field or existing attribute first
        # This prevents conflicts with actual attributes like 'plan', 'database'
        if hasattr(self.__class__, name):
            # Check if it's a dataclass field
            import dataclasses

            if name in {f.name for f in dataclasses.fields(self)}:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )
            # Check if it's a method or property
            attr = getattr(self.__class__, name, None)
            if attr is not None:
                raise AttributeError(
                    f"'{self.__class__.__name__}' object has no attribute '{name}'"
                )

        # If we get here, treat it as a column name
        from ..expressions.column import col

        return col(name)

    # ---------------------------------------------------------------- utilities
    def _with_plan(self, plan: LogicalPlan) -> "AsyncDataFrame":
        """Create a new AsyncDataFrame with a different plan."""
        return AsyncDataFrame(
            plan=plan,
            database=self.database,
        )

    def _normalize_projection(self, expr: Union[Column, str]) -> Column:
        """Normalize a projection expression to a Column."""
        if isinstance(expr, Column):
            return expr
        return col(expr)

    def _is_window_function(self, col_expr: Column) -> bool:
        """Check if a Column expression is a window function.

        Args:
            col_expr: Column expression to check

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
        """Extract column name from a Column expression.

        Args:
            col_expr: Column expression to extract name from

        Returns:
            Column name string, or None if cannot be determined
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
        from ..logical.plan import (
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

        # If this is a Project, check if child is also a Project
        if isinstance(plan, Project):
            child_base = self._find_base_plan(plan.child)
            # If child is also a Project, return the child (more specific)
            if isinstance(child_base, Project):
                return child_base
            # Otherwise, return this Project (it's the final projection)
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
        from ..logical.plan import (
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
            for proj in base_plan.projections:
                col_name = self._extract_column_name(proj)
                if col_name:
                    column_names.append(col_name)
                elif proj.op == "star":
                    # For "*", need to get all columns from underlying plan
                    child_names = self._extract_column_names(base_plan.child)
                    column_names.extend(child_names)
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
            from ..utils.inspector import get_table_columns

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
        from ..logical.plan import (
            Aggregate,
            Explode,
            FileScan,
            Join,
            Project,
            RawSQL,
            TableScan,
        )
        from ..utils.inspector import ColumnInfo

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
            from ..utils.inspector import get_table_columns

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
        """Normalize a sort expression to a SortOrder."""
        if expr.op == "sort_desc":
            return operators.sort_order(expr.args[0], descending=True)
        if expr.op == "sort_asc":
            return operators.sort_order(expr.args[0], descending=False)
        return operators.sort_order(expr, descending=False)

    def _normalize_join_condition(
        self,
        on: Optional[
            Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
        ],
    ) -> Union[Sequence[Tuple[str, str]], Column]:
        """Normalize join condition to either tuple pairs or a Column expression.

        Returns:
            - Sequence[Tuple[str, str]]: For tuple/string-based joins (backward compatible)
            - Column: For PySpark-style Column expression joins
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


class AsyncNullHandling:
    """Helper class for null handling operations on AsyncDataFrames.

    Accessed via the `na` property on AsyncDataFrame instances.
    """

    def __init__(self, df: AsyncDataFrame):
        self._df = df

    def drop(self, how: str = "any", subset: Optional[Sequence[str]] = None) -> AsyncDataFrame:
        """Drop rows with null values.

        This is a convenience wrapper around AsyncDataFrame.dropna().

        Args:
            how: "any" (drop if any null) or "all" (drop if all null) (default: "any")
            subset: Optional list of column names to check. If None, checks all columns.

        Returns:
            New AsyncDataFrame with null rows removed

        Example:
            >>> await df.na.drop().collect()  # Drop rows with any null values
            >>> await df.na.drop(how="all").collect()  # Drop rows where all values are null
            >>> await df.na.drop(subset=["col1", "col2"]).collect()  # Only check specific columns
        """
        return self._df.dropna(how=how, subset=subset)

    def fill(
        self, value: Union[object, Dict[str, object]], subset: Optional[Sequence[str]] = None
    ) -> AsyncDataFrame:
        """Fill null values with a specified value.

        This is a convenience wrapper around AsyncDataFrame.fillna().

        Args:
            value: Value to use for filling nulls. Can be a single value or a dict mapping column names to values.
            subset: Optional list of column names to fill. If None, fills all columns.

        Returns:
            New AsyncDataFrame with null values filled

        Example:
            >>> await df.na.fill(0).collect()  # Fill all nulls with 0
            >>> await df.na.fill({"col1": 0, "col2": "unknown"}).collect()  # Fill different columns with different values
            >>> await df.na.fill(0, subset=["col1", "col2"]).collect()  # Fill specific columns with 0
        """
        return self._df.fillna(value=value, subset=subset)

    def polars(self) -> "AsyncPolarsDataFrame":
        """Convert this AsyncDataFrame to an AsyncPolarsDataFrame for Polars-style operations.

        Returns:
            AsyncPolarsDataFrame wrapping this AsyncDataFrame

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> df = await db.load.csv("data.csv")
            >>> polars_df = df.polars()
            >>> results = await polars_df.collect()
        """
        from .async_polars_dataframe import AsyncPolarsDataFrame

        return AsyncPolarsDataFrame.from_dataframe(self._df)

    def pandas(self) -> "AsyncPandasDataFrame":
        """Convert this AsyncDataFrame to an AsyncPandasDataFrame for Pandas-style operations.

        Returns:
            AsyncPandasDataFrame wrapping this AsyncDataFrame

        Example:
            >>> from moltres import async_connect
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> df = await db.load.csv("data.csv")
            >>> pandas_df = df.pandas()
            >>> results = await pandas_df.collect()
        """
        from .async_pandas_dataframe import AsyncPandasDataFrame

        return AsyncPandasDataFrame.from_dataframe(self._df)
