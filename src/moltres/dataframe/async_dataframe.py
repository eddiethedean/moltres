"""Async lazy DataFrame representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    overload,
)

from ..expressions.column import Column, col
from ..logical import operators
from ..logical.plan import FileScan, LogicalPlan, RawSQL
from ..sql.compiler import compile_plan
from .dataframe_helpers import DataFrameHelpersMixin

if TYPE_CHECKING:
    from sqlalchemy.sql import Select
    from ..io.records import AsyncRecords
    from ..table.async_table import AsyncDatabase, AsyncTableHandle
    from ..utils.inspector import ColumnInfo
    from .async_groupby import AsyncGroupedDataFrame
    from .async_pandas_dataframe import AsyncPandasDataFrame
    from .async_polars_dataframe import AsyncPolarsDataFrame
    from .async_writer import AsyncDataFrameWriter
    from .pyspark_column import PySparkColumn


@dataclass(frozen=True)
class AsyncDataFrame(DataFrameHelpersMixin):
    """Async lazy DataFrame representation."""

    plan: LogicalPlan
    database: Optional["AsyncDatabase"] = None
    model: Optional[Type[Any]] = None  # SQLModel class, if attached

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_table(
        cls, table_handle: "AsyncTableHandle", columns: Optional[Sequence[str]] = None
    ) -> "AsyncDataFrame":
        """Create an AsyncDataFrame from a table handle."""
        plan = operators.scan(table_handle.name)
        # Check if table_handle has a model attached (SQLModel, Pydantic, or SQLAlchemy)
        model = None
        if hasattr(table_handle, "model") and table_handle.model is not None:
            # Check if it's a SQLModel or Pydantic model
            from ..utils.sqlmodel_integration import is_model_class

            if is_model_class(table_handle.model):
                model = table_handle.model
        df = cls(plan=plan, database=table_handle.database, model=model)
        if columns:
            df = df.select(*columns)
        return df

    @classmethod
    def from_sqlalchemy(
        cls, select_stmt: "Select", database: Optional["AsyncDatabase"] = None
    ) -> "AsyncDataFrame":
        """Create an AsyncDataFrame from a SQLAlchemy Select statement.

        This allows you to integrate existing SQLAlchemy queries with Moltres
        AsyncDataFrame operations. The SQLAlchemy statement is wrapped as a RawSQL
        logical plan, which can then be further chained with Moltres operations.

        Args:
            select_stmt: SQLAlchemy Select statement to convert
            database: Optional AsyncDatabase instance to attach to the DataFrame.
                     If provided, allows the DataFrame to be executed with collect().

        Returns:
            AsyncDataFrame that can be further chained with Moltres operations

        Example:
            >>> from sqlalchemy import create_engine, select, table, column
            >>> from moltres import AsyncDataFrame
            >>> engine = create_engine("sqlite:///:memory:")
            >>> # Create a SQLAlchemy select statement
            >>> users = table("users", column("id"), column("name"))
            >>> sa_stmt = select(users.c.id, users.c.name).where(users.c.id > 1)
            >>> # Convert to Moltres AsyncDataFrame
            >>> df = AsyncDataFrame.from_sqlalchemy(sa_stmt)
            >>> # Can now chain Moltres operations
            >>> df2 = df.select("id")
        """
        from sqlalchemy.sql import Select

        if not isinstance(select_stmt, Select):
            raise TypeError(f"Expected SQLAlchemy Select statement, got {type(select_stmt)}")

        # Compile to SQL string
        sql_str = str(select_stmt.compile(compile_kwargs={"literal_binds": True}))

        # Create RawSQL logical plan
        plan = RawSQL(sql=sql_str, params=None)

        return cls(plan=plan, database=database)

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

    def withColumns(self, cols_map: Dict[str, Union[Column, str]]) -> "AsyncDataFrame":
        """Add or replace multiple columns in the DataFrame.

        Args:
            cols_map: Dictionary mapping column names to Column expressions or column names

        Returns:
            New AsyncDataFrame with the added/replaced columns

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = await connect("sqlite:///:memory:")
            >>> await db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "amount": 100.0}], _database=db).insert_into("orders")
            >>> df = await db.table("orders").select()
            >>> # Add multiple columns at once
            >>> df2 = await df.withColumns({
            ...     "amount_with_tax": col("amount") * 1.1,
            ...     "amount_doubled": col("amount") * 2
            ... })
            >>> results = await df2.collect()
            >>> results[0]["amount_with_tax"]
            110.0
            >>> results[0]["amount_doubled"]
            200.0
            >>> await db.close()
        """
        # Apply each column addition/replacement sequentially
        result_df = self
        for col_name, col_expr in cols_map.items():
            result_df = result_df.withColumn(col_name, col_expr)
        return result_df

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

    async def nunique(self, column: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """Count distinct values in column(s).

        Args:
            column: Column name to count. If None, counts distinct values for all columns.

        Returns:
            If column is specified: integer count of distinct values.
            If column is None: dictionary mapping column names to distinct counts.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = await connect("sqlite:///:memory:")
            >>> await db.create_table("users", [column("id", "INTEGER"), column("country", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "country": "USA", "age": 25}, {"id": 2, "country": "USA", "age": 30}, {"id": 3, "country": "UK", "age": 25}], _database=db).insert_into("users")
            >>> df = await db.table("users").select()
            >>> # Count distinct values in a column
            >>> await df.nunique("country")
            2
            >>> # Count distinct for all columns
            >>> counts = await df.nunique()
            >>> counts["country"]
            2
            >>> await db.close()
        """
        from ..expressions.functions import count_distinct

        if column is not None:
            # Count distinct values in the column
            count_df = self.select(count_distinct(col(column)).alias("count"))
            result = await count_df.collect()
            if result and isinstance(result, list) and len(result) > 0:
                row = result[0]
                if isinstance(row, dict):
                    count_val = row.get("count", 0)
                    return int(count_val) if isinstance(count_val, (int, float)) else 0
            return 0
        else:
            # Count distinct for all columns
            counts: Dict[str, int] = {}
            for col_name in self.columns:
                count_df = self.select(count_distinct(col(col_name)).alias("count"))
                result = await count_df.collect()
                if result and isinstance(result, list) and len(result) > 0:
                    row = result[0]
                    if isinstance(row, dict):
                        count_val = row.get("count", 0)
                        counts[col_name] = (
                            int(count_val) if isinstance(count_val, (int, float)) else 0
                        )
                    else:
                        counts[col_name] = 0
                else:
                    counts[col_name] = 0
            return counts

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

    def to_sqlalchemy(self, dialect: Optional[str] = None) -> "Select":
        """Convert AsyncDataFrame's logical plan to a SQLAlchemy Select statement.

        This method allows you to use Moltres AsyncDataFrames with existing SQLAlchemy
        async connections, sessions, or other SQLAlchemy infrastructure.

        Args:
            dialect: Optional SQL dialect name (e.g., "postgresql", "mysql", "sqlite").
                    If not provided, uses the dialect from the attached AsyncDatabase,
                    or defaults to "ansi" if no AsyncDatabase is attached.

        Returns:
            SQLAlchemy Select statement that can be executed with any SQLAlchemy connection

        Example:
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> from sqlalchemy.ext.asyncio import create_async_engine
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            ...     df = await db.table("users")
            ...     df = df.select().where(col("id") > 1)
            ...     # Convert to SQLAlchemy statement
            ...     stmt = df.to_sqlalchemy()
            ...     # Execute with existing SQLAlchemy async connection
            ...     engine = create_async_engine("sqlite+aiosqlite:///:memory:")
            ...     async with engine.connect() as conn:
            ...         result = await conn.execute(stmt)
            ...         rows = result.fetchall()
            ...     await db.close()
        """
        # Determine dialect to use
        if dialect is None:
            if self.database is not None:
                dialect = self.database._dialect_name
            else:
                dialect = "ansi"

        # Compile logical plan to SQLAlchemy Select statement
        return compile_plan(self.plan, dialect=dialect)

    @overload
    async def collect(self, stream: Literal[False] = False) -> List[Dict[str, object]]: ...

    @overload
    async def collect(self, stream: Literal[True]) -> AsyncIterator[List[Dict[str, object]]]: ...

    async def collect(
        self, stream: bool = False
    ) -> Union[
        List[Dict[str, object]],
        AsyncIterator[List[Dict[str, object]]],
        List[Any],
        AsyncIterator[List[Any]],
    ]:
        """Collect DataFrame results asynchronously.

        Args:
            stream: If True, return an async iterator of row chunks. If False (default),
                   materialize all rows into a list.

        Returns:
            If stream=False and no model attached: List of dictionaries representing rows.
            If stream=False and model attached: List of SQLModel or Pydantic instances.
            If stream=True and no model attached: AsyncIterator of row chunks (each chunk is a list of dicts).
            If stream=True and model attached: AsyncIterator of row chunks (each chunk is a list of model instances).

        Raises:
            RuntimeError: If DataFrame is not bound to an AsyncDatabase
            ImportError: If model is attached but Pydantic or SQLModel is not installed

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

        # Helper function to convert rows to model instances if model is attached
        def _convert_rows(
            rows: List[Dict[str, object]],
        ) -> Union[List[Dict[str, object]], List[Any]]:
            from .materialization_helpers import convert_rows_to_models

            return convert_rows_to_models(rows, self.model)

        # Handle RawSQL at root level - execute directly for efficiency
        if isinstance(self.plan, RawSQL):
            if stream:
                # For streaming, we need to use execute_plan_stream which expects a compiled plan
                # So we'll compile the RawSQL plan
                plan = await self._materialize_filescan(self.plan)

                async def stream_gen() -> AsyncIterator[Union[List[Dict[str, object]], List[Any]]]:
                    async for chunk in self.database.execute_plan_stream(plan):  # type: ignore[union-attr]
                        yield _convert_rows(chunk)

                return stream_gen()
            else:
                # Execute RawSQL directly
                from .materialization_helpers import convert_result_rows

                result = await self.database.execute_sql(self.plan.sql, params=self.plan.params)
                rows = convert_result_rows(result.rows)
                return _convert_rows(rows)

        # Handle FileScan by materializing file data into a temporary table
        plan = await self._materialize_filescan(self.plan)

        if stream:
            # For SQL queries, use streaming execution
            if self.database is None:
                raise RuntimeError("Cannot collect a plan without an attached AsyncDatabase")

            async def stream_gen() -> AsyncIterator[Union[List[Dict[str, object]], List[Any]]]:
                async for chunk in self.database.execute_plan_stream(plan):  # type: ignore[union-attr]
                    yield _convert_rows(chunk)

            return stream_gen()

        result = await self.database.execute_plan(plan, model=self.model)
        if result.rows is None:
            return []
        # If result.rows is already a list of SQLModel instances (from .exec()), return directly
        if isinstance(result.rows, list) and len(result.rows) > 0:
            # Check if first item is a SQLModel instance
            try:
                from sqlmodel import SQLModel

                if isinstance(result.rows[0], SQLModel):
                    # Already SQLModel instances from .exec(), return as-is
                    return result.rows
            except ImportError:
                pass
        # Convert to list if it's a DataFrame
        if hasattr(result.rows, "to_dict"):
            records = result.rows.to_dict("records")  # type: ignore[call-overload]
            # Convert Hashable keys to str keys
            rows = [{str(k): v for k, v in row.items()} for row in records]
            return _convert_rows(rows)
        if hasattr(result.rows, "to_dicts"):
            records = list(result.rows.to_dicts())
            # Convert Hashable keys to str keys
            rows = [{str(k): v for k, v in row.items()} for row in records]
            return _convert_rows(rows)
        return _convert_rows(result.rows)

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

        from .file_io_helpers import route_file_read

        records = await route_file_read(
            format_name=filescan.format,
            path=filescan.path,
            database=self.database,
            schema=filescan.schema,
            options=filescan.options,
            column_name=filescan.column_name,
            async_mode=True,
        )

        # AsyncRecords.rows() returns a coroutine, so we need to await it
        return await records.rows()  # type: ignore[no-any-return]

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

        from .file_io_helpers import route_file_read_streaming

        return await route_file_read_streaming(  # type: ignore[no-any-return]
            format_name=filescan.format,
            path=filescan.path,
            database=self.database,
            schema=filescan.schema,
            options=filescan.options,
            column_name=filescan.column_name,
            async_mode=True,
        )

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

    def __getitem__(
        self, key: Union[str, Sequence[str], Column]
    ) -> Union["AsyncDataFrame", Column, "PySparkColumn"]:
        """Enable bracket notation column access (e.g., df["col"], df[["col1", "col2"]]).

        Supports:
        - df['col'] - Returns Column expression with string/date accessors
        - df[['col1', 'col2']] - Returns new AsyncDataFrame with selected columns
        - df[df['age'] > 25] - Boolean indexing (filtering via Column condition)

        Args:
            key: Column name(s) or boolean Column condition

        Returns:
            - For single column string: PySparkColumn (with .str and .dt accessors)
            - For list of columns: AsyncDataFrame with selected columns
            - For boolean Column condition: AsyncDataFrame with filtered rows

        Example:
            >>> df = await db.table("users").select()
            >>> df['age']  # Returns PySparkColumn with .str and .dt accessors
            >>> df[['id', 'name']]  # Returns AsyncDataFrame with selected columns
            >>> df[df['age'] > 25]  # Returns filtered AsyncDataFrame
        """
        # Import here to avoid circular imports
        try:
            from .pyspark_column import PySparkColumn
        except ImportError:
            PySparkColumn = None  # type: ignore

        # Single column string: df['col'] - return Column-like object with accessors
        if isinstance(key, str):
            column_expr = col(key)
            # Wrap in PySparkColumn to enable .str and .dt accessors
            if PySparkColumn is not None:
                return PySparkColumn(column_expr)
            return column_expr

        # List of columns: df[['col1', 'col2']] - select columns
        if isinstance(key, (list, tuple)):
            if len(key) == 0:
                return self.select()
            # Convert all to strings/Columns and select
            columns = [col(c) if isinstance(c, str) else c for c in key]
            return self.select(*columns)

        # Column expression - if it's a boolean condition, use as filter
        if isinstance(key, Column):
            # This is likely a boolean condition like df['age'] > 25
            # We should filter using it
            return self.where(key)

        # Handle PySparkColumn wrapper (which wraps a Column)
        if PySparkColumn is not None and hasattr(key, "_column"):
            # This might be a PySparkColumn - extract underlying Column
            return self.where(key._column)

        raise TypeError(
            f"Invalid key type for __getitem__: {type(key)}. Expected str, list, tuple, or Column."
        )

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
            model=self.model,
        )

    def _with_model(self, model: Optional[Type[Any]]) -> "AsyncDataFrame":
        """Create a new AsyncDataFrame with a SQLModel attached.

        Args:
            model: SQLModel model class to attach, or None to remove model

        Returns:
            New AsyncDataFrame with the model attached
        """
        return AsyncDataFrame(
            plan=self.plan,
            database=self.database,
            model=model,
        )

    def with_model(self, model: Type[Any]) -> "AsyncDataFrame":
        """Attach a SQLModel or Pydantic model to this AsyncDataFrame.

        When a model is attached, `collect()` will return model instances
        instead of dictionaries. This provides type safety and validation.

        Args:
            model: SQLModel or Pydantic model class to attach

        Returns:
            New AsyncDataFrame with the model attached

        Raises:
            TypeError: If model is not a SQLModel or Pydantic class
            ImportError: If required dependencies are not installed

        Example:
            >>> from sqlmodel import SQLModel, Field
            >>> class User(SQLModel, table=True):
            ...     id: int = Field(primary_key=True)
            ...     name: str
            >>> df = await db.table("users")
            >>> df = df.select()
            >>> df_with_model = df.with_model(User)
            >>> results = await df_with_model.collect()  # Returns list of User instances

            >>> from pydantic import BaseModel
            >>> class UserData(BaseModel):
            ...     id: int
            ...     name: str
            >>> df_with_pydantic = df.with_model(UserData)
            >>> results = await df_with_pydantic.collect()  # Returns list of UserData instances
        """
        from ..utils.sqlmodel_integration import is_model_class

        if not is_model_class(model):
            raise TypeError(f"Expected SQLModel or Pydantic class, got {type(model)}")

        return self._with_model(model)


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
