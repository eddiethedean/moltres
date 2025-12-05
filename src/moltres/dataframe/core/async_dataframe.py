"""Async lazy :class:`DataFrame` representation."""

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
    cast,
    overload,
)

from ...expressions.column import Column, LiteralValue, col
from ...logical import operators
from ...logical.plan import FileScan, LogicalPlan, RawSQL
from ...sql.compiler import compile_plan
from ..helpers.dataframe_helpers import DataFrameHelpersMixin

if TYPE_CHECKING:
    from sqlalchemy.sql import Select
    from ...io.records import AsyncRecords
    from ...logical.plan import Project
    from ...table.async_table import AsyncDatabase, AsyncTableHandle
    from ...utils.inspector import ColumnInfo
    from ..groupby.async_groupby import AsyncGroupedDataFrame
    from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame
    from ..interfaces.async_polars_dataframe import AsyncPolarsDataFrame
    from ..io.async_writer import AsyncDataFrameWriter
    from ..columns.pyspark_column import PySparkColumn
else:
    AsyncDatabase = Any
    AsyncTableHandle = Any
    AsyncGroupedDataFrame = Any
    AsyncPandasDataFrame = Any
    AsyncPolarsDataFrame = Any
    AsyncDataFrameWriter = Any
    PySparkColumn = Any
    AsyncRecords = Any
    Select = Any
    AsyncNullHandling = Any


@dataclass(frozen=True)
class AsyncDataFrame(DataFrameHelpersMixin):
    """Async lazy :class:`DataFrame` representation."""

    plan: LogicalPlan
    database: Optional[AsyncDatabase] = None
    model: Optional[Type[Any]] = None  # SQLModel class, if attached

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_table(
        cls, table_handle: AsyncTableHandle, columns: Optional[Sequence[str]] = None
    ) -> AsyncDataFrame:
        """Create an AsyncDataFrame from a table handle."""
        plan = operators.scan(table_handle.name)
        # Check if table_handle has a model attached (SQLModel, Pydantic, or SQLAlchemy)
        model = None
        if hasattr(table_handle, "model") and table_handle.model is not None:
            # Check if it's a SQLModel or Pydantic model
            from ...utils.sqlmodel_integration import is_model_class

            if is_model_class(table_handle.model):
                model = table_handle.model
        df = cls(plan=plan, database=table_handle.database, model=model)
        if columns:
            df = df.select(*columns)
        return df

    @classmethod
    def from_sqlalchemy(
        cls, select_stmt: Select, database: Optional[AsyncDatabase] = None
    ) -> AsyncDataFrame:
        """Create an AsyncDataFrame from a SQLAlchemy Select statement.

        This allows you to integrate existing SQLAlchemy queries with Moltres
        AsyncDataFrame operations. The SQLAlchemy statement is wrapped as a RawSQL
        logical plan, which can then be further chained with Moltres operations.

        Args:
            select_stmt: SQLAlchemy Select statement to convert
            database: Optional :class:`AsyncDatabase` instance to attach to the :class:`DataFrame`.
                     If provided, allows the :class:`DataFrame` to be executed with collect().

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

    def select(self, *columns: Union[Column, str]) -> AsyncDataFrame:
        """Select columns from the :class:`DataFrame`.

        Args:
            *columns: :class:`Column` names or :class:`Column` expressions to select.
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
            ...     from moltres.io.records import :class:`AsyncRecords`
            ...     records = :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice", "email": "alice@example.com"}], _database=db)
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
        from ..operations.dataframe_operations import build_select_operation

        result = build_select_operation(self, columns)
        return self._with_plan(result.plan) if result.should_apply else self

    def selectExpr(self, *exprs: str) -> AsyncDataFrame:
        """Select columns using SQL expressions (async version).

        This method allows you to write SQL expressions directly instead of
        building :class:`Column` objects manually, similar to PySpark's selectExpr().

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
            ...     from moltres.io.records import :class:`AsyncRecords`
            ...     records = :class:`AsyncRecords`(_data=[{"id": 1, "amount": 100.0, "name": "Alice"}], _database=db)
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
        from ...expressions.sql_parser import parse_sql_expr

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
        except (AttributeError, TypeError, KeyError) as e:
            # Column extraction may fail due to:
            # - AttributeError: Plan structure doesn't match expected format
            # - TypeError: Unexpected type in projections
            # - KeyError: Missing expected attributes
            # This is acceptable - the SQL parser will still work without column context
            # Log at debug level for troubleshooting
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                "Could not extract column names from plan for selectExpr() context: %s. "
                "SQL parsing will continue without column validation.",
                e,
            )
        except Exception as e:
            # Catch any other unexpected exceptions during column extraction
            # This broad catch is acceptable because column extraction is optional
            # and we want selectExpr() to work even if extraction fails
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                "Unexpected error extracting column names from plan for selectExpr(): %s. "
                "SQL parsing will continue without column validation.",
                e,
            )

        # Parse each SQL expression into a Column expression
        parsed_columns = []
        for expr_str in exprs:
            parsed_col = parse_sql_expr(expr_str, available_columns)
            parsed_columns.append(parsed_col)

        # Use the existing select() method with parsed columns
        return self.select(*parsed_columns)

    def where(self, predicate: Union[Column, str]) -> AsyncDataFrame:
        """Filter rows based on a predicate.

        Args:
            predicate: :class:`Column` expression or SQL string representing the filter condition.
                      Can be a :class:`Column` object or a SQL string like "age > 18".

        Returns:
            New AsyncDataFrame with filtered rows

        Example:
            >>> import asyncio
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            ...     from moltres.io.records import :class:`AsyncRecords`
            ...     records = :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice", "age": 25}, {"id": 2, "name": "Bob", "age": 17}], _database=db)
            ...     await records.insert_into("users")
            ...     # Filter by condition using :class:`Column`
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
        from ..operations.dataframe_operations import build_where_operation

        return self._with_plan(build_where_operation(self, predicate))

    filter = where

    def join(
        self,
        other: AsyncDataFrame,
        on: Optional[
            Union[str, Sequence[str], Sequence[Tuple[str, str]], Column, Sequence[Column]]
        ] = None,
        how: str = "inner",
    ) -> AsyncDataFrame:
        """Join with another :class:`DataFrame`.

        Args:
            other: Another :class:`DataFrame` to join with
            on: Join condition - can be:
                - A single column name (assumes same name in both DataFrames): ``on="order_id"``
                - A sequence of column names (assumes same names in both): ``on=["col1", "col2"]``
                - A sequence of (left_column, right_column) tuples: ``on=[("id", "customer_id")]``
                - A :class:`Column` expression (PySpark-style): ``on=[col("left_col") == col("right_col")]``
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
            ...     from moltres.io.records import :class:`AsyncRecords`
            ...     records1 = :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice"}], _database=db)
            ...     await records1.insert_into("customers")
            ...     records2 = :class:`AsyncRecords`(_data=[{"id": 1, "customer_id": 1, "amount": 100.0}], _database=db)
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
        from ..operations.dataframe_operations import join_dataframes

        return join_dataframes(self, other, on=on, how=how, lateral=False, hints=None)

    def crossJoin(self, other: AsyncDataFrame) -> AsyncDataFrame:
        """Perform a cross join (Cartesian product) with another :class:`DataFrame`.

        Args:
            other: Another :class:`DataFrame` to cross join with

        Returns:
            New :class:`DataFrame` containing the Cartesian product of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`AsyncDatabase`
        """
        return self.join(other, how="cross")

    def semi_join(
        self,
        other: AsyncDataFrame,
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
    ) -> AsyncDataFrame:
        """Perform a semi-join: return rows from this :class:`DataFrame` where a matching row exists in other.

        This is equivalent to filtering with EXISTS subquery.

        Args:
            other: Another :class:`DataFrame` to semi-join with (used as EXISTS subquery)
            on: Join condition - can be:
                - A single column name (assumes same name in both DataFrames)
                - A sequence of column names (assumes same names in both)
                - A sequence of (left_column, right_column) tuples

        Returns:
            New :class:`DataFrame` containing rows from this :class:`DataFrame` that have matches in other

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`AsyncDatabase`
        """
        from ..operations.dataframe_operations import semi_join_dataframes

        return semi_join_dataframes(self, other, on=on)

    def anti_join(
        self,
        other: AsyncDataFrame,
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
    ) -> AsyncDataFrame:
        """Perform an anti-join: return rows from this :class:`DataFrame` where no matching row exists in other.

        This is equivalent to filtering with NOT EXISTS subquery.

        Args:
            other: Another :class:`DataFrame` to anti-join with (used as NOT EXISTS subquery)
            on: Join condition - can be:
                - A single column name (assumes same name in both DataFrames)
                - A sequence of column names (assumes same names in both)
                - A sequence of (left_column, right_column) tuples

        Returns:
            New :class:`DataFrame` containing rows from this :class:`DataFrame` that have no matches in other

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`AsyncDatabase`
        """
        from ..operations.dataframe_operations import anti_join_dataframes

        return anti_join_dataframes(self, other, on=on)

    def group_by(self, *columns: Union[Column, str]) -> AsyncGroupedDataFrame:
        """Group by the specified columns.

        Example:
            >>> import asyncio
            >>> from moltres import async_connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("sales", [column("category", "TEXT"), column("amount", "REAL")]).collect()
            ...     from moltres.io.records import :class:`AsyncRecords`
            ...     records = :class:`AsyncRecords`(_data=[{"category": "A", "amount": 100.0}, {"category": "A", "amount": 200.0}, {"category": "B", "amount": 150.0}], _database=db)
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
        from ..groupby.async_groupby import AsyncGroupedDataFrame

        normalized = tuple(self._normalize_projection(col) for col in columns)
        grouped_plan = operators.aggregate(self.plan, keys=normalized, aggregates=())
        return AsyncGroupedDataFrame(plan=grouped_plan, database=self.database)

    groupBy = group_by

    def order_by(self, *columns: Union[Column, str]) -> AsyncDataFrame:
        """Sort rows by one or more columns.

        Args:
            *columns: :class:`Column` expressions or column names to sort by. Use .asc() or .desc() for sort order.
                     Can be strings (column names) or :class:`Column` objects.

        Returns:
            New AsyncDataFrame with sorted rows

        Example:
            >>> from moltres import col
            >>> # Sort ascending with string column name
            >>> df = await db.table("users").select().order_by("name")
            >>> # SQL: SELECT * FROM users ORDER BY name

            >>> # Sort ascending with :class:`Column` object
            >>> df = await db.table("users").select().order_by(col("name"))
            >>> # SQL: SELECT * FROM users ORDER BY name

            >>> # Sort descending
            >>> df = await db.table("orders").select().order_by(col("amount").desc())
            >>> # SQL: SELECT * FROM orders ORDER BY amount DESC

            >>> # Multiple sort columns (mixed string and :class:`Column`)
            >>> df = (
            ...     await db.table("sales")
            ...     .select()
            ...     .order_by("region", col("amount").desc())
            ... )
            >>> # SQL: SELECT * FROM sales ORDER BY region, amount DESC
        """
        from ..operations.dataframe_operations import build_order_by_operation

        return self._with_plan(build_order_by_operation(self, columns))

    orderBy = order_by
    sort = order_by  # PySpark-style alias

    def _find_or_create_project_for_locking(
        self, plan: LogicalPlan, for_update: bool, for_share: bool, nowait: bool, skip_locked: bool
    ) -> tuple[Project, LogicalPlan, bool]:
        """Find or create a Project node for row-level locking.

        Traverses the plan tree to find the topmost Project node, or creates one
        if needed. Returns the Project node, the updated plan, and whether wrapping
        was needed.

        Args:
            plan: The logical plan to process
            for_update: Whether to use FOR UPDATE locking
            for_share: Whether to use FOR SHARE locking
            nowait: Whether to use NOWAIT option
            skip_locked: Whether to use SKIP LOCKED option

        Returns:
            Tuple of (Project node, updated plan, needs_wrap flag)
        """
        from ...logical.plan import (
            Aggregate,
            Distinct,
            Explode,
            Filter,
            Join,
            Limit,
            Project,
            Sample,
            SemiJoin,
            AntiJoin,
            Sort,
            TableScan,
        )
        from dataclasses import replace

        # Helper to find the topmost Project node by traversing the plan tree
        def find_project_node(p: LogicalPlan) -> tuple[Project | None, LogicalPlan]:
            """Find the topmost Project node, or return None if not found."""
            if isinstance(p, Project):
                return p, p
            elif isinstance(p, (Filter, Aggregate, Sort, Limit, Distinct, Sample, Explode)):
                # These have a single child - recurse
                found_project, child_plan = find_project_node(p.child)
                if found_project is not None:
                    return found_project, p
                # No Project found in child - we'll need to wrap
                return None, p
            elif isinstance(p, (Join, SemiJoin, AntiJoin)):
                # These have left and right children - check left first (typically the main query)
                found_project, left_plan = find_project_node(p.left)
                if found_project is not None:
                    return found_project, p
                # Check right child
                found_project, right_plan = find_project_node(p.right)
                if found_project is not None:
                    return found_project, p
                # No Project found - we'll need to wrap
                return None, p
            elif isinstance(p, TableScan):
                # TableScan needs to be wrapped in a Project
                return None, p
            else:
                # Other plan types (CTE, Union, etc.) - wrap the entire plan
                return None, p

        project_node, root_plan = find_project_node(plan)

        if project_node is not None:
            # Found a Project node - update it with locking flags
            updated_project = replace(
                project_node,
                for_update=for_update,
                for_share=for_share,
                for_update_nowait=nowait,
                for_update_skip_locked=skip_locked,
            )
            # Rebuild the plan tree with the updated Project
            updated_plan = self._rebuild_plan_with_updated_project(
                root_plan, project_node, updated_project
            )
            return updated_project, updated_plan, False
        else:
            # No Project found - wrap the entire plan in a Project
            new_project = Project(
                child=root_plan,
                projections=(),  # Empty means select all
                for_update=for_update,
                for_share=for_share,
                for_update_nowait=nowait,
                for_update_skip_locked=skip_locked,
            )
            return new_project, new_project, True

    def _rebuild_plan_with_updated_project(
        self, plan: LogicalPlan, old_project: Project, new_project: Project
    ) -> LogicalPlan:
        """Rebuild the plan tree with an updated Project node.

        Args:
            plan: The root plan to rebuild
            old_project: The Project node to replace
            new_project: The new Project node to use

        Returns:
            Updated plan tree
        """
        from ...logical.plan import (
            Aggregate,
            Distinct,
            Explode,
            Filter,
            Join,
            Limit,
            Project,
            Sample,
            SemiJoin,
            AntiJoin,
            Sort,
        )
        from dataclasses import replace

        if plan is old_project:
            return new_project
        elif isinstance(plan, (Filter, Aggregate, Sort, Limit, Distinct, Sample, Explode)):
            updated_child = self._rebuild_plan_with_updated_project(
                plan.child, old_project, new_project
            )
            return cast(LogicalPlan, replace(plan, child=updated_child))
        elif isinstance(plan, (Join, SemiJoin, AntiJoin)):
            updated_left = self._rebuild_plan_with_updated_project(
                plan.left, old_project, new_project
            )
            updated_right = self._rebuild_plan_with_updated_project(
                plan.right, old_project, new_project
            )
            if isinstance(plan, Join):
                return cast(LogicalPlan, replace(plan, left=updated_left, right=updated_right))
            elif isinstance(plan, SemiJoin):
                return cast(LogicalPlan, replace(plan, left=updated_left, right=updated_right))
            else:  # AntiJoin
                return cast(LogicalPlan, replace(plan, left=updated_left, right=updated_right))
        elif isinstance(plan, Project):
            # This shouldn't happen if old_project is found, but handle it
            if plan is old_project:
                return new_project
            updated_child = self._rebuild_plan_with_updated_project(
                plan.child, old_project, new_project
            )
            return cast(LogicalPlan, replace(plan, child=updated_child))
        else:
            # Other plan types - no change needed
            return plan

    def select_for_update(
        self, nowait: bool = False, skip_locked: bool = False
    ) -> "AsyncDataFrame":
        """Select rows with FOR UPDATE lock.

        This method adds a FOR UPDATE clause to the SELECT statement, which locks
        the selected rows for update until the transaction commits or rolls back.
        This is useful for preventing concurrent modifications.

        Args:
            nowait: If True, don't wait for lock - raise error if rows are locked.
                   Requires database support (PostgreSQL, MySQL 8.0+).
            skip_locked: If True, skip locked rows instead of waiting or erroring.
                        Requires database support (PostgreSQL, MySQL 8.0+).

        Returns:
            New AsyncDataFrame with FOR UPDATE locking enabled

        Raises:
            ValueError: If nowait or skip_locked is requested but not supported by dialect.

        Example:
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> db = async_connect("sqlite+aiosqlite:///:memory:")
            >>> await db.create_table("orders", [column("id", "INTEGER"), column("status", "TEXT")]).collect()
            >>> from moltres.io.records import AsyncRecords
            >>> _ = await AsyncRecords(_data=[{"id": 1, "status": "pending"}], _database=db).insert_into("orders")
            >>> async with db.transaction() as txn:
            ...     table_handle = await db.table("orders")
            ...     df = table_handle.select().where(col("status") == "pending")
            ...     locked_df = df.select_for_update(nowait=True)
            ...     results = await locked_df.collect()
            ...     # Rows are now locked for update
        """

        # Check dialect support first
        if self.database:
            dialect = self.database.dialect
            if nowait and not dialect.supports_for_update_nowait:
                raise ValueError(f"Dialect '{dialect.name}' does not support FOR UPDATE NOWAIT")
            if skip_locked and not dialect.supports_for_update_skip_locked:
                raise ValueError(
                    f"Dialect '{dialect.name}' does not support FOR UPDATE SKIP LOCKED"
                )
            if not dialect.supports_row_locking:
                raise ValueError(f"Dialect '{dialect.name}' does not support row-level locking")

        # Get the current plan
        plan = self.plan

        try:
            # Find or create a Project plan for locking
            project_plan, updated_plan, needs_wrap = self._find_or_create_project_for_locking(
                plan, for_update=True, for_share=False, nowait=nowait, skip_locked=skip_locked
            )

            # Return AsyncDataFrame with updated plan
            return self._with_plan(updated_plan)
        except Exception as e:
            # Provide better error message with plan type information
            plan_type = type(plan).__name__
            raise ValueError(
                f"select_for_update() failed on plan type '{plan_type}'. "
                f"This may indicate an unsupported plan structure. "
                f"Original error: {e}"
            ) from e

    def select_for_share(self, nowait: bool = False, skip_locked: bool = False) -> "AsyncDataFrame":
        """Select rows with FOR SHARE lock.

        This method adds a FOR SHARE clause to the SELECT statement, which locks
        the selected rows for shared (read) access. Other transactions can still
        read the rows but cannot modify them until the transaction commits.

        This method works with any plan structure (joins, aggregations, sorts, etc.)
        by finding or creating the appropriate Project node in the plan tree.

        Args:
            nowait: If True, don't wait for lock - raise error if rows are locked.
                   Requires database support (PostgreSQL, MySQL 8.0+).
            skip_locked: If True, skip locked rows instead of waiting or erroring.
                        Requires database support (PostgreSQL, MySQL 8.0+).

        Returns:
            New AsyncDataFrame with FOR SHARE locking enabled

        Raises:
            ValueError: If nowait or skip_locked is requested but not supported by dialect,
                       or if the plan structure cannot support row-level locking.

        Example:
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> db = async_connect("sqlite+aiosqlite:///:memory:")
            >>> await db.create_table("products", [column("id", "INTEGER"), column("stock", "INTEGER")]).collect()
            >>> from moltres.io.records import AsyncRecords
            >>> _ = await AsyncRecords(_data=[{"id": 1, "stock": 10}], _database=db).insert_into("products")
            >>> async with db.transaction() as txn:
            ...     table_handle = await db.table("products")
            ...     df = table_handle.select().where(col("id") == 1)
            ...     locked_df = df.select_for_share()
            ...     results = await locked_df.collect()
            ...     # Rows are now locked for shared access

        Example with joins:
            >>> orders = await db.table("orders").select()
            >>> customers = await db.table("customers").select()
            >>> joined = orders.join(customers, on=[col("orders.customer_id") == col("customers.id")])
            >>> locked_joined = joined.select_for_share()
            >>> results = await locked_joined.collect()
        """
        # Check dialect support first
        if self.database:
            dialect = self.database.dialect
            if nowait and not dialect.supports_for_update_nowait:
                raise ValueError(f"Dialect '{dialect.name}' does not support FOR UPDATE NOWAIT")
            if skip_locked and not dialect.supports_for_update_skip_locked:
                raise ValueError(
                    f"Dialect '{dialect.name}' does not support FOR UPDATE SKIP LOCKED"
                )
            if not dialect.supports_row_locking:
                raise ValueError(f"Dialect '{dialect.name}' does not support row-level locking")

        # Get the current plan
        plan = self.plan

        try:
            # Find or create a Project plan for locking
            project_plan, updated_plan, needs_wrap = self._find_or_create_project_for_locking(
                plan, for_update=False, for_share=True, nowait=nowait, skip_locked=skip_locked
            )

            # Return AsyncDataFrame with updated plan
            return self._with_plan(updated_plan)
        except Exception as e:
            # Provide better error message with plan type information
            plan_type = type(plan).__name__
            raise ValueError(
                f"select_for_share() failed on plan type '{plan_type}'. "
                f"This may indicate an unsupported plan structure. "
                f"Original error: {e}"
            ) from e

    def limit(self, count: int) -> AsyncDataFrame:
        """Limit the number of rows returned."""
        from ..operations.dataframe_operations import build_limit_operation

        return self._with_plan(build_limit_operation(self.plan, count))

    def sample(self, fraction: float, seed: Optional[int] = None) -> AsyncDataFrame:
        """Sample a fraction of rows from the :class:`DataFrame`.

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
        from ..operations.dataframe_operations import build_sample_operation

        return self._with_plan(build_sample_operation(self.plan, fraction, seed))

    def union(self, other: AsyncDataFrame) -> AsyncDataFrame:
        """Union this :class:`DataFrame` with another :class:`DataFrame` (distinct rows only)."""
        from ..operations.dataframe_operations import union_dataframes

        return union_dataframes(self, other, distinct=True)

    def unionAll(self, other: AsyncDataFrame) -> AsyncDataFrame:
        """Union this :class:`DataFrame` with another :class:`DataFrame` (all rows, including duplicates)."""
        from ..operations.dataframe_operations import union_dataframes

        return union_dataframes(self, other, distinct=False)

    def intersect(self, other: AsyncDataFrame) -> AsyncDataFrame:
        """Intersect this :class:`DataFrame` with another :class:`DataFrame` (distinct rows only)."""
        from ..operations.dataframe_operations import intersect_dataframes

        return intersect_dataframes(self, other, distinct=True)

    def except_(self, other: AsyncDataFrame) -> AsyncDataFrame:
        """Return rows in this :class:`DataFrame` that are not in another :class:`DataFrame` (distinct rows only)."""
        from ..operations.dataframe_operations import except_dataframes

        return except_dataframes(self, other, distinct=True)

    def cte(self, name: str) -> AsyncDataFrame:
        """Create a Common Table Expression (CTE) from this :class:`DataFrame`.

        Args:
            name: Name for the CTE

        Returns:
            New AsyncDataFrame representing the CTE
        """
        from ..operations.dataframe_operations import cte_dataframe

        return cte_dataframe(self, name)

    def recursive_cte(
        self, name: str, recursive: AsyncDataFrame, union_all: bool = False
    ) -> AsyncDataFrame:
        """Create a Recursive Common Table Expression (WITH RECURSIVE) from this :class:`DataFrame`.

        Args:
            name: Name for the recursive CTE
            recursive: :class:`AsyncDataFrame` representing the recursive part (references the CTE)
            union_all: If True, use UNION ALL; if False, use UNION (distinct)

        Returns:
            New AsyncDataFrame representing the recursive CTE

        Example:
            >>> # Fibonacci sequence example
            >>> from moltres.expressions import functions as F
            >>> initial = await db.table("seed").select(F.lit(1).alias("n"), F.lit(1).alias("fib"))
            >>> recursive = initial.select(...)  # Recursive part
            >>> fib_cte = initial.recursive_cte("fib", recursive)
        """
        from ..operations.dataframe_operations import recursive_cte_dataframe

        return recursive_cte_dataframe(self, name, recursive, union_all)

    def explode(self, column: Union[Column, str], alias: str = "value") -> AsyncDataFrame:
        """Explode an array/JSON column into multiple rows (one row per element).

        Args:
            column: :class:`Column` expression or column name to explode (must be array or JSON)
            alias: Alias for the exploded value column (default: "value")

        Returns:
            New AsyncDataFrame with exploded rows

        Example:
            >>> from moltres import async_connect, col
            >>> from moltres.table.schema import column
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> # Note: explode() requires array/JSON support which varies by database
            >>> # This example shows the API usage pattern
            >>> await db.create_table("users", [column("id", "INTEGER"), column("tags", "TEXT")]).collect()
            >>> from moltres.io.records import AsyncRecords
            >>> _ = await AsyncRecords(_data=[{"id": 1, "tags": '["python", "sql"]'}], _database=db).insert_into("users")
            >>> # Explode a JSON array column (database-specific support required)
            >>> table_handle = await db.table("users")
            >>> df = table_handle.select()
            >>> exploded = df.explode(col("tags"), alias="tag")
            >>> # Each row in exploded will have one tag per row
            >>> # Note: Actual execution depends on database JSON/array support
            >>> await db.close()
        """
        from ..operations.dataframe_operations import explode_dataframe

        return explode_dataframe(self, column, alias)

    def pivot(
        self,
        pivot_column: str,
        value_column: str,
        agg_func: str = "sum",
        pivot_values: Optional[Sequence[str]] = None,
    ) -> AsyncDataFrame:
        """Pivot the :class:`DataFrame` to reshape data from long to wide format.

        Args:
            pivot_column: :class:`Column` to pivot on (values become column headers)
            value_column: :class:`Column` containing values to aggregate
            agg_func: Aggregation function to apply (default: "sum")
            pivot_values: Optional list of specific values to pivot (if None, uses all distinct values)

        Returns:
            New AsyncDataFrame with pivoted data

        Example:
            >>> from moltres import async_connect
            >>> from moltres.table.schema import column
            >>> db = await async_connect("sqlite+aiosqlite:///:memory:")
            >>> await db.create_table("sales", [column("date", "TEXT"), column("product", "TEXT"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import AsyncRecords
            >>> _ = await AsyncRecords(_data=[{"date": "2024-01-01", "product": "A", "amount": 100.0}, {"date": "2024-01-01", "product": "B", "amount": 200.0}, {"date": "2024-01-02", "product": "A", "amount": 150.0}], _database=db).insert_into("sales")
            >>> # Pivot sales data by product
            >>> table_handle = await db.table("sales")
            >>> df = table_handle.select("date", "product", "amount")
            >>> pivoted = df.pivot(pivot_column="product", value_column="amount", agg_func="sum")
            >>> results = await pivoted.collect()
            >>> len(results) > 0
            True
            >>> await db.close()
        """
        from ..operations.dataframe_operations import pivot_dataframe

        return pivot_dataframe(self, pivot_column, value_column, agg_func, pivot_values)

    def distinct(self) -> AsyncDataFrame:
        """Return a new :class:`DataFrame` with distinct rows."""
        return self._with_plan(operators.distinct(self.plan))

    def dropDuplicates(self, subset: Optional[Sequence[str]] = None) -> AsyncDataFrame:
        """Return a new :class:`DataFrame` with duplicate rows removed."""
        if subset is None:
            return self.distinct()
        # Simplified implementation
        return self.group_by(*subset).agg()

    def withColumn(self, colName: str, col_expr: Union[Column, str]) -> AsyncDataFrame:
        """Add or replace a column in the :class:`DataFrame`.

        Args:
            colName: Name of the column to add or replace
            col_expr: :class:`Column` expression or column name

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
        from ...expressions.column import Column
        from ...logical.plan import Project
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
            from ...expressions.column import Column

            star_col = Column(op="star", args=(), _alias=None)
            new_projections = [star_col, new_col]

        return self._with_plan(operators.project(self.plan, tuple(new_projections)))

    def withColumns(self, cols_map: Dict[str, Union[Column, str]]) -> AsyncDataFrame:
        """Add or replace multiple columns in the :class:`DataFrame`.

        Args:
            cols_map: Dictionary mapping column names to :class:`Column` expressions or column names

        Returns:
            New AsyncDataFrame with the added/replaced columns

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = await connect("sqlite:///:memory:")
            >>> await db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "amount": 100.0}], _database=db).insert_into("orders")
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

    def withColumnRenamed(self, existing: str, new: str) -> AsyncDataFrame:
        """Rename a column in the :class:`DataFrame`."""
        from ...logical.plan import Project
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

    def drop(self, *cols: Union[str, Column]) -> AsyncDataFrame:
        """Drop one or more columns from the :class:`DataFrame`.

        Args:
            *cols: :class:`Column` names or :class:`Column` objects to drop

        Returns:
            New AsyncDataFrame with the specified columns removed

        Example:
            >>> # Drop by string column name
            >>> await df.drop("col1", "col2").collect()
            >>> # Drop by :class:`Column` object
            >>> await df.drop(col("col1"), col("col2")).collect()
            >>> # Mixed usage
            >>> await df.drop("col1", col("col2")).collect()
        """
        from ...logical.plan import Project

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
                    if (
                        col_expr.op == "column"
                        and col_expr.args[0]
                        and isinstance(col_expr.args[0], str)
                    ):
                        col_name = col_expr._alias or col_expr.args[0]
                    else:
                        col_name = col_expr._alias
                    if col_name not in cols_to_drop:
                        new_projections.append(col_expr)
                else:
                    new_projections.append(col_expr)
            return self._with_plan(operators.project(self.plan.child, tuple(new_projections)))
        else:
            return self

    async def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the :class:`DataFrame`."""
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
            column: :class:`Column` name to count. If None, counts distinct values for all columns.

        Returns:
            If column is specified: integer count of distinct values.
            If column is None: dictionary mapping column names to distinct counts.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = await connect("sqlite:///:memory:")
            >>> await db.create_table("users", [column("id", "INTEGER"), column("country", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "country": "USA", "age": 25}, {"id": 2, "country": "USA", "age": 30}, {"id": 3, "country": "UK", "age": 25}], _database=db).insert_into("users")
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
        from ...expressions.functions import count_distinct

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
        """Return the number of rows in the :class:`DataFrame`."""
        from ...expressions.functions import count as count_func

        count_col = count_func("*").alias("count")
        result_df = self._with_plan(operators.aggregate(self.plan, (), (count_col,)))
        results = await result_df.collect()
        if not isinstance(results, list):
            raise TypeError("count() requires collect() to return a list, not an iterator")
        if results:
            count_value = results[0].get("count", 0)
            return int(count_value) if isinstance(count_value, (int, float, str)) else 0
        return 0

    async def describe(self, *cols: str) -> AsyncDataFrame:
        """Compute basic statistics for numeric columns."""
        from ...expressions.functions import count, avg, min, max

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

    async def summary(self, *statistics: str) -> AsyncDataFrame:
        """Compute summary statistics for numeric columns."""
        return await self.describe()

    def fillna(
        self,
        value: Union[LiteralValue, Dict[str, LiteralValue]],
        subset: Optional[Sequence[str]] = None,
    ) -> AsyncDataFrame:
        """Replace null values with a specified value."""
        from ...expressions.functions import coalesce, lit

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
                filled_col = coalesce(col_expr, lit(fill_value)).alias(col_name)
                new_projections.append(filled_col)
            else:
                new_projections.append(col_expr)

        return self.select(*new_projections)

    def dropna(self, how: str = "any", subset: Optional[Sequence[str]] = None) -> AsyncDataFrame:
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
        """Compile the :class:`DataFrame` plan to SQL."""
        from sqlalchemy.sql import Select

        stmt = (
            self.database.compile_plan(self.plan)
            if self.database is not None
            else compile_plan(self.plan)
        )
        if isinstance(stmt, Select):
            return str(stmt.compile(compile_kwargs={"literal_binds": True}))
        return str(stmt)

    def to_sqlalchemy(self, dialect: Optional[str] = None) -> Select:
        """Convert AsyncDataFrame's logical plan to a SQLAlchemy Select statement.

        This method allows you to use Moltres AsyncDataFrames with existing SQLAlchemy
        async connections, sessions, or other SQLAlchemy infrastructure.

        Args:
            dialect: Optional SQL dialect name (e.g., "postgresql", "mysql", "sqlite").
                    If not provided, uses the dialect from the attached :class:`AsyncDatabase`,
                    or defaults to "ansi" if no :class:`AsyncDatabase` is attached.

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
        """Collect :class:`DataFrame` results asynchronously.

        Args:
            stream: If True, return an async iterator of row chunks. If False (default),
                   materialize all rows into a list.

        Returns:
            If stream=False and no model attached: List of dictionaries representing rows.
            If stream=False and model attached: List of SQLModel or Pydantic instances.
            If stream=True and no model attached: AsyncIterator of row chunks (each chunk is a list of dicts).
            If stream=True and model attached: AsyncIterator of row chunks (each chunk is a list of model instances).

        Raises:
            RuntimeError: If :class:`DataFrame` is not bound to an :class:`AsyncDatabase`
            ImportError: If model is attached but Pydantic or SQLModel is not installed

        Example:
            >>> import asyncio
            >>> from moltres import async_connect
            >>> from moltres.table.schema import column
            >>> async def example():
            ...     db = await async_connect("sqlite+aiosqlite:///:memory:")
            ...     await db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            ...     from moltres.io.records import :class:`AsyncRecords`
            ...     records = :class:`AsyncRecords`(_data=[{"id": 1, "name": "Alice"}], _database=db)
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
            from ..helpers.materialization_helpers import convert_rows_to_models

            return convert_rows_to_models(rows, self.model)

        # Handle RawSQL at root level - execute directly for efficiency
        if isinstance(self.plan, RawSQL):
            if stream:
                # For streaming, we need to use execute_plan_stream which expects a compiled plan
                # So we'll compile the RawSQL plan
                plan = await self._materialize_filescan(self.plan)

                async def stream_gen() -> AsyncIterator[Union[List[Dict[str, object]], List[Any]]]:
                    if self.database is None:
                        raise RuntimeError("Cannot stream without an attached AsyncDatabase")
                    async for chunk in self.database.execute_plan_stream(plan):
                        yield _convert_rows(chunk)

                return stream_gen()
            else:
                # Execute RawSQL directly
                from ..helpers.materialization_helpers import convert_result_rows

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
                if self.database is None:
                    raise RuntimeError("Cannot stream without an attached AsyncDatabase")
                async for chunk in self.database.execute_plan_stream(plan):
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
                from .create_dataframe import create_temp_table_from_streaming_async
                from ...logical.operators import scan

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
        from ...logical.plan import (
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

        from ..helpers.file_io_helpers import route_file_read

        records = await route_file_read(  # type: ignore[misc]
            format_name=filescan.format,
            path=filescan.path,
            database=self.database,
            schema=filescan.schema,
            options=filescan.options,
            column_name=filescan.column_name,
            async_mode=True,
        )

        # AsyncRecords.rows() returns a coroutine, so we need to await it
        from typing import cast

        return cast("list[dict[str, object]]", await records.rows())

    async def _read_file_streaming(self, filescan: FileScan) -> AsyncRecords:
        """Read a file in streaming mode (chunked, safe for large files).

        Args:
            filescan: FileScan logical plan node

        Returns:
            :class:`AsyncRecords` object with _generator set (streaming mode)

        Note:
            This method returns :class:`AsyncRecords` with a generator, allowing chunked processing
            without loading the entire file into memory. Use this for large files.
        """
        if self.database is None:
            raise RuntimeError("Cannot read file without an attached AsyncDatabase")

        from ..helpers.file_io_helpers import route_file_read_streaming

        result = await route_file_read_streaming(  # type: ignore[misc]
            format_name=filescan.format,
            path=filescan.path,
            database=self.database,
            schema=filescan.schema,
            options=filescan.options,
            column_name=filescan.column_name,
            async_mode=True,
        )
        return cast(AsyncRecords, result)

    @property
    def na(self) -> AsyncNullHandling:
        """Access null handling methods via the `na` property.

        Returns:
            AsyncNullHandling helper object with drop() and fill() methods

        Example:
            >>> await df.na.drop().collect()  # Drop rows with nulls
            >>> await df.na.fill(0).collect()  # Fill nulls with 0
        """
        return AsyncNullHandling(self)

    @property
    def write(self) -> AsyncDataFrameWriter:
        """Return an AsyncDataFrameWriter for writing this :class:`DataFrame` to a table."""
        from ..io.async_writer import AsyncDataFrameWriter

        return AsyncDataFrameWriter(self)

    @property
    def columns(self) -> List[str]:
        """Return a list of column names in this :class:`DataFrame`.

        Similar to PySpark's :class:`DataFrame`.columns property, this extracts column
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
        """Return the schema of this :class:`DataFrame` as a list of ColumnInfo objects.

        Similar to PySpark's :class:`DataFrame`.schema property, this extracts column
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

        Similar to PySpark's :class:`DataFrame`.dtypes property, this returns a list
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
        """Print the schema of this :class:`DataFrame` in a tree format.

        Similar to PySpark's :class:`DataFrame`.printSchema() method, this prints
        a formatted representation of the :class:`DataFrame`'s schema.

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
    ) -> Union[AsyncDataFrame, Column, PySparkColumn]:
        """Enable bracket notation column access (e.g., df["col"], df[["col1", "col2"]]).

        Supports:
        - df['col'] - Returns :class:`Column` expression with string/date accessors
        - df[['col1', 'col2']] - Returns new AsyncDataFrame with selected columns
        - df[df['age'] > 25] - Boolean indexing (filtering via :class:`Column` condition)

        Args:
            key: :class:`Column` name(s) or boolean :class:`Column` condition

        Returns:
            - For single column string: PySparkColumn (with .str and .dt accessors)
            - For list of columns: AsyncDataFrame with selected columns
            - For boolean :class:`Column` condition: AsyncDataFrame with filtered rows

        Example:
            >>> df = await db.table("users").select()
            >>> df['age']  # Returns PySparkColumn with .str and .dt accessors
            >>> df[['id', 'name']]  # Returns AsyncDataFrame with selected columns
            >>> df[df['age'] > 25]  # Returns filtered AsyncDataFrame
        """
        # Import here to avoid circular imports
        PySparkColumn: Optional[Type[Any]] = None
        try:
            from ..columns.pyspark_column import PySparkColumn as _PySparkColumn

            PySparkColumn = _PySparkColumn
        except ImportError:
            pass

        # Single column string: df['col'] - return Column-like object with accessors
        if isinstance(key, str):
            column_expr = col(key)
            # Wrap in PySparkColumn to enable .str and .dt accessors
            if PySparkColumn is not None:
                return PySparkColumn(column_expr)  # type: ignore[no-any-return]
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
            name: :class:`Column` name to access

        Returns:
            :class:`Column` object for the specified column name

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
        from ...expressions.column import col

        return col(name)

    # ---------------------------------------------------------------- utilities
    def _with_plan(self, plan: LogicalPlan) -> AsyncDataFrame:
        """Create a new AsyncDataFrame with a different plan."""
        return AsyncDataFrame(
            plan=plan,
            database=self.database,
            model=self.model,
        )

    def _with_model(self, model: Optional[Type[Any]]) -> AsyncDataFrame:
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

    def with_model(self, model: Type[Any]) -> AsyncDataFrame:
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
        from ...utils.sqlmodel_integration import is_model_class

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
        self,
        value: Union[LiteralValue, Dict[str, LiteralValue]],
        subset: Optional[Sequence[str]] = None,
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

    def polars(self) -> AsyncPolarsDataFrame:
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
        from ..interfaces.async_polars_dataframe import AsyncPolarsDataFrame

        return AsyncPolarsDataFrame.from_dataframe(self._df)

    def pandas(self) -> AsyncPandasDataFrame:
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
        from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame

        return AsyncPandasDataFrame.from_dataframe(self._df)
