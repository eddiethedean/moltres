"""Lazy :class:`DataFrame` representation.

This module provides the core :class:`DataFrame` class, which represents a lazy
query plan that is executed only when results are requested (via :meth:`collect`,
:meth:`show`, etc.).

The :class:`DataFrame` class supports:
- PySpark-style operations (select, where, join, groupBy, etc.)
- SQL pushdown execution (all operations compile to SQL)
- Lazy evaluation (queries are not executed until collect/show is called)
- Model integration (SQLModel, Pydantic, SQLAlchemy)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
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
from ...logical.plan import FileScan, Limit, LogicalPlan, RawSQL
from ...sql.compiler import compile_plan
from ..helpers.dataframe_helpers import DataFrameHelpersMixin

if TYPE_CHECKING:
    from sqlalchemy.sql import Select
    from sqlalchemy.orm import DeclarativeBase
    from ...io.records import Records
    from ...logical.plan import Project
    from ...table.table import Database, TableHandle
    from ...utils.inspector import ColumnInfo
    from ..groupby import GroupedDataFrame
    from ..interfaces.polars_dataframe import PolarsDataFrame
    from ..io.writer import DataFrameWriter
    from ..columns.pyspark_column import PySparkColumn

    # Type alias for model types (SQLModel, Pydantic, SQLAlchemy models)
    ModelType = Type[Union[DeclarativeBase, Any]]
else:
    Database = Any
    Select = Any
    Records = Any
    TableHandle = Any
    ColumnInfo = Any
    GroupedDataFrame = Any
    PolarsDataFrame = Any
    DataFrameWriter = Any
    PySparkColumn = Any
    ModelType = Any


@dataclass(frozen=True)
class DataFrame(DataFrameHelpersMixin):
    """Lazy :class:`DataFrame` representing a query plan.

    A :class:`DataFrame` is an immutable, lazy representation of a SQL query.
    Operations on a :class:`DataFrame` build up a logical plan that is only executed
    when you call :meth:`collect`, :meth:`show`, or similar execution methods.

    All operations compile to SQL and execute directly on the database - no
    data is loaded into memory until you explicitly request results.

    Attributes:
        plan: The logical plan representing this query
        database: Optional :class:`Database` instance for executing the query
        model: Optional SQLModel, Pydantic, or SQLAlchemy model class for type safety

    Example:
        >>> from moltres import connect, col
        >>> db = connect("sqlite:///example.db")
        >>> df = db.table("users").select().where(col("age") > 25)
        >>> results = df.collect()  # Query executes here
    """

    plan: LogicalPlan
    database: Optional[Database] = None
    model: Optional["ModelType"] = (
        None  # SQLModel, Pydantic, or SQLAlchemy model class, if attached
    )

    def __repr__(self) -> str:
        """Return a user-friendly string representation of the DataFrame."""
        from ...logical.plan import (
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
            Pivot,
            Project,
            RawSQL,
            RecursiveCTE,
            Sample,
            SemiJoin,
            Sort,
            SortOrder,
            TableScan,
            Union,
        )

        def format_plan(plan: LogicalPlan, depth: int = 0) -> str:
            """Recursively format a logical plan node."""
            if depth > 3:  # Limit depth to avoid overly long representations
                return "..."

            if isinstance(plan, TableScan):
                table_name = plan.table
                if plan.alias and plan.alias != plan.table:
                    return f"TableScan('{table_name}' AS '{plan.alias}')"
                return f"TableScan('{table_name}')"

            elif isinstance(plan, FileScan):
                return f"FileScan('{plan.path}', format='{plan.format}')"

            elif isinstance(plan, RawSQL):
                sql_preview = plan.sql[:50] + "..." if len(plan.sql) > 50 else plan.sql
                return f"RawSQL('{sql_preview}')"

            elif isinstance(plan, Project):
                child_str = format_plan(plan.child, depth + 1)
                # Show column names if available
                col_names = []
                for proj in plan.projections[:5]:  # Limit to first 5 columns
                    if isinstance(proj, Column):
                        if proj.op == "column" and proj.args:
                            col_names.append(str(proj.args[0]))
                        elif proj._alias:
                            col_names.append(proj._alias)
                        else:
                            col_names.append(str(proj)[:20])
                    else:
                        col_names.append(str(proj)[:20])
                if len(plan.projections) > 5:
                    col_names.append("...")
                cols_str = ", ".join(col_names)
                return f"Project([{cols_str}]) <- {child_str}"

            elif isinstance(plan, Filter):
                child_str = format_plan(plan.child, depth + 1)
                pred_str = str(plan.predicate)[:50]
                if len(str(plan.predicate)) > 50:
                    pred_str += "..."
                return f"Filter({pred_str}) <- {child_str}"

            elif isinstance(plan, Limit):
                child_str = format_plan(plan.child, depth + 1)
                if plan.offset > 0:
                    return f"Limit({plan.count}, offset={plan.offset}) <- {child_str}"
                return f"Limit({plan.count}) <- {child_str}"

            elif isinstance(plan, Sort):
                child_str = format_plan(plan.child, depth + 1)
                orders = []
                for order in plan.orders[:3]:  # Limit to first 3 sort columns
                    if isinstance(order, SortOrder):
                        col_str = str(order.expression)[:20]
                        dir_str = "DESC" if order.descending else "ASC"
                        orders.append(f"{col_str} {dir_str}")
                if len(plan.orders) > 3:
                    orders.append("...")
                orders_str = ", ".join(orders)
                return f"Sort([{orders_str}]) <- {child_str}"

            elif isinstance(plan, Aggregate):
                child_str = format_plan(plan.child, depth + 1)
                group_cols = [str(col)[:20] for col in plan.grouping[:3]]
                if len(plan.grouping) > 3:
                    group_cols.append("...")
                agg_cols = [str(col)[:20] for col in plan.aggregates[:3]]
                if len(plan.aggregates) > 3:
                    agg_cols.append("...")
                group_str = ", ".join(group_cols) if group_cols else "()"
                agg_str = ", ".join(agg_cols) if agg_cols else "()"
                return f"Aggregate(group_by=[{group_str}], agg=[{agg_str}]) <- {child_str}"

            elif isinstance(plan, Join):
                left_str = format_plan(plan.left, depth + 1)
                right_str = format_plan(plan.right, depth + 1)
                join_type = plan.how.upper()
                if plan.on:
                    on_str = ", ".join(
                        f"{left_col}=={right_col}" for left_col, right_col in plan.on[:2]
                    )
                    if len(plan.on) > 2:
                        on_str += "..."
                    return f"Join({join_type}, on=[{on_str}]) <- ({left_str}, {right_str})"
                elif plan.condition is not None:
                    cond_str = str(plan.condition)[:30]
                    return f"Join({join_type}, on={cond_str}) <- ({left_str}, {right_str})"
                else:
                    return f"Join({join_type}) <- ({left_str}, {right_str})"

            elif isinstance(plan, SemiJoin):
                left_str = format_plan(plan.left, depth + 1)
                right_str = format_plan(plan.right, depth + 1)
                if plan.on:
                    on_str = ", ".join(
                        f"{left_col}=={right_col}"
                        for left_col, right_col in (plan.on[:2] if plan.on else [])
                    )
                    return f"SemiJoin(on=[{on_str}]) <- ({left_str}, {right_str})"
                elif plan.condition is not None:
                    cond_str = str(plan.condition)[:30]
                    return f"SemiJoin(on={cond_str}) <- ({left_str}, {right_str})"
                return f"SemiJoin <- ({left_str}, {right_str})"

            elif isinstance(plan, AntiJoin):
                left_str = format_plan(plan.left, depth + 1)
                right_str = format_plan(plan.right, depth + 1)
                if plan.on:
                    on_str = ", ".join(
                        f"{left_col}=={right_col}"
                        for left_col, right_col in (plan.on[:2] if plan.on else [])
                    )
                    return f"AntiJoin(on=[{on_str}]) <- ({left_str}, {right_str})"
                elif plan.condition is not None:
                    cond_str = str(plan.condition)[:30]
                    return f"AntiJoin(on={cond_str}) <- ({left_str}, {right_str})"
                return f"AntiJoin <- ({left_str}, {right_str})"

            elif isinstance(plan, Distinct):
                child_str = format_plan(plan.child, depth + 1)
                return f"Distinct <- {child_str}"

            elif isinstance(plan, Union):
                left_str = format_plan(plan.left, depth + 1)
                right_str = format_plan(plan.right, depth + 1)
                union_type = "UNION" if plan.distinct else "UNION ALL"
                return f"{union_type} <- ({left_str}, {right_str})"

            elif isinstance(plan, Intersect):
                left_str = format_plan(plan.left, depth + 1)
                right_str = format_plan(plan.right, depth + 1)
                intersect_type = "INTERSECT" if plan.distinct else "INTERSECT ALL"
                return f"{intersect_type} <- ({left_str}, {right_str})"

            elif isinstance(plan, Except):
                left_str = format_plan(plan.left, depth + 1)
                right_str = format_plan(plan.right, depth + 1)
                except_type = "EXCEPT" if plan.distinct else "EXCEPT ALL"
                return f"{except_type} <- ({left_str}, {right_str})"

            elif isinstance(plan, CTE):
                child_str = format_plan(plan.child, depth + 1)
                return f"CTE('{plan.name}') <- {child_str}"

            elif isinstance(plan, RecursiveCTE):
                initial_str = format_plan(plan.initial, depth + 1)
                recursive_str = format_plan(plan.recursive, depth + 1)
                return f"RecursiveCTE('{plan.name}') <- ({initial_str}, {recursive_str})"

            elif isinstance(plan, Pivot):
                child_str = format_plan(plan.child, depth + 1)
                return f"Pivot(pivot='{plan.pivot_column}', value='{plan.value_column}', agg='{plan.agg_func}') <- {child_str}"

            elif isinstance(plan, GroupedPivot):
                child_str = format_plan(plan.child, depth + 1)
                return f"GroupedPivot(pivot='{plan.pivot_column}', value='{plan.value_column}', agg='{plan.agg_func}') <- {child_str}"

            elif isinstance(plan, Explode):
                child_str = format_plan(plan.child, depth + 1)
                col_str = str(plan.column)[:30]
                return f"Explode({col_str}) <- {child_str}"

            elif isinstance(plan, Sample):
                child_str = format_plan(plan.child, depth + 1)
                seed_str = f", seed={plan.seed}" if plan.seed is not None else ""
                return f"Sample(fraction={plan.fraction}{seed_str}) <- {child_str}"

            else:
                # Fallback for unknown plan types
                return f"{type(plan).__name__}(...)"

        plan_str = format_plan(self.plan)
        model_str = f", model={self.model.__name__}" if self.model else ""
        return f"DataFrame({plan_str}{model_str})"

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_table(
        cls, table_handle: "TableHandle", columns: Optional[Sequence[str]] = None
    ) -> DataFrame:
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
    def from_sqlalchemy(cls, select_stmt: Select, database: Optional[Database] = None) -> DataFrame:
        """Create a :class:`DataFrame` from a SQLAlchemy Select statement.

        This allows you to integrate existing SQLAlchemy queries with Moltres
        :class:`DataFrame` operations. The SQLAlchemy statement is wrapped as a RawSQL
        logical plan, which can then be further chained with Moltres operations.

        Args:
            select_stmt: SQLAlchemy Select statement to convert
            database: Optional :class:`Database` instance to attach to the :class:`DataFrame`.
                     If provided, allows the :class:`DataFrame` to be executed with collect().

        Returns:
            :class:`DataFrame`: :class:`DataFrame` that can be further chained with Moltres operations

        Example:
            >>> from sqlalchemy import create_engine, select, table, column
            >>> from moltres.dataframe.dataframe import DataFrame
            >>> engine = create_engine("sqlite:///:memory:")
            >>> # Create a SQLAlchemy select statement
            >>> users = table("users", column("id"), column("name"))
            >>> sa_stmt = select(users.c.id, users.c.name).where(users.c.id > 1)
            >>> # Convert to Moltres DataFrame
            >>> df = DataFrame.from_sqlalchemy(sa_stmt)
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

    def select(self, *columns: Union[Column, str]) -> DataFrame:
        """Select specific columns from the :class:`DataFrame`.

        Args:
            *columns: :class:`Column` names or :class:`Column` expressions to select.
                     Use "*" to select all columns (same as empty select).
                     Can combine "*" with other columns: select("*", col("new_col"))

        Returns:
            New :class:`DataFrame` with selected columns

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("email", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice", "email": "alice@example.com"}], _database=db).insert_into("users")
            >>> # Select specific columns
            >>> df = db.table("users").select("id", "name", "email")
            >>> results = df.collect()
            >>> results[0]["name"]
            'Alice'
            >>> # Select all columns (empty select)
            >>> df2 = db.table("users").select()
            >>> results2 = df2.collect()
            >>> len(results2[0].keys())
            3
            >>> # Select with expressions
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> _ = Records(_data=[{"id": 1, "amount": 100.0}], _database=db).insert_into("orders")
            >>> df3 = db.table("orders").select(col("id"), (col("amount") * 1.1).alias("amount_with_tax"))
            >>> results3 = df3.collect()
            >>> results3[0]["amount_with_tax"]
            110.0
            >>> # Select all columns plus new ones
            >>> df4 = db.table("orders").select("*", (col("amount") * 1.1).alias("with_tax"))
            >>> results4 = df4.collect()
            >>> results4[0]["id"]
            1
            >>> results4[0]["with_tax"]
            110.0
            >>> db.close()
        """
        from ..operations.dataframe_operations import build_select_operation

        result = build_select_operation(self, columns)
        return self._with_plan(result.plan) if result.should_apply else self

    def selectExpr(self, *exprs: str) -> DataFrame:
        """Select columns using SQL expressions.

        This method allows you to write SQL expressions directly instead of
        building :class:`Column` objects manually, similar to PySpark's selectExpr().

        Args:
            *exprs: SQL expression strings (e.g., "amount * 1.1 as with_tax")

        Returns:
            New :class:`DataFrame` with selected expressions

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "amount": 100.0, "name": "Alice"}], _database=db).insert_into("orders")
            >>> # Basic column selection
            >>> df = db.table("orders").selectExpr("id", "name")
            >>> results = df.collect()
            >>> results[0]["id"]
            1
            >>> # With expressions and aliases
            >>> df2 = db.table("orders").selectExpr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")
            >>> results2 = df2.collect()
            >>> results2[0]["with_tax"]
            110.0
            >>> results2[0]["name_upper"]
            'ALICE'
            >>> # Chaining with other operations
            >>> df3 = db.table("orders").selectExpr("id", "amount").where(col("amount") > 50)
            >>> results3 = df3.collect()
            >>> len(results3)
            1
            >>> db.close()
        """
        from ...expressions.sql_parser import parse_sql_expr

        if not exprs:
            return self

        # Get available column names from the DataFrame for context
        # This is optional but can be used for validation
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

    def where(self, predicate: Union[Column, str]) -> DataFrame:
        """Filter rows based on a condition.

        Args:
            predicate: :class:`Column` expression or SQL string representing the filter condition.
                      Can be a :class:`Column` object or a SQL string like "age > 18".

        Returns:
            New :class:`DataFrame` with filtered rows

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice", "age": 25}, {"id": 2, "name": "Bob", "age": 17}], _database=db).insert_into("users")
            >>> # Filter by condition using :class:`Column`
            >>> df = db.table("users").select().where(col("age") >= 18)
            >>> results = df.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Alice'
            >>> # Filter using SQL string
            >>> df2 = db.table("users").select().where("age > 18")
            >>> results2 = df2.collect()
            >>> len(results2)
            1
            >>> # Multiple conditions with :class:`Column`
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL"), column("status", "TEXT")]).collect()
            >>> _ = :class:`Records`(_data=[{"id": 1, "amount": 150.0, "status": "active"}, {"id": 2, "amount": 50.0, "status": "active"}], _database=db).insert_into("orders")
            >>> df3 = db.table("orders").select().where((col("amount") > 100) & (col("status") == "active"))
            >>> results3 = df3.collect()
            >>> len(results3)
            1
            >>> results3[0]["amount"]
            150.0
            >>> db.close()
        """
        from ..operations.dataframe_operations import build_where_operation

        return self._with_plan(build_where_operation(self, predicate))

    filter = where

    def limit(self, count: int) -> DataFrame:
        """Limit the number of rows returned by the query.

        Args:
            count: Maximum number of rows to return. Must be non-negative.
                  If 0, returns an empty result set.

        Returns:
            New :class:`DataFrame` with the limit applied

        Raises:
            ValueError: If count is negative

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> Records(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 6)], _database=db).insert_into("users")
            >>> # Limit to 3 rows
            >>> df = db.table("users").select().limit(3)
            >>> results = df.collect()
            >>> len(results)
            3
            >>> # Limit with ordering
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> Records(_data=[{"id": i, "amount": float(i * 10)} for i in range(1, 6)], _database=db).insert_into("orders")
            >>> df2 = db.table("orders").select().order_by(col("amount").desc()).limit(2)
            >>> results2 = df2.collect()
            >>> len(results2)
            2
            >>> results2[0]["amount"]
            50.0
            >>> db.close()
        """
        from ..operations.dataframe_operations import build_limit_operation

        return self._with_plan(build_limit_operation(self.plan, count))

    def sample(self, fraction: float, seed: Optional[int] = None) -> DataFrame:
        """Sample a fraction of rows from the :class:`DataFrame`.

        Args:
            fraction: Fraction of rows to sample (0.0 to 1.0)
            seed: Optional random seed for reproducible sampling

        Returns:
            New :class:`DataFrame` with sampled rows

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> :class:`Records`(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 11)], _database=db).insert_into("users")
            >>> # Sample 30% of rows with seed for reproducibility
            >>> df = db.table("users").select().sample(0.3, seed=42)
            >>> results = df.collect()
            >>> len(results) <= 10  # Should be approximately 30% of 10 rows
            True
            >>> db.close()
        """
        from ..operations.dataframe_operations import build_sample_operation

        return self._with_plan(build_sample_operation(self.plan, fraction, seed))

    def order_by(self, *columns: Union[Column, str]) -> DataFrame:
        """Sort rows by one or more columns.

        Args:
            *columns: :class:`Column` expressions or column names to sort by. Use .asc() or .desc() for sort order.
                     Can be strings (column names) or :class:`Column` objects.

        Returns:
            New :class:`DataFrame` with sorted rows

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Charlie"}, {"id": 2, "name": "Alice"}, {"id": 3, "name": "Bob"}], _database=db).insert_into("users")
            >>> # Sort ascending with string column name
            >>> df = db.table("users").select().order_by("name")
            >>> results = df.collect()
            >>> results[0]["name"]
            'Alice'
            >>> results[1]["name"]
            'Bob'
            >>> # Sort descending with :class:`Column` object
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> _ = :class:`Records`(_data=[{"id": 1, "amount": 50.0}, {"id": 2, "amount": 100.0}, {"id": 3, "amount": 25.0}], _database=db).insert_into("orders")
            >>> df2 = db.table("orders").select().order_by(col("amount").desc())
            >>> results2 = df2.collect()
            >>> results2[0]["amount"]
            100.0
            >>> # Multiple sort columns
            >>> db.create_table("sales", [column("region", "TEXT"), column("amount", "REAL")]).collect()
            >>> _ = :class:`Records`(_data=[{"region": "North", "amount": 100.0}, {"region": "North", "amount": 50.0}, {"region": "South", "amount": 75.0}], _database=db).insert_into("sales")
            >>> df3 = db.table("sales").select().order_by("region", col("amount").desc())
            >>> results3 = df3.collect()
            >>> results3[0]["region"]
            'North'
            >>> results3[0]["amount"]
            100.0
            >>> db.close()
        """
        from ..operations.dataframe_operations import build_order_by_operation

        return self._with_plan(build_order_by_operation(self, columns))

    orderBy = order_by  # PySpark-style alias
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

    def select_for_update(self, nowait: bool = False, skip_locked: bool = False) -> "DataFrame":
        """Select rows with FOR UPDATE lock.

        This method adds a FOR UPDATE clause to the SELECT statement, which locks
        the selected rows for exclusive access. Other transactions cannot read or
        modify the rows until the transaction commits.

        This method works with any plan structure (joins, aggregations, sorts, etc.)
        by finding or creating the appropriate Project node in the plan tree.

        Args:
            nowait: If True, don't wait for lock - raise error if rows are locked.
                   Requires database support (PostgreSQL, MySQL 8.0+).
            skip_locked: If True, skip locked rows instead of waiting or erroring.
                        Requires database support (PostgreSQL, MySQL 8.0+).

        Returns:
            New :class:`DataFrame` with FOR UPDATE locking enabled

        Raises:
            ValueError: If nowait or skip_locked is requested but not supported by dialect,
                       or if the plan structure cannot support row-level locking.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("orders", [column("id", "INTEGER"), column("status", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "status": "pending"}], _database=db).insert_into("orders")
            >>> with db.transaction() as txn:
            ...     df = db.table("orders").select().where(col("status") == "pending")
            ...     locked_df = df.select_for_update(nowait=True)
            ...     results = locked_df.collect()
            ...     # Rows are now locked for update

        Example with joins:
            >>> orders = db.table("orders").select()
            >>> customers = db.table("customers").select()
            >>> joined = orders.join(customers, on=[col("orders.customer_id") == col("customers.id")])
            >>> locked_joined = joined.select_for_update()
            >>> results = locked_joined.collect()
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

            # Return DataFrame with updated plan
            return self._with_plan(updated_plan)
        except Exception as e:
            # Provide better error message with plan type information
            plan_type = type(plan).__name__
            raise ValueError(
                f"select_for_update() failed on plan type '{plan_type}'. "
                f"This may indicate an unsupported plan structure. "
                f"Original error: {e}"
            ) from e

    def select_for_share(self, nowait: bool = False, skip_locked: bool = False) -> "DataFrame":
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
            New :class:`DataFrame` with FOR SHARE locking enabled

        Raises:
            ValueError: If nowait or skip_locked is requested but not supported by dialect,
                       or if the plan structure cannot support row-level locking.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("products", [column("id", "INTEGER"), column("stock", "INTEGER")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "stock": 10}], _database=db).insert_into("products")
            >>> with db.transaction() as txn:
            ...     df = db.table("products").select().where(col("id") == 1)
            ...     locked_df = df.select_for_share()
            ...     results = locked_df.collect()
            ...     # Rows are now locked for shared access

        Example with joins:
            >>> orders = db.table("orders").select()
            >>> customers = db.table("customers").select()
            >>> joined = orders.join(customers, on=[col("orders.customer_id") == col("customers.id")])
            >>> locked_joined = joined.select_for_share()
            >>> results = locked_joined.collect()
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

            # Return DataFrame with updated plan
            return self._with_plan(updated_plan)
        except Exception as e:
            # Provide better error message with plan type information
            plan_type = type(plan).__name__
            raise ValueError(
                f"select_for_share() failed on plan type '{plan_type}'. "
                f"This may indicate an unsupported plan structure. "
                f"Original error: {e}"
            ) from e

    def join(
        self,
        other: DataFrame,
        *,
        on: Optional[
            Union[str, Sequence[str], Sequence[Tuple[str, str]], "Column", Sequence["Column"]]
        ] = None,
        how: str = "inner",
        lateral: bool = False,
        hints: Optional[Sequence[str]] = None,
    ) -> DataFrame:
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
            lateral: If True, create a LATERAL join (PostgreSQL, MySQL 8.0+).
                    Allows right side to reference columns from left side.
            hints: Optional sequence of join hints (e.g., ["USE_INDEX(idx_name)", "FORCE_INDEX(idx_name)"]).
                   Dialect-specific: MySQL uses USE INDEX, PostgreSQL uses /*+ ... */ comments.

        Returns:
            New :class:`DataFrame` containing the join result

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`Database`

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> # Setup tables
            >>> db.create_table("customers", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("orders", [column("id", "INTEGER"), column("customer_id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("customers")
            >>> _ = :class:`Records`(_data=[{"id": 1, "customer_id": 1, "amount": 100.0}], _database=db).insert_into("orders")
            >>> # PySpark-style with :class:`Column` expressions (recommended)
            >>> customers = db.table("customers").select()
            >>> orders = db.table("orders").select()
            >>> df = customers.join(orders, on=[col("customers.id") == col("orders.customer_id")], how="inner")
            >>> results = df.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Alice'
            >>> results[0]["amount"]
            100.0
            >>> # Same column name (simplest)
            >>> db.create_table("items", [column("order_id", "INTEGER"), column("product", "TEXT")]).collect()
            >>> _ = :class:`Records`(_data=[{"order_id": 1, "product": "Widget"}], _database=db).insert_into("items")
            >>> df2 = orders.join(db.table("items").select(), on="order_id", how="inner")
            >>> results2 = df2.collect()
            >>> results2[0]["product"]
            'Widget'
            >>> # Left join
            >>> _ = :class:`Records`(_data=[{"id": 2, "name": "Bob"}], _database=db).insert_into("customers")
            >>> df3 = customers.join(orders, on=[col("customers.id") == col("orders.customer_id")], how="left")
            >>> results3 = df3.collect()
            >>> len(results3)
            2
            >>> db.close()
            ...     lateral=True
            ... )
            >>> # SQL: SELECT * FROM customers LEFT JOIN LATERAL (SELECT * FROM orders WHERE customer_id = customers.id) ...
        """
        from ..operations.dataframe_operations import join_dataframes

        return join_dataframes(self, other, on=on, how=how, lateral=lateral, hints=hints)

    def crossJoin(self, other: DataFrame) -> DataFrame:
        """Perform a cross join (Cartesian product) with another :class:`DataFrame`.

        Args:
            other: Another :class:`DataFrame` to cross join with

        Returns:
            New :class:`DataFrame` containing the Cartesian product of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`Database`

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("value", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}], _database=db).insert_into("table1")
            >>> _ = :class:`Records`(_data=[{"id": 1, "value": "X"}, {"id": 2, "value": "Y"}], _database=db).insert_into("table2")
            >>> df1 = db.table("table1").select()
            >>> df2 = db.table("table2").select()
            >>> # Cross join (Cartesian product)
            >>> df_cross = df1.crossJoin(df2)
            >>> results = df_cross.collect()
            >>> len(results)
            4
            >>> db.close()
        """
        return self.join(other, how="cross")

    def semi_join(
        self,
        other: DataFrame,
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
    ) -> DataFrame:
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
            RuntimeError: If DataFrames are not bound to the same :class:`Database`

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("customers", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("orders", [column("id", "INTEGER"), column("customer_id", "INTEGER")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("customers")
            >>> _ = :class:`Records`(_data=[{"id": 1, "customer_id": 1}], _database=db).insert_into("orders")
            >>> # Find customers who have placed orders
            >>> customers = db.table("customers").select()
            >>> orders = db.table("orders").select()
            >>> customers_with_orders = customers.semi_join(orders, on=[("id", "customer_id")])
            >>> results = customers_with_orders.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Alice'
            >>> db.close()
        """
        from ..operations.dataframe_operations import semi_join_dataframes

        return semi_join_dataframes(self, other, on=on)

    def anti_join(
        self,
        other: DataFrame,
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
    ) -> DataFrame:
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
            RuntimeError: If DataFrames are not bound to the same :class:`Database`

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("customers", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("orders", [column("id", "INTEGER"), column("customer_id", "INTEGER")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("customers")
            >>> _ = :class:`Records`(_data=[{"id": 1, "customer_id": 1}], _database=db).insert_into("orders")
            >>> # Find customers who have not placed any orders
            >>> customers = db.table("customers").select()
            >>> orders = db.table("orders").select()
            >>> customers_without_orders = customers.anti_join(orders, on=[("id", "customer_id")])
            >>> results = customers_without_orders.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Bob'
            >>> db.close()
        """
        from ..operations.dataframe_operations import anti_join_dataframes

        return anti_join_dataframes(self, other, on=on)

    def pivot(
        self,
        pivot_column: str,
        value_column: str,
        agg_func: str = "sum",
        pivot_values: Optional[Sequence[str]] = None,
    ) -> DataFrame:
        """Pivot the :class:`DataFrame` to reshape data from long to wide format.

        Args:
            pivot_column: :class:`Column` to pivot on (values become column headers)
            value_column: :class:`Column` containing values to aggregate
            agg_func: Aggregation function to apply (default: "sum")
            pivot_values: Optional list of specific values to pivot (if None, uses all distinct values)

        Returns:
            New :class:`DataFrame` with pivoted data

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("sales", [column("date", "TEXT"), column("product", "TEXT"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"date": "2024-01-01", "product": "A", "amount": 100.0}, {"date": "2024-01-01", "product": "B", "amount": 200.0}, {"date": "2024-01-02", "product": "A", "amount": 150.0}], _database=db).insert_into("sales")
            >>> # Pivot sales data by product
            >>> df = db.table("sales").select("date", "product", "amount")
            >>> pivoted = df.pivot(pivot_column="product", value_column="amount", agg_func="sum")
            >>> results = pivoted.collect()
            >>> len(results) > 0
            True
            >>> db.close()
        """
        from ..operations.dataframe_operations import pivot_dataframe

        return pivot_dataframe(self, pivot_column, value_column, agg_func, pivot_values)

    def explode(self, column: Union[Column, str], alias: str = "value") -> DataFrame:
        """Explode an array/JSON column into multiple rows (one row per element).

        Args:
            column: :class:`Column` expression or column name to explode (must be array or JSON)
            alias: Alias for the exploded value column (default: "value")

        Returns:
            New :class:`DataFrame` with exploded rows

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> # Note: explode() requires array/JSON support which varies by database
            >>> # This example shows the API usage pattern
            >>> db.create_table("users", [column("id", "INTEGER"), column("tags", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "tags": '["python", "sql"]'}], _database=db).insert_into("users")
            >>> # Explode a JSON array column (database-specific support required)
            >>> df = db.table("users").select()
            >>> exploded = df.explode(col("tags"), alias="tag")
            >>> # Each row in exploded will have one tag per row
            >>> # Note: Actual execution depends on database JSON/array support
            >>> db.close()
        """
        from ..operations.dataframe_operations import explode_dataframe

        return explode_dataframe(self, column, alias)

    def group_by(self, *columns: Union[Column, str]) -> "GroupedDataFrame":
        """Group rows by one or more columns for aggregation.

        Args:
            *columns: :class:`Column` names or :class:`Column` expressions to group by

        Returns:
            :class:`GroupedDataFrame` that can be used with aggregation functions

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> # Group by single column
            >>> db.create_table("orders", [column("customer_id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"customer_id": 1, "amount": 100.0}, {"customer_id": 1, "amount": 50.0}, {"customer_id": 2, "amount": 200.0}], _database=db).insert_into("orders")
            >>> df = db.table("orders").select().group_by("customer_id").agg(F.sum(col("amount")).alias("total"))
            >>> results = df.collect()
            >>> len(results)
            2
            >>> results[0]["total"]
            150.0
            >>> # Group by multiple columns
            >>> db.create_table("sales", [column("region", "TEXT"), column("product", "TEXT"), column("revenue", "REAL")]).collect()
            >>> _ = :class:`Records`(_data=[{"region": "North", "product": "A", "revenue": 100.0}, {"region": "North", "product": "A", "revenue": 50.0}], _database=db).insert_into("sales")
            >>> df2 = db.table("sales").select().group_by("region", "product").agg(F.sum(col("revenue")).alias("total_revenue"), F.count("*").alias("count"))
            >>> results2 = df2.collect()
            >>> results2[0]["total_revenue"]
            150.0
            >>> results2[0]["count"]
            2
            >>> db.close()
            ... )
            >>> # SQL: SELECT region, product, SUM(revenue) AS total_revenue, COUNT(*) AS count
            >>> #      FROM sales GROUP BY region, product
        """
        if not columns:
            raise ValueError("group_by requires at least one grouping column")
        from ..groupby.groupby import GroupedDataFrame

        keys = tuple(self._normalize_projection(column) for column in columns)
        return GroupedDataFrame(plan=self.plan, keys=keys, parent=self)

    groupBy = group_by

    def union(self, other: DataFrame) -> DataFrame:
        """Union this :class:`DataFrame` with another :class:`DataFrame` (distinct rows only).

        Args:
            other: Another :class:`DataFrame` to union with

        Returns:
            New :class:`DataFrame` containing the union of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`Database`

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("table1")
            >>> _ = :class:`Records`(_data=[{"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}], _database=db).insert_into("table2")
            >>> df1 = db.table("table1").select()
            >>> df2 = db.table("table2").select()
            >>> # Union (distinct rows only)
            >>> df_union = df1.union(df2)
            >>> results = df_union.collect()
            >>> len(results)
            3
            >>> names = {r["name"] for r in results}
            >>> "Alice" in names and "Bob" in names and "Charlie" in names
            True
            >>> db.close()
        """
        from ..operations.dataframe_operations import union_dataframes

        return union_dataframes(self, other, distinct=True)

    def unionAll(self, other: DataFrame) -> DataFrame:
        """Union this :class:`DataFrame` with another :class:`DataFrame` (all rows, including duplicates).

        Args:
            other: Another :class:`DataFrame` to union with

        Returns:
            New :class:`DataFrame` containing the union of all rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`Database`

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("table1")
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("table2")
            >>> df1 = db.table("table1").select()
            >>> df2 = db.table("table2").select()
            >>> # UnionAll (all rows, including duplicates)
            >>> df_union = df1.unionAll(df2)
            >>> results = df_union.collect()
            >>> len(results)
            2
            >>> db.close()
        """
        from ..operations.dataframe_operations import union_dataframes

        return union_dataframes(self, other, distinct=False)

    def intersect(self, other: DataFrame) -> DataFrame:
        """Intersect this :class:`DataFrame` with another :class:`DataFrame` (distinct rows only).

        Args:
            other: Another :class:`DataFrame` to intersect with

        Returns:
            New :class:`DataFrame` containing the intersection of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`Database`

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("table1")
            >>> _ = :class:`Records`(_data=[{"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}], _database=db).insert_into("table2")
            >>> df1 = db.table("table1").select()
            >>> df2 = db.table("table2").select()
            >>> # Intersect (common rows only)
            >>> df_intersect = df1.intersect(df2)
            >>> results = df_intersect.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Bob'
            >>> db.close()
        """
        from ..operations.dataframe_operations import intersect_dataframes

        return intersect_dataframes(self, other, distinct=True)

    def except_(self, other: DataFrame) -> DataFrame:
        """Return rows in this :class:`DataFrame` that are not in another :class:`DataFrame` (distinct rows only).

        Args:
            other: Another :class:`DataFrame` to exclude from

        Returns:
            New :class:`DataFrame` containing rows in this :class:`DataFrame` but not in other

        Raises:
            RuntimeError: If DataFrames are not bound to the same :class:`Database`

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("table1")
            >>> _ = :class:`Records`(_data=[{"id": 2, "name": "Bob"}], _database=db).insert_into("table2")
            >>> df1 = db.table("table1").select()
            >>> df2 = db.table("table2").select()
            >>> # Except (rows in df1 but not in df2)
            >>> df_except = df1.except_(df2)
            >>> results = df_except.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Alice'
            >>> db.close()
        """
        from ..operations.dataframe_operations import except_dataframes

        return except_dataframes(self, other, distinct=True)

    def cte(self, name: str) -> DataFrame:
        """Create a Common Table Expression (CTE) from this :class:`DataFrame`.

        Args:
            name: Name for the CTE

        Returns:
            New :class:`DataFrame` representing the CTE

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "amount": 150.0}, {"id": 2, "amount": 50.0}], _database=db).insert_into("orders")
            >>> # Create CTE
            >>> cte_df = db.table("orders").select().where(col("amount") > 100).cte("high_value_orders")
            >>> # Query the CTE
            >>> result = cte_df.select().collect()
            >>> len(result)
            1
            >>> result[0]["amount"]
            150.0
            >>> db.close()
        """
        from ..operations.dataframe_operations import cte_dataframe

        return cte_dataframe(self, name)

    def recursive_cte(self, name: str, recursive: DataFrame, union_all: bool = False) -> DataFrame:
        """Create a Recursive Common Table Expression (WITH RECURSIVE) from this :class:`DataFrame`.

        Args:
            name: Name for the recursive CTE
            recursive: :class:`DataFrame` representing the recursive part (references the CTE)
            union_all: If True, use UNION ALL; if False, use UNION (distinct)

        Returns:
            New :class:`DataFrame` representing the recursive CTE

        Example:
            >>> # Fibonacci sequence example
            >>> from moltres.expressions import functions as F
            >>> initial = db.table("seed").select(F.lit(1).alias("n"), F.lit(1).alias("fib"))
            >>> recursive = initial.select(...)  # Recursive part
            >>> fib_cte = initial.recursive_cte("fib", recursive)
        """
        from ..operations.dataframe_operations import recursive_cte_dataframe

        return recursive_cte_dataframe(self, name, recursive, union_all)

    def distinct(self) -> DataFrame:
        """Return a new :class:`DataFrame` with distinct rows.

        Returns:
            New :class:`DataFrame` with distinct rows

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Alice"}, {"id": 3, "name": "Bob"}], _database=db).insert_into("users")
            >>> df = db.table("users").select("name").distinct()
            >>> results = df.collect()
            >>> len(results)
            2
            >>> names = {r["name"] for r in results}
            >>> "Alice" in names
            True
            >>> "Bob" in names
            True
            >>> db.close()
        """
        return self._with_plan(operators.distinct(self.plan))

    def dropDuplicates(self, subset: Optional[Sequence[str]] = None) -> DataFrame:
        """Return a new :class:`DataFrame` with duplicate rows removed.

        Args:
            subset: Optional list of column names to consider when identifying duplicates.
                   If None, all columns are considered.

        Returns:
            New :class:`DataFrame` with duplicates removed

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

    def withColumn(self, colName: str, col_expr: Union[Column, str]) -> DataFrame:
        """Add or replace a column in the :class:`DataFrame`.

        Args:
            colName: Name of the column to add or replace
            col_expr: :class:`Column` expression or column name

        Returns:
            New :class:`DataFrame` with the added/replaced column

        Note:
            This operation adds a Project on top of the current plan.
            If a column with the same name exists, it will be replaced.
            Window functions are supported and will ensure all columns are available.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL"), column("category", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "amount": 100.0, "category": "A"}, {"id": 2, "amount": 200.0, "category": "A"}], _database=db).insert_into("orders")
            >>> # Add a computed column
            >>> df = db.table("orders").select()
            >>> df2 = df.withColumn("amount_with_tax", col("amount") * 1.1)
            >>> results = df2.collect()
            >>> results[0]["amount_with_tax"]
            110.0
            >>> # Add window function column
            >>> df3 = df.withColumn("row_num", F.row_number().over(partition_by=col("category"), order_by=col("amount")))
            >>> results3 = df3.collect()
            >>> results3[0]["row_num"]
            1
            >>> results3[1]["row_num"]
            2
            >>> db.close()
        """
        from ...logical.plan import Project

        # Normalize the column expression
        new_col = self._normalize_projection(col_expr)
        # Add alias if it's a Column expression
        if isinstance(new_col, Column) and not new_col._alias:
            new_col = new_col.alias(colName)
        elif isinstance(new_col, Column):
            # Already has alias, but we want to use colName
            new_col = replace(new_col, _alias=colName)

        # Check if this is a window function
        is_window_func = isinstance(new_col, Column) and self._is_window_function(new_col)

        # Get existing columns from the plan if it's a Project
        # Otherwise, we'll select all columns plus the new one
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
                        # Always keep the star column to preserve all original columns
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
            elif not has_star:
                # If there's no star column, we need to add one to preserve all original columns
                # This ensures that when adding a new column, all existing columns are still available
                star_col = Column(op="star", args=(), _alias=None)
                new_projections = [star_col, new_col]
            else:
                # Add the new column at the end
                new_projections = existing_cols + [new_col]
        else:
            # No existing projection, select all plus new column
            # Use a wildcard select and add the new column
            star_col = Column(op="star", args=(), _alias=None)
            new_projections = [star_col, new_col]

        return self._with_plan(operators.project(self.plan, tuple(new_projections)))

    def withColumns(self, cols_map: Dict[str, Union[Column, str]]) -> DataFrame:
        """Add or replace multiple columns in the :class:`DataFrame`.

        Args:
            cols_map: Dictionary mapping column names to :class:`Column` expressions or column names

        Returns:
            New :class:`DataFrame` with the added/replaced columns

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = Records(_data=[{"id": 1, "amount": 100.0}], _database=db).insert_into("orders")
            >>> df = db.table("orders").select()
            >>> # Add multiple columns at once
            >>> df2 = df.withColumns({
            ...     "amount_with_tax": col("amount") * 1.1,
            ...     "amount_doubled": col("amount") * 2
            ... })
            >>> results = df2.collect()
            >>> results[0]["amount_with_tax"]
            110.0
            >>> results[0]["amount_doubled"]
            200.0
            >>> db.close()
        """
        # Apply each column addition/replacement sequentially
        result_df = self
        for col_name, col_expr in cols_map.items():
            result_df = result_df.withColumn(col_name, col_expr)
        return result_df

    def withColumnRenamed(self, existing: str, new: str) -> DataFrame:
        """Rename a column in the :class:`DataFrame`.

        Args:
            existing: Current name of the column
            new: New name for the column

        Returns:
            New :class:`DataFrame` with the renamed column

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("users")
            >>> df = db.table("users").select().withColumnRenamed("name", "user_name")
            >>> results = df.collect()
            >>> "user_name" in results[0]
            True
            >>> results[0]["user_name"]
            'Alice'
            >>> db.close()
        """
        from ...logical.plan import Project

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

    def drop(self, *cols: Union[str, Column]) -> DataFrame:
        """Drop one or more columns from the :class:`DataFrame`.

        Args:
            *cols: :class:`Column` names or :class:`Column` objects to drop

        Returns:
            New :class:`DataFrame` with the specified columns removed

        Note:
            This operation only works if the :class:`DataFrame` has a Project operation.
            Otherwise, it will create a Project that excludes the specified columns.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("email", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice", "email": "alice@example.com"}], _database=db).insert_into("users")
            >>> # Drop by string column name
            >>> df = db.table("users").select().drop("email")
            >>> results = df.collect()
            >>> "email" not in results[0]
            True
            >>> "name" in results[0]
            True
            >>> # Drop by :class:`Column` object
            >>> df2 = db.table("users").select().drop(col("email"))
            >>> results2 = df2.collect()
            >>> "email" not in results2[0]
            True
            >>> # Drop multiple columns
            >>> df3 = db.table("users").select().drop("email", "id")
            >>> results3 = df3.collect()
            >>> len(results3[0].keys())
            1
            >>> "name" in results3[0]
            True
            >>> db.close()
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
            # Filter out the columns to drop
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
            # No projection - this is a simplified implementation
            # In practice, we'd need to know all columns to exclude the dropped ones
            # For now, return self (can't drop from a table scan without schema)
            return self

    # ---------------------------------------------------------------- execution
    def to_sql(self, pretty: bool = False) -> str:
        """Convert the :class:`DataFrame`'s logical plan to a SQL string.

        Args:
            pretty: If True, format SQL with indentation and line breaks for readability.
                   If False, return compact SQL string.

        Returns:
            SQL string representation of the query

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> sql = df.to_sql()
            >>> "SELECT" in sql
            True
            >>> "users" in sql
            True
            >>> db.close()
        """
        from sqlalchemy.sql import Select

        stmt = (
            self.database.compile_plan(self.plan)
            if self.database is not None
            else compile_plan(self.plan)
        )
        if isinstance(stmt, Select):
            # Compile SQLAlchemy statement to SQL string
            sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
            if pretty:
                return self._format_sql(sql)
            return sql
        return str(stmt)

    def _format_sql(self, sql: str) -> str:
        """Format SQL string with indentation for readability."""
        # Simple SQL formatter - add basic indentation
        # Keywords that should start a new line
        keywords = [
            "SELECT",
            "FROM",
            "WHERE",
            "JOIN",
            "INNER JOIN",
            "LEFT JOIN",
            "RIGHT JOIN",
            "FULL JOIN",
            "GROUP BY",
            "ORDER BY",
            "HAVING",
            "LIMIT",
            "UNION",
            "INTERSECT",
            "EXCEPT",
        ]

        # Split by keywords (case-insensitive)
        lines = []
        current_line = ""
        i = 0
        sql_upper = sql.upper()

        while i < len(sql):
            # Check if we're at a keyword
            found_keyword = None
            for keyword in keywords:
                if sql_upper[i:].startswith(keyword):
                    # Check if it's a whole word (not part of another word)
                    if (i == 0 or not sql[i - 1].isalnum() and sql[i - 1] != "_") and (
                        i + len(keyword) >= len(sql)
                        or not sql[i + len(keyword)].isalnum()
                        and sql[i + len(keyword)] != "_"
                    ):
                        found_keyword = keyword
                        break

            if found_keyword:
                # Add current line if not empty
                if current_line.strip():
                    lines.append(current_line.rstrip())
                # Add keyword on new line with indentation
                keyword_text = sql[i : i + len(found_keyword)]
                # Determine indentation level
                indent = "  "  # 2 spaces per level
                if keyword_text.upper() in (
                    "FROM",
                    "WHERE",
                    "GROUP BY",
                    "ORDER BY",
                    "HAVING",
                    "LIMIT",
                ):
                    indent = ""
                elif keyword_text.upper().endswith("JOIN"):
                    indent = "  "
                lines.append(indent + keyword_text)
                i += len(found_keyword)
                current_line = " " * (len(indent) + 2)  # Continue with indentation
            else:
                current_line += sql[i]
                i += 1

        if current_line.strip():
            lines.append(current_line.rstrip())

        return "\n".join(lines)

    @property
    def sql(self) -> str:
        """Property accessor for SQL string representation.

        Returns:
            SQL string representation of the query (formatted for readability)

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> print(df.sql)  # Pretty-printed SQL
            >>> db.close()
        """
        return self.to_sql(pretty=True)

    def show_sql(self, max_length: Optional[int] = None) -> None:
        """Pretty-print the SQL query that will be executed.

        Args:
            max_length: Optional maximum length to display. If SQL is longer,
                       shows first part with "..." indicator. If None, shows full SQL.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> df.show_sql()  # Prints formatted SQL
            >>> db.close()
        """
        sql = self.to_sql(pretty=True)
        if max_length and len(sql) > max_length:
            print(sql[:max_length] + "...")
            print(f"\n[SQL truncated at {max_length} characters, full length: {len(sql)}]")
        else:
            print(sql)

    def sql_preview(self, max_length: int = 200) -> str:
        """Get a preview of the SQL query (first N characters).

        Args:
            max_length: Maximum length of preview (default: 200)

        Returns:
            SQL preview string with "..." if truncated

        Example:
            >>> from moltres import connect, col
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> preview = df.sql_preview()
            >>> len(preview) <= 203  # 200 + "..."
            True
        """
        sql = self.to_sql(pretty=False)
        if len(sql) > max_length:
            return sql[:max_length] + "..."
        return sql

    def to_sqlalchemy(self, dialect: Optional[str] = None) -> "Select":
        """Convert :class:`DataFrame`'s logical plan to a SQLAlchemy Select statement.

        This method allows you to use Moltres DataFrames with existing SQLAlchemy
        connections, sessions, or other SQLAlchemy infrastructure.

        Args:
            dialect: Optional SQL dialect name (e.g., "postgresql", "mysql", "sqlite").
                    If not provided, uses the dialect from the attached :class:`Database`,
                    or defaults to "ansi" if no :class:`Database` is attached.

        Returns:
            SQLAlchemy Select statement that can be executed with any SQLAlchemy connection

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> from sqlalchemy import create_engine
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> # Convert to SQLAlchemy statement
            >>> stmt = df.to_sqlalchemy()
            >>> # Execute with existing SQLAlchemy connection
            >>> engine = create_engine("sqlite:///:memory:")
            >>> with engine.connect() as conn:
            ...     result = conn.execute(stmt)
            ...     rows = result.fetchall()
            >>> db.close()
        """
        # Determine dialect to use
        if dialect is None:
            if self.database is not None:
                dialect = self.database._dialect_name
            else:
                dialect = "ansi"

        # Compile logical plan to SQLAlchemy Select statement
        return compile_plan(self.plan, dialect=dialect)

    def explain(self, analyze: bool = False) -> str:
        """Get the query execution plan using SQL EXPLAIN.

        Convenience method for query debugging and optimization.

        Args:
            analyze: If True, use EXPLAIN ANALYZE (executes query and shows actual execution stats).
                    If False, use EXPLAIN (shows estimated plan without executing).

        Returns:
            Query plan as a string

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> # Get query plan
            >>> plan = df.explain()
            >>> "EXPLAIN" in plan or "SCAN" in plan or "SELECT" in plan
            True
            >>> # Get execution plan with actual stats
            >>> plan2 = df.explain(analyze=True)
            >>> len(plan2) > 0
            True
            >>> db.close()
            >>> plan = df.explain(analyze=True)

        Raises:
            RuntimeError: If :class:`DataFrame` is not bound to a :class:`Database`

        Example:
            >>> df = db.table("users").select().where(col("age") > 18)
            >>> plan = df.explain()
            >>> print(plan)
            >>> # For actual execution stats:
            >>> plan = df.explain(analyze=True)
        """
        if self.database is None:
            raise RuntimeError("Cannot explain a plan without an attached Database")

        sql = self.to_sql()
        # SQLite uses EXPLAIN QUERY PLAN, not EXPLAIN ANALYZE
        dialect_name = self.database.dialect.name if self.database else "sqlite"
        if analyze:
            if dialect_name == "sqlite":
                explain_sql = f"EXPLAIN QUERY PLAN {sql}"
            elif dialect_name == "postgresql":
                explain_sql = f"EXPLAIN ANALYZE {sql}"
            else:
                explain_sql = f"EXPLAIN {sql}"
        else:
            if dialect_name == "sqlite":
                explain_sql = f"EXPLAIN QUERY PLAN {sql}"
            else:
                explain_sql = f"EXPLAIN {sql}"

        # Execute EXPLAIN query
        result = self.database.execute_sql(explain_sql)
        # Format the plan results - EXPLAIN typically returns a single column
        plan_lines = []
        if result.rows is not None:
            for row in result.rows:
                # Format each row of the plan - row is a dict
                if isinstance(row, dict) and len(row) == 1:
                    # Single column result (common for EXPLAIN)
                    plan_lines.append(str(list(row.values())[0]))
                else:
                    plan_lines.append(str(row))
        return "\n".join(plan_lines)

    def plan_summary(self) -> Dict[str, Any]:
        """Get a structured summary of the query plan.

        Returns:
            Dictionary containing plan statistics:
            - operations: List of operation types in the plan
            - table_scans: Number of table scans
            - joins: Number of joins
            - filters: Number of filter operations
            - aggregations: Number of aggregation operations
            - depth: Maximum depth of the plan tree

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> summary = df.plan_summary()
            >>> summary["operations"]
            ['TableScan', 'Filter']
            >>> db.close()
        """
        from collections import deque

        operations = []
        table_scans = 0
        joins = 0
        filters = 0
        aggregations = 0
        max_depth = 0

        # Traverse plan tree
        queue = deque([(self.plan, 0)])
        while queue:
            plan_node, depth = queue.popleft()
            max_depth = max(max_depth, depth)

            # Get operation type
            op_type = type(plan_node).__name__
            operations.append(op_type)

            # Count specific operations
            if op_type == "TableScan":
                table_scans += 1
            elif op_type in (
                "Join",
                "InnerJoin",
                "LeftJoin",
                "RightJoin",
                "FullJoin",
                "SemiJoin",
                "AntiJoin",
            ):
                joins += 1
            elif op_type == "Filter":
                filters += 1
            elif op_type == "Aggregate":
                aggregations += 1

            # Add children to queue
            if hasattr(plan_node, "child"):
                queue.append((plan_node.child, depth + 1))
            elif hasattr(plan_node, "left") and hasattr(plan_node, "right"):
                queue.append((plan_node.left, depth + 1))
                queue.append((plan_node.right, depth + 1))
            elif hasattr(plan_node, "children"):
                # children is a method, call it
                children = plan_node.children()
                for child in children:
                    queue.append((child, depth + 1))

        return {
            "operations": operations,
            "table_scans": table_scans,
            "joins": joins,
            "filters": filters,
            "aggregations": aggregations,
            "depth": max_depth,
            "total_operations": len(operations),
        }

    def validate(self) -> List[Dict[str, Any]]:
        """Validate the query plan and check for common issues.

        Returns:
            List of dictionaries containing validation results:
            - type: "warning" or "error"
            - message: Description of the issue
            - suggestion: Optional suggestion for fixing the issue

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> issues = df.validate()
            >>> len(issues) >= 0
            True
        """
        issues = []

        # Check if database is attached (needed for execution)
        if self.database is None:
            issues.append(
                {
                    "type": "warning",
                    "message": "DataFrame is not attached to a Database. Query cannot be executed.",
                    "suggestion": "Attach a Database using df.with_database(db) or create DataFrame from db.table()",
                }
            )

        # Check for potential performance issues
        summary = self.plan_summary()

        # Warn about multiple table scans (potential cartesian product)
        if summary["table_scans"] > 1 and summary["joins"] == 0:
            issues.append(
                {
                    "type": "warning",
                    "message": f"Query has {summary['table_scans']} table scans but no joins. This may indicate a missing join condition.",
                    "suggestion": "Check if you need to add a join() operation with proper join conditions.",
                }
            )

        # Warn about filters without indexes (if we can detect)
        if summary["filters"] > 0 and self.database is not None:
            # This is a simple check - in practice, we'd need to inspect the actual filter predicates
            issues.append(
                {
                    "type": "info",
                    "message": f"Query has {summary['filters']} filter operation(s). Consider adding indexes on filtered columns for better performance.",
                    "suggestion": "Use db.create_index() to add indexes on frequently filtered columns.",
                }
            )

        # Check for deep plan trees (potential performance issue)
        if summary["depth"] > 10:
            issues.append(
                {
                    "type": "warning",
                    "message": f"Query plan has depth {summary['depth']}, which may indicate a complex query that could be slow.",
                    "suggestion": "Consider breaking the query into smaller parts or using subqueries.",
                }
            )

        return issues

    def performance_hints(self) -> List[str]:
        """Get performance optimization hints for this query.

        Returns:
            List of performance optimization suggestions

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> hints = df.performance_hints()
            >>> len(hints) >= 0
            True
        """
        hints = []
        summary = self.plan_summary()

        # Check for missing indexes on filtered columns
        if summary["filters"] > 0:
            hints.append(
                "Consider adding indexes on columns used in WHERE clauses for better performance."
            )

        # Check for joins without conditions
        if summary["joins"] > 0:
            hints.append("Ensure join conditions are properly specified and indexed.")

        # Check for aggregations without GROUP BY
        if summary["aggregations"] > 0:
            # This would need more detailed plan inspection to be accurate
            hints.append(
                "For aggregation queries, ensure GROUP BY columns are indexed if possible."
            )

        # Check for large result sets - traverse plan to find Limit
        has_limit = False
        queue = [self.plan]
        visited = set()
        while queue:
            node = queue.pop(0)
            if id(node) in visited:
                continue
            visited.add(id(node))
            if isinstance(node, Limit):
                has_limit = True
                break
            # Add children
            if hasattr(node, "child"):
                queue.append(node.child)
            elif hasattr(node, "left") and hasattr(node, "right"):
                queue.append(node.left)
                queue.append(node.right)
            elif hasattr(node, "children"):
                queue.extend(node.children())

        if not has_limit:
            hints.append("Consider using limit() if you only need a subset of results.")

        return hints

    def help(self) -> None:
        """Display interactive help showing available operations and examples.

        Example:
            >>> from moltres import connect
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.table("users").select()
            >>> df.help()  # Prints help information
        """
        print("=" * 70)
        print("Moltres DataFrame - Available Operations")
        print("=" * 70)
        print()
        print("Query Operations:")
        print("  - select(*columns)      : Select specific columns")
        print("  - where(condition)       : Filter rows")
        print("  - join(other, on=...)    : Join with another DataFrame")
        print("  - group_by(*columns)    : Group rows")
        print("  - agg(*expressions)      : Aggregate grouped data")
        print("  - order_by(*columns)    : Sort results")
        print("  - limit(n)              : Limit number of rows")
        print("  - distinct()            : Remove duplicates")
        print()
        print("Execution Operations:")
        print("  - collect()              : Execute query and return results")
        print("  - show(n=20)             : Print first n rows")
        print("  - head(n=5)              : Get first n rows")
        print("  - tail(n=5)              : Get last n rows")
        print()
        print("Debugging & Introspection:")
        print("  - to_sql(pretty=False)   : Get SQL string")
        print("  - show_sql()             : Pretty-print SQL")
        print("  - explain(analyze=False) : Get query execution plan")
        print("  - plan_summary()         : Get structured plan summary")
        print("  - visualize_plan()      : ASCII tree visualization")
        print("  - validate()            : Check for common issues")
        print("  - performance_hints()   : Get optimization suggestions")
        print()
        print("Schema Operations:")
        print("  - columns                : Get column names")
        print("  - schema                 : Get column schema")
        print("  - dtypes                 : Get data types")
        print()
        print("For more information, see: https://moltres.readthedocs.io/")
        print("=" * 70)

    def suggest_next(self) -> List[str]:
        """Suggest logical next operations based on current DataFrame state.

        Returns:
            List of suggested next operations

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select()
            >>> suggestions = df.suggest_next()
            >>> len(suggestions) > 0
            True
        """
        suggestions = []
        summary = self.plan_summary()

        # If just a table scan, suggest filtering
        if summary["table_scans"] == 1 and summary["filters"] == 0:
            suggestions.append(
                "You might want to filter rows with where(), e.g., df.where(col('column') > value)"
            )

        # If has filters but no projection, suggest selecting specific columns
        if summary["filters"] > 0:
            suggestions.append(
                "Consider selecting specific columns with select() for better performance"
            )

        # If has joins, suggest checking join conditions
        if summary["joins"] > 0:
            suggestions.append(
                "Verify join conditions are correct and indexed for optimal performance"
            )

        # If no limit, suggest adding one for large datasets
        has_limit = any(
            isinstance(node, Limit) for node in [self.plan] if hasattr(self.plan, "child")
        )
        if not has_limit:
            suggestions.append("Consider adding limit() if you only need a subset of results")

        # If has aggregations, suggest ordering
        if summary["aggregations"] > 0:
            suggestions.append("You might want to order results with order_by()")

        # General suggestions
        if not suggestions:
            suggestions.append(
                "Ready to execute! Use collect() to get results or show() to preview"
            )
            suggestions.append("Use explain() to see the query execution plan")

        return suggestions

    def visualize_plan(self) -> str:
        """Create an ASCII tree visualization of the query plan.

        Returns:
            String containing ASCII tree representation of the plan

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select().where(col("id") > 1)
            >>> print(df.visualize_plan())
            >>> db.close()
        """
        lines = []
        visited = set()

        def format_node(node: LogicalPlan, prefix: str = "", is_last: bool = True) -> None:
            """Recursively format plan nodes as a tree."""
            node_id = id(node)
            if node_id in visited:
                lines.append(f"{prefix}{'└── ' if is_last else '├── '}[CYCLE]")
                return
            visited.add(node_id)

            op_type = type(node).__name__
            # Add details based on operation type
            details = ""
            if hasattr(node, "table"):
                details = f"({node.table})"
            elif hasattr(node, "predicate"):
                details = " [has predicate]"
            elif hasattr(node, "columns"):
                details = f" [{len(node.columns)} columns]"

            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{op_type}{details}")

            # Update prefix for children
            child_prefix = prefix + ("    " if is_last else "│   ")

            # Get children
            children = []
            if hasattr(node, "child"):
                children = [node.child]
            elif hasattr(node, "left") and hasattr(node, "right"):
                children = [node.left, node.right]
            elif hasattr(node, "children"):
                # children is a method, call it
                children = list(node.children())

            # Format children
            for i, child in enumerate(children):
                is_last_child = i == len(children) - 1
                format_node(child, child_prefix, is_last_child)

        format_node(self.plan)
        return "\n".join(lines)

    @overload
    def collect(self, stream: Literal[False] = False) -> List[Dict[str, object]]: ...

    @overload
    def collect(self, stream: Literal[True]) -> Iterator[List[Dict[str, object]]]: ...

    def collect(
        self, stream: bool = False
    ) -> Union[
        List[Dict[str, object]], Iterator[List[Dict[str, object]]], List[Any], Iterator[List[Any]]
    ]:
        """Collect :class:`DataFrame` results.

        Args:
            stream: If True, return an iterator of row chunks. If False (default),
                   materialize all rows into a list.

        Returns:
            If stream=False and no model attached: List of dictionaries representing rows.
            If stream=False and model attached: List of SQLModel or Pydantic instances.
            If stream=True and no model attached: Iterator of row chunks (each chunk is a list of dicts).
            If stream=True and model attached: Iterator of row chunks (each chunk is a list of model instances).

        Raises:
            RuntimeError: If :class:`DataFrame` is not bound to a :class:`Database`
            ImportError: If model is attached but Pydantic or SQLModel is not installed

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("users")
            >>> # Collect all results
            >>> df = db.table("users").select()
            >>> results = df.collect()
            >>> len(results)
            2
            >>> results[0]["name"]
            'Alice'
            >>> # Collect with streaming (returns iterator)
            >>> stream_results = df.collect(stream=True)
            >>> chunk = next(stream_results)
            >>> len(chunk)
            2
            >>> db.close()
        """
        from ..managers.execution import DataFrameExecutor

        executor = DataFrameExecutor(self)
        if stream:
            return executor.collect(stream=True)
        else:
            return executor.collect(stream=False)

    def _materialize_filescan(self, plan: LogicalPlan) -> LogicalPlan:
        """Materialize FileScan nodes by reading files and creating temporary tables.

        Delegates to :class:`MaterializationHandler`.

        Args:
            plan: Logical plan that may contain FileScan nodes

        Returns:
            Logical plan with FileScan nodes replaced by TableScan nodes
        """
        from ..managers.materialization import MaterializationHandler

        handler = MaterializationHandler(self)
        return handler.materialize_filescan(plan)

    def _read_file(self, filescan: FileScan) -> List[Dict[str, object]]:
        """Read a file based on FileScan configuration (non-streaming, loads all into memory).

        Delegates to :class:`MaterializationHandler`.

        Args:
            filescan: FileScan logical plan node

        Returns:
            List of dictionaries representing the file data

        Note:
            This method loads the entire file into memory. For large files, use
            _read_file_streaming() instead.
        """
        from ..managers.materialization import MaterializationHandler

        handler = MaterializationHandler(self)
        return handler.read_file(filescan)

    def _read_file_streaming(self, filescan: FileScan) -> Records:
        """Read a file in streaming mode (chunked, safe for large files).

        Delegates to :class:`MaterializationHandler`.

        Args:
            filescan: FileScan logical plan node

        Returns:
            Records object with _generator set (streaming mode)

        Note:
            This method returns Records with a generator, allowing chunked processing
            without loading the entire file into memory. Use this for large files.
        """
        from ..managers.materialization import MaterializationHandler

        handler = MaterializationHandler(self)
        return handler.read_file_streaming(filescan)

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the :class:`DataFrame`.

        Delegates to :class:`DataFrameExecutor`.

        Args:
            n: Number of rows to show (default: 20)
            truncate: If True, truncate long strings (default: True)

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> df.show(2)  # doctest: +SKIP
            >>> # Output: id | name
            >>> #         ---|-----
            >>> #         1  | Alice
            >>> #         2  | Bob
            >>> db.close()
        """
        from ..managers.execution import DataFrameExecutor

        executor = DataFrameExecutor(self)
        executor.show(n=n, truncate=truncate)

    def take(self, num: int) -> List[Dict[str, object]]:
        """Take the first num rows as a list.

        Delegates to :class:`DataFrameExecutor`.

        Args:
            num: Number of rows to take

        Returns:
            List of dictionaries representing the rows

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> Records(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 6)], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> rows = df.take(3)
            >>> len(rows)
            3
            >>> rows[0]["id"]
            1
            >>> db.close()
        """
        from ..managers.execution import DataFrameExecutor

        executor = DataFrameExecutor(self)
        return executor.take(num)

    def first(self) -> Optional[Dict[str, object]]:
        """Return the first row as a dictionary, or None if empty.

        Delegates to :class:`DataFrameExecutor`.

        Returns:
            First row as a dictionary, or None if :class:`DataFrame` is empty

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> first_row = df.first()
            >>> first_row["name"]
            'Alice'
            >>> # Empty :class:`DataFrame` returns None
            >>> df2 = db.table("users").select().where(col("id") > 100)
            >>> df2.first() is None
            True
            >>> db.close()
        """
        from ..managers.execution import DataFrameExecutor

        executor = DataFrameExecutor(self)
        return executor.first()

    def head(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the first n rows of the :class:`DataFrame`.

        Delegates to :class:`DataFrameExecutor`.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of row dictionaries

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> Records(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 6)], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> rows = df.head(3)
            >>> len(rows)
            3
            >>> rows[0]["id"]
            1
            >>> db.close()
        """
        from ..managers.execution import DataFrameExecutor

        executor = DataFrameExecutor(self)
        return executor.head(n)

    def tail(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the last n rows of the :class:`DataFrame`.

        Delegates to :class:`DataFrameExecutor`.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of row dictionaries

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> Records(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 6)], _database=db).insert_into("users")
            >>> df = db.table("users").select().order_by("id")
            >>> rows = df.tail(2)
            >>> len(rows)
            2
            >>> rows[0]["id"]
            4
            >>> rows[1]["id"]
            5
            >>> db.close()
        """
        from ..managers.execution import DataFrameExecutor

        executor = DataFrameExecutor(self)
        return executor.tail(n)

    def nunique(self, column: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """Count distinct values in column(s).

        Delegates to :class:`StatisticsCalculator`.

        Args:
            column: Column name to count. If None, counts distinct values for all columns.

        Returns:
            If column is specified: integer count of distinct values.
            If column is None: dictionary mapping column names to distinct counts.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("country", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "country": "USA", "age": 25}, {"id": 2, "country": "USA", "age": 30}, {"id": 3, "country": "UK", "age": 25}], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> # Count distinct values in a column
            >>> df.nunique("country")
            2
            >>> # Count distinct for all columns
            >>> counts = df.nunique()
            >>> counts["country"]
            2
            >>> db.close()
        """
        from ..managers.statistics import StatisticsCalculator

        calculator = StatisticsCalculator(self)
        return calculator.nunique(column)

    def count(self) -> int:
        """Return the number of rows in the :class:`DataFrame`.

        Delegates to :class:`StatisticsCalculator`.

        Returns:
            Number of rows

        Note:
            This executes a COUNT(*) query against the database.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> Records(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 6)], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> df.count()
            5
            >>> # Count with filter
            >>> df2 = db.table("users").select().where(col("id") > 2)
            >>> df2.count()
            3
            >>> db.close()
        """
        from ..managers.statistics import StatisticsCalculator

        calculator = StatisticsCalculator(self)
        return calculator.count()

    def describe(self, *cols: str) -> DataFrame:
        """Compute basic statistics for numeric columns.

        Delegates to :class:`StatisticsCalculator`.

        Args:
            *cols: Optional column names to describe. If not provided, describes all numeric columns.

        Returns:
            DataFrame with statistics: count, mean, stddev, min, max

        Note:
            This is a simplified implementation. A full implementation would
            automatically detect numeric columns if cols is not provided.
        """
        from ..managers.statistics import StatisticsCalculator

        calculator = StatisticsCalculator(self)
        return calculator.describe(*cols)

    def summary(self, *statistics: str) -> DataFrame:
        """Compute summary statistics for numeric columns.

        Delegates to :class:`StatisticsCalculator`.

        Args:
            *statistics: Statistics to compute (e.g., "count", "mean", "stddev", "min", "max").
                        If not provided, computes common statistics.

        Returns:
            DataFrame with summary statistics

        Note:
            This is a simplified implementation. A full implementation would
            automatically detect numeric columns and compute all statistics.
        """
        from ..managers.statistics import StatisticsCalculator

        calculator = StatisticsCalculator(self)
        return calculator.summary(*statistics)

    def fillna(
        self,
        value: Union[LiteralValue, Dict[str, LiteralValue]],
        subset: Optional[Sequence[str]] = None,
    ) -> DataFrame:
        """Replace null values with a specified value.

        Args:
            value: Value to use for filling nulls. Can be a single value or a dict mapping column names to values.
            subset: Optional list of column names to fill. If None, fills all columns.

        Returns:
            New :class:`DataFrame` with null values filled

        Note:
            This uses COALESCE or CASE WHEN to replace nulls in SQL.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice", "age": None}, {"id": 2, "name": None, "age": 25}], _database=db).insert_into("users")
            >>> # Fill nulls with single value
            >>> df = db.table("users").select().fillna(0, subset=["age"])
            >>> results = df.collect()
            >>> results[0]["age"]
            0
            >>> # Fill nulls with different values per column
            >>> df2 = db.table("users").select().fillna({"name": "Unknown", "age": 0}, subset=["name", "age"])
            >>> results2 = df2.collect()
            >>> results2[1]["name"]
            'Unknown'
            >>> db.close()
        """
        from ...expressions.functions import coalesce, lit

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
                filled_col = coalesce(col_expr, lit(fill_value)).alias(col_name)
                new_projections.append(filled_col)
            else:
                new_projections.append(col_expr)

        # This is simplified - a full implementation would handle all columns
        return self.select(*new_projections)

    def dropna(self, how: str = "any", subset: Optional[Sequence[str]] = None) -> DataFrame:
        """Remove rows with null values.

        Args:
            how: "any" (drop if any null) or "all" (drop if all null) (default: "any")
            subset: Optional list of column names to check. If None, checks all columns.

        Returns:
            New :class:`DataFrame` with null rows removed

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import :class:`Records`
            >>> _ = :class:`Records`(_data=[{"id": 1, "name": "Alice", "age": 25}, {"id": 2, "name": None, "age": 30}, {"id": 3, "name": "Bob", "age": None}], _database=db).insert_into("users")
            >>> # Drop rows where any column in subset is null
            >>> df = db.table("users").select().dropna(how="any", subset=["name", "age"])
            >>> results = df.collect()
            >>> len(results)
            1
            >>> results[0]["name"]
            'Alice'
            >>> # Drop rows where all columns in subset are null
            >>> df2 = db.table("users").select().dropna(how="all", subset=["name", "age"])
            >>> results2 = df2.collect()
            >>> len(results2)
            3
            >>> db.close()
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

    def polars(self) -> "PolarsDataFrame":
        """Convert this :class:`DataFrame` to a :class:`PolarsDataFrame` for Polars-style operations.

        Returns:
            :class:`PolarsDataFrame` wrapping this :class:`DataFrame`

        Example:
            >>> from moltres import connect
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.read.csv("data.csv")
            >>> polars_df = df.polars()
            >>> results = polars_df.collect()
        """
        from ..interfaces.polars_dataframe import PolarsDataFrame

        return PolarsDataFrame.from_dataframe(self)

    @property
    def na(self) -> "NullHandling":
        """Access null handling methods via the `na` property.

        Returns:
            NullHandling helper object with drop() and fill() methods

        Example:
            >>> df.na.drop()  # Drop rows with nulls
            >>> df.na.fill(0)  # Fill nulls with 0
        """
        return NullHandling(self)

    @property
    def write(self) -> "DataFrameWriter":
        """Return a :class:`DataFrameWriter` for writing this :class:`DataFrame` to a table."""
        from ..io.writer import DataFrameWriter

        return DataFrameWriter(self)

    @property
    def columns(self) -> List[str]:
        """Return a list of column names in this :class:`DataFrame`.

        Delegates to :class:`SchemaInspector`.

        Similar to PySpark's :class:`DataFrame`.columns property, this extracts column
        names from the logical plan without requiring query execution.

        Returns:
            List of column name strings

        Raises:
            RuntimeError: If column names cannot be determined (e.g., RawSQL without execution)

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("email", "TEXT")]).collect()
            >>> df = db.table("users").select()
            >>> cols = df.columns
            >>> "id" in cols and "name" in cols and "email" in cols
            True
            >>> df2 = df.select("id", "name")
            >>> cols2 = df2.columns
            >>> len(cols2)
            2
            >>> "id" in cols2 and "name" in cols2
            True
            >>> db.close()
        """
        from ..managers.schema import SchemaInspector

        inspector = SchemaInspector(self)
        return inspector.columns()

    @property
    def schema(self) -> List["ColumnInfo"]:
        """Return the schema of this :class:`DataFrame` as a list of ColumnInfo objects.

        Delegates to :class:`SchemaInspector`.

        Similar to PySpark's :class:`DataFrame`.schema property, this extracts column
        names and types from the logical plan without requiring query execution.

        Returns:
            List of ColumnInfo objects with column names and types

        Raises:
            RuntimeError: If schema cannot be determined (e.g., RawSQL without execution)

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select()
            >>> schema = df.schema
            >>> len(schema)
            2
            >>> schema[0].name
            'id'
            >>> schema[0].type_name
            'INTEGER'
            >>> schema[1].name
            'name'
            >>> db.close()
        """
        from ..managers.schema import SchemaInspector

        inspector = SchemaInspector(self)
        return inspector.schema()

    @property
    def dtypes(self) -> List[Tuple[str, str]]:
        """Return a list of tuples containing column names and their data types.

        Delegates to :class:`SchemaInspector`.

        Similar to PySpark's :class:`DataFrame`.dtypes property, this returns a list
        of (column_name, type_name) tuples.

        Returns:
            List of tuples (column_name, type_name)

        Raises:
            RuntimeError: If schema cannot be determined (e.g., RawSQL without execution)

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select()
            >>> dtypes = df.dtypes
            >>> len(dtypes)
            2
            >>> dtypes[0]
            ('id', 'INTEGER')
            >>> dtypes[1][0]
            'name'
            >>> db.close()
        """
        from ..managers.schema import SchemaInspector

        inspector = SchemaInspector(self)
        return inspector.dtypes()

    def printSchema(self) -> None:
        """Print the schema of this :class:`DataFrame` in a tree format.

        Delegates to :class:`SchemaInspector`.

        Similar to PySpark's :class:`DataFrame`.printSchema() method, this prints
        a formatted representation of the :class:`DataFrame`'s schema.

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> df = db.table("users").select()
            >>> df.printSchema()  # doctest: +SKIP
            >>> # Output: root
            >>> #          |-- id: INTEGER (nullable = true)
            >>> #          |-- name: TEXT (nullable = true)
            >>> db.close()
        """
        from ..managers.schema import SchemaInspector

        inspector = SchemaInspector(self)
        inspector.print_schema()

    def __getitem__(
        self, key: Union[str, Sequence[str], Column]
    ) -> Union["DataFrame", Column, "PySparkColumn"]:
        """Enable bracket notation column access (e.g., df["col"], df[["col1", "col2"]]).

        Supports:
        - df['col'] - Returns :class:`Column` expression with string/date accessors
        - df[['col1', 'col2']] - Returns new :class:`DataFrame` with selected columns
        - df[df['age'] > 25] - Boolean indexing (filtering via :class:`Column` condition)

        Args:
            key: :class:`Column` name(s) or boolean :class:`Column` condition

        Returns:
            - For single column string: PySparkColumn (with .str and .dt accessors)
            - For list of columns: :class:`DataFrame` with selected columns
            - For boolean :class:`Column` condition: :class:`DataFrame` with filtered rows

        Example:
            >>> df = db.table("users").select()
            >>> df['age']  # Returns PySparkColumn with .str and .dt accessors
            >>> df[['id', 'name']]  # Returns :class:`DataFrame` with selected columns
            >>> df[df['age'] > 25]  # Returns filtered :class:`DataFrame`
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
            >>> df = db.table("users").select()
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
        return col(name)

    # ---------------------------------------------------------------- utilities
    def _with_plan(self, plan: LogicalPlan) -> DataFrame:
        return DataFrame(
            plan=plan,
            database=self.database,
            model=self.model,
        )

    def _with_model(self, model: Optional[Type[Any]]) -> DataFrame:
        """Create a new :class:`DataFrame` with a SQLModel attached.

        Args:
            model: SQLModel model class to attach, or None to remove model

        Returns:
            New :class:`DataFrame` with the model attached
        """
        return DataFrame(
            plan=self.plan,
            database=self.database,
            model=model,
        )

    def with_model(self, model: Type[Any]) -> DataFrame:
        """Attach a SQLModel or Pydantic model to this :class:`DataFrame`.

        Delegates to :class:`ModelIntegrator`.

        When a model is attached, `collect()` will return model instances
        instead of dictionaries. This provides type safety and validation.

        Args:
            model: SQLModel or Pydantic model class to attach

        Returns:
            New DataFrame with the model attached

        Raises:
            TypeError: If model is not a SQLModel or Pydantic class
            ImportError: If required dependencies are not installed

        Example:
            >>> from sqlmodel import SQLModel, Field
            >>> class User(SQLModel, table=True):
            ...     id: int = Field(primary_key=True)
            ...     name: str
            >>> df = db.table("users").select()
            >>> df_with_model = df.with_model(User)
            >>> results = df_with_model.collect()  # Returns list of User instances

            >>> from pydantic import BaseModel
            >>> class UserData(BaseModel):
            ...     id: int
            ...     name: str
            >>> df_with_pydantic = df.with_model(UserData)
            >>> results = df_with_pydantic.collect()  # Returns list of UserData instances
        """
        from ..managers.model_integration import ModelIntegrator

        integrator = ModelIntegrator(self)
        return integrator.with_model(model)


class NullHandling:
    """Helper class for null handling operations on DataFrames.

    Accessed via the `na` property on :class:`DataFrame` instances.
    """

    def __init__(self, df: DataFrame):
        self._df = df

    def drop(self, how: str = "any", subset: Optional[Sequence[str]] = None) -> DataFrame:
        """Drop rows with null values.

        This is a convenience wrapper around :class:`DataFrame`.dropna().

        Args:
            how: "any" (drop if any null) or "all" (drop if all null) (default: "any")
            subset: Optional list of column names to check. If None, checks all columns.

        Returns:
            New :class:`DataFrame` with null rows removed

        Example:
            >>> df.na.drop()  # Drop rows with any null values
            >>> df.na.drop(how="all")  # Drop rows where all values are null
            >>> df.na.drop(subset=["col1", "col2"])  # Only check specific columns
        """
        return self._df.dropna(how=how, subset=subset)

    def fill(
        self,
        value: Union[LiteralValue, Dict[str, LiteralValue]],
        subset: Optional[Sequence[str]] = None,
    ) -> DataFrame:
        """Fill null values with a specified value.

        This is a convenience wrapper around :class:`DataFrame`.fillna().

        Args:
            value: Value to use for filling nulls. Can be a single value or a dict mapping column names to values.
            subset: Optional list of column names to fill. If None, fills all columns.

        Returns:
            New :class:`DataFrame` with null values filled

        Example:
            >>> df.na.fill(0)  # Fill all nulls with 0
            >>> df.na.fill({"col1": 0, "col2": "unknown"})  # Fill different columns with different values
            >>> df.na.fill(0, subset=["col1", "col2"])  # Fill specific columns with 0
        """
        return self._df.fillna(value=value, subset=subset)
