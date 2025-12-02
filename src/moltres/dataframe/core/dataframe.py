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
    overload,
)

from ...expressions.column import Column, LiteralValue, col
from ...logical import operators
from ...logical.plan import FileScan, LogicalPlan, RawSQL
from ...sql.compiler import compile_plan
from ..helpers.dataframe_helpers import DataFrameHelpersMixin

if TYPE_CHECKING:
    from sqlalchemy.sql import Select
    from ...io.records import Records
    from ...table.table import Database, TableHandle
    from ...utils.inspector import ColumnInfo
    from ..groupby import GroupedDataFrame
    from ..interfaces.polars_dataframe import PolarsDataFrame
    from ..io.writer import DataFrameWriter
    from ..columns.pyspark_column import PySparkColumn
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
    model: Optional[Type[Any]] = None  # SQLModel class, if attached

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
    def to_sql(self) -> str:
        """Convert the :class:`DataFrame`'s logical plan to a SQL string.

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
            return str(stmt.compile(compile_kwargs={"literal_binds": True}))
        return str(stmt)

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
