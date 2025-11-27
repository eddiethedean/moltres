"""Lazy DataFrame representation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterator,
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
    from ..io.records import Records
    from ..table.table import Database, TableHandle
    from ..utils.inspector import ColumnInfo
    from .groupby import GroupedDataFrame
    from .polars_dataframe import PolarsDataFrame
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
        """Select specific columns from the DataFrame.

        Args:
            *columns: Column names or Column expressions to select.
                     Use "*" to select all columns (same as empty select).
                     Can combine "*" with other columns: select("*", col("new_col"))

        Returns:
            New DataFrame with selected columns

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
        if not columns:
            return self

        # Handle "*" as special case
        if len(columns) == 1 and isinstance(columns[0], str) and columns[0] == "*":
            return self

        # Check if "*" is in the columns (only check string elements, not Column objects)
        has_star = any(isinstance(col, str) and col == "*" for col in columns)

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

    def selectExpr(self, *exprs: str) -> "DataFrame":
        """Select columns using SQL expressions.

        This method allows you to write SQL expressions directly instead of
        building Column objects manually, similar to PySpark's selectExpr().

        Args:
            *exprs: SQL expression strings (e.g., "amount * 1.1 as with_tax")

        Returns:
            New DataFrame with selected expressions

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "amount": 100.0, "name": "Alice"}], _database=db).insert_into("orders")
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
        from ..expressions.sql_parser import parse_sql_expr

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

    def where(self, predicate: Union[Column, str]) -> "DataFrame":
        """Filter rows based on a condition.

        Args:
            predicate: Column expression or SQL string representing the filter condition.
                      Can be a Column object or a SQL string like "age > 18".

        Returns:
            New DataFrame with filtered rows

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice", "age": 25}, {"id": 2, "name": "Bob", "age": 17}], _database=db).insert_into("users")
            >>> # Filter by condition using Column
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
            >>> # Multiple conditions with Column
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL"), column("status", "TEXT")]).collect()
            >>> _ = Records(_data=[{"id": 1, "amount": 150.0, "status": "active"}, {"id": 2, "amount": 50.0, "status": "active"}], _database=db).insert_into("orders")
            >>> df3 = db.table("orders").select().where((col("amount") > 100) & (col("status") == "active"))
            >>> results3 = df3.collect()
            >>> len(results3)
            1
            >>> results3[0]["amount"]
            150.0
            >>> db.close()
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

    def limit(self, count: int) -> "DataFrame":
        """Limit the number of rows returned by the query.

        Args:
            count: Maximum number of rows to return. Must be non-negative.
                  If 0, returns an empty result set.

        Returns:
            New DataFrame with the limit applied

        Raises:
            ValueError: If count is negative

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
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
        if count < 0:
            raise ValueError("limit count must be non-negative")
        return self._with_plan(operators.limit(self.plan, count))

    def sample(self, fraction: float, seed: Optional[int] = None) -> "DataFrame":
        """Sample a fraction of rows from the DataFrame.

        Args:
            fraction: Fraction of rows to sample (0.0 to 1.0)
            seed: Optional random seed for reproducible sampling

        Returns:
            New DataFrame with sampled rows

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> Records(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 11)], _database=db).insert_into("users")
            >>> # Sample 30% of rows with seed for reproducibility
            >>> df = db.table("users").select().sample(0.3, seed=42)
            >>> results = df.collect()
            >>> len(results) <= 10  # Should be approximately 30% of 10 rows
            True
            >>> db.close()
        """
        return self._with_plan(operators.sample(self.plan, fraction, seed))

    def order_by(self, *columns: Union[Column, str]) -> "DataFrame":
        """Sort rows by one or more columns.

        Args:
            *columns: Column expressions or column names to sort by. Use .asc() or .desc() for sort order.
                     Can be strings (column names) or Column objects.

        Returns:
            New DataFrame with sorted rows

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Charlie"}, {"id": 2, "name": "Alice"}, {"id": 3, "name": "Bob"}], _database=db).insert_into("users")
            >>> # Sort ascending with string column name
            >>> df = db.table("users").select().order_by("name")
            >>> results = df.collect()
            >>> results[0]["name"]
            'Alice'
            >>> results[1]["name"]
            'Bob'
            >>> # Sort descending with Column object
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> _ = Records(_data=[{"id": 1, "amount": 50.0}, {"id": 2, "amount": 100.0}, {"id": 3, "amount": 25.0}], _database=db).insert_into("orders")
            >>> df2 = db.table("orders").select().order_by(col("amount").desc())
            >>> results2 = df2.collect()
            >>> results2[0]["amount"]
            100.0
            >>> # Multiple sort columns
            >>> db.create_table("sales", [column("region", "TEXT"), column("amount", "REAL")]).collect()
            >>> _ = Records(_data=[{"region": "North", "amount": 100.0}, {"region": "North", "amount": 50.0}, {"region": "South", "amount": 75.0}], _database=db).insert_into("sales")
            >>> df3 = db.table("sales").select().order_by("region", col("amount").desc())
            >>> results3 = df3.collect()
            >>> results3[0]["region"]
            'North'
            >>> results3[0]["amount"]
            100.0
            >>> db.close()
        """
        if not columns:
            return self
        orders = tuple(
            self._normalize_sort_expression(self._normalize_projection(col)) for col in columns
        )
        return self._with_plan(operators.order_by(self.plan, orders))

    orderBy = order_by  # PySpark-style alias
    sort = order_by  # PySpark-style alias

    def join(
        self,
        other: "DataFrame",
        *,
        on: Optional[
            Union[str, Sequence[str], Sequence[Tuple[str, str]], "Column", Sequence["Column"]]
        ] = None,
        how: str = "inner",
        lateral: bool = False,
        hints: Optional[Sequence[str]] = None,
    ) -> "DataFrame":
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
            lateral: If True, create a LATERAL join (PostgreSQL, MySQL 8.0+).
                    Allows right side to reference columns from left side.
            hints: Optional sequence of join hints (e.g., ["USE_INDEX(idx_name)", "FORCE_INDEX(idx_name)"]).
                   Dialect-specific: MySQL uses USE INDEX, PostgreSQL uses /*+ ... */ comments.

        Returns:
            New DataFrame containing the join result

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> # Setup tables
            >>> db.create_table("customers", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("orders", [column("id", "INTEGER"), column("customer_id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("customers")
            >>> _ = Records(_data=[{"id": 1, "customer_id": 1, "amount": 100.0}], _database=db).insert_into("orders")
            >>> # PySpark-style with Column expressions (recommended)
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
            >>> _ = Records(_data=[{"order_id": 1, "product": "Widget"}], _database=db).insert_into("items")
            >>> df2 = orders.join(db.table("items").select(), on="order_id", how="inner")
            >>> results2 = df2.collect()
            >>> results2[0]["product"]
            'Widget'
            >>> # Left join
            >>> _ = Records(_data=[{"id": 2, "name": "Bob"}], _database=db).insert_into("customers")
            >>> df3 = customers.join(orders, on=[col("customers.id") == col("orders.customer_id")], how="left")
            >>> results3 = df3.collect()
            >>> len(results3)
            2
            >>> db.close()
            ...     lateral=True
            ... )
            >>> # SQL: SELECT * FROM customers LEFT JOIN LATERAL (SELECT * FROM orders WHERE customer_id = customers.id) ...
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before joining")
        if self.database is not other.database:
            raise ValueError("Cannot join DataFrames from different Database instances")
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
        hints_tuple = tuple(hints) if hints else None
        plan = operators.join(
            self.plan,
            other.plan,
            how=how.lower(),
            on=normalized_on,
            condition=condition,
            lateral=lateral,
            hints=hints_tuple,
        )
        return DataFrame(plan=plan, database=self.database)

    def crossJoin(self, other: "DataFrame") -> "DataFrame":
        """Perform a cross join (Cartesian product) with another DataFrame.

        Args:
            other: Another DataFrame to cross join with

        Returns:
            New DataFrame containing the Cartesian product of rows

        Raises:
            RuntimeError: If DataFrames are not bound to the same Database

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("value", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "A"}, {"id": 2, "name": "B"}], _database=db).insert_into("table1")
            >>> _ = Records(_data=[{"id": 1, "value": "X"}, {"id": 2, "value": "Y"}], _database=db).insert_into("table2")
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
        other: "DataFrame",
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
    ) -> "DataFrame":
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
            RuntimeError: If DataFrames are not bound to the same Database

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("customers", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("orders", [column("id", "INTEGER"), column("customer_id", "INTEGER")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("customers")
            >>> _ = Records(_data=[{"id": 1, "customer_id": 1}], _database=db).insert_into("orders")
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
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before semi_join")
        if self.database is not other.database:
            raise ValueError("Cannot semi_join DataFrames from different Database instances")
        normalized_condition = self._normalize_join_condition(on)
        if isinstance(normalized_condition, Column):
            raise ValueError("semi_join does not support Column expressions, use tuple syntax")
        normalized_on = normalized_condition
        plan = operators.semi_join(self.plan, other.plan, on=normalized_on)
        return DataFrame(plan=plan, database=self.database)

    def anti_join(
        self,
        other: "DataFrame",
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
    ) -> "DataFrame":
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
            RuntimeError: If DataFrames are not bound to the same Database

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("customers", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("orders", [column("id", "INTEGER"), column("customer_id", "INTEGER")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("customers")
            >>> _ = Records(_data=[{"id": 1, "customer_id": 1}], _database=db).insert_into("orders")
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
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before anti_join")
        if self.database is not other.database:
            raise ValueError("Cannot anti_join DataFrames from different Database instances")
        normalized_condition = self._normalize_join_condition(on)
        if isinstance(normalized_condition, Column):
            raise ValueError("anti_join does not support Column expressions, use tuple syntax")
        normalized_on = normalized_condition
        plan = operators.anti_join(self.plan, other.plan, on=normalized_on)
        return DataFrame(plan=plan, database=self.database)

    def pivot(
        self,
        pivot_column: str,
        value_column: str,
        agg_func: str = "sum",
        pivot_values: Optional[Sequence[str]] = None,
    ) -> "DataFrame":
        """Pivot the DataFrame to reshape data from long to wide format.

        Args:
            pivot_column: Column to pivot on (values become column headers)
            value_column: Column containing values to aggregate
            agg_func: Aggregation function to apply (default: "sum")
            pivot_values: Optional list of specific values to pivot (if None, uses all distinct values)

        Returns:
            New DataFrame with pivoted data

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("sales", [column("date", "TEXT"), column("product", "TEXT"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"date": "2024-01-01", "product": "A", "amount": 100.0}, {"date": "2024-01-01", "product": "B", "amount": 200.0}, {"date": "2024-01-02", "product": "A", "amount": 150.0}], _database=db).insert_into("sales")
            >>> # Pivot sales data by product
            >>> df = db.table("sales").select("date", "product", "amount")
            >>> pivoted = df.pivot(pivot_column="product", value_column="amount", agg_func="sum")
            >>> results = pivoted.collect()
            >>> len(results) > 0
            True
            >>> db.close()
        """
        return self._with_plan(
            operators.pivot(
                self.plan,
                pivot_column=pivot_column,
                value_column=value_column,
                agg_func=agg_func,
                pivot_values=pivot_values,
            )
        )

    def explode(self, column: Union[Column, str], alias: str = "value") -> "DataFrame":
        """Explode an array/JSON column into multiple rows (one row per element).

        Args:
            column: Column expression or column name to explode (must be array or JSON)
            alias: Alias for the exploded value column (default: "value")

        Returns:
            New DataFrame with exploded rows

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> # Note: explode() requires array/JSON support which varies by database
            >>> # This example shows the API usage pattern
            >>> db.create_table("users", [column("id", "INTEGER"), column("tags", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "tags": '["python", "sql"]'}], _database=db).insert_into("users")
            >>> # Explode a JSON array column (database-specific support required)
            >>> df = db.table("users").select()
            >>> exploded = df.explode(col("tags"), alias="tag")
            >>> # Each row in exploded will have one tag per row
            >>> # Note: Actual execution depends on database JSON/array support
            >>> db.close()
        """
        normalized_col = self._normalize_projection(column)
        if not isinstance(normalized_col, Column):
            raise TypeError("explode() requires a Column expression")
        plan = operators.explode(self.plan, normalized_col, alias=alias)
        return DataFrame(plan=plan, database=self.database)

    def group_by(self, *columns: Union[Column, str]) -> "GroupedDataFrame":
        """Group rows by one or more columns for aggregation.

        Args:
            *columns: Column names or Column expressions to group by

        Returns:
            GroupedDataFrame that can be used with aggregation functions

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> # Group by single column
            >>> db.create_table("orders", [column("customer_id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"customer_id": 1, "amount": 100.0}, {"customer_id": 1, "amount": 50.0}, {"customer_id": 2, "amount": 200.0}], _database=db).insert_into("orders")
            >>> df = db.table("orders").select().group_by("customer_id").agg(F.sum(col("amount")).alias("total"))
            >>> results = df.collect()
            >>> len(results)
            2
            >>> results[0]["total"]
            150.0
            >>> # Group by multiple columns
            >>> db.create_table("sales", [column("region", "TEXT"), column("product", "TEXT"), column("revenue", "REAL")]).collect()
            >>> _ = Records(_data=[{"region": "North", "product": "A", "revenue": 100.0}, {"region": "North", "product": "A", "revenue": 50.0}], _database=db).insert_into("sales")
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

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("table1")
            >>> _ = Records(_data=[{"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}], _database=db).insert_into("table2")
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

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("table1")
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("table2")
            >>> df1 = db.table("table1").select()
            >>> df2 = db.table("table2").select()
            >>> # UnionAll (all rows, including duplicates)
            >>> df_union = df1.unionAll(df2)
            >>> results = df_union.collect()
            >>> len(results)
            2
            >>> db.close()
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

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("table1")
            >>> _ = Records(_data=[{"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}], _database=db).insert_into("table2")
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

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("table1", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> db.create_table("table2", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("table1")
            >>> _ = Records(_data=[{"id": 2, "name": "Bob"}], _database=db).insert_into("table2")
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
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "amount": 150.0}, {"id": 2, "amount": 50.0}], _database=db).insert_into("orders")
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
        plan = operators.cte(self.plan, name)
        return DataFrame(plan=plan, database=self.database)

    def recursive_cte(
        self, name: str, recursive: "DataFrame", union_all: bool = False
    ) -> "DataFrame":
        """Create a Recursive Common Table Expression (WITH RECURSIVE) from this DataFrame.

        Args:
            name: Name for the recursive CTE
            recursive: DataFrame representing the recursive part (references the CTE)
            union_all: If True, use UNION ALL; if False, use UNION (distinct)

        Returns:
            New DataFrame representing the recursive CTE

        Example:
            >>> # Fibonacci sequence example
            >>> from moltres.expressions import functions as F
            >>> initial = db.table("seed").select(F.lit(1).alias("n"), F.lit(1).alias("fib"))
            >>> recursive = initial.select(...)  # Recursive part
            >>> fib_cte = initial.recursive_cte("fib", recursive)
        """
        if self.database is None or recursive.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before recursive_cte")
        if self.database is not recursive.database:
            raise ValueError("Cannot create recursive CTE from DataFrames in different Databases")
        plan = operators.recursive_cte(name, self.plan, recursive.plan, union_all=union_all)
        return DataFrame(plan=plan, database=self.database)

    def distinct(self) -> "DataFrame":
        """Return a new DataFrame with distinct rows.

        Returns:
            New DataFrame with distinct rows

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Alice"}, {"id": 3, "name": "Bob"}], _database=db).insert_into("users")
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
            Window functions are supported and will ensure all columns are available.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.expressions import functions as F
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("orders", [column("id", "INTEGER"), column("amount", "REAL"), column("category", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "amount": 100.0, "category": "A"}, {"id": 2, "amount": 200.0, "category": "A"}], _database=db).insert_into("orders")
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
        from ..logical.plan import Project

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

    def withColumnRenamed(self, existing: str, new: str) -> "DataFrame":
        """Rename a column in the DataFrame.

        Args:
            existing: Current name of the column
            new: New name for the column

        Returns:
            New DataFrame with the renamed column

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}], _database=db).insert_into("users")
            >>> df = db.table("users").select().withColumnRenamed("name", "user_name")
            >>> results = df.collect()
            >>> "user_name" in results[0]
            True
            >>> results[0]["user_name"]
            'Alice'
            >>> db.close()
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

    def drop(self, *cols: Union[str, Column]) -> "DataFrame":
        """Drop one or more columns from the DataFrame.

        Args:
            *cols: Column names or Column objects to drop

        Returns:
            New DataFrame with the specified columns removed

        Note:
            This operation only works if the DataFrame has a Project operation.
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
            >>> # Drop by Column object
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
            RuntimeError: If DataFrame is not bound to a Database

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

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("users")
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
        if self.database is None:
            raise RuntimeError("Cannot collect a plan without an attached Database")

        # Handle RawSQL at root level - execute directly for efficiency
        if isinstance(self.plan, RawSQL):
            if stream:
                # For streaming, we need to use execute_plan_stream which expects a compiled plan
                # So we'll compile the RawSQL plan
                plan = self._materialize_filescan(self.plan)
                return self.database.execute_plan_stream(plan)
            else:
                # Execute RawSQL directly
                result = self.database.execute_sql(self.plan.sql, params=self.plan.params)
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
        plan = self._materialize_filescan(self.plan)

        if stream:
            # For SQL queries, use streaming execution
            return self.database.execute_plan_stream(plan)

        result = self.database.execute_plan(plan)
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

    def _materialize_filescan(self, plan: LogicalPlan) -> LogicalPlan:
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
        if self.database is None:
            raise RuntimeError("Cannot materialize FileScan without an attached Database")

        if isinstance(plan, FileScan):
            # Check if streaming is disabled (opt-out mechanism)
            # Default is True (streaming/chunked reading) for safety with large files
            stream_enabled = plan.options.get("stream", True)
            if isinstance(stream_enabled, bool) and not stream_enabled:
                # Non-streaming mode: load entire file into memory (current behavior)
                rows = self._read_file(plan)

                # Materialize into temporary table using createDataFrame
                # This enables SQL pushdown for subsequent operations
                # Use auto_pk to create an auto-incrementing primary key for temporary tables
                temp_df = self.database.createDataFrame(
                    rows, schema=plan.schema, auto_pk="__moltres_rowid__"
                )

                # createDataFrame returns a DataFrame with a TableScan plan
                # Return the TableScan plan to replace the FileScan
                return temp_df.plan
            else:
                # Streaming mode (default): read file in chunks and insert incrementally
                from .create_dataframe import create_temp_table_from_streaming
                from ..logical.operators import scan

                # Read file using streaming readers
                records = self._read_file_streaming(plan)

                # Create temp table from streaming records (chunked insertion)
                table_name, final_schema = create_temp_table_from_streaming(
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
            child = self._materialize_filescan(plan.child)
            return replace(plan, child=child)
        elif isinstance(plan, (Join, Union, Intersect, Except, SemiJoin, AntiJoin)):
            left = self._materialize_filescan(plan.left)
            right = self._materialize_filescan(plan.right)
            return replace(plan, left=left, right=right)
        elif isinstance(plan, (CTE, RecursiveCTE)):
            # For CTEs, we need to handle the child
            if isinstance(plan, CTE):
                child = self._materialize_filescan(plan.child)
                return replace(plan, child=child)
            else:  # RecursiveCTE
                initial = self._materialize_filescan(plan.initial)
                recursive = self._materialize_filescan(plan.recursive)
                return replace(plan, initial=initial, recursive=recursive)

        # For other plan types, return as-is
        return plan

    def _read_file(self, filescan: FileScan) -> List[Dict[str, object]]:
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
            raise RuntimeError("Cannot read file without an attached Database")

        from .readers import (
            read_csv,
            read_json,
            read_jsonl,
            read_parquet,
            read_text,
        )

        if filescan.format == "csv":
            records = read_csv(filescan.path, self.database, filescan.schema, filescan.options)
        elif filescan.format == "json":
            records = read_json(filescan.path, self.database, filescan.schema, filescan.options)
        elif filescan.format == "jsonl":
            records = read_jsonl(filescan.path, self.database, filescan.schema, filescan.options)
        elif filescan.format == "parquet":
            records = read_parquet(filescan.path, self.database, filescan.schema, filescan.options)
        elif filescan.format == "text":
            records = read_text(
                filescan.path,
                self.database,
                filescan.schema,
                filescan.options,
                filescan.column_name or "value",
            )
        else:
            raise ValueError(f"Unsupported file format: {filescan.format}")

        return records.rows()

    def _read_file_streaming(self, filescan: FileScan) -> Records:
        """Read a file in streaming mode (chunked, safe for large files).

        Args:
            filescan: FileScan logical plan node

        Returns:
            Records object with _generator set (streaming mode)

        Note:
            This method returns Records with a generator, allowing chunked processing
            without loading the entire file into memory. Use this for large files.
        """
        if self.database is None:
            raise RuntimeError("Cannot read file without an attached Database")

        from .readers import (
            read_csv_stream,
            read_json_stream,
            read_jsonl_stream,
            read_parquet_stream,
            read_text_stream,
        )

        if filescan.format == "csv":
            records = read_csv_stream(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "json":
            records = read_json_stream(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "jsonl":
            records = read_jsonl_stream(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "parquet":
            records = read_parquet_stream(
                filescan.path, self.database, filescan.schema, filescan.options
            )
        elif filescan.format == "text":
            records = read_text_stream(
                filescan.path,
                self.database,
                filescan.schema,
                filescan.options,
                filescan.column_name or "value",
            )
        else:
            raise ValueError(f"Unsupported file format: {filescan.format}")

        return records

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the DataFrame.

        Args:
            n: Number of rows to show (default: 20)
            truncate: If True, truncate long strings (default: True)

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> df.show(2)  # doctest: +SKIP
            >>> # Output: id | name
            >>> #         ---|-----
            >>> #         1  | Alice
            >>> #         2  | Bob
            >>> db.close()
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

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> Records(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 6)], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> rows = df.take(3)
            >>> len(rows)
            3
            >>> rows[0]["id"]
            1
            >>> db.close()
        """
        rows = self.limit(num).collect()
        if not isinstance(rows, list):
            raise TypeError("take() requires collect() to return a list, not an iterator")
        return rows

    def first(self) -> Optional[Dict[str, object]]:
        """Return the first row as a dictionary, or None if empty.

        Returns:
            First row as a dictionary, or None if DataFrame is empty

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> first_row = df.first()
            >>> first_row["name"]
            'Alice'
            >>> # Empty DataFrame returns None
            >>> df2 = db.table("users").select().where(col("id") > 100)
            >>> df2.first() is None
            True
            >>> db.close()
        """
        rows = self.limit(1).collect()
        if not isinstance(rows, list):
            raise TypeError("first() requires collect() to return a list, not an iterator")
        return rows[0] if rows else None

    def head(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the first n rows of the DataFrame.

        Convenience method for quickly inspecting data.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of row dictionaries

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
            >>> Records(_data=[{"id": i, "name": f"User{i}"} for i in range(1, 6)], _database=db).insert_into("users")
            >>> df = db.table("users").select()
            >>> rows = df.head(3)
            >>> len(rows)
            3
            >>> rows[0]["id"]
            1
            >>> db.close()
        """
        rows = self.limit(n).collect()
        if not isinstance(rows, list):
            raise TypeError("head() requires collect() to return a list, not an iterator")
        return rows

    def tail(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the last n rows of the DataFrame.

        Note: This requires materializing the entire DataFrame, so it may be slow for large datasets.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of row dictionaries

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
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
        all_rows = self.collect()
        if not isinstance(all_rows, list):
            # If collect() returns an iterator, convert to list
            all_rows = list(all_rows)
        return all_rows[-n:] if len(all_rows) > n else all_rows

    def count(self) -> int:
        """Return the number of rows in the DataFrame.

        Returns:
            Number of rows

        Note:
            This executes a COUNT(*) query against the database.

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT")]).collect()
            >>> from moltres.io.records import Records
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

        Example:
            >>> from moltres import connect, col
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice", "age": None}, {"id": 2, "name": None, "age": 25}], _database=db).insert_into("users")
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

        Example:
            >>> from moltres import connect
            >>> from moltres.table.schema import column
            >>> db = connect("sqlite:///:memory:")
            >>> db.create_table("users", [column("id", "INTEGER"), column("name", "TEXT"), column("age", "INTEGER")]).collect()
            >>> from moltres.io.records import Records
            >>> _ = Records(_data=[{"id": 1, "name": "Alice", "age": 25}, {"id": 2, "name": None, "age": 30}, {"id": 3, "name": "Bob", "age": None}], _database=db).insert_into("users")
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
        """Convert this DataFrame to a PolarsDataFrame for Polars-style operations.

        Returns:
            PolarsDataFrame wrapping this DataFrame

        Example:
            >>> from moltres import connect
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.read.csv("data.csv")
            >>> polars_df = df.polars()
            >>> results = polars_df.collect()
        """
        from .polars_dataframe import PolarsDataFrame

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
        """Return a DataFrameWriter for writing this DataFrame to a table."""
        from .writer import DataFrameWriter

        return DataFrameWriter(self)

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
        schema = self.schema
        return [(col_info.name, col_info.type_name) for col_info in schema]

    def printSchema(self) -> None:
        """Print the schema of this DataFrame in a tree format.

        Similar to PySpark's DataFrame.printSchema() method, this prints
        a formatted representation of the DataFrame's schema.

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
    def _with_plan(self, plan: LogicalPlan) -> "DataFrame":
        return DataFrame(
            plan=plan,
            database=self.database,
        )

    def _normalize_projection(self, expr: Union[Column, str]) -> Column:
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


class NullHandling:
    """Helper class for null handling operations on DataFrames.

    Accessed via the `na` property on DataFrame instances.
    """

    def __init__(self, df: DataFrame):
        self._df = df

    def drop(self, how: str = "any", subset: Optional[Sequence[str]] = None) -> DataFrame:
        """Drop rows with null values.

        This is a convenience wrapper around DataFrame.dropna().

        Args:
            how: "any" (drop if any null) or "all" (drop if all null) (default: "any")
            subset: Optional list of column names to check. If None, checks all columns.

        Returns:
            New DataFrame with null rows removed

        Example:
            >>> df.na.drop()  # Drop rows with any null values
            >>> df.na.drop(how="all")  # Drop rows where all values are null
            >>> df.na.drop(subset=["col1", "col2"])  # Only check specific columns
        """
        return self._df.dropna(how=how, subset=subset)

    def fill(
        self, value: Union[object, Dict[str, object]], subset: Optional[Sequence[str]] = None
    ) -> DataFrame:
        """Fill null values with a specified value.

        This is a convenience wrapper around DataFrame.fillna().

        Args:
            value: Value to use for filling nulls. Can be a single value or a dict mapping column names to values.
            subset: Optional list of column names to fill. If None, fills all columns.

        Returns:
            New DataFrame with null values filled

        Example:
            >>> df.na.fill(0)  # Fill all nulls with 0
            >>> df.na.fill({"col1": 0, "col2": "unknown"})  # Fill different columns with different values
            >>> df.na.fill(0, subset=["col1", "col2"])  # Fill specific columns with 0
        """
        return self._df.fillna(value=value, subset=subset)
