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
        """Select specific columns from the DataFrame.

        Args:
            *columns: Column names or Column expressions to select.
                     Use "*" to select all columns (same as empty select).
                     Can combine "*" with other columns: select("*", col("new_col"))

        Returns:
            New DataFrame with selected columns

        Example:
            >>> # Select specific columns
            >>> df = db.table("users").select("id", "name", "email")
            >>> # SQL: SELECT id, name, email FROM users

            >>> # Select all columns (empty select or "*")
            >>> df = db.table("users").select()  # or .select("*")
            >>> # SQL: SELECT * FROM users

            >>> # Select with expressions
            >>> from moltres import col
            >>> df = db.table("orders").select(
            ...     col("id"),
            ...     (col("amount") * 1.1).alias("amount_with_tax")
            ... )
            >>> # SQL: SELECT id, amount * 1.1 AS amount_with_tax FROM orders

            >>> # Select all columns plus new ones
            >>> df = db.table("orders").select("*", (col("amount") * 1.1).alias("with_tax"))
            >>> # SQL: SELECT *, amount * 1.1 AS with_tax FROM orders
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
            >>> # Basic column selection
            >>> df.selectExpr("id", "name", "email")

            >>> # With expressions and aliases
            >>> df.selectExpr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")

            >>> # Complex expressions
            >>> df.selectExpr("(amount + tax) * 1.1 as total", "CASE WHEN status = 'active' THEN 1 ELSE 0 END as is_active")

            >>> # Chaining with other operations
            >>> df.selectExpr("id", "amount").where(col("amount") > 100)
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
            >>> from moltres import col
            >>> # Filter by condition using Column
            >>> df = db.table("users").select().where(col("age") >= 18)
            >>> # SQL: SELECT * FROM users WHERE age >= 18

            >>> # Filter using SQL string
            >>> df = db.table("users").select().where("age > 18")
            >>> # SQL: SELECT * FROM users WHERE age > 18

            >>> # Multiple conditions with Column
            >>> df = db.table("orders").select().where(
            ...     (col("amount") > 100) & (col("status") == "active")
            ... )
            >>> # SQL: SELECT * FROM orders WHERE amount > 100 AND status = 'active'

            >>> # Multiple conditions with SQL string
            >>> df = db.table("orders").select().where("amount > 100 AND status = 'active'")
            >>> # SQL: SELECT * FROM orders WHERE amount > 100 AND status = 'active'
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
            >>> # Limit to 10 rows
            >>> df = db.table("users").select().limit(10)
            >>> # SQL: SELECT * FROM users LIMIT 10

            >>> # Limit with ordering
            >>> from moltres import col
            >>> df = (
            ...     db.table("orders")
            ...     .select()
            ...     .order_by(col("amount").desc())
            ...     .limit(5)
            ... )
            >>> # SQL: SELECT * FROM orders ORDER BY amount DESC LIMIT 5
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
            >>> df = db.table("users").select().sample(0.1)  # Sample 10% of rows
            >>> # SQL (PostgreSQL): SELECT * FROM users TABLESAMPLE BERNOULLI(10)
            >>> # SQL (SQLite): SELECT * FROM users ORDER BY RANDOM() LIMIT (COUNT(*) * 0.1)
        """
        return self._with_plan(operators.sample(self.plan, fraction, seed))

    def order_by(self, *columns: Column) -> "DataFrame":
        """Sort rows by one or more columns.

        Args:
            *columns: Column expressions to sort by. Use .asc() or .desc() for sort order.

        Returns:
            New DataFrame with sorted rows

        Example:
            >>> from moltres import col
            >>> # Sort ascending
            >>> df = db.table("users").select().order_by(col("name"))
            >>> # SQL: SELECT * FROM users ORDER BY name

            >>> # Sort descending
            >>> df = db.table("orders").select().order_by(col("amount").desc())
            >>> # SQL: SELECT * FROM orders ORDER BY amount DESC

            >>> # Multiple sort columns
            >>> df = (
            ...     db.table("sales")
            ...     .select()
            ...     .order_by(col("region"), col("amount").desc())
            ... )
            >>> # SQL: SELECT * FROM sales ORDER BY region, amount DESC
        """
        if not columns:
            return self
        orders = tuple(self._normalize_sort_expression(column) for column in columns)
        return self._with_plan(operators.order_by(self.plan, orders))

    orderBy = order_by  # PySpark-style alias
    sort = order_by  # PySpark-style alias

    def join(
        self,
        other: "DataFrame",
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        how: str = "inner",
        lateral: bool = False,
        hints: Optional[Sequence[str]] = None,
    ) -> "DataFrame":
        """Join with another DataFrame.

        Args:
            other: Another DataFrame to join with
            on: Join condition - can be:
                - A single column name (assumes same name in both DataFrames)
                - A sequence of column names (assumes same names in both)
                - A sequence of (left_column, right_column) tuples
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
            >>> # Inner join with column pairs
            >>> customers = db.table("customers").select()
            >>> orders = db.table("orders").select()
            >>> df = customers.join(orders, on=[("id", "customer_id")], how="inner")
            >>> # SQL: SELECT * FROM customers INNER JOIN orders ON customers.id = orders.customer_id

            >>> # Left join
            >>> df = customers.join(orders, on=[("id", "customer_id")], how="left")
            >>> # SQL: SELECT * FROM customers LEFT JOIN orders ON customers.id = orders.customer_id

            >>> # LATERAL join (PostgreSQL/MySQL 8.0+)
            >>> from moltres import col
            >>> df = customers.join(
            ...     orders.select().where(col("customer_id") == col("customers.id")),
            ...     how="left",
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
        else:
            normalized_on = self._normalize_join_keys(on)
        hints_tuple = tuple(hints) if hints else None
        plan = operators.join(
            self.plan,
            other.plan,
            how=how.lower(),
            on=normalized_on,
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
            >>> # Find customers who have placed orders
            >>> customers = db.table("customers").select()
            >>> orders = db.table("orders").select()
            >>> customers_with_orders = customers.semi_join(orders, on="customer_id")
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before semi_join")
        if self.database is not other.database:
            raise ValueError("Cannot semi_join DataFrames from different Database instances")
        normalized_on = self._normalize_join_keys(on)
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
            >>> # Find customers who have not placed any orders
            >>> customers = db.table("customers").select()
            >>> orders = db.table("orders").select()
            >>> customers_without_orders = customers.anti_join(orders, on="customer_id")
        """
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before anti_join")
        if self.database is not other.database:
            raise ValueError("Cannot anti_join DataFrames from different Database instances")
        normalized_on = self._normalize_join_keys(on)
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
            >>> # Pivot sales data by product
            >>> df = db.table("sales").select("date", "product", "amount")
            >>> pivoted = df.pivot(pivot_column="product", value_column="amount", agg_func="sum")
            >>> # SQL: Uses CASE WHEN with aggregation for cross-dialect compatibility
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
            >>> # Explode a JSON array column
            >>> df = db.table("users").select()
            >>> exploded = df.explode(col("tags"))  # tags is a JSON array
            >>> # Each row in exploded will have one tag per row
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
            >>> from moltres import col
            >>> from moltres.expressions import functions as F
            >>> # Group by single column
            >>> df = (
            ...     db.table("orders")
            ...     .select()
            ...     .group_by("customer_id")
            ...     .agg(F.sum(col("amount")).alias("total"))
            ... )
            >>> # SQL: SELECT customer_id, SUM(amount) AS total FROM orders GROUP BY customer_id

            >>> # Group by multiple columns
            >>> df = (
            ...     db.table("sales")
            ...     .select()
            ...     .group_by("region", "product")
            ...     .agg(
            ...         F.sum(col("revenue")).alias("total_revenue"),
            ...         F.count("*").alias("count")
            ...     )
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
            # Remove any column with the same name (replace behavior)
            existing_cols = []
            for col_expr in self.plan.projections:
                if isinstance(col_expr, Column):
                    # Check if this column matches the colName (by alias or column name)
                    col_alias = col_expr._alias
                    col_name = (
                        col_expr.args[0] if col_expr.op == "column" and col_expr.args else None
                    )
                    if col_alias == colName or col_name == colName:
                        # Skip this column - it will be replaced by new_col
                        continue
                existing_cols.append(col_expr)
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

    def explain(self, analyze: bool = False) -> str:
        """Get the query execution plan using SQL EXPLAIN.

        Args:
            analyze: If True, use EXPLAIN ANALYZE (executes query and shows actual execution stats).
                    If False, use EXPLAIN (shows estimated plan without executing).

        Returns:
            Query plan as a string

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
        explain_sql = f"EXPLAIN ANALYZE {sql}" if analyze else f"EXPLAIN {sql}"

        # Execute EXPLAIN query
        result = self.database.execute_sql(explain_sql)
        # Format the plan results - EXPLAIN typically returns a single column
        plan_lines = []
        for row in result.rows:
            # Format each row of the plan - row is a dict
            if len(row) == 1:
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
                return result.rows  # type: ignore[no-any-return]

        # Handle FileScan by materializing file data into a temporary table
        plan = self._materialize_filescan(self.plan)

        if stream:
            # For SQL queries, use streaming execution
            return self.database.execute_plan_stream(plan)

        result = self.database.execute_plan(plan)
        return result.rows  # type: ignore[no-any-return]

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
