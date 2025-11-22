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
    Tuple,
    Union,
    overload,
)

from ..expressions.column import Column, col
from ..logical import operators
from ..logical.plan import LogicalPlan, SortOrder
from ..sql.compiler import compile_plan

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase, AsyncTableHandle
    from .async_groupby import AsyncGroupedDataFrame
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
        """Select columns from the DataFrame."""
        if not columns:
            return self
        normalized = tuple(self._normalize_projection(column) for column in columns)
        return self._with_plan(operators.project(self.plan, normalized))

    def where(self, predicate: Column) -> "AsyncDataFrame":
        """Filter rows based on a predicate."""
        return self._with_plan(operators.filter(self.plan, predicate))

    filter = where

    def join(
        self,
        other: "AsyncDataFrame",
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        how: str = "inner",
    ) -> "AsyncDataFrame":
        """Join with another DataFrame."""
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to an AsyncDatabase before joining")
        if self.database is not other.database:
            raise ValueError("Cannot join DataFrames from different AsyncDatabase instances")

        # Cross joins don't require an 'on' clause
        if how.lower() == "cross":
            normalized_on = None
        else:
            normalized_on = self._normalize_join_keys(on)
        return self._with_plan(
            operators.join(self.plan, other.plan, how=how.lower(), on=normalized_on)
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
        normalized_on = self._normalize_join_keys(on)
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
        normalized_on = self._normalize_join_keys(on)
        plan = operators.anti_join(self.plan, other.plan, on=normalized_on)
        return AsyncDataFrame(plan=plan, database=self.database)

    def group_by(self, *columns: Union[Column, str]) -> "AsyncGroupedDataFrame":
        """Group by the specified columns."""
        from .async_groupby import AsyncGroupedDataFrame

        normalized = tuple(self._normalize_projection(col) for col in columns)
        grouped_plan = operators.aggregate(self.plan, keys=normalized, aggregates=())
        return AsyncGroupedDataFrame(plan=grouped_plan, database=self.database)

    groupBy = group_by

    def order_by(self, *columns: Union[Column, str]) -> "AsyncDataFrame":
        """Sort by the specified columns."""
        sort_orders = tuple(
            self._normalize_sort_expression(self._normalize_projection(col)) for col in columns
        )
        return self._with_plan(operators.order_by(self.plan, sort_orders))

    orderBy = order_by

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
        """Add or replace a column in the DataFrame."""
        from ..logical.plan import Project
        from dataclasses import replace as dataclass_replace

        new_col = self._normalize_projection(col_expr)
        if isinstance(new_col, Column) and not new_col._alias:
            new_col = new_col.alias(colName)
        elif isinstance(new_col, Column):
            new_col = dataclass_replace(new_col, _alias=colName)

        if isinstance(self.plan, Project):
            new_projections = list(self.plan.projections) + [new_col]
        else:
            new_projections = [new_col]

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

    def drop(self, *cols: str) -> "AsyncDataFrame":
        """Drop one or more columns from the DataFrame."""
        from ..logical.plan import Project

        cols_to_drop = set(cols)

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
        """
        if self.database is None:
            raise RuntimeError("Cannot collect a plan without an attached AsyncDatabase")

        if stream:
            # For SQL queries, use streaming execution
            if self.database is None:
                raise RuntimeError("Cannot collect a plan without an attached AsyncDatabase")

            async def stream_gen() -> AsyncIterator[List[Dict[str, object]]]:
                async for chunk in self.database.execute_plan_stream(self.plan):  # type: ignore[union-attr]
                    yield chunk

            return stream_gen()

        result = await self.database.execute_plan(self.plan)
        return result.rows  # type: ignore[no-any-return]

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

    def _normalize_sort_expression(self, expr: Column) -> SortOrder:
        """Normalize a sort expression to a SortOrder."""
        if expr.op == "sort_desc":
            return operators.sort_order(expr.args[0], descending=True)
        if expr.op == "sort_asc":
            return operators.sort_order(expr.args[0], descending=False)
        return operators.sort_order(expr, descending=False)

    def _normalize_join_keys(
        self, on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]]
    ) -> Sequence[Tuple[str, str]]:
        """Normalize join keys to a sequence of (left, right) column pairs."""
        if isinstance(on, str):
            return [(on, on)]
        if isinstance(on, (list, tuple)) and on and isinstance(on[0], str):
            return [(key, key) for key in on]
        if isinstance(on, (list, tuple)) and on and isinstance(on[0], (list, tuple)):
            return [tuple(pair) for pair in on]
        raise ValueError(f"Invalid join keys: {on}")


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
