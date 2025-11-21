"""Async lazy DataFrame representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
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
        on: Union[str, Sequence[str], Sequence[Tuple[str, str]]],
        how: str = "inner",
    ) -> "AsyncDataFrame":
        """Join with another DataFrame."""
        if how not in ("inner", "left", "right", "outer"):
            raise ValueError(f"Unsupported join type: {how}")

        # Normalize join keys
        join_keys = self._normalize_join_keys(on)
        return self._with_plan(operators.join(self.plan, other.plan, how=how, on=join_keys))

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
