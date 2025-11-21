"""Async lazy DataFrame representation."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, cast

from ..expressions.column import Column, col
from ..logical import operators
from ..logical.plan import LogicalPlan, SortOrder
from ..sql.compiler import compile_plan

if TYPE_CHECKING:
    from ..table.async_table import AsyncDatabase, AsyncTableHandle
    from ..table.schema import ColumnDef
    from .async_groupby import AsyncGroupedDataFrame
    from .async_writer import AsyncDataFrameWriter


@dataclass(frozen=True)
class AsyncDataFrame:
    """Async lazy DataFrame representation."""

    plan: LogicalPlan
    database: AsyncDatabase | None = None
    _materialized_data: list[dict[str, object]] | None = None
    _stream_generator: Callable[[], AsyncIterator[list[dict[str, object]]]] | None = None
    _stream_schema: Sequence[ColumnDef] | None = None

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_table(
        cls, table_handle: AsyncTableHandle, columns: Sequence[str] | None = None
    ) -> AsyncDataFrame:
        """Create an AsyncDataFrame from a table handle."""
        plan = operators.scan(table_handle.name)
        df = cls(plan=plan, database=table_handle.database)
        if columns:
            df = df.select(*columns)
        return df

    def select(self, *columns: Column | str) -> AsyncDataFrame:
        """Select columns from the DataFrame."""
        if not columns:
            return self
        normalized = tuple(self._normalize_projection(column) for column in columns)
        return self._with_plan(operators.project(self.plan, normalized))

    def where(self, predicate: Column) -> AsyncDataFrame:
        """Filter rows based on a predicate."""
        return self._with_plan(operators.filter(self.plan, predicate))

    filter = where

    def join(
        self,
        other: AsyncDataFrame,
        on: str | Sequence[str] | Sequence[tuple[str, str]],
        how: str = "inner") -> AsyncDataFrame:
        """Join with another DataFrame."""
        if how not in ("inner", "left", "right", "outer"):
            raise ValueError(f"Unsupported join type: {how}")

        # Normalize join keys
        join_keys = self._normalize_join_keys(on)
        return self._with_plan(operators.join(self.plan, other.plan, how=how, on=join_keys))

    def group_by(self, *columns: Column | str) -> AsyncGroupedDataFrame:
        """Group by the specified columns."""
        from .async_groupby import AsyncGroupedDataFrame

        normalized = tuple(self._normalize_projection(col) for col in columns)
        grouped_plan = operators.aggregate(self.plan, keys=normalized, aggregates=())
        return AsyncGroupedDataFrame(
            plan=grouped_plan, database=self.database, _materialized_data=self._materialized_data
        )

    groupBy = group_by

    def order_by(self, *columns: Column | str) -> AsyncDataFrame:
        """Sort by the specified columns."""
        sort_orders = tuple(
            self._normalize_sort_expression(self._normalize_projection(col)) for col in columns
        )
        return self._with_plan(operators.order_by(self.plan, sort_orders))

    orderBy = order_by

    def limit(self, count: int) -> AsyncDataFrame:
        """Limit the number of rows returned."""
        if count < 0:
            raise ValueError("limit() requires a non-negative integer")
        return self._with_plan(operators.limit(self.plan, count))

    # ---------------------------------------------------------------- execution
    def to_sql(self) -> str:
        """Compile the DataFrame plan to SQL."""
        if self.database is not None:
            return self.database.compile_plan(self.plan)
        return compile_plan(self.plan)

    async def collect(
        self, stream: bool = False
    ) -> list[dict[str, object]] | AsyncIterator[list[dict[str, object]]]:
        """Collect DataFrame results asynchronously.

        Args:
            stream: If True, return an async iterator of row chunks. If False (default),
                   materialize all rows into a list.

        Returns:
            If stream=False: list of dictionaries representing rows.
            If stream=True: AsyncIterator of row chunks (each chunk is a list of dicts).

        Raises:
            RuntimeError: If DataFrame is not bound to an AsyncDatabase and has no materialized data
        """
        # Check if DataFrame has streaming generator
        if self._stream_generator is not None:
            if stream:
                # _stream_generator is already an async generator function
                return self._stream_generator()
            # Materialize all chunks
            all_rows: list[dict[str, object]] = []
            async for chunk in self._stream_generator():
                all_rows.extend(chunk)
            return all_rows

        # Check if DataFrame has materialized data (from file readers)
        if self._materialized_data is not None:
            if stream:
                # Return async iterator with single chunk
                async def single_chunk() -> AsyncIterator[list[dict[str, object]]]:
                    yield self._materialized_data  # type: ignore[misc]

                return single_chunk()
            # _materialized_data is already list[dict[str, object]] when not None
            return self._materialized_data

        if self.database is None:
            raise RuntimeError("Cannot collect a plan without an attached AsyncDatabase")

        if stream:
            # For SQL queries, use streaming execution
            async def stream_gen() -> AsyncIterator[list[dict[str, object]]]:
                assert self.database is not None  # Type narrowing
                async for chunk in self.database.execute_plan_stream(self.plan):
                    yield chunk

            return stream_gen()

        assert self.database is not None  # Type narrowing
        result = await self.database.execute_plan(self.plan)
        return cast(list[dict[str, object]], result.rows)

    @property
    def write(self) -> AsyncDataFrameWriter:
        """Return an AsyncDataFrameWriter for writing this DataFrame to a table."""
        from .async_writer import AsyncDataFrameWriter

        return AsyncDataFrameWriter(self)

    # ---------------------------------------------------------------- utilities
    def _with_plan(self, plan: LogicalPlan) -> AsyncDataFrame:
        """Create a new AsyncDataFrame with a different plan."""
        return AsyncDataFrame(
            plan=plan,
            database=self.database,
            _materialized_data=self._materialized_data,
            _stream_generator=self._stream_generator,
            _stream_schema=self._stream_schema)

    def _normalize_projection(self, expr: Column | str) -> Column:
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
        self, on: str | Sequence[str] | Sequence[tuple[str, str]] | None
    ) -> Sequence[tuple[str, str]]:
        """Normalize join keys to a sequence of (left, right) column pairs."""
        if isinstance(on, str):
            return [(on, on)]
        if isinstance(on, (list, tuple)) and on and isinstance(on[0], str):
            return [(key, key) for key in on]
        if isinstance(on, (list, tuple)) and on and isinstance(on[0], (list, tuple)):
            return [tuple(pair) for pair in on]  # type: ignore[misc]
        raise ValueError(f"Invalid join keys: {on}")
