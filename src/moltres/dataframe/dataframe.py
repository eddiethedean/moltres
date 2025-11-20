"""Lazy DataFrame representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from ..expressions.column import Column, col
from ..logical import operators
from ..logical.plan import LogicalPlan, SortOrder
from ..sql.compiler import compile_plan

if TYPE_CHECKING:
    from ..table.schema import ColumnDef
    from ..table.table import Database, TableHandle
    from .groupby import GroupedDataFrame
    from .writer import DataFrameWriter


@dataclass(frozen=True)
class DataFrame:
    plan: LogicalPlan
    database: Optional["Database"] = None
    _materialized_data: Optional[List[dict[str, object]]] = None
    _stream_generator: Optional[Callable[[], Iterator[List[dict[str, object]]]]] = None
    _stream_schema: Optional[Sequence["ColumnDef"]] = None

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
        if not columns:
            return self
        normalized = tuple(self._normalize_projection(column) for column in columns)
        return self._with_plan(operators.project(self.plan, normalized))

    def where(self, predicate: Column) -> "DataFrame":
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
        """
        if count < 0:
            raise ValueError("limit count must be non-negative")
        return self._with_plan(operators.limit(self.plan, count))

    def order_by(self, *columns: Column) -> "DataFrame":
        if not columns:
            return self
        orders = tuple(self._normalize_sort_expression(column) for column in columns)
        return self._with_plan(operators.order_by(self.plan, orders))

    def join(
        self,
        other: "DataFrame",
        *,
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        how: str = "inner",
    ) -> "DataFrame":
        if self.database is None or other.database is None:
            raise RuntimeError("Both DataFrames must be bound to a Database before joining")
        if self.database is not other.database:
            raise ValueError("Cannot join DataFrames from different Database instances")
        normalized_on = self._normalize_join_keys(on)
        plan = operators.join(self.plan, other.plan, how=how.lower(), on=normalized_on)
        return DataFrame(plan=plan, database=self.database)

    def group_by(self, *columns: Union[Column, str]) -> "GroupedDataFrame":
        if not columns:
            raise ValueError("group_by requires at least one grouping column")
        from .groupby import GroupedDataFrame

        keys = tuple(self._normalize_projection(column) for column in columns)
        return GroupedDataFrame(plan=self.plan, keys=keys, parent=self)

    groupBy = group_by

    # ---------------------------------------------------------------- execution
    def to_sql(self) -> str:
        if self.database is not None:
            return self.database.compile_plan(self.plan)
        return compile_plan(self.plan)

    def collect(
        self, stream: bool = False
    ) -> Union[List[Dict[str, object]], Iterator[List[Dict[str, object]]]]:
        """Collect DataFrame results.

        Args:
            stream: If True, return an iterator of row chunks. If False (default),
                   materialize all rows into a list (backward compatible).

        Returns:
            If stream=False: List of dictionaries representing rows.
            If stream=True: Iterator of row chunks (each chunk is a list of dicts).

        Raises:
            RuntimeError: If DataFrame is not bound to a Database and has no materialized data
        """
        # Check if DataFrame has streaming generator
        if self._stream_generator is not None:
            if stream:
                return self._stream_generator()
            else:
                # Materialize all chunks for backward compatibility
                all_rows = []
                for chunk in self._stream_generator():
                    all_rows.extend(chunk)
                return all_rows

        # Check if DataFrame has materialized data (from file readers)
        if self._materialized_data is not None:
            if stream:
                # Return iterator with single chunk
                return iter([self._materialized_data])
            return self._materialized_data

        if self.database is None:
            raise RuntimeError("Cannot collect a plan without an attached Database")

        if stream:
            # For SQL queries, use streaming execution
            return self.database.execute_plan_stream(self.plan)

        result = self.database.execute_plan(self.plan)
        return result.rows

    @property
    def write(self) -> "DataFrameWriter":
        """Return a DataFrameWriter for writing this DataFrame to a table."""
        from .writer import DataFrameWriter

        return DataFrameWriter(self)

    # ---------------------------------------------------------------- utilities
    def _with_plan(self, plan: LogicalPlan) -> "DataFrame":
        return DataFrame(
            plan=plan,
            database=self.database,
            _materialized_data=self._materialized_data,
            _stream_generator=self._stream_generator,
            _stream_schema=self._stream_schema,
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
