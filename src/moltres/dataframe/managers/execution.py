"""DataFrame execution operations.

This module handles execution of DataFrame queries, including collect, show, take, first, head, and tail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Literal, Optional, Union, overload

if TYPE_CHECKING:
    from ..core.dataframe import DataFrame


class DataFrameExecutor:
    """Handles execution of DataFrame queries."""

    def __init__(self, df: "DataFrame"):
        """Initialize executor with a DataFrame.

        Args:
            df: The DataFrame to execute
        """
        self._df = df

    @overload
    def collect(self, stream: Literal[False] = False) -> List[Dict[str, object]]: ...

    @overload
    def collect(self, stream: Literal[True]) -> Iterator[List[Dict[str, object]]]: ...

    def collect(
        self, stream: bool = False
    ) -> Union[
        List[Dict[str, object]], Iterator[List[Dict[str, object]]], List[Any], Iterator[List[Any]]
    ]:
        """Collect DataFrame results.

        Args:
            stream: If True, return an iterator of row chunks. If False (default),
                   materialize all rows into a list.

        Returns:
            If stream=False and no model attached: List of dictionaries representing rows.
            If stream=False and model attached: List of SQLModel or Pydantic instances.
            If stream=True and no model attached: Iterator of row chunks (each chunk is a list of dicts).
            If stream=True and model attached: Iterator of row chunks (each chunk is a list of model instances).

        Raises:
            RuntimeError: If DataFrame is not bound to a Database
        """
        if self._df.database is None:
            raise RuntimeError("Cannot collect a plan without an attached Database")

        # Helper function to convert rows to model instances if model is attached
        def _convert_rows(
            rows: List[Dict[str, object]],
        ) -> Union[List[Dict[str, object]], List[Any]]:
            from ..helpers.materialization_helpers import convert_rows_to_models

            return convert_rows_to_models(rows, self._df.model)

        # Handle RawSQL at root level - execute directly for efficiency
        from ...logical.plan import RawSQL

        if isinstance(self._df.plan, RawSQL):
            if stream:
                # For streaming, we need to use execute_plan_stream which expects a compiled plan
                # So we'll compile the RawSQL plan
                plan = self._df._materialize_filescan(self._df.plan)
                stream_iter = self._df.database.execute_plan_stream(plan)
                # Convert each chunk to SQLModel instances if model is attached
                if self._df.model is not None:

                    def _convert_stream() -> Iterator[List[Any]]:
                        for chunk in stream_iter:
                            yield _convert_rows(chunk)

                    return _convert_stream()
                return stream_iter
            else:
                # Execute RawSQL directly
                from ..helpers.materialization_helpers import convert_result_rows

                result = self._df.database.execute_sql(
                    self._df.plan.sql, params=self._df.plan.params
                )
                rows = convert_result_rows(result.rows)
                return _convert_rows(rows)

        # Handle FileScan by materializing file data into a temporary table
        plan = self._df._materialize_filescan(self._df.plan)

        if stream:
            # For SQL queries, use streaming execution
            stream_iter = self._df.database.execute_plan_stream(plan)
            # Convert each chunk to SQLModel instances if model is attached
            if self._df.model is not None:

                def _convert_stream() -> Iterator[List[Any]]:
                    for chunk in stream_iter:
                        yield _convert_rows(chunk)

                return _convert_stream()
            return stream_iter

        result = self._df.database.execute_plan(plan, model=self._df.model)
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

    def show(self, n: int = 20, truncate: bool = True) -> None:
        """Print the first n rows of the DataFrame in a tabular format.

        Args:
            n: Number of rows to show (default: 20)
            truncate: If True, truncate long strings to 20 characters (default: True)
        """
        if self._df.database is None:
            raise RuntimeError("Cannot show a plan without an attached Database")

        rows = self.take(n)
        if not rows:
            print("Empty DataFrame")
            return

        # Get column names from first row
        columns = list(rows[0].keys())

        # Calculate column widths
        col_widths: Dict[str, int] = {}
        for col_name in columns:
            col_widths[col_name] = len(col_name)

        # Calculate widths from data
        for row in rows:
            for col_name in columns:
                value = row.get(col_name, "")
                if truncate and isinstance(value, str) and len(value) > 20:
                    value_str = value[:17] + "..."
                else:
                    value_str = str(value)
                col_widths[col_name] = max(col_widths[col_name], len(value_str))

        # Print header
        header = " | ".join(col.ljust(col_widths[col]) for col in columns)
        print(header)
        print("-" * len(header))

        # Print rows
        for row in rows:
            row_strs = []
            for col in columns:
                value = row.get(col, "")
                if truncate and isinstance(value, str) and len(value) > 20:
                    value_str = value[:17] + "..."
                else:
                    value_str = str(value)
                row_strs.append(value_str.ljust(col_widths[col]))
            print(" | ".join(row_strs))

        # Print summary
        total_rows = self._df.count() if self._df.database else len(rows)
        if total_rows > n:
            print(f"\nshowing top {n} of {total_rows} rows")

    def take(self, num: int) -> List[Dict[str, object]]:
        """Take the first num rows from the DataFrame.

        Args:
            num: Number of rows to take

        Returns:
            List of dictionaries representing the first num rows
        """
        if self._df.database is None:
            raise RuntimeError("Cannot take rows from a plan without an attached Database")

        limited_df = self._df.limit(num)
        result = limited_df.collect()
        if not isinstance(result, list):
            raise TypeError("take() requires collect() to return a list, not an iterator")
        return result

    def first(self) -> Optional[Dict[str, object]]:
        """Return the first row of the DataFrame as a dictionary.

        Returns:
            First row as a dictionary, or None if DataFrame is empty
        """
        if self._df.database is None:
            raise RuntimeError("Cannot get first row from a plan without an attached Database")

        rows = self.take(1)
        return rows[0] if rows else None

    def head(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the first n rows of the DataFrame.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of dictionaries representing the first n rows
        """
        return self.take(n)

    def tail(self, n: int = 5) -> List[Dict[str, object]]:
        """Return the last n rows of the DataFrame.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            List of dictionaries representing the last n rows

        Note:
            This materializes all rows, reverses the order, and takes the first n.
            For large DataFrames, this may be slow.
        """
        if self._df.database is None:
            raise RuntimeError("Cannot get tail rows from a plan without an attached Database")

        # Materialize all rows, reverse, and take first n
        # This preserves the original ordering from the DataFrame
        all_rows = self._df.collect()
        if not isinstance(all_rows, list):
            # If collect() returns an iterator, convert to list
            all_rows = list(all_rows)
        # Reverse to get last n rows
        reversed_rows = list(reversed(all_rows))
        return reversed_rows[:n] if len(reversed_rows) > n else reversed_rows
