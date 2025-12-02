"""Async Pandas-style indexers for Moltres DataFrames."""

from __future__ import annotations

from typing import Any

from ...expressions.column import Column
from ..interfaces.async_pandas_dataframe import AsyncPandasDataFrame


class _AsyncLocIndexer:
    """Indexer for pandas-style loc accessor (async)."""

    _df: AsyncPandasDataFrame

    def __init__(self, df: AsyncPandasDataFrame):
        self._df = df

    def __getitem__(self, key: Any) -> AsyncPandasDataFrame:
        """Access rows and columns using loc.

        Supports:
        - df.loc[df['age'] > 25] - Row filtering
        - df.loc[:, ['col1', 'col2']] - :class:`Column` selection
        - df.loc[df['age'] > 25, 'col1'] - Combined filter and select
        """
        # Handle different key types
        if isinstance(key, tuple) and len(key) == 2:
            # Two-dimensional indexing: df.loc[rows, cols]
            row_key, col_key = key
            result_df = self._df._df

            # Apply row filter if not ':' or Ellipsis
            # Need to check type first to avoid boolean evaluation of Column
            if isinstance(row_key, Column):
                # Boolean condition
                result_df = result_df.where(row_key)
            elif row_key is not Ellipsis:
                # Check if it's slice(None) without triggering comparison
                if not (
                    isinstance(row_key, slice)
                    and row_key.start is None
                    and row_key.stop is None
                    and row_key.step is None
                ):
                    raise NotImplementedError(
                        "loc row indexing only supports boolean conditions or :"
                    )

            # Apply column selection
            if col_key is Ellipsis or (
                isinstance(col_key, slice)
                and col_key.start is None
                and col_key.stop is None
                and col_key.step is None
            ):
                # Select all columns (no change)
                pass
            elif isinstance(col_key, (list, tuple)):
                # Select specific columns
                str_columns = [c for c in col_key if isinstance(c, str)]
                if str_columns:
                    self._df._validate_columns_exist(str_columns, "loc column selection")
                from ...expressions.column import col

                columns = [col(c) if isinstance(c, str) else c for c in col_key]
                result_df = result_df.select(*columns)
            elif isinstance(col_key, str):
                # Single column
                self._df._validate_columns_exist([col_key], "loc column selection")
                from ...expressions.column import col

                result_df = result_df.select(col(col_key))
            else:
                raise TypeError(f"Invalid column key type for loc: {type(col_key)}")

            return self._df._with_dataframe(result_df)
        elif isinstance(key, Column):
            # Single boolean condition - filter rows
            return self._df._with_dataframe(self._df._df.where(key))
        elif isinstance(key, (list, tuple)):
            # Column selection only
            str_columns = [c for c in key if isinstance(c, str)]
            if str_columns:
                self._df._validate_columns_exist(str_columns, "loc column selection")
            from ...expressions.column import col

            columns = [col(c) if isinstance(c, str) else c for c in key]
            return self._df._with_dataframe(self._df._df.select(*columns))
        else:
            raise TypeError(f"Invalid key type for loc: {type(key)}")


class _AsyncILocIndexer:
    """Indexer for pandas-style iloc accessor (async)."""

    _df: AsyncPandasDataFrame

    def __init__(self, df: AsyncPandasDataFrame):
        self._df = df

    def __getitem__(self, key: Any) -> AsyncPandasDataFrame:
        """Access rows and columns using iloc (integer position).

        Note:
            Full iloc functionality requires materialization.
            Only boolean array filtering is supported for lazy evaluation.
        """
        # For lazy evaluation, we can only support boolean filtering
        if isinstance(key, Column):
            # Boolean condition
            return self._df._with_dataframe(self._df._df.where(key))
        else:
            raise NotImplementedError(
                "iloc positional indexing requires materialization. "
                "Use limit() or boolean filtering instead."
            )
