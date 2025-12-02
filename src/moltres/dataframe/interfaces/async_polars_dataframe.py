"""Async Polars-style interface for Moltres DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

from ...expressions.column import Column, col
from ...logical.plan import LogicalPlan
from ...utils.typing import FillValue
from ..core.async_dataframe import AsyncDataFrame
from ..core.interface_common import AsyncInterfaceCommonMixin

# Import from polars_operations
from ..operations.polars_operations import sql_type_to_polars_dtype

if TYPE_CHECKING:
    import polars as pl

    from ...table.async_table import AsyncDatabase
    from ..groupby.async_polars_groupby import AsyncPolarsGroupBy
    from ..columns.polars_column import PolarsColumn


@dataclass(frozen=True)
class AsyncPolarsDataFrame(AsyncInterfaceCommonMixin):
    """Async Polars-style interface wrapper around Moltres AsyncDataFrame.

    Provides familiar Polars LazyFrame API methods while maintaining lazy evaluation
    and SQL pushdown execution. All operations remain lazy until await collect() is called.

    Example:
        >>> df = await db.table('users').polars()
        >>> # Polars-style operations
        >>> df.filter(col('age') > 25).select(['id', 'name'])
        >>> # Returns actual Polars :class:`DataFrame`
        >>> result = await df.collect()  # pl.:class:`DataFrame`
    """

    _df: AsyncDataFrame
    _height_cache: Optional[int] = field(default=None, repr=False, compare=False)
    _schema_cache: Optional[List[Tuple[str, str]]] = field(default=None, repr=False, compare=False)

    @property
    def plan(self) -> LogicalPlan:
        """Get the underlying logical plan."""
        return self._df.plan

    @property
    def database(self) -> Optional["AsyncDatabase"]:
        """Get the associated database."""
        return self._df.database

    @classmethod
    def from_dataframe(cls, df: AsyncDataFrame) -> "AsyncPolarsDataFrame":
        """Create an AsyncPolarsDataFrame from an AsyncDataFrame.

        Args:
            df: The AsyncDataFrame to wrap

        Returns:
            AsyncPolarsDataFrame wrapping the provided AsyncDataFrame
        """
        return cls(_df=df)

    def _with_dataframe(self, df: AsyncDataFrame) -> "AsyncPolarsDataFrame":
        """Create a new AsyncPolarsDataFrame with a different underlying AsyncDataFrame.

        Args:
            df: The new underlying AsyncDataFrame

        Returns:
            New AsyncPolarsDataFrame instance
        """
        # Clear caches when creating new DataFrame instance
        return AsyncPolarsDataFrame(_df=df, _height_cache=None, _schema_cache=None)

    def _validate_columns_exist(
        self, column_names: Sequence[str], operation: str = "operation"
    ) -> None:
        """Validate that all specified columns exist in the :class:`DataFrame`.

        Args:
            column_names: List of column names to validate
            operation: Name of the operation being performed (for error messages)

        Raises:
            ValidationError: If any column does not exist, with helpful suggestions

        Note:
            Validation only occurs if columns can be determined from the plan.
            For complex plans (e.g., RawSQL), validation is skipped to avoid false positives.
        """
        from ..operations.polars_operations import validate_columns_exist

        try:
            available_columns = self.columns
            # Only validate if we successfully got column names
            if available_columns:
                validate_columns_exist(column_names, available_columns, operation)
        except RuntimeError:
            # If we can't determine columns (e.g., RawSQL, complex plans),
            # skip validation - the error will be caught at execution time
            pass
        except Exception:
            # For other exceptions, also skip validation to be safe
            pass

    @property
    def columns(self) -> List[str]:
        """Get column names (Polars-style property).

        Returns:
            List of column names

        Example:
            >>> df.columns  # ['id', 'name', 'age']
        """
        try:
            return self._df._extract_column_names(self._df.plan)
        except Exception:
            # If we can't extract columns, return empty list
            return []

    @property
    def width(self) -> int:
        """Get number of columns (Polars-style property).

        Returns:
            Number of columns

        Example:
            >>> df.width  # 3
        """
        return len(self.columns)

    async def height(self) -> int:
        """Get number of rows (Polars-style property, async).

        Returns:
            Number of rows

        Note:
            Getting row count requires executing a COUNT query,
            which can be expensive for large datasets. The result is cached
            for the lifetime of this :class:`DataFrame` instance.

        Warning:
            This operation executes a SQL query. For large tables, consider
            using limit() or filtering first.

        Example:
            >>> num_rows = await df.height()
        """
        # Return cached height if available
        if self._height_cache is not None:
            return self._height_cache

        if self.database is None:
            raise RuntimeError("Cannot get height without an attached AsyncDatabase")

        # Create a count query
        from ...expressions.functions import count

        count_df = self._df.select(count("*").alias("count"))
        result = await count_df.collect()
        num_rows: int = 0
        if result and isinstance(result, list) and len(result) > 0:
            row = result[0]
            if isinstance(row, dict):
                count_val = row.get("count", 0)
                if isinstance(count_val, int):
                    num_rows = count_val
                elif count_val is not None:
                    try:
                        if isinstance(count_val, (str, float)):
                            num_rows = int(count_val)
                        else:
                            num_rows = 0
                    except (ValueError, TypeError):
                        num_rows = 0

        # Note: We can't update the cache in a frozen dataclass, but we return the result
        # The cache field will be set when a new instance is created
        return num_rows

    async def schema(self) -> List[Tuple[str, str]]:
        """Get schema as Polars format (list of (name, dtype) tuples, async).

        Returns:
            List of tuples mapping column names to Polars dtype strings

        Note:
            Schema is cached after first access to avoid redundant queries.

        Example:
            >>> schema = await df.schema()  # [('id', 'Int64'), ('name', 'Utf8'), ('age', 'Int64')]
        """
        # Return cached schema if available
        if self._schema_cache is not None:
            return self._schema_cache

        if self.database is None:
            # Cannot get schema without database connection
            return []

        try:
            # Try to extract schema from the logical plan
            schema = self._df._extract_schema_from_plan(self._df.plan)

            # Map ColumnInfo to Polars dtypes
            schema_list: List[Tuple[str, str]] = []
            for col_info in schema:
                polars_dtype = sql_type_to_polars_dtype(col_info.type_name)
                schema_list.append((col_info.name, polars_dtype))

            # Cache the result (Note: we can't modify frozen dataclass, but we return the list)
            # The cache will be set on the next DataFrame operation that creates a new instance
            return schema_list
        except Exception:
            # If schema extraction fails, return empty list
            return []

    def lazy(self) -> "AsyncPolarsDataFrame":
        """Return self (for API compatibility, AsyncPolarsDataFrame is already lazy).

        Returns:
            Self (AsyncPolarsDataFrame is always lazy)

        Example:
            >>> df.lazy()  # Returns self
        """
        return self

    def select(self, *exprs: Union[str, Column]) -> "AsyncPolarsDataFrame":
        """Select columns/expressions (Polars-style).

        Args:
            *exprs: :class:`Column` names or :class:`Column` expressions to select

        Returns:
            AsyncPolarsDataFrame with selected columns

        Example:
            >>> df.select('id', 'name')
            >>> df.select(col('id'), (col('amount') * 1.1).alias('with_tax'))
        """
        from ..helpers.polars_dataframe_helpers import build_polars_select_operation

        result_df = build_polars_select_operation(self, *exprs)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def filter(self, predicate: Column) -> "AsyncPolarsDataFrame":
        """Filter rows (Polars-style, uses 'filter' instead of 'where').

        Args:
            predicate: :class:`Column` expression for filtering condition

        Returns:
            Filtered AsyncPolarsDataFrame

        Example:
            >>> df.filter(col('age') > 25)
            >>> df.filter((col('age') > 25) & (col('active') == True))
        """
        from ..helpers.polars_dataframe_helpers import build_polars_filter_operation

        result_df = build_polars_filter_operation(self, predicate)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def with_columns(self, *exprs: Union[Column, Tuple[str, Column]]) -> "AsyncPolarsDataFrame":
        """Add or modify columns (Polars primary method for adding columns).

        Args:
            *exprs: :class:`Column` expressions or (name, expression) tuples

        Returns:
            AsyncPolarsDataFrame with new/modified columns

        Example:
            >>> df.with_columns((col('amount') * 1.1).alias('with_tax'))
            >>> df.with_columns(('with_tax', col('amount') * 1.1))
        """
        from ..helpers.polars_dataframe_helpers import build_polars_with_columns_operation

        result_df = build_polars_with_columns_operation(self, *exprs)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def with_column(self, expr: Union[Column, Tuple[str, Column]]) -> "AsyncPolarsDataFrame":
        """Add or modify a single column (alias for with_columns with one expression).

        Args:
            expr: :class:`Column` expression or (name, expression) tuple

        Returns:
            AsyncPolarsDataFrame with new/modified column

        Example:
            >>> df.with_column((col('amount') * 1.1).alias('with_tax'))
        """
        return self.with_columns(expr)

    def drop(self, *columns: Union[str, Column]) -> "AsyncPolarsDataFrame":
        """Drop columns (Polars-style).

        Args:
            *columns: :class:`Column` names to drop

        Returns:
            AsyncPolarsDataFrame with dropped columns

        Example:
            >>> df.drop('col1', 'col2')
        """
        from ..helpers.polars_dataframe_helpers import build_polars_drop_operation

        if not columns:
            return self

        result_df = build_polars_drop_operation(self, *columns)
        if result_df is None:
            return self
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def rename(self, mapping: Dict[str, str]) -> "AsyncPolarsDataFrame":
        """Rename columns (Polars-style).

        Args:
            mapping: Dictionary mapping old names to new names

        Returns:
            AsyncPolarsDataFrame with renamed columns

        Example:
            >>> df.rename({'old_name': 'new_name'})
        """
        from ..helpers.polars_dataframe_helpers import build_polars_rename_operation

        result_df = build_polars_rename_operation(self, mapping)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def sort(
        self,
        *columns: Union[str, Column],
        descending: Union[bool, Sequence[bool]] = False,
    ) -> "AsyncPolarsDataFrame":
        """Sort by columns (Polars-style).

        Args:
            *columns: :class:`Column` names or :class:`Column` expressions to sort by
            descending: Sort order - single bool or sequence of bools for each column

        Returns:
            Sorted AsyncPolarsDataFrame

        Example:
            >>> df.sort('age')
            >>> df.sort('age', 'name', descending=[True, False])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_sort_operation

        if not columns:
            return self

        result_df = build_polars_sort_operation(self, *columns, descending=descending)
        if result_df is None:
            return self
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def limit(self, n: int) -> "AsyncPolarsDataFrame":
        """Limit number of rows.

        Args:
            n: Number of rows to return

        Returns:
            AsyncPolarsDataFrame with limited rows

        Example:
            >>> df.limit(10)
        """
        from ..helpers.polars_dataframe_helpers import build_polars_limit_operation

        result_df = build_polars_limit_operation(self, n)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def head(self, n: int = 5) -> "AsyncPolarsDataFrame":
        """Return the first n rows.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            AsyncPolarsDataFrame with first n rows

        Example:
            >>> df.head(10)  # First 10 rows
        """
        return self.limit(n)

    def tail(self, n: int = 5) -> "AsyncPolarsDataFrame":
        """Return the last n rows.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            AsyncPolarsDataFrame with last n rows

        Note:
            This is a simplified implementation. For proper tail() behavior with lazy
            evaluation, this method sorts all columns in descending order and takes
            the first n rows. For better performance, consider using limit() directly
            or collecting and using polars tail().

        Example:
            >>> df.tail(10)  # Last 10 rows
        """
        from ..helpers.polars_dataframe_helpers import build_polars_tail_operation

        result_df = build_polars_tail_operation(self, n)
        if result_df is None:
            return self
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def sample(
        self,
        fraction: Optional[float] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "AsyncPolarsDataFrame":
        """Random sampling (Polars-style).

        Args:
            fraction: Fraction of rows to sample (0.0 to 1.0)
            n: Number of rows to sample (if provided, fraction is ignored)
            seed: Random seed for reproducibility

        Returns:
            Sampled AsyncPolarsDataFrame

        Example:
            >>> df.sample(fraction=0.1, seed=42)
            >>> df.sample(n=100, seed=42)
        """
        from ..helpers.polars_dataframe_helpers import build_polars_sample_operation

        result_df = build_polars_sample_operation(self, fraction=fraction, n=n, seed=seed)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def group_by(self, *columns: Union[str, Column]) -> "AsyncPolarsGroupBy":
        """Group rows by one or more columns (Polars-style).

        Args:
            *columns: :class:`Column` name(s) to group by

        Returns:
            AsyncPolarsGroupBy object for aggregation

        Example:
            >>> df.group_by('country')
            >>> df.group_by('country', 'region')
        """
        from ..groupby.async_polars_groupby import AsyncPolarsGroupBy

        # Validate columns exist
        str_columns = [c for c in columns if isinstance(c, str)]
        if str_columns:
            self._validate_columns_exist(str_columns, "group_by")

        # Use the underlying AsyncDataFrame's group_by method
        grouped_df = self._df.group_by(*columns)

        # Wrap it in AsyncPolarsGroupBy
        return AsyncPolarsGroupBy(_grouped=grouped_df)

    def join(
        self,
        other: "AsyncPolarsDataFrame",
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        how: str = "inner",
        left_on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
        suffix: str = "_right",
    ) -> "AsyncPolarsDataFrame":
        """Join with another AsyncPolarsDataFrame (Polars-style).

        Args:
            other: Right :class:`DataFrame` to join with
            on: :class:`Column` name(s) to join on (must exist in both DataFrames)
            how: Type of join ('inner', 'left', 'right', 'outer', 'anti', 'semi')
            left_on: :class:`Column` name(s) in left :class:`DataFrame`
            right_on: :class:`Column` name(s) in right :class:`DataFrame`
            suffix: Suffix to append to overlapping column names from right :class:`DataFrame`

        Returns:
            Joined AsyncPolarsDataFrame

        Example:
            >>> df1.join(df2, on='id')
            >>> df1.join(df2, left_on='customer_id', right_on='id', how='left')
        """
        from ..helpers.polars_dataframe_helpers import build_polars_join_operation

        result_df = build_polars_join_operation(
            self, other, on=on, how=how, left_on=left_on, right_on=right_on
        )
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def semi_join(
        self,
        other: "AsyncPolarsDataFrame",
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        left_on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
    ) -> "AsyncPolarsDataFrame":
        """Semi-join: filter rows in left :class:`DataFrame` that have matches in right :class:`DataFrame`.

        Args:
            other: Right :class:`DataFrame` to join with
            on: :class:`Column` name(s) to join on (must exist in both DataFrames)
            left_on: :class:`Column` name(s) in left :class:`DataFrame`
            right_on: :class:`Column` name(s) in right :class:`DataFrame`

        Returns:
            AsyncPolarsDataFrame with rows from left that have matches in right

        Example:
            >>> df1.semi_join(df2, on='id')
            >>> df1.semi_join(df2, left_on='customer_id', right_on='id')
        """
        # Use the join method with how='semi'
        return self.join(other, on=on, left_on=left_on, right_on=right_on, how="semi")

    def anti_join(
        self,
        other: "AsyncPolarsDataFrame",
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        left_on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
    ) -> "AsyncPolarsDataFrame":
        """Anti-join: filter rows in left :class:`DataFrame` that don't have matches in right :class:`DataFrame`.

        Args:
            other: Right :class:`DataFrame` to join with
            on: :class:`Column` name(s) to join on (must exist in both DataFrames)
            left_on: :class:`Column` name(s) in left :class:`DataFrame`
            right_on: :class:`Column` name(s) in right :class:`DataFrame`

        Returns:
            AsyncPolarsDataFrame with rows from left that don't have matches in right

        Example:
            >>> df1.anti_join(df2, on='id')
            >>> df1.anti_join(df2, left_on='customer_id', right_on='id')
        """
        # Use the join method with how='anti'
        return self.join(other, on=on, left_on=left_on, right_on=right_on, how="anti")

    def unique(
        self, subset: Optional[Union[str, Sequence[str]]] = None, keep: str = "first"
    ) -> "AsyncPolarsDataFrame":
        """Remove duplicate rows (Polars-style).

        Args:
            subset: :class:`Column` name(s) to consider for duplicates (None means all columns)
            keep: Which duplicate to keep ('first' or 'last')

        Returns:
            AsyncPolarsDataFrame with duplicates removed

        Example:
            >>> df.unique()
            >>> df.unique(subset=['col1', 'col2'])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_unique_operation

        result_df = build_polars_unique_operation(self, subset=subset, keep=keep)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def distinct(self) -> "AsyncPolarsDataFrame":
        """Remove duplicate rows (alias for unique()).

        Returns:
            AsyncPolarsDataFrame with duplicates removed

        Example:
            >>> df.distinct()
        """
        return self.unique()

    def drop_nulls(
        self, subset: Optional[Union[str, Sequence[str]]] = None
    ) -> "AsyncPolarsDataFrame":
        """Drop rows with null values (Polars-style).

        Args:
            subset: :class:`Column` name(s) to check for nulls (None means all columns)

        Returns:
            AsyncPolarsDataFrame with null rows removed

        Example:
            >>> df.drop_nulls()
            >>> df.drop_nulls(subset=['col1', 'col2'])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_drop_nulls_operation

        result_df = build_polars_drop_nulls_operation(self, subset=subset)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def fill_null(
        self,
        value: Optional[FillValue] = None,
        strategy: Optional[str] = None,
        limit: Optional[int] = None,
        subset: Optional[Union[str, Sequence[str]]] = None,
    ) -> "AsyncPolarsDataFrame":
        """Fill null values (Polars-style).

        Args:
            value: Value to fill nulls with
            strategy: Fill strategy (e.g., 'forward', 'backward') - not fully supported
            limit: Maximum number of consecutive nulls to fill - not fully supported
            subset: :class:`Column` name(s) to fill nulls in (None means all columns)

        Returns:
            AsyncPolarsDataFrame with nulls filled

        Example:
            >>> df.fill_null(0)
            >>> df.fill_null(value='unknown', subset=['name'])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_fill_null_operation

        result_df = build_polars_fill_null_operation(
            self, value=value, strategy=strategy, limit=limit, subset=subset
        )
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def __getitem__(
        self, key: Union[str, Sequence[str], Column]
    ) -> Union["AsyncPolarsDataFrame", Column, "PolarsColumn"]:
        """Polars-style column access.

        Supports:
        - df['col'] - Returns :class:`Column` expression for filtering/expressions
        - df[['col1', 'col2']] - Returns new AsyncPolarsDataFrame with selected columns
        - df[df['age'] > 25] - Boolean indexing (filtering via :class:`Column` condition)

        Args:
            key: :class:`Column` name(s) or boolean :class:`Column` condition

        Returns:
            - For single column string: :class:`Column` expression
            - For list of columns: AsyncPolarsDataFrame with selected columns
            - For boolean :class:`Column` condition: AsyncPolarsDataFrame with filtered rows

        Example:
            >>> df['age']  # Returns :class:`Column` expression
            >>> df[['id', 'name']]  # Returns AsyncPolarsDataFrame
            >>> df[df['age'] > 25]  # Returns filtered AsyncPolarsDataFrame
        """
        # Single column string: df['col'] - return PolarsColumn with str/dt accessors
        if isinstance(key, str):
            self._validate_columns_exist([key], "column access")
            from ..columns.polars_column import PolarsColumn

            return PolarsColumn(col(key))

        # List of columns: df[['col1', 'col2']] - select columns
        if isinstance(key, (list, tuple)):
            if len(key) == 0:
                return self._with_dataframe(self._df.select())
            str_columns = [c for c in key if isinstance(c, str)]
            if str_columns:
                self._validate_columns_exist(str_columns, "column selection")
            columns = [col(c) if isinstance(c, str) else c for c in key]
            return self._with_dataframe(self._df.select(*columns))

        # Column expression - if it's a boolean condition, use as filter
        if isinstance(key, Column):
            return self._with_dataframe(self._df.where(key))

        raise TypeError(
            f"Invalid key type for __getitem__: {type(key)}. Expected str, list, tuple, or Column."
        )

    @overload
    async def collect(self, stream: Literal[False] = False) -> "pl.DataFrame": ...

    @overload
    async def collect(
        self, stream: Literal[True]
    ) -> AsyncIterator[Union["pl.DataFrame", List[Dict[str, Any]]]]: ...

    async def collect(
        self, stream: bool = False
    ) -> Union[
        "pl.DataFrame",
        AsyncIterator[Union["pl.DataFrame", List[Dict[str, Any]]]],
        List[Dict[str, Any]],
    ]:
        """Collect results as Polars :class:`DataFrame` (async).

        Args:
            stream: If True, return an async iterator of Polars :class:`DataFrame` chunks.
                   If False (default), return a single Polars :class:`DataFrame`.

        Returns:
            If stream=False: Polars :class:`DataFrame` (if polars installed) or list of dicts
            If stream=True: AsyncIterator of Polars :class:`DataFrame` chunks

        Example:
            >>> pdf = await df.collect()  # Returns pl.:class:`DataFrame`
            >>> async for chunk in await df.collect(stream=True):  # Streaming
            ...     process(chunk)
        """
        # Collect results from underlying AsyncDataFrame
        if stream:
            # Streaming mode
            async def _stream_chunks() -> AsyncIterator[
                Union["pl.DataFrame", List[Dict[str, Any]]]
            ]:
                try:
                    import polars as pl
                except ImportError:
                    # Fall back to list of dicts if polars not available
                    async for chunk in await self._df.collect(stream=True):
                        yield chunk
                    return

                async for chunk in await self._df.collect(stream=True):
                    df_chunk = pl.DataFrame(chunk)
                    yield df_chunk

            return _stream_chunks()
        else:
            # Single result
            results = await self._df.collect(stream=False)

            try:
                import polars as pl
            except ImportError:
                # Return list of dicts if polars not available
                return results

            return pl.DataFrame(results)

    async def fetch(self, n: int) -> "pl.DataFrame":
        """Fetch first n rows without full collection (async).

        Args:
            n: Number of rows to fetch

        Returns:
            Polars :class:`DataFrame` with first n rows

        Example:
            >>> pdf = await df.fetch(10)  # First 10 rows as Polars :class:`DataFrame`
        """
        limited_df = self.limit(n)
        return await limited_df.collect()

    async def write_csv(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this AsyncPolarsDataFrame to a CSV file (Polars-style, async).

        Args:
            path: Path to write the CSV file
            mode: Write mode ("overwrite", "append", "error_if_exists")
            **options: Format-specific options (e.g., header=True, delimiter=",")

        Example:
            >>> await df.write_csv("output.csv", header=True)
            >>> await df.write_csv("output.csv", mode="append", header=True, delimiter=",")
        """
        writer = self._df.write.mode(mode)
        if options:
            writer = writer.options(**options)
        await writer.csv(path)

    async def write_json(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this AsyncPolarsDataFrame to a JSON file (Polars-style, async).

        Args:
            path: Path to write the JSON file
            mode: Write mode ("overwrite", "append", "error_if_exists")
            **options: Format-specific options

        Example:
            >>> await df.write_json("output.json")
        """
        writer = self._df.write.mode(mode)
        if options:
            writer = writer.options(**options)
        await writer.json(path)

    async def write_jsonl(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this AsyncPolarsDataFrame to a JSONL file (Polars-style, async).

        Args:
            path: Path to write the JSONL file
            mode: Write mode ("overwrite", "append", "error_if_exists")
            **options: Format-specific options

        Example:
            >>> await df.write_jsonl("output.jsonl")
        """
        writer = self._df.write.mode(mode)
        if options:
            writer = writer.options(**options)
        await writer.jsonl(path)

    async def write_parquet(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this AsyncPolarsDataFrame to a Parquet file (Polars-style, async).

        Args:
            path: Path to write the Parquet file
            mode: Write mode ("overwrite", "append", "error_if_exists")
            **options: Format-specific options

        Raises:
            RuntimeError: If pandas or pyarrow are not installed

        Example:
            >>> await df.write_parquet("output.parquet")
        """
        writer = self._df.write.mode(mode)
        if options:
            writer = writer.options(**options)
        await writer.parquet(path)

    # ========================================================================
    # Additional Polars Features
    # ========================================================================

    def explode(self, columns: Union[str, Sequence[str]]) -> "AsyncPolarsDataFrame":
        """Explode array/JSON columns into multiple rows (Polars-style).

        Args:
            columns: :class:`Column` name(s) to explode

        Returns:
            AsyncPolarsDataFrame with exploded rows

        Example:
            >>> df.explode('tags')
            >>> df.explode(['tags', 'categories'])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_explode_operation

        result_df = build_polars_explode_operation(self, columns)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def unnest(self, columns: Union[str, Sequence[str]]) -> "AsyncPolarsDataFrame":
        """Unnest struct columns (Polars-style).

        Note: This is similar to explode but for struct types.
        For now, we'll use explode as the implementation.

        Args:
            columns: :class:`Column` name(s) to unnest

        Returns:
            AsyncPolarsDataFrame with unnested columns

        Example:
            >>> df.unnest('struct_col')
        """
        # For now, unnest is similar to explode
        return self.explode(columns)

    def pivot(
        self,
        values: Union[str, Sequence[str]],
        index: Optional[Union[str, Sequence[str]]] = None,
        columns: Optional[str] = None,
        aggregate_function: Optional[str] = None,
    ) -> "AsyncPolarsDataFrame":
        """Pivot :class:`DataFrame` (Polars-style).

        Args:
            values: :class:`Column`(s) to aggregate
            index: :class:`Column`(s) to use as index (rows)
            columns: :class:`Column` to use as columns (pivot column)
            aggregate_function: Aggregation function (e.g., 'sum', 'mean', 'count')

        Returns:
            Pivoted AsyncPolarsDataFrame

        Example:
            >>> df.pivot(values='amount', index='category', columns='status', aggregate_function='sum')
        """
        from ..helpers.polars_dataframe_helpers import build_polars_pivot_operation

        if aggregate_function is None:
            aggregate_function = "sum"

        result_df = build_polars_pivot_operation(self, values, columns, aggregate_function)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def melt(
        self,
        id_vars: Optional[Union[str, Sequence[str]]] = None,
        value_vars: Optional[Union[str, Sequence[str]]] = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> "AsyncPolarsDataFrame":
        """Melt :class:`DataFrame` from wide to long format (Polars-style).

        Args:
            id_vars: :class:`Column`(s) to use as identifier variables
            value_vars: :class:`Column`(s) to unpivot (if None, unpivot all except id_vars)
            variable_name: Name for the variable column
            value_name: Name for the value column

        Returns:
            Melted AsyncPolarsDataFrame

        Example:
            >>> df.melt(id_vars=['id'], value_vars=['col1', 'col2'])
        """
        # Melt is not yet implemented in DataFrame, so we'll raise NotImplementedError
        # This would require implementing UNPIVOT in SQL
        raise NotImplementedError(
            "melt() is not yet implemented. "
            "This would require UNPIVOT SQL support which varies by database."
        )

    def slice(self, offset: int, length: Optional[int] = None) -> "AsyncPolarsDataFrame":
        """Slice :class:`DataFrame` (Polars-style).

        Args:
            offset: Starting row index
            length: Number of rows to return (if None, returns all remaining rows)

        Returns:
            Sliced AsyncPolarsDataFrame

        Example:
            >>> df.slice(10, 5)  # Rows 10-14
            >>> df.slice(10)  # All rows from 10 onwards
        """
        from ..helpers.polars_dataframe_helpers import build_polars_slice_operation

        result_df = build_polars_slice_operation(self, offset, length)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def gather_every(self, n: int, offset: int = 0) -> "AsyncPolarsDataFrame":
        """Sample every nth row (Polars-style).

        Args:
            n: Sample every nth row
            offset: Starting offset

        Returns:
            Sampled AsyncPolarsDataFrame

        Example:
            >>> df.gather_every(10)  # Every 10th row
            >>> df.gather_every(5, offset=2)  # Every 5th row starting from row 2
        """
        from ..helpers.polars_dataframe_helpers import build_polars_gather_every_operation

        result_df = build_polars_gather_every_operation(self, n, offset)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def interpolate(self, method: str = "linear") -> "AsyncPolarsDataFrame":
        """Interpolate missing values (Polars-style).

        Args:
            method: Interpolation method ('linear', 'nearest', etc.)

        Returns:
            AsyncPolarsDataFrame with interpolated values

        Note:
            Full interpolation support depends on database capabilities.
            This is a placeholder for the API.

        Example:
            >>> df.interpolate()
        """
        # Interpolation in SQL is complex and database-specific
        # For now, we'll raise NotImplementedError
        raise NotImplementedError(
            "interpolate() is not yet implemented. "
            "This would require database-specific interpolation functions."
        )

    def quantile(
        self,
        quantile: Union[float, Sequence[float]],
        interpolation: str = "linear",
    ) -> "AsyncPolarsDataFrame":
        """Compute quantiles (Polars-style).

        Args:
            quantile: Quantile value(s) (0.0 to 1.0)
            interpolation: Interpolation method (not used, for API compatibility)

        Returns:
            AsyncPolarsDataFrame with quantile values

        Note:
            Quantile computation requires database-specific functions.
            This is a simplified implementation.

        Example:
            >>> df.quantile(0.5)  # Median
            >>> df.quantile([0.25, 0.5, 0.75])  # Quartiles
        """
        # Quantile requires aggregation
        # For now, we'll compute basic statistics instead
        # Full quantile support would require PERCENTILE_CONT or similar
        from ...expressions import functions as F

        if isinstance(quantile, (int, float)):
            quantile = [quantile]

        # For each numeric column, compute approximate quantiles using median
        # Full implementation would use database-specific percentile functions
        numeric_cols = [c for c in self.columns if self._is_numeric_column(c)]
        if not numeric_cols:
            return self

        # Use percentile_cont for quantiles
        quantile_exprs = []
        for col_name in numeric_cols:
            for q in quantile:
                quantile_exprs.append(
                    F.percentile_cont(col(col_name), q).alias(f"{col_name}_q{int(q * 100)}")
                )

        result_df = self._df.select(*quantile_exprs)
        return self._with_dataframe(result_df)

    async def describe(self) -> "AsyncPolarsDataFrame":
        """Compute descriptive statistics (Polars-style, async).

        Returns:
            AsyncPolarsDataFrame with statistics (count, mean, std, min, max, etc.)

        Note:
            Standard deviation (std) may not be available in all databases
            (e.g., SQLite). In such cases, std will be omitted.

        Example:
            >>> stats = await df.describe()
        """
        from ..helpers.polars_dataframe_helpers import build_polars_describe_operation

        result_df = build_polars_describe_operation(self)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    async def explain(self, format: str = "string") -> str:
        """Explain the query plan (Polars-style, async).

        Args:
            format: Output format ('string' or 'tree')

        Returns:
            Query plan as string

        Example:
            >>> plan = await df.explain()
            >>> print(plan)
        """
        result = await self._df.explain(analyze=False)  # type: ignore[operator]
        return str(result)

    def _is_numeric_column(self, col_name: str) -> bool:
        """Check if a column is numeric based on schema."""
        # Note: This is sync but uses schema which is async
        # For now, we'll use a simplified check based on column name patterns
        # A better implementation would await schema() but that's not possible in a sync method
        # This is a limitation - users should use describe() for proper statistics
        numeric_patterns = ["id", "count", "amount", "price", "age", "score", "value"]
        col_lower = col_name.lower()
        return any(pattern in col_lower for pattern in numeric_patterns)

    # ========================================================================
    # Set Operations (Polars-style)
    # ========================================================================

    def concat(
        self,
        *others: "AsyncPolarsDataFrame",
        how: str = "vertical",
        rechunk: bool = True,
    ) -> "AsyncPolarsDataFrame":
        """Concatenate DataFrames (Polars-style).

        Args:
            *others: Other AsyncPolarsDataFrames to concatenate
            how: Concatenation mode - "vertical" (union) or "diagonal" (union with different schemas)
            rechunk: If True, rechunk the result (not used, for API compatibility)

        Returns:
            Concatenated AsyncPolarsDataFrame

        Example:
            >>> df1.concat(df2)  # Vertical concatenation
            >>> df1.concat(df2, df3, how="vertical")
        """
        from ..helpers.polars_dataframe_helpers import build_polars_concat_operation

        result_df = build_polars_concat_operation(self, [other._df for other in others], how=how)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def hstack(
        self,
        *others: "AsyncPolarsDataFrame",
    ) -> "AsyncPolarsDataFrame":
        """Horizontally stack DataFrames (Polars-style).

        Args:
            *others: Other AsyncPolarsDataFrames to stack horizontally

        Returns:
            Horizontally stacked AsyncPolarsDataFrame

        Example:
            >>> df1.hstack(df2)  # Combine columns from df1 and df2
        """
        from ..helpers.polars_dataframe_helpers import build_polars_hstack_operation

        result_df = build_polars_hstack_operation(self, [other._df for other in others])
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def vstack(
        self,
        *others: "AsyncPolarsDataFrame",
    ) -> "AsyncPolarsDataFrame":
        """Vertically stack DataFrames (Polars-style alias for concat).

        Args:
            *others: Other AsyncPolarsDataFrames to stack vertically

        Returns:
            Vertically stacked AsyncPolarsDataFrame

        Example:
            >>> df1.vstack(df2)  # Same as df1.concat(df2)
        """
        return self.concat(*others, how="vertical")

    def union(
        self,
        other: "AsyncPolarsDataFrame",
        *,
        distinct: bool = True,
    ) -> "AsyncPolarsDataFrame":
        """Union with another AsyncPolarsDataFrame (Polars-style).

        Args:
            other: Another AsyncPolarsDataFrame to union with
            distinct: If True, return distinct rows only (default: True)

        Returns:
            Unioned AsyncPolarsDataFrame

        Example:
            >>> df1.union(df2)  # Union distinct
            >>> df1.union(df2, distinct=False)  # Union all
        """
        from ..helpers.polars_dataframe_helpers import build_polars_union_operation

        result_df = build_polars_union_operation(self, other._df, distinct=distinct)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def intersect(
        self,
        other: "AsyncPolarsDataFrame",
    ) -> "AsyncPolarsDataFrame":
        """Intersect with another AsyncPolarsDataFrame (Polars-style).

        Args:
            other: Another AsyncPolarsDataFrame to intersect with

        Returns:
            Intersected AsyncPolarsDataFrame (common rows only)

        Example:
            >>> df1.intersect(df2)  # Common rows
        """
        from ..helpers.polars_dataframe_helpers import build_polars_intersect_operation

        result_df = build_polars_intersect_operation(self, other._df)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def difference(
        self,
        other: "AsyncPolarsDataFrame",
    ) -> "AsyncPolarsDataFrame":
        """Return rows in this :class:`DataFrame` that are not in another (Polars-style).

        Args:
            other: Another AsyncPolarsDataFrame to exclude from

        Returns:
            AsyncPolarsDataFrame with rows in this but not in other

        Example:
            >>> df1.difference(df2)  # Rows in df1 but not in df2
        """
        from ..helpers.polars_dataframe_helpers import build_polars_difference_operation

        result_df = build_polars_difference_operation(self, other._df)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def cross_join(
        self,
        other: "AsyncPolarsDataFrame",
    ) -> "AsyncPolarsDataFrame":
        """Perform a cross join (Cartesian product) with another AsyncPolarsDataFrame (Polars-style).

        Args:
            other: Another AsyncPolarsDataFrame to cross join with

        Returns:
            Cross-joined AsyncPolarsDataFrame

        Example:
            >>> df1.cross_join(df2)  # Cartesian product
        """
        from ..helpers.polars_dataframe_helpers import build_polars_cross_join_operation

        result_df = build_polars_cross_join_operation(self, other._df)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    # ========================================================================
    # SQL Expression Selection
    # ========================================================================

    def select_expr(
        self,
        *exprs: str,
    ) -> "AsyncPolarsDataFrame":
        """Select columns using SQL expressions (Polars-style).

        Args:
            *exprs: SQL expression strings (e.g., "amount * 1.1 as with_tax")

        Returns:
            AsyncPolarsDataFrame with selected expressions

        Example:
            >>> df.select_expr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")
        """
        from ..helpers.polars_dataframe_helpers import build_polars_select_expr_operation

        result_df = build_polars_select_expr_operation(self, *exprs)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    # ========================================================================
    # Common Table Expressions (CTEs)
    # ========================================================================

    def with_columns_renamed(
        self,
        mapping: Dict[str, str],
    ) -> "AsyncPolarsDataFrame":
        """Rename columns using a mapping (Polars-style alias for rename).

        Args:
            mapping: Dictionary mapping old column names to new names

        Returns:
            AsyncPolarsDataFrame with renamed columns

        Example:
            >>> df.with_columns_renamed({"old_name": "new_name"})
        """
        return self.rename(mapping)

    def with_row_count(
        self,
        name: str = "row_nr",
        offset: int = 0,
    ) -> "AsyncPolarsDataFrame":
        """Add a row number column (Polars-style).

        Args:
            name: Name for the row number column (default: "row_nr")
            offset: Starting offset for row numbers (default: 0)

        Returns:
            AsyncPolarsDataFrame with row number column

        Example:
            >>> df.with_row_count("row_id")
        """
        from ...expressions import functions as F

        # Add row number using window function
        row_num_col = F.row_number().over()
        if offset != 0:
            # Add offset to row number
            row_num_col = (row_num_col + offset).alias(name)
        else:
            row_num_col = row_num_col.alias(name)

        return self.with_columns(row_num_col)

    def with_context(
        self,
        *contexts: "AsyncPolarsDataFrame",
    ) -> "AsyncPolarsDataFrame":
        """Add context DataFrames for use in expressions (Polars-style).

        Note: This is a placeholder for Polars' with_context feature.
        In Moltres, CTEs serve a similar purpose.

        Args:
            *contexts: Context DataFrames to add

        Returns:
            AsyncPolarsDataFrame with context

        Example:
            >>> df.with_context(context_df)
        """
        # For now, this is a no-op as Moltres doesn't have the same context system
        # Users should use CTEs instead
        return self

    async def summary(self, *statistics: str) -> "AsyncPolarsDataFrame":
        """Compute summary statistics for numeric columns (Polars-style).

        Args:
            *statistics: Statistics to compute (e.g., "count", "mean", "stddev", "min", "max").
                        If not provided, computes common statistics.

        Returns:
            AsyncPolarsDataFrame with summary statistics

        Example:
            >>> await df.summary()
            >>> await df.summary("count", "mean", "max")
        """
        # Call summary on the underlying AsyncDataFrame directly (it's async)
        result_df = await self._df.summary(*statistics)
        return self._with_dataframe(result_df)

    # ========================================================================
    # Common Table Expressions (CTEs) - Moltres-specific but Polars-style API
    # ========================================================================

    def cte(
        self,
        name: str,
    ) -> "AsyncPolarsDataFrame":
        """Create a Common Table Expression (CTE) from this :class:`DataFrame`.

        Args:
            name: Name for the CTE

        Returns:
            AsyncPolarsDataFrame representing the CTE

        Example:
            >>> cte_df = df.filter(col("age") > 25).cte("adults")
            >>> result = await cte_df.select().collect()
        """
        from ..helpers.polars_dataframe_helpers import build_polars_cte_operation

        result_df = build_polars_cte_operation(self, name)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))

    def with_recursive(
        self,
        name: str,
        recursive: "AsyncPolarsDataFrame",
        *,
        union_all: bool = False,
    ) -> "AsyncPolarsDataFrame":
        """Create a Recursive Common Table Expression (WITH RECURSIVE).

        Args:
            name: Name for the recursive CTE
            recursive: AsyncPolarsDataFrame representing the recursive part
            union_all: If True, use UNION ALL; if False, use UNION (distinct)

        Returns:
            AsyncPolarsDataFrame representing the recursive CTE

        Example:
            >>> initial = await db.table("seed").polars()
            >>> recursive = initial.select(...)  # Recursive part
            >>> fib_cte = initial.with_recursive("fib", recursive)
        """
        from ..helpers.polars_dataframe_helpers import build_polars_recursive_cte_operation

        result_df = build_polars_recursive_cte_operation(self, name, recursive, union_all=union_all)
        return self._with_dataframe(cast(AsyncDataFrame, result_df))
