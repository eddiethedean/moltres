"""Pandas-style interface for Moltres DataFrames."""

from __future__ import annotations

from dataclasses import dataclass
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
    Union,
    overload,
)

from ..expressions.column import Column, col
from ..logical.plan import LogicalPlan
from .dataframe import DataFrame

if TYPE_CHECKING:
    import pandas as pd
    from ..table.table import Database
    from .pandas_groupby import PandasGroupBy


@dataclass(frozen=True)
class PandasDataFrame:
    """Pandas-style interface wrapper around Moltres DataFrame.

    Provides familiar pandas API methods while maintaining lazy evaluation
    and SQL pushdown execution. All operations remain lazy until collect() is called.

    Example:
        >>> df = db.table('users').pandas()
        >>> # Pandas-style column access
        >>> df[['id', 'name']].query('age > 25')
        >>> # Pandas-style groupby
        >>> df.groupby('country').agg({'amount': 'sum'})
        >>> # Returns actual pandas DataFrame
        >>> result = df.collect()  # pd.DataFrame
    """

    _df: DataFrame

    @property
    def plan(self) -> LogicalPlan:
        """Get the underlying logical plan."""
        return self._df.plan

    @property
    def database(self) -> Optional["Database"]:
        """Get the associated database."""
        return self._df.database

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> "PandasDataFrame":
        """Create a PandasDataFrame from a regular DataFrame.

        Args:
            df: The DataFrame to wrap

        Returns:
            PandasDataFrame wrapping the provided DataFrame
        """
        return cls(_df=df)

    def _with_dataframe(self, df: DataFrame) -> "PandasDataFrame":
        """Create a new PandasDataFrame with a different underlying DataFrame.

        Args:
            df: The new underlying DataFrame

        Returns:
            New PandasDataFrame instance
        """
        return PandasDataFrame(_df=df)

    def __getitem__(
        self, key: Union[str, Sequence[str], Column]
    ) -> Union["PandasDataFrame", Column]:
        """Pandas-style column access.

        Supports:
        - df['col'] - Returns Column expression for filtering/expressions
        - df[['col1', 'col2']] - Returns new PandasDataFrame with selected columns
        - df[df['age'] > 25] - Boolean indexing (filtering via Column condition)

        Args:
            key: Column name(s) or boolean Column condition

        Returns:
            - For single column string: Column expression
            - For list of columns: PandasDataFrame with selected columns
            - For boolean Column condition: PandasDataFrame with filtered rows

        Example:
            >>> df['age']  # Returns Column expression
            >>> df[['id', 'name']]  # Returns PandasDataFrame
            >>> df[df['age'] > 25]  # Returns filtered PandasDataFrame
        """
        # Single column string: df['col'] - return Column for expressions
        if isinstance(key, str):
            return col(key)

        # List of columns: df[['col1', 'col2']] - select columns
        if isinstance(key, (list, tuple)):
            if len(key) == 0:
                return self._with_dataframe(self._df.select())
            # Convert all to strings/Columns and select
            columns = [col(c) if isinstance(c, str) else c for c in key]
            return self._with_dataframe(self._df.select(*columns))

        # Column expression - if it's a boolean condition, use as filter
        if isinstance(key, Column):
            # This is likely a boolean condition like df['age'] > 25
            # We should filter using it
            return self._with_dataframe(self._df.where(key))

        raise TypeError(
            f"Invalid key type for __getitem__: {type(key)}. Expected str, list, tuple, or Column."
        )

    def query(self, expr: str) -> "PandasDataFrame":
        """Filter DataFrame using a pandas-style query string.

        Args:
            expr: Query string with pandas-style syntax (e.g., "age > 25 and status == 'active'")

        Returns:
            Filtered PandasDataFrame

        Example:
            >>> df.query('age > 25')
            >>> df.query("age > 25 and status == 'active'")
            >>> df.query("name in ['Alice', 'Bob']")
        """
        from ..expressions.sql_parser import parse_sql_expr

        # Get available column names for context
        available_columns: Optional[Set[str]] = None
        try:
            if hasattr(self._df, "plan") and hasattr(self._df.plan, "projections"):
                available_columns = set()
                for proj in self._df.plan.projections:
                    if isinstance(proj, Column) and proj.op == "column" and proj.args:
                        available_columns.add(str(proj.args[0]))
        except Exception:
            pass

        # Parse query string to Column expression
        predicate = parse_sql_expr(expr, available_columns)

        # Apply filter
        return self._with_dataframe(self._df.where(predicate))

    def groupby(self, by: Union[str, Sequence[str]], *args: Any, **kwargs: Any) -> "PandasGroupBy":
        """Group rows by one or more columns (pandas-style).

        Args:
            by: Column name(s) to group by
            *args: Additional positional arguments (for pandas compatibility)
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            PandasGroupBy object for aggregation

        Example:
            >>> df.groupby('country')
            >>> df.groupby(['country', 'region'])
        """
        from .pandas_groupby import PandasGroupBy

        if isinstance(by, str):
            columns = (by,)
        elif isinstance(by, (list, tuple)):
            columns = tuple(by)
        else:
            raise TypeError(f"by must be str or sequence of str, got {type(by)}")

        # Use the underlying DataFrame's group_by method to get GroupedDataFrame
        grouped_df = self._df.group_by(*columns)

        # Wrap it in PandasGroupBy
        return PandasGroupBy(_grouped=grouped_df)

    def merge(
        self,
        right: "PandasDataFrame",
        *,
        on: Optional[Union[str, Sequence[str]]] = None,
        left_on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
        how: str = "inner",
        **kwargs: Any,
    ) -> "PandasDataFrame":
        """Merge two DataFrames (pandas-style join).

        Args:
            right: Right DataFrame to merge with
            on: Column name(s) to join on (must exist in both DataFrames)
            left_on: Column name(s) in left DataFrame
            right_on: Column name(s) in right DataFrame
            how: Type of join ('inner', 'left', 'right', 'outer')
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            Merged PandasDataFrame

        Example:
            >>> df1.merge(df2, on='id')
            >>> df1.merge(df2, left_on='customer_id', right_on='id')
            >>> df1.merge(df2, on='id', how='left')
        """
        # Normalize how parameter
        how_map = {
            "inner": "inner",
            "left": "left",
            "right": "right",
            "outer": "outer",
            "full": "outer",
            "full_outer": "outer",
        }
        join_how = how_map.get(how.lower(), "inner")

        # Determine join keys
        if on is not None:
            # Same column names in both DataFrames
            if isinstance(on, str):
                join_on = [(on, on)]
            else:
                join_on = [(col, col) for col in on]
        elif left_on is not None and right_on is not None:
            # Different column names
            if isinstance(left_on, str) and isinstance(right_on, str):
                join_on = [(left_on, right_on)]
            elif isinstance(left_on, (list, tuple)) and isinstance(right_on, (list, tuple)):
                if len(left_on) != len(right_on):
                    raise ValueError("left_on and right_on must have the same length")
                join_on = list(zip(left_on, right_on))
            else:
                raise TypeError("left_on and right_on must both be str or both be sequences")
        else:
            raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")

        # Perform join
        result_df = self._df.join(right._df, on=join_on, how=join_how)
        return self._with_dataframe(result_df)

    def sort_values(
        self,
        by: Union[str, Sequence[str]],
        ascending: Union[bool, Sequence[bool]] = True,
        **kwargs: Any,
    ) -> "PandasDataFrame":
        """Sort DataFrame by column(s) (pandas-style).

        Args:
            by: Column name(s) to sort by
            ascending: Sort order (True for ascending, False for descending)
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            Sorted PandasDataFrame

        Example:
            >>> df.sort_values('age')
            >>> df.sort_values(['age', 'name'], ascending=[False, True])
        """
        if isinstance(by, str):
            columns = [by]
            ascending_list = [ascending] if isinstance(ascending, bool) else list(ascending)
        else:
            columns = list(by)
            if isinstance(ascending, bool):
                ascending_list = [ascending] * len(columns)
            else:
                ascending_list = list(ascending)
                if len(ascending_list) != len(columns):
                    raise ValueError("ascending must have same length as by")

        # Build order_by list - use Column expressions with .desc() for descending
        order_by_cols = []
        for col_name, asc in zip(columns, ascending_list):
            col_expr = col(col_name)
            if not asc:
                # Descending order
                col_expr = col_expr.desc()
            order_by_cols.append(col_expr)

        result_df = self._df.order_by(*order_by_cols)
        return self._with_dataframe(result_df)

    def rename(self, columns: Dict[str, str], **kwargs: Any) -> "PandasDataFrame":
        """Rename columns (pandas-style).

        Args:
            columns: Dictionary mapping old names to new names
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            PandasDataFrame with renamed columns

        Example:
            >>> df.rename(columns={'old_name': 'new_name'})
        """
        result_df = self._df
        for old_name, new_name in columns.items():
            result_df = result_df.withColumnRenamed(old_name, new_name)
        return self._with_dataframe(result_df)

    def drop(
        self, columns: Optional[Union[str, Sequence[str]]] = None, **kwargs: Any
    ) -> "PandasDataFrame":
        """Drop columns (pandas-style).

        Args:
            columns: Column name(s) to drop
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            PandasDataFrame with dropped columns

        Example:
            >>> df.drop(columns=['col1', 'col2'])
            >>> df.drop(columns='col1')
        """
        if columns is None:
            return self

        if isinstance(columns, str):
            cols_to_drop = [columns]
        else:
            cols_to_drop = list(columns)

        result_df = self._df.drop(*cols_to_drop)
        return self._with_dataframe(result_df)

    def drop_duplicates(
        self, subset: Optional[Union[str, Sequence[str]]] = None, **kwargs: Any
    ) -> "PandasDataFrame":
        """Remove duplicate rows (pandas-style).

        Args:
            subset: Column name(s) to consider for duplicates (None means all columns)
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            PandasDataFrame with duplicates removed

        Example:
            >>> df.drop_duplicates()
            >>> df.drop_duplicates(subset=['col1', 'col2'])
        """
        if subset is None:
            result_df = self._df.distinct()
        else:
            # Use distinct on selected columns
            # This requires a different approach - group by columns and take first
            # For now, just use distinct on all columns
            result_df = self._df.distinct()

        return self._with_dataframe(result_df)

    def assign(self, **kwargs: Union[Column, Any]) -> "PandasDataFrame":
        """Assign new columns (pandas-style).

        Args:
            **kwargs: Column name = value pairs where value can be a Column expression or literal

        Returns:
            PandasDataFrame with new columns

        Example:
            >>> df.assign(total=df['amount'] * 1.1)
        """
        result_df = self._df
        for col_name, value in kwargs.items():
            if isinstance(value, Column):
                result_df = result_df.withColumn(col_name, value)
            else:
                # Literal value
                from ..expressions import lit

                result_df = result_df.withColumn(col_name, lit(value))
        return self._with_dataframe(result_df)

    @overload
    def collect(self, stream: Literal[False] = False) -> "pd.DataFrame": ...

    @overload
    def collect(self, stream: Literal[True]) -> Iterator["pd.DataFrame"]: ...

    def collect(self, stream: bool = False) -> Union["pd.DataFrame", Iterator["pd.DataFrame"]]:
        """Collect results as pandas DataFrame.

        Args:
            stream: If True, return an iterator of pandas DataFrame chunks.
                   If False (default), return a single pandas DataFrame.

        Returns:
            If stream=False: pandas DataFrame
            If stream=True: Iterator of pandas DataFrame chunks

        Example:
            >>> pdf = df.collect()  # Returns pd.DataFrame
            >>> for chunk in df.collect(stream=True):  # Streaming
            ...     process(chunk)
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to use PandasDataFrame.collect(). "
                "Install with: pip install pandas"
            )

        # Collect results from underlying DataFrame
        if stream:
            # Streaming mode
            def _stream_chunks() -> Iterator["pd.DataFrame"]:
                for chunk in self._df.collect(stream=True):
                    df_chunk = pd.DataFrame(chunk)
                    yield df_chunk

            return _stream_chunks()
        else:
            # Single result
            results = self._df.collect(stream=False)
            return pd.DataFrame(results)

    @property
    def columns(self) -> List[str]:
        """Get column names (pandas-style property).

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
    def dtypes(self) -> Dict[str, str]:
        """Get column data types (pandas-style property).

        Returns:
            Dictionary mapping column names to type strings

        Note:
            This is a simplified implementation. Full type information
            may require schema inspection which can be expensive.
        """
        # This is a placeholder - full implementation would require schema inspection
        # For now, return empty dict or attempt to infer from plan
        return {}

    @property
    def shape(self) -> Tuple[int, int]:
        """Get DataFrame shape (rows, columns) (pandas-style property).

        Returns:
            Tuple of (number of rows, number of columns)

        Note:
            Getting row count requires materializing the query,
            which can be expensive for large datasets.
        """
        num_cols = len(self.columns)

        # To get row count, we need to execute a COUNT query
        # This is expensive, so we'll only do it if requested
        if self.database is None:
            raise RuntimeError("Cannot get shape without an attached Database")

        # Create a count query
        from ..expressions.functions import count

        count_df = self._df.select(count("*").alias("count"))
        result = count_df.collect()
        num_rows: int = 0
        if result and isinstance(result, list) and len(result) > 0:
            row = result[0]
            if isinstance(row, dict):
                count_val = row.get("count", 0)
                if isinstance(count_val, int):
                    num_rows = count_val
                elif count_val is not None:
                    try:
                        # Type narrowing: count_val is not None and not int
                        if isinstance(count_val, (str, float)):
                            num_rows = int(count_val)
                        else:
                            num_rows = 0
                    except (ValueError, TypeError):
                        num_rows = 0

        return (num_rows, num_cols)

    @property
    def empty(self) -> bool:
        """Check if DataFrame is empty (pandas-style property).

        Returns:
            True if DataFrame has no rows, False otherwise

        Note:
            This requires executing a query to check row count.
        """
        try:
            rows, _ = self.shape
            return rows == 0
        except Exception:
            # If we can't determine, return False as a safe default
            return False

    @property
    def loc(self) -> "_LocIndexer":
        """Access a group of rows and columns by label(s) or boolean array (pandas-style).

        Returns:
            LocIndexer for label-based indexing

        Example:
            >>> df.loc[df['age'] > 25]  # Filter rows
            >>> df.loc[:, ['col1', 'col2']]  # Select columns
        """
        return _LocIndexer(self)

    @property
    def iloc(self) -> "_ILocIndexer":
        """Access a group of rows and columns by integer position (pandas-style).

        Returns:
            ILocIndexer for integer-based indexing

        Note:
            Full iloc functionality is limited by lazy evaluation.
            Only row filtering via boolean arrays is supported.
        """
        return _ILocIndexer(self)


@dataclass(frozen=True)
class _LocIndexer:
    """Indexer for pandas-style loc accessor."""

    _df: PandasDataFrame

    def __getitem__(self, key: Any) -> PandasDataFrame:
        """Access rows and columns using loc.

        Supports:
        - df.loc[df['age'] > 25] - Row filtering
        - df.loc[:, ['col1', 'col2']] - Column selection
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

            # Apply column selection if not ':' or Ellipsis
            if isinstance(col_key, (list, tuple)):
                result_df = result_df.select(*col_key)
            elif isinstance(col_key, str):
                result_df = result_df.select(col_key)
            elif col_key is not Ellipsis:
                # Check if it's slice(None) without triggering comparison
                if not (
                    isinstance(col_key, slice)
                    and col_key.start is None
                    and col_key.stop is None
                    and col_key.step is None
                ):
                    raise TypeError(f"Invalid column key type: {type(col_key)}")

            return self._df._with_dataframe(result_df)
        else:
            # Single-dimensional indexing
            if isinstance(key, Column):
                # Boolean condition - filter rows
                return self._df._with_dataframe(self._df._df.where(key))
            else:
                raise NotImplementedError("loc only supports boolean conditions for row filtering")


@dataclass(frozen=True)
class _ILocIndexer:
    """Indexer for pandas-style iloc accessor."""

    _df: PandasDataFrame

    def __getitem__(self, key: Any) -> PandasDataFrame:
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
