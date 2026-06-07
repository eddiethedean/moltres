"""Pandas-style interface for Moltres DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Type,
    cast,
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    overload,
)

from ...expressions.column import Column, col
from ...logical.plan import LogicalPlan
from ...utils.typing import FillValue
from ..core.dataframe import DataFrame
from ..core.interface_common import InterfaceCommonMixin

# Import PandasColumn wrapper for string accessor support
PandasColumn: Optional[Type[Any]] = None
try:
    from ..columns.pandas_column import PandasColumn as _PandasColumn

    PandasColumn = _PandasColumn
except ImportError:
    pass

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.sql import Select
    from ...table.table import Database
    from ..groupby.pandas_groupby import PandasGroupBy


@dataclass(frozen=True)
class PandasDataFrame(InterfaceCommonMixin):
    """Pandas-style interface wrapper around Moltres :class:`DataFrame`.

    Provides familiar pandas API methods while maintaining lazy evaluation
    and SQL pushdown execution. All operations remain lazy until collect() is called.

    Example:
        >>> df = db.table('users').pandas()
        >>> # Pandas-style column access
        >>> df[['id', 'name']].query('age > 25')
        >>> # Pandas-style groupby
        >>> df.groupby('country').agg({'amount': 'sum'})
        >>> # Returns actual pandas :class:`DataFrame`
        >>> result = df.collect()  # pd.:class:`DataFrame`
    """

    _df: DataFrame
    _shape_cache: Optional[Tuple[int, int]] = field(default=None, repr=False, compare=False)
    _dtypes_cache: Optional[Dict[str, str]] = field(default=None, repr=False, compare=False)

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
        """Create a :class:`PandasDataFrame` from a regular :class:`DataFrame`.

        Args:
            df: The :class:`DataFrame` to wrap

        Returns:
            :class:`PandasDataFrame` wrapping the provided :class:`DataFrame`
        """
        return cls(_df=df)

    def _with_dataframe(self, df: DataFrame) -> "PandasDataFrame":
        """Create a new :class:`PandasDataFrame` with a different underlying :class:`DataFrame`.

        Args:
            df: The new underlying :class:`DataFrame`

        Returns:
            New :class:`PandasDataFrame` instance
        """
        # Clear caches when creating new DataFrame instance
        return PandasDataFrame(_df=df, _shape_cache=None, _dtypes_cache=None)

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
        from ...utils.validation import validate_columns_exist

        try:
            available_columns = set(self.columns)
            # Only validate if we successfully got column names
            if available_columns:
                validate_columns_exist(column_names, available_columns, operation)
        except RuntimeError:
            # If we can't determine columns (e.g., RawSQL, complex plans),
            # skip validation - the error will be caught at execution time
            # This is a RuntimeError from _extract_column_names when it can't determine columns
            pass
        except Exception:
            # For other exceptions, also skip validation to be safe
            pass

    def __getitem__(
        self, key: Union[str, Sequence[str], Column]
    ) -> Union["PandasDataFrame", Column, Any]:
        """Pandas-style column access.

        Supports:
        - df['col'] - Returns :class:`Column` expression for filtering/expressions
        - df[['col1', 'col2']] - Returns new :class:`PandasDataFrame` with selected columns
        - df[df['age'] > 25] - Boolean indexing (filtering via :class:`Column` condition)

        Args:
            key: :class:`Column` name(s) or boolean :class:`Column` condition

        Returns:
            - For single column string: :class:`Column` expression
            - For list of columns: :class:`PandasDataFrame` with selected columns
            - For boolean :class:`Column` condition: :class:`PandasDataFrame` with filtered rows

        Example:
            >>> df['age']  # Returns :class:`Column` expression
            >>> df[['id', 'name']]  # Returns :class:`PandasDataFrame`
            >>> df[df['age'] > 25]  # Returns filtered :class:`PandasDataFrame`
        """
        # Single column string: df['col'] - return Column-like object for expressions
        if isinstance(key, str):
            # Validate column exists
            self._validate_columns_exist([key], "column access")
            column_expr = col(key)
            # Wrap in PandasColumn to enable .str accessor
            if PandasColumn is not None:
                return PandasColumn(column_expr)
            return column_expr

        # List of columns: df[['col1', 'col2']] - select columns
        if isinstance(key, (list, tuple)):
            if len(key) == 0:
                return self._with_dataframe(self._df.select())
            # Validate column names if they're strings
            str_columns = [c for c in key if isinstance(c, str)]
            if str_columns:
                self._validate_columns_exist(str_columns, "column selection")
            # Convert all to strings/Columns and select
            columns = [col(c) if isinstance(c, str) else c for c in key]
            return self._with_dataframe(self._df.select(*columns))

        # Column expression or PandasColumn - if it's a boolean condition, use as filter
        if isinstance(key, Column):
            # This is likely a boolean condition like df['age'] > 25
            # We should filter using it
            return self._with_dataframe(self._df.where(key))

        # Handle PandasColumn wrapper (which wraps a Column)
        # Note: Comparisons on PandasColumn return Column, so this may not be needed,
        # but it's here for completeness
        if PandasColumn is not None and hasattr(key, "_column"):
            # This might be a PandasColumn - extract underlying Column
            return self._with_dataframe(self._df.where(key._column))

        raise TypeError(
            f"Invalid key type for __getitem__: {type(key)}. Expected str, list, tuple, or Column."
        )

    def query(self, expr: str) -> "PandasDataFrame":
        """Filter :class:`DataFrame` using a pandas-style query string.

        Args:
            expr: Query string with pandas-style syntax (e.g., "age > 25 and status == 'active'")
                  Supports both '=' and '==' for equality comparisons.
                  Supports 'and'/'or' keywords in addition to '&'/'|' operators.

        Returns:
            Filtered :class:`PandasDataFrame`

        Raises:
            ValueError: If the query string cannot be parsed
            ValidationError: If referenced columns do not exist

        Example:
            >>> df.query('age > 25')
            >>> df.query("age > 25 and status == 'active'")
            >>> df.query("name in ['Alice', 'Bob']")
            >>> df.query("age = 30")  # Both = and == work
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_query_operation

        result_df = build_pandas_query_operation(self, expr)
        return self._with_dataframe(cast(DataFrame, result_df))

    def groupby(self, by: Union[str, Sequence[str]], *args: Any, **kwargs: Any) -> "PandasGroupBy":
        """Group rows by one or more columns (pandas-style).

        Args:
            by: :class:`Column` name(s) to group by
            *args: Additional positional arguments (for pandas compatibility)
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            PandasGroupBy object for aggregation

        Example:
            >>> df.groupby('country')
            >>> df.groupby(['country', 'region'])
        """
        from ..groupby.pandas_groupby import PandasGroupBy

        from ..operations.pandas_operations import normalize_groupby_by

        columns = normalize_groupby_by(by)

        # Validate columns exist
        self._validate_columns_exist(list(columns), "groupby")

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
            right: Right :class:`DataFrame` to merge with
            on: :class:`Column` name(s) to join on (must exist in both DataFrames)
            left_on: :class:`Column` name(s) in left :class:`DataFrame`
            right_on: :class:`Column` name(s) in right :class:`DataFrame`
            how: Type of join ('inner', 'left', 'right', 'outer')
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            Merged :class:`PandasDataFrame`

        Example:
            >>> df1.merge(df2, on='id')
            >>> df1.merge(df2, left_on='customer_id', right_on='id')
            >>> df1.merge(df2, on='id', how='left')
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_merge_operation

        result_df = build_pandas_merge_operation(
            self, right, on=on, left_on=left_on, right_on=right_on, how=how
        )
        return self._with_dataframe(cast(DataFrame, result_df))

    def crossJoin(self, other: "PandasDataFrame") -> "PandasDataFrame":
        """Perform a cross join (Cartesian product) with another :class:`DataFrame`.

        Args:
            other: Another :class:`PandasDataFrame` to cross join with

        Returns:
            New :class:`PandasDataFrame` containing the Cartesian product of rows

        Example:
            >>> df1 = db.table("table1").pandas()
            >>> df2 = db.table("table2").pandas()
            >>> df_cross = df1.crossJoin(df2)
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_cross_join_operation

        result_df = build_pandas_cross_join_operation(self, other)
        return self._with_dataframe(cast(DataFrame, result_df))

    cross_join = crossJoin  # Alias for consistency

    def sort_values(
        self,
        by: Union[str, Sequence[str]],
        ascending: Union[bool, Sequence[bool]] = True,
        **kwargs: Any,
    ) -> "PandasDataFrame":
        """Sort :class:`DataFrame` by column(s) (pandas-style).

        Args:
            by: :class:`Column` name(s) to sort by
            ascending: Sort order (True for ascending, False for descending)
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            Sorted :class:`PandasDataFrame`

        Example:
            >>> df.sort_values('age')
            >>> df.sort_values(['age', 'name'], ascending=[False, True])
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_sort_values_operation

        result_df = build_pandas_sort_values_operation(self, by, ascending)
        return self._with_dataframe(cast(DataFrame, result_df))

    def rename(self, columns: Dict[str, str], **kwargs: Any) -> "PandasDataFrame":
        """Rename columns (pandas-style).

        Args:
            columns: Dictionary mapping old names to new names
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            :class:`PandasDataFrame` with renamed columns

        Example:
            >>> df.rename(columns={'old_name': 'new_name'})
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_rename_operation

        result_df = build_pandas_rename_operation(self, columns)
        return self._with_dataframe(cast(DataFrame, result_df))

    def drop(
        self, columns: Optional[Union[str, Sequence[str]]] = None, **kwargs: Any
    ) -> "PandasDataFrame":
        """Drop columns (pandas-style).

        Args:
            columns: :class:`Column` name(s) to drop
            **kwargs: Additional keyword arguments (for pandas compatibility)

        Returns:
            :class:`PandasDataFrame` with dropped columns

        Example:
            >>> df.drop(columns=['col1', 'col2'])
            >>> df.drop(columns='col1')
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_drop_operation

        if columns is None:
            return self

        result_df = build_pandas_drop_operation(self, columns)
        return self._with_dataframe(cast(DataFrame, result_df))

    def drop_duplicates(
        self, subset: Optional[Union[str, Sequence[str]]] = None, **kwargs: Any
    ) -> "PandasDataFrame":
        """Remove duplicate rows (pandas-style).

        Args:
            subset: :class:`Column` name(s) to consider for duplicates (None means all columns)
            **kwargs: Additional keyword arguments (for pandas compatibility)
                - keep: 'first' (default) or 'last' - which duplicate to keep

        Returns:
            :class:`PandasDataFrame` with duplicates removed

        Example:
            >>> df.drop_duplicates()
            >>> df.drop_duplicates(subset=['col1', 'col2'])
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_drop_duplicates_operation

        keep = kwargs.get("keep", "first")
        result_df = build_pandas_drop_duplicates_operation(self, subset, keep)
        return self._with_dataframe(cast(DataFrame, result_df))

    def select(self, *columns: Union[str, Column]) -> "PandasDataFrame":
        """Select columns from the :class:`DataFrame` (pandas-style wrapper).

        Args:
            *columns: :class:`Column` names or :class:`Column` expressions to select

        Returns:
            :class:`PandasDataFrame` with selected columns

        Example:
            >>> df.select('id', 'name')
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_select_operation

        result_df = build_pandas_select_operation(self, *columns)
        return self._with_dataframe(cast(DataFrame, result_df))

    def assign(self, **kwargs: Union[Column, Any]) -> "PandasDataFrame":
        """Assign new columns (pandas-style).

        Args:
            **kwargs: :class:`Column` name = value pairs where value can be a :class:`Column` expression or literal

        Returns:
            :class:`PandasDataFrame` with new columns

        Example:
            >>> df.assign(total=df['amount'] * 1.1)
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_assign_operation

        result_df = build_pandas_assign_operation(self, **kwargs)
        return self._with_dataframe(cast(DataFrame, result_df))

    @overload
    def collect(self, stream: Literal[False] = False) -> "pd.DataFrame": ...

    @overload
    def collect(self, stream: Literal[True]) -> Iterator["pd.DataFrame"]: ...

    def collect(self, stream: bool = False) -> Union["pd.DataFrame", Iterator["pd.DataFrame"]]:
        """Collect results as pandas :class:`DataFrame`.

        Args:
            stream: If True, return an iterator of pandas :class:`DataFrame` chunks.
                   If False (default), return a single pandas :class:`DataFrame`.

        Returns:
            If stream=False: pandas :class:`DataFrame`
            If stream=True: Iterator of pandas :class:`DataFrame` chunks

        Example:
            >>> pdf = df.collect()  # Returns pd.:class:`DataFrame`
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

    def to_sqlalchemy(self, dialect: Optional[str] = None) -> "Select":
        """Convert :class:`PandasDataFrame`'s logical plan to a SQLAlchemy Select statement.

        This method delegates to the underlying :class:`DataFrame`'s to_sqlalchemy() method,
        allowing you to use :class:`PandasDataFrame` with existing SQLAlchemy infrastructure.

        Args:
            dialect: Optional SQL dialect name. If not provided, uses the dialect
                    from the attached :class:`Database`, or defaults to "ansi"

        Returns:
            SQLAlchemy Select statement that can be executed with any SQLAlchemy connection

        Example:
            >>> from moltres import connect, col
            >>> from sqlalchemy import create_engine
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.table("users").pandas()
            >>> stmt = df.to_sqlalchemy()
            >>> # Execute with existing SQLAlchemy connection
            >>> engine = create_engine("sqlite:///:memory:")
            >>> with engine.connect() as conn:
            ...     result = conn.execute(stmt)
        """
        return self._df.to_sqlalchemy(dialect=dialect)

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
            Dictionary mapping column names to pandas dtype strings (e.g., 'int64', 'object', 'float64')

        Note:
            This uses schema inspection which may require a database query if not cached.
            Types are cached after first access.
        """
        # Return cached dtypes if available
        if self._dtypes_cache is not None:
            return self._dtypes_cache

        if self.database is None:
            # Cannot get types without database connection
            return {}

        try:
            from ...utils.inspector import sql_type_to_pandas_dtype

            # Try to extract schema from the logical plan
            schema = self._df._extract_schema_from_plan(self._df.plan)

            # Map ColumnInfo to pandas dtypes
            dtypes_dict: Dict[str, str] = {}
            for col_info in schema:
                pandas_dtype = sql_type_to_pandas_dtype(col_info.type_name)
                dtypes_dict[col_info.name] = pandas_dtype

            # Cache the result (Note: we can't modify frozen dataclass, but we can return the dict)
            # The cache will be set on the next DataFrame operation that creates a new instance
            return dtypes_dict
        except Exception:
            # If schema extraction fails, return empty dict
            return {}

    @property
    def shape(self) -> Tuple[int, int]:
        """Get :class:`DataFrame` shape (rows, columns) (pandas-style property).

        Returns:
            Tuple of (number of rows, number of columns)

        Note:
            Getting row count requires executing a COUNT query,
            which can be expensive for large datasets. The result is cached
            for the lifetime of this :class:`DataFrame` instance.

        Warning:
            This operation executes a SQL query. For large tables, consider
            using limit() or filtering first.
        """
        # Return cached shape if available
        if self._shape_cache is not None:
            return self._shape_cache

        num_cols = len(self.columns)

        # To get row count, we need to execute a COUNT query
        # This is expensive, so we'll only do it if requested
        if self.database is None:
            raise RuntimeError("Cannot get shape without an attached Database")

        # Create a count query
        from ...expressions.functions import count

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

        shape_result = (num_rows, num_cols)
        # Note: We can't update the cache in a frozen dataclass, but we return the result
        # The cache field will be set when a new instance is created
        return shape_result

    @property
    def empty(self) -> bool:
        """Check if :class:`DataFrame` is empty (pandas-style property).

        Returns:
            True if :class:`DataFrame` has no rows, False otherwise

        Note:
            This requires executing a query to check row count.
        """
        try:
            rows, _ = self.shape
            return rows == 0
        except Exception:
            # If we can't determine, return False as a safe default
            return False

    def head(self, n: int = 5) -> "PandasDataFrame":
        """Return the first n rows (pandas-style).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            :class:`PandasDataFrame` with first n rows

        Example:
            >>> df.head(10)  # First 10 rows
        """
        return self._with_dataframe(self._df.limit(n))

    def tail(self, n: int = 5) -> "PandasDataFrame":
        """Return the last n rows (pandas-style).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            :class:`PandasDataFrame` with last n rows

        Note:
            This is a simplified implementation. For proper tail() behavior with lazy
            evaluation, this method sorts all columns in descending order and takes
            the first n rows. For better performance, consider using limit() directly
            or collecting and using pandas tail().

        Example:
            >>> df.tail(10)  # Last 10 rows
        """
        # To get last n rows with lazy evaluation, we:
        # 1. Sort by all columns in descending order
        # 2. Limit to n rows
        # Note: This doesn't preserve original order, but provides last n rows

        cols = self.columns
        if not cols:
            return self

        # Sort by all columns in descending order, then limit
        from ...expressions.column import col

        # Create a composite sort key - sort by all columns descending
        sorted_df = self._df
        for col_name in cols:
            sorted_df = sorted_df.order_by(col(col_name).desc())

        limited_df = sorted_df.limit(n)
        return self._with_dataframe(limited_df)

    def describe(self) -> "pd.DataFrame":
        """Generate descriptive statistics (pandas-style).

        Returns:
            pandas :class:`DataFrame` with summary statistics

        Note:
            This executes the query and requires pandas to be installed.

        Example:
            >>> stats = df.describe()
        """
        import importlib.util

        if importlib.util.find_spec("pandas") is None:
            raise ImportError(
                "pandas is required to use describe(). Install with: pip install pandas"
            )

        # Collect the full DataFrame
        pdf = self.collect()

        # Use pandas describe
        return pdf.describe()

    def info(self) -> None:
        """Print a concise summary of the :class:`DataFrame` (pandas-style).

        Prints column names, types, non-null counts, and memory usage.

        Example:
            >>> df.info()
        """
        import importlib.util

        if importlib.util.find_spec("pandas") is None:
            raise ImportError("pandas is required to use info(). Install with: pip install pandas")

        # Collect the DataFrame
        pdf = self.collect()

        # Use pandas info
        pdf.info()

    def nunique(self, column: Optional[str] = None) -> Union[int, Dict[str, int]]:
        """Count distinct values in column(s) (pandas-style).

        Args:
            column: :class:`Column` name to count. If None, counts distinct values for all columns.

        Returns:
            If column is specified: integer count of distinct values.
            If column is None: dictionary mapping column names to distinct counts.

        Example:
            >>> df.nunique('country')  # Count distinct countries
            >>> df.nunique()  # Count distinct for all columns
        """
        from ...expressions.column import col
        from ...expressions.functions import count_distinct

        if column is not None:
            # Validate column exists
            self._validate_columns_exist([column], "nunique")
            # Count distinct values in the column
            count_df = self._df.select(count_distinct(col(column)).alias("count"))
            result = count_df.collect()
            if result and isinstance(result, list) and len(result) > 0:
                row = result[0]
                if isinstance(row, dict):
                    count_val = row.get("count", 0)
                    return int(count_val) if isinstance(count_val, (int, float)) else 0
            return 0
        else:
            # Count distinct for all columns
            from ...expressions.column import col

            counts = {}
            for col_name in self.columns:
                count_df = self._df.select(count_distinct(col(col_name)).alias("count"))
                result = count_df.collect()
                if result and isinstance(result, list) and len(result) > 0:
                    row = result[0]
                    if isinstance(row, dict):
                        count_val = row.get("count", 0)
                        counts[col_name] = (
                            int(count_val) if isinstance(count_val, (int, float)) else 0
                        )
                    else:
                        counts[col_name] = 0
                else:
                    counts[col_name] = 0
            return counts

    def value_counts(
        self, column: str, normalize: bool = False, ascending: bool = False
    ) -> "pd.DataFrame":
        """Count value frequencies (pandas-style).

        Args:
            column: :class:`Column` name to count values for
            normalize: If True, return proportions instead of counts
            ascending: If True, sort in ascending order

        Returns:
            pandas :class:`DataFrame` with value counts

        Note:
            This executes the query and requires pandas to be installed.

        Example:
            >>> df.value_counts('country')
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required to use value_counts(). Install with: pip install pandas"
            )

        # Validate column exists
        self._validate_columns_exist([column], "value_counts")

        # Group by column and count
        from ...expressions.functions import count

        grouped = self._df.group_by(column)
        count_df = grouped.agg(count("*").alias("count"))

        # Sort by count column using underlying DataFrame
        from ...expressions.column import col

        if ascending:
            sorted_df = count_df.order_by(col("count"))
        else:
            sorted_df = count_df.order_by(col("count").desc())

        # Collect results and convert to pandas DataFrame
        results = sorted_df.collect()
        pdf = pd.DataFrame(results)

        # Normalize if requested
        if normalize and len(pdf) > 0:
            total = pdf["count"].sum()
            if total > 0:
                pdf["proportion"] = pdf["count"] / total
                pdf = pdf.drop(columns=["count"])
                pdf = pdf.rename(columns={"proportion": "count"})

        return pdf

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

    # ========================================================================
    # Additional Pandas Features
    # ========================================================================

    def explode(
        self,
        column: Union[str, Sequence[str]],
        ignore_index: bool = False,
    ) -> "PandasDataFrame":
        """Explode array/JSON columns into multiple rows (pandas-style).

        Args:
            column: :class:`Column` name(s) to explode
            ignore_index: If True, reset index (not used, for API compatibility)

        Returns:
            :class:`PandasDataFrame` with exploded rows

        Example:
            >>> df.explode('tags')
            >>> df.explode(['tags', 'categories'])
        """
        if isinstance(column, str):
            columns = [column]
        else:
            columns = list(column)
        self._validate_columns_exist(columns, "explode")

        result_df = self._df
        for col_name in columns:
            result_df = result_df.explode(col(col_name), alias=col_name)
        return self._with_dataframe(result_df)

    def pivot(
        self,
        index: Optional[Union[str, Sequence[str]]] = None,
        columns: Optional[str] = None,
        values: Optional[Union[str, Sequence[str]]] = None,
        aggfunc: Union[str, Dict[str, str]] = "sum",
    ) -> "PandasDataFrame":
        """Pivot :class:`DataFrame` (pandas-style).

        Args:
            index: :class:`Column`(s) to use as index (rows)
            columns: :class:`Column` to use as columns (pivot column)
            values: :class:`Column`(s) to aggregate
            aggfunc: Aggregation function(s) - string or dict mapping column to function

        Returns:
            Pivoted :class:`PandasDataFrame`

        Example:
            >>> df.pivot(index='category', columns='status', values='amount', aggfunc='sum')
        """
        if columns is None:
            raise ValueError("pivot() requires 'columns' parameter")
        if values is None:
            raise ValueError("pivot() requires 'values' parameter")

        if isinstance(values, (list, tuple)) and len(values) > 0:
            value_col: str = str(values[0])
        else:
            value_col = str(values)
        agg_func = aggfunc if isinstance(aggfunc, str) else list(aggfunc.values())[0]

        result_df = self._df.pivot(
            pivot_column=columns,
            value_column=value_col,
            agg_func=agg_func,
            pivot_values=None,
        )
        return self._with_dataframe(result_df)

    def pivot_table(
        self,
        values: Optional[Union[str, Sequence[str]]] = None,
        index: Optional[Union[str, Sequence[str]]] = None,
        columns: Optional[str] = None,
        aggfunc: Union[str, Dict[str, str]] = "mean",
        fill_value: Optional[FillValue] = None,
        margins: bool = False,
    ) -> "PandasDataFrame":
        """Create a pivot table (pandas-style).

        Args:
            values: :class:`Column`(s) to aggregate
            index: :class:`Column`(s) to use as index (rows)
            columns: :class:`Column` to use as columns (pivot column)
            aggfunc: Aggregation function(s)
            fill_value: Value to fill missing values (not used, for API compatibility)
            margins: Add row/column margins (not supported, for API compatibility)

        Returns:
            Pivot table :class:`PandasDataFrame`

        Example:
            >>> df.pivot_table(values='amount', index='category', columns='status', aggfunc='mean')
        """
        # pivot_table is similar to pivot but with different defaults
        return self.pivot(
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
        )

    def melt(
        self,
        id_vars: Optional[Union[str, Sequence[str]]] = None,
        value_vars: Optional[Union[str, Sequence[str]]] = None,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> "PandasDataFrame":
        """Melt :class:`DataFrame` from wide to long format (pandas-style).

        Args:
            id_vars: :class:`Column`(s) to use as identifier variables
            value_vars: :class:`Column`(s) to unpivot (if None, unpivot all except id_vars)
            var_name: Name for the variable column
            value_name: Name for the value column

        Returns:
            Melted :class:`PandasDataFrame`

        Example:
            >>> df.melt(id_vars=['id'], value_vars=['col1', 'col2'])
        """
        # Melt is not yet implemented in DataFrame
        raise NotImplementedError(
            "melt() is not yet implemented. "
            "This would require UNPIVOT SQL support which varies by database."
        )

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        weights: Optional[Union[str, Sequence[float]]] = None,
        random_state: Optional[int] = None,
    ) -> "PandasDataFrame":
        """Sample rows from :class:`DataFrame` (pandas-style).

        Args:
            n: Number of rows to sample
            frac: Fraction of rows to sample (0.0 to 1.0)
            replace: Sample with replacement (not supported, for API compatibility)
            weights: Sampling weights (not supported, for API compatibility)
            random_state: Random seed (alias for seed)

        Returns:
            Sampled :class:`PandasDataFrame`

        Example:
            >>> df.sample(n=10, random_state=42)
            >>> df.sample(frac=0.1, random_state=42)
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_sample_operation

        if replace:
            raise NotImplementedError(
                "Sampling with replacement is not yet supported. "
                "Alternative: Use sample() with replace=False, or materialize the DataFrame "
                "and use pandas' sample() method with replace=True. "
                "See https://github.com/eddiethedean/moltres/issues for feature requests."
            )

        result_df = build_pandas_sample_operation(self, n=n, frac=frac, random_state=random_state)
        return self._with_dataframe(cast(DataFrame, result_df))

    def limit(self, n: int) -> "PandasDataFrame":
        """Limit number of rows (pandas-style alias).

        Args:
            n: Number of rows to return

        Returns:
            :class:`PandasDataFrame` with limited rows

        Example:
            >>> df.limit(10)
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_limit_operation

        result_df = build_pandas_limit_operation(self, n)
        return self._with_dataframe(cast(DataFrame, result_df))

    def append(
        self,
        other: "PandasDataFrame",
        ignore_index: bool = False,
        verify_integrity: bool = False,
    ) -> "PandasDataFrame":
        """Append rows from another :class:`DataFrame` (pandas-style).

        Note: pandas deprecated append() in favor of concat(). This is provided for compatibility.

        Args:
            other: Another :class:`PandasDataFrame` to append
            ignore_index: If True, reset index (not used, for API compatibility)
            verify_integrity: Check for duplicate indices (not used, for API compatibility)

        Returns:
            Appended :class:`PandasDataFrame`

        Example:
            >>> df1.append(df2)
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_append_operation

        result_df = build_pandas_append_operation(self, other)
        return self._with_dataframe(cast(DataFrame, result_df))

    def concat(
        self,
        *others: "PandasDataFrame",
        axis: Union[int, str] = 0,
        join: str = "outer",
        ignore_index: bool = False,
    ) -> "PandasDataFrame":
        """Concatenate DataFrames (pandas-style).

        Args:
            *others: Other PandasDataFrames to concatenate
            axis: Concatenation axis (0 for vertical, 1 for horizontal)
            join: How to handle indexes on other axis (not used, for API compatibility)
            ignore_index: If True, reset index (not used, for API compatibility)

        Returns:
            Concatenated :class:`PandasDataFrame`

        Example:
            >>> pd.concat([df1, df2])  # pandas style
            >>> df1.concat(df2)  # method style
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_concat_operation

        if not others:
            return self

        result_df = build_pandas_concat_operation(self, others, axis=axis)
        return self._with_dataframe(cast(DataFrame, result_df))

    def isin(self, values: Union[Dict[str, Sequence[Any]], Sequence[Any]]) -> "PandasDataFrame":
        """Filter rows where values are in a sequence (pandas-style).

        Args:
            values: Dictionary mapping column names to sequences, or sequence for all columns

        Returns:
            Filtered :class:`PandasDataFrame`

        Example:
            >>> df.isin({'age': [25, 30, 35]})
            >>> df.isin([1, 2, 3])  # Check all columns
        """
        if isinstance(values, dict):
            # Multiple columns
            condition = None
            for col_name, val_list in values.items():
                self._validate_columns_exist([col_name], "isin")
                col_condition = col(col_name).isin(val_list)
                if condition is None:
                    condition = col_condition
                else:
                    condition = condition & col_condition
            if condition is None:
                return self
            return self._with_dataframe(self._df.where(condition))
        else:
            # Single sequence - check all columns
            # This is tricky in SQL, so we'll check the first column
            if not self.columns:
                return self
            first_col = self.columns[0]
            return self._with_dataframe(self._df.where(col(first_col).isin(values)))

    def between(
        self,
        left: Union[Any, Dict[str, Any]],
        right: Union[Any, Dict[str, Any]],
        inclusive: Union[str, bool] = "both",
    ) -> "PandasDataFrame":
        """Filter rows where values are between left and right (pandas-style).

        Args:
            left: Left boundary (scalar or dict mapping column to value)
            right: Right boundary (scalar or dict mapping column to value)
            inclusive: Include boundaries - "both", "neither", "left", "right", or bool

        Returns:
            Filtered :class:`PandasDataFrame`

        Example:
            >>> df.between(left=20, right=30)  # All numeric columns
            >>> df.between(left={'age': 20}, right={'age': 30})  # Specific column
        """
        if isinstance(left, dict) and isinstance(right, dict):
            # Multiple columns
            condition = None
            for col_name in left.keys():
                if col_name not in right:
                    continue
                self._validate_columns_exist([col_name], "between")
                col_expr = col(col_name)
                left_val: Any = left[col_name]
                right_val: Any = right[col_name]

                if inclusive in ("both", True):
                    col_condition = (col_expr >= left_val) & (col_expr <= right_val)
                elif inclusive == "left":
                    col_condition = (col_expr >= left_val) & (col_expr < right_val)
                elif inclusive == "right":
                    col_condition = (col_expr > left_val) & (col_expr <= right_val)
                else:  # "neither" or False
                    col_condition = (col_expr > left_val) & (col_expr < right_val)

                if condition is None:
                    condition = col_condition
                else:
                    condition = condition & col_condition
            if condition is None:
                return self
            return self._with_dataframe(self._df.where(condition))
        else:
            # Single value - apply to all numeric columns
            numeric_cols = [c for c in self.columns if self._is_numeric_column(c)]
            if not numeric_cols:
                return self

            condition = None
            left_scalar: Any = left
            right_scalar: Any = right
            for col_name in numeric_cols:
                col_expr = col(col_name)
                if inclusive in ("both", True):
                    col_condition = (col_expr >= left_scalar) & (col_expr <= right_scalar)
                elif inclusive == "left":
                    col_condition = (col_expr >= left_scalar) & (col_expr < right_scalar)
                elif inclusive == "right":
                    col_condition = (col_expr > left_scalar) & (col_expr <= right_scalar)
                else:  # "neither" or False
                    col_condition = (col_expr > left_scalar) & (col_expr < right_scalar)

                if condition is None:
                    condition = col_condition
                else:
                    condition = condition | col_condition
            if condition is None:
                return self
            return self._with_dataframe(self._df.where(condition))

    def _is_numeric_column(self, col_name: str) -> bool:
        """Check if a column is numeric based on dtypes."""
        dtypes = self.dtypes
        dtype = dtypes.get(col_name, "")
        numeric_dtypes = ["int64", "int32", "float64", "float32"]
        return dtype in numeric_dtypes

    def select_expr(self, *exprs: str) -> "PandasDataFrame":
        """Select columns using SQL expressions (pandas-style).

        Args:
            *exprs: SQL expression strings (e.g., "amount * 1.1 as with_tax")

        Returns:
            :class:`PandasDataFrame` with selected expressions

        Example:
            >>> df.select_expr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_select_expr_operation

        result_df = build_pandas_select_expr_operation(self, *exprs)
        return self._with_dataframe(cast(DataFrame, result_df))

    def cte(self, name: str) -> "PandasDataFrame":
        """Create a Common Table Expression (CTE) from this :class:`DataFrame`.

        Args:
            name: Name for the CTE

        Returns:
            :class:`PandasDataFrame` representing the CTE

        Example:
            >>> cte_df = df.query('age > 25').cte('adults')
            >>> result = cte_df.collect()
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_cte_operation

        result_df = build_pandas_cte_operation(self, name)
        return self._with_dataframe(cast(DataFrame, result_df))

    def summary(self, *statistics: str) -> "PandasDataFrame":
        """Compute summary statistics for numeric columns (pandas-style).

        Args:
            *statistics: Statistics to compute (e.g., "count", "mean", "stddev", "min", "max").
                        If not provided, computes common statistics.

        Returns:
            :class:`PandasDataFrame` with summary statistics

        Example:
            >>> df.summary()
            >>> df.summary("count", "mean", "max")
        """
        from ..helpers.pandas_dataframe_helpers import build_pandas_summary_operation

        result_df = build_pandas_summary_operation(self, *statistics)
        return self._with_dataframe(cast(DataFrame, result_df))


@dataclass(frozen=True)
class _LocIndexer:
    """Indexer for pandas-style loc accessor."""

    _df: PandasDataFrame

    def __getitem__(self, key: Any) -> PandasDataFrame:
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
                raise NotImplementedError(
                    "loc only supports boolean conditions for row filtering in lazy evaluation. "
                    "Alternative: Use query() or where() for filtering, or materialize the DataFrame "
                    "and use pandas' loc accessor for label-based indexing. "
                    "Example: df.query('column == value') instead of df.loc[df['column'] == 'value']"
                )


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
                "iloc positional indexing requires materialization in lazy evaluation. "
                "Alternatives: "
                "1. Use limit(n) to get the first n rows, "
                "2. Use boolean filtering with query() or where(), "
                "3. Materialize the DataFrame with collect() and use pandas' iloc accessor. "
                "Example: df.limit(10) or df.query('condition').limit(5)"
            )
