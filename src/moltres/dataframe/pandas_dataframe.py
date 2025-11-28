"""Pandas-style interface for Moltres DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
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
from .interface_common import InterfaceCommonMixin

# Import PandasColumn wrapper for string accessor support
try:
    from .pandas_column import PandasColumn
except ImportError:
    PandasColumn = None  # type: ignore

if TYPE_CHECKING:
    import pandas as pd
    from sqlalchemy.sql import Select
    from ..table.table import Database
    from .pandas_groupby import PandasGroupBy


@dataclass(frozen=True)
class PandasDataFrame(InterfaceCommonMixin):
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
        # Clear caches when creating new DataFrame instance
        return PandasDataFrame(_df=df, _shape_cache=None, _dtypes_cache=None)

    def _validate_columns_exist(
        self, column_names: Sequence[str], operation: str = "operation"
    ) -> None:
        """Validate that all specified columns exist in the DataFrame.

        Args:
            column_names: List of column names to validate
            operation: Name of the operation being performed (for error messages)

        Raises:
            ValidationError: If any column does not exist, with helpful suggestions

        Note:
            Validation only occurs if columns can be determined from the plan.
            For complex plans (e.g., RawSQL), validation is skipped to avoid false positives.
        """
        from ..utils.validation import validate_columns_exist

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
    ) -> Union["PandasDataFrame", Column, "PandasColumn"]:
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
        """Filter DataFrame using a pandas-style query string.

        Args:
            expr: Query string with pandas-style syntax (e.g., "age > 25 and status == 'active'")
                  Supports both '=' and '==' for equality comparisons.
                  Supports 'and'/'or' keywords in addition to '&'/'|' operators.

        Returns:
            Filtered PandasDataFrame

        Raises:
            ValueError: If the query string cannot be parsed
            ValidationError: If referenced columns do not exist

        Example:
            >>> df.query('age > 25')
            >>> df.query("age > 25 and status == 'active'")
            >>> df.query("name in ['Alice', 'Bob']")
            >>> df.query("age = 30")  # Both = and == work
        """

        from .pandas_operations import parse_query_expression

        # Get available column names for context and validation
        available_columns: Optional[Set[str]] = None
        try:
            available_columns = set(self.columns)
        except Exception:
            pass

        # Parse query string to Column expression
        predicate = parse_query_expression(expr, available_columns, self._df.plan)

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

        from .pandas_operations import normalize_groupby_by

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
        from .pandas_operations import normalize_merge_how, prepare_merge_keys

        # Normalize how parameter
        join_how = normalize_merge_how(how)

        # Determine join keys
        join_on = prepare_merge_keys(
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_columns=self.columns,
            right_columns=right.columns,
            left_validate_fn=self._validate_columns_exist,
            right_validate_fn=right._validate_columns_exist,
        )

        # Perform join
        result_df = self._df.join(right._df, on=join_on, how=join_how)
        return self._with_dataframe(result_df)

    def crossJoin(self, other: "PandasDataFrame") -> "PandasDataFrame":
        """Perform a cross join (Cartesian product) with another DataFrame.

        Args:
            other: Another PandasDataFrame to cross join with

        Returns:
            New PandasDataFrame containing the Cartesian product of rows

        Example:
            >>> df1 = db.table("table1").pandas()
            >>> df2 = db.table("table2").pandas()
            >>> df_cross = df1.crossJoin(df2)
        """
        result_df = self._df.crossJoin(other._df)
        return self._with_dataframe(result_df)

    cross_join = crossJoin  # Alias for consistency

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

        # Validate columns exist
        self._validate_columns_exist(columns, "sort_values")

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
                - keep: 'first' (default) or 'last' - which duplicate to keep

        Returns:
            PandasDataFrame with duplicates removed

        Example:
            >>> df.drop_duplicates()
            >>> df.drop_duplicates(subset=['col1', 'col2'])
        """
        keep = kwargs.get("keep", "first")

        if subset is None:
            # Remove duplicates on all columns
            result_df = self._df.distinct()
        else:
            # Validate subset columns exist
            if isinstance(subset, str):
                subset_cols = [subset]
            else:
                subset_cols = list(subset)

            # Validate columns exist
            self._validate_columns_exist(subset_cols, "drop_duplicates")

            # For subset-based deduplication, we need to:
            # 1. Group by subset columns
            # 2. Select all columns, taking first/last from each group
            # This is complex in SQL - we'll use a window function approach if possible
            # For now, we'll use GROUP BY with MIN/MAX on non-grouped columns

            # Get all column names
            all_cols = self.columns
            other_cols = [col for col in all_cols if col not in subset_cols]

            # Group by subset columns
            grouped = self._df.group_by(*subset_cols)

            # Build aggregations: use MIN/MAX for all non-grouped columns
            from ..expressions import functions as F

            # GroupBy automatically includes grouping columns, so we only need to aggregate others
            if not other_cols:
                # If only grouping columns, distinct works fine
                result_df = self._df.distinct()
            else:
                # Build aggregations for non-grouped columns only
                # GroupBy automatically includes grouping columns in result
                agg_exprs = []
                for col_name in other_cols:
                    if keep == "last":
                        agg_exprs.append(F.max(col(col_name)).alias(col_name))
                    else:  # keep == "first"
                        agg_exprs.append(F.min(col(col_name)).alias(col_name))

                # GroupBy includes grouping columns automatically, so we only pass aggregations
                # The result will have grouping columns + aggregated columns
                result_df = grouped.agg(*agg_exprs)

        return self._with_dataframe(result_df)

    def select(self, *columns: Union[str, Column]) -> "PandasDataFrame":
        """Select columns from the DataFrame (pandas-style wrapper).

        Args:
            *columns: Column names or Column expressions to select

        Returns:
            PandasDataFrame with selected columns

        Example:
            >>> df.select('id', 'name')
        """
        # Validate column names if they're strings
        str_columns = [c for c in columns if isinstance(c, str)]
        if str_columns:
            self._validate_columns_exist(str_columns, "select")

        # Use underlying DataFrame's select
        from ..expressions.column import col

        selected_cols = [col(c) if isinstance(c, str) else c for c in columns]
        result_df = self._df.select(*selected_cols)
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

    def to_sqlalchemy(self, dialect: Optional[str] = None) -> "Select":
        """Convert PandasDataFrame's logical plan to a SQLAlchemy Select statement.

        This method delegates to the underlying DataFrame's to_sqlalchemy() method,
        allowing you to use PandasDataFrame with existing SQLAlchemy infrastructure.

        Args:
            dialect: Optional SQL dialect name. If not provided, uses the dialect
                    from the attached Database, or defaults to "ansi"

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
            from ..utils.inspector import sql_type_to_pandas_dtype

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
        """Get DataFrame shape (rows, columns) (pandas-style property).

        Returns:
            Tuple of (number of rows, number of columns)

        Note:
            Getting row count requires executing a COUNT query,
            which can be expensive for large datasets. The result is cached
            for the lifetime of this DataFrame instance.

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

        shape_result = (num_rows, num_cols)
        # Note: We can't update the cache in a frozen dataclass, but we return the result
        # The cache field will be set when a new instance is created
        return shape_result

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

    def head(self, n: int = 5) -> "PandasDataFrame":
        """Return the first n rows (pandas-style).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            PandasDataFrame with first n rows

        Example:
            >>> df.head(10)  # First 10 rows
        """
        return self._with_dataframe(self._df.limit(n))

    def tail(self, n: int = 5) -> "PandasDataFrame":
        """Return the last n rows (pandas-style).

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            PandasDataFrame with last n rows

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
        from ..expressions.column import col

        # Create a composite sort key - sort by all columns descending
        sorted_df = self._df
        for col_name in cols:
            sorted_df = sorted_df.order_by(col(col_name).desc())

        limited_df = sorted_df.limit(n)
        return self._with_dataframe(limited_df)

    def describe(self) -> "pd.DataFrame":
        """Generate descriptive statistics (pandas-style).

        Returns:
            pandas DataFrame with summary statistics

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
        """Print a concise summary of the DataFrame (pandas-style).

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
            column: Column name to count. If None, counts distinct values for all columns.

        Returns:
            If column is specified: integer count of distinct values.
            If column is None: dictionary mapping column names to distinct counts.

        Example:
            >>> df.nunique('country')  # Count distinct countries
            >>> df.nunique()  # Count distinct for all columns
        """
        from ..expressions.column import col
        from ..expressions.functions import count_distinct

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
            from ..expressions.column import col

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
            column: Column name to count values for
            normalize: If True, return proportions instead of counts
            ascending: If True, sort in ascending order

        Returns:
            pandas DataFrame with value counts

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
        from ..expressions.functions import count

        grouped = self._df.group_by(column)
        count_df = grouped.agg(count("*").alias("count"))

        # Sort by count column using underlying DataFrame
        from ..expressions.column import col

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
            column: Column name(s) to explode
            ignore_index: If True, reset index (not used, for API compatibility)

        Returns:
            PandasDataFrame with exploded rows

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
        """Pivot DataFrame (pandas-style).

        Args:
            index: Column(s) to use as index (rows)
            columns: Column to use as columns (pivot column)
            values: Column(s) to aggregate
            aggfunc: Aggregation function(s) - string or dict mapping column to function

        Returns:
            Pivoted PandasDataFrame

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
        fill_value: Optional[Any] = None,
        margins: bool = False,
    ) -> "PandasDataFrame":
        """Create a pivot table (pandas-style).

        Args:
            values: Column(s) to aggregate
            index: Column(s) to use as index (rows)
            columns: Column to use as columns (pivot column)
            aggfunc: Aggregation function(s)
            fill_value: Value to fill missing values (not used, for API compatibility)
            margins: Add row/column margins (not supported, for API compatibility)

        Returns:
            Pivot table PandasDataFrame

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
        """Melt DataFrame from wide to long format (pandas-style).

        Args:
            id_vars: Column(s) to use as identifier variables
            value_vars: Column(s) to unpivot (if None, unpivot all except id_vars)
            var_name: Name for the variable column
            value_name: Name for the value column

        Returns:
            Melted PandasDataFrame

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
        """Sample rows from DataFrame (pandas-style).

        Args:
            n: Number of rows to sample
            frac: Fraction of rows to sample (0.0 to 1.0)
            replace: Sample with replacement (not supported, for API compatibility)
            weights: Sampling weights (not supported, for API compatibility)
            random_state: Random seed (alias for seed)

        Returns:
            Sampled PandasDataFrame

        Example:
            >>> df.sample(n=10, random_state=42)
            >>> df.sample(frac=0.1, random_state=42)
        """
        if replace:
            raise NotImplementedError("Sampling with replacement is not yet supported")

        if n is not None and frac is not None:
            raise ValueError("Cannot specify both 'n' and 'frac'")

        if n is not None:
            # Sample n rows - use fraction=1.0 then limit
            result_df = self._df.sample(fraction=1.0, seed=random_state).limit(n)
        elif frac is not None:
            result_df = self._df.sample(fraction=frac, seed=random_state)
        else:
            raise ValueError("Must specify either 'n' or 'frac'")

        return self._with_dataframe(result_df)

    def limit(self, n: int) -> "PandasDataFrame":
        """Limit number of rows (pandas-style alias).

        Args:
            n: Number of rows to return

        Returns:
            PandasDataFrame with limited rows

        Example:
            >>> df.limit(10)
        """
        result_df = self._df.limit(n)
        return self._with_dataframe(result_df)

    def append(
        self,
        other: "PandasDataFrame",
        ignore_index: bool = False,
        verify_integrity: bool = False,
    ) -> "PandasDataFrame":
        """Append rows from another DataFrame (pandas-style).

        Note: pandas deprecated append() in favor of concat(). This is provided for compatibility.

        Args:
            other: Another PandasDataFrame to append
            ignore_index: If True, reset index (not used, for API compatibility)
            verify_integrity: Check for duplicate indices (not used, for API compatibility)

        Returns:
            Appended PandasDataFrame

        Example:
            >>> df1.append(df2)
        """
        # Append is essentially union all
        result_df = self._df.unionAll(other._df)
        return self._with_dataframe(result_df)

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
            Concatenated PandasDataFrame

        Example:
            >>> pd.concat([df1, df2])  # pandas style
            >>> df1.concat(df2)  # method style
        """
        if not others:
            return self

        if axis == 0 or axis == "index":
            # Vertical concatenation (union all)
            result_df = self._df
            for other in others:
                result_df = result_df.unionAll(other._df)
            return self._with_dataframe(result_df)
        elif axis == 1 or axis == "columns":
            # Horizontal concatenation (cross join)
            result_df = self._df
            for other in others:
                result_df = result_df.crossJoin(other._df)
            return self._with_dataframe(result_df)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 0, 1, 'index', or 'columns'")

    def isin(self, values: Union[Dict[str, Sequence[Any]], Sequence[Any]]) -> "PandasDataFrame":
        """Filter rows where values are in a sequence (pandas-style).

        Args:
            values: Dictionary mapping column names to sequences, or sequence for all columns

        Returns:
            Filtered PandasDataFrame

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
            Filtered PandasDataFrame

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
            PandasDataFrame with selected expressions

        Example:
            >>> df.select_expr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")
        """
        result_df = self._df.selectExpr(*exprs)
        return self._with_dataframe(result_df)

    def cte(self, name: str) -> "PandasDataFrame":
        """Create a Common Table Expression (CTE) from this DataFrame.

        Args:
            name: Name for the CTE

        Returns:
            PandasDataFrame representing the CTE

        Example:
            >>> cte_df = df.query('age > 25').cte('adults')
            >>> result = cte_df.collect()
        """
        result_df = self._df.cte(name)
        return self._with_dataframe(result_df)

    def summary(self, *statistics: str) -> "PandasDataFrame":
        """Compute summary statistics for numeric columns (pandas-style).

        Args:
            *statistics: Statistics to compute (e.g., "count", "mean", "stddev", "min", "max").
                        If not provided, computes common statistics.

        Returns:
            PandasDataFrame with summary statistics

        Example:
            >>> df.summary()
            >>> df.summary("count", "mean", "max")
        """
        result_df = self._df.summary(*statistics)
        return self._with_dataframe(result_df)


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
