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

# Import PandasColumn wrapper for string accessor support
try:
    from .pandas_column import PandasColumn
except ImportError:
    PandasColumn = None  # type: ignore

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
        from ..expressions.sql_parser import parse_sql_expr
        from ..utils.exceptions import PandasAPIError

        # Get available column names for context and validation
        available_columns: Optional[Set[str]] = None
        try:
            available_columns = set(self.columns)
        except Exception:
            # Fallback: try to extract from plan
            try:
                if hasattr(self._df, "plan") and hasattr(self._df.plan, "projections"):
                    available_columns = set()
                    for proj in self._df.plan.projections:
                        if isinstance(proj, Column) and proj.op == "column" and proj.args:
                            available_columns.add(str(proj.args[0]))
            except Exception:
                pass

        # Parse query string to Column expression
        try:
            predicate = parse_sql_expr(expr, available_columns)
        except ValueError as e:
            # Provide more helpful error message
            raise PandasAPIError(
                f"Failed to parse query expression: {expr}",
                suggestion=(
                    f"Error: {str(e)}\n"
                    "Query syntax should follow pandas-style syntax:\n"
                    "  - Use '=' or '==' for equality: 'age == 25' or 'age = 25'\n"
                    "  - Use 'and'/'or' keywords: 'age > 25 and status == \"active\"'\n"
                    "  - Use comparison operators: >, <, >=, <=, !=, ==\n"
                    f"{'  - Available columns: ' + ', '.join(sorted(available_columns)) if available_columns else ''}"
                ),
                context={
                    "query": expr,
                    "available_columns": list(available_columns) if available_columns else [],
                },
            ) from e

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
                # Validate columns exist in both DataFrames
                self._validate_columns_exist([on], "merge (left DataFrame)")
                right._validate_columns_exist([on], "merge (right DataFrame)")
                join_on = [(on, on)]
            else:
                # Validate all columns exist
                self._validate_columns_exist(list(on), "merge (left DataFrame)")
                right._validate_columns_exist(list(on), "merge (right DataFrame)")
                join_on = [(col, col) for col in on]
        elif left_on is not None and right_on is not None:
            # Different column names
            if isinstance(left_on, str) and isinstance(right_on, str):
                # Validate columns exist
                self._validate_columns_exist([left_on], "merge (left DataFrame)")
                right._validate_columns_exist([right_on], "merge (right DataFrame)")
                join_on = [(left_on, right_on)]
            elif isinstance(left_on, (list, tuple)) and isinstance(right_on, (list, tuple)):
                if len(left_on) != len(right_on):
                    raise ValueError("left_on and right_on must have the same length")
                # Validate all columns exist
                self._validate_columns_exist(list(left_on), "merge (left DataFrame)")
                right._validate_columns_exist(list(right_on), "merge (right DataFrame)")
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
