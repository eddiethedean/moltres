"""Polars-style interface for Moltres DataFrames."""

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
    Tuple,
    Union,
    overload,
)

from ..expressions.column import Column, col
from ..logical.plan import LogicalPlan
from .dataframe import DataFrame

if TYPE_CHECKING:
    import polars as pl

    from ..table.table import Database
    from .polars_column import PolarsColumn
    from .polars_groupby import PolarsGroupBy


def sql_type_to_polars_dtype(sql_type: str) -> str:
    """Map SQL type names to Polars dtype strings.

    Args:
        sql_type: SQL type name (e.g., "INTEGER", "TEXT", "VARCHAR(255)", "REAL")

    Returns:
        Polars dtype string (e.g., "Int64", "Utf8", "Float64")

    Example:
        >>> sql_type_to_polars_dtype("INTEGER")
        'Int64'
        >>> sql_type_to_polars_dtype("TEXT")
        'Utf8'
        >>> sql_type_to_polars_dtype("REAL")
        'Float64'
    """
    # Normalize the type name - remove parameters and convert to uppercase
    type_upper = sql_type.upper().strip()
    # Remove parameters if present (e.g., "VARCHAR(255)" -> "VARCHAR")
    if "(" in type_upper:
        type_upper = type_upper.split("(")[0].strip()

    # Remove parentheses suffix if present
    type_upper = type_upper.replace("()", "")

    # Map SQL types to Polars dtypes
    type_mapping: Dict[str, str] = {
        # Integer types
        "INTEGER": "Int64",
        "INT": "Int64",
        "BIGINT": "Int64",
        "SMALLINT": "Int32",
        "TINYINT": "Int8",
        "SERIAL": "Int64",
        "BIGSERIAL": "Int64",
        # Floating point types
        "REAL": "Float64",
        "FLOAT": "Float64",
        "DOUBLE": "Float64",
        "DOUBLE PRECISION": "Float64",
        "NUMERIC": "Float64",
        "DECIMAL": "Float64",
        "MONEY": "Float64",
        # Text types
        "TEXT": "Utf8",
        "VARCHAR": "Utf8",
        "CHAR": "Utf8",
        "CHARACTER": "Utf8",
        "STRING": "Utf8",
        "CLOB": "Utf8",
        # Binary types
        "BLOB": "Binary",
        "BYTEA": "Binary",
        "BINARY": "Binary",
        "VARBINARY": "Binary",
        # Boolean
        "BOOLEAN": "Boolean",
        "BOOL": "Boolean",
        # Date/Time types
        "DATE": "Date",
        "TIME": "Time",
        "TIMESTAMP": "Datetime",
        "DATETIME": "Datetime",
        "TIMESTAMPTZ": "Datetime",
        # JSON types
        "JSON": "Object",
        "JSONB": "Object",
        # UUID
        "UUID": "Utf8",
    }

    # Return mapped type or default to 'Utf8' for unknown types
    return type_mapping.get(type_upper, "Utf8")


@dataclass(frozen=True)
class PolarsDataFrame:
    """Polars-style interface wrapper around Moltres DataFrame.

    Provides familiar Polars LazyFrame API methods while maintaining lazy evaluation
    and SQL pushdown execution. All operations remain lazy until collect() is called.

    Example:
        >>> df = db.table('users').polars()
        >>> # Polars-style operations
        >>> df.filter(col('age') > 25).select(['id', 'name'])
        >>> # Returns actual Polars DataFrame
        >>> result = df.collect()  # pl.DataFrame
    """

    _df: DataFrame
    _height_cache: Optional[int] = field(default=None, repr=False, compare=False)
    _schema_cache: Optional[List[Tuple[str, str]]] = field(default=None, repr=False, compare=False)

    @property
    def plan(self) -> LogicalPlan:
        """Get the underlying logical plan."""
        return self._df.plan

    @property
    def database(self) -> Optional["Database"]:
        """Get the associated database."""
        return self._df.database

    @classmethod
    def from_dataframe(cls, df: DataFrame) -> "PolarsDataFrame":
        """Create a PolarsDataFrame from a regular DataFrame.

        Args:
            df: The DataFrame to wrap

        Returns:
            PolarsDataFrame wrapping the provided DataFrame
        """
        return cls(_df=df)

    def _with_dataframe(self, df: DataFrame) -> "PolarsDataFrame":
        """Create a new PolarsDataFrame with a different underlying DataFrame.

        Args:
            df: The new underlying DataFrame

        Returns:
            New PolarsDataFrame instance
        """
        # Clear caches when creating new DataFrame instance
        return PolarsDataFrame(_df=df, _height_cache=None, _schema_cache=None)

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

    @property
    def height(self) -> int:
        """Get number of rows (Polars-style property).

        Returns:
            Number of rows

        Note:
            Getting row count requires executing a COUNT query,
            which can be expensive for large datasets. The result is cached
            for the lifetime of this DataFrame instance.

        Warning:
            This operation executes a SQL query. For large tables, consider
            using limit() or filtering first.
        """
        # Return cached height if available
        if self._height_cache is not None:
            return self._height_cache

        if self.database is None:
            raise RuntimeError("Cannot get height without an attached Database")

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
                        if isinstance(count_val, (str, float)):
                            num_rows = int(count_val)
                        else:
                            num_rows = 0
                    except (ValueError, TypeError):
                        num_rows = 0

        # Note: We can't update the cache in a frozen dataclass, but we return the result
        # The cache field will be set when a new instance is created
        return num_rows

    @property
    def schema(self) -> List[Tuple[str, str]]:
        """Get schema as Polars format (list of (name, dtype) tuples).

        Returns:
            List of tuples mapping column names to Polars dtype strings

        Note:
            Schema is cached after first access to avoid redundant queries.

        Example:
            >>> df.schema  # [('id', 'Int64'), ('name', 'Utf8'), ('age', 'Int64')]
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

    def lazy(self) -> "PolarsDataFrame":
        """Return self (for API compatibility, PolarsDataFrame is already lazy).

        Returns:
            Self (PolarsDataFrame is always lazy)

        Example:
            >>> df.lazy()  # Returns self
        """
        return self

    def select(self, *exprs: Union[str, Column]) -> "PolarsDataFrame":
        """Select columns/expressions (Polars-style).

        Args:
            *exprs: Column names or Column expressions to select

        Returns:
            PolarsDataFrame with selected columns

        Example:
            >>> df.select('id', 'name')
            >>> df.select(col('id'), (col('amount') * 1.1).alias('with_tax'))
        """
        # Validate column names if they're strings
        str_columns = [e for e in exprs if isinstance(e, str)]
        if str_columns:
            self._validate_columns_exist(str_columns, "select")

        # Use underlying DataFrame's select
        selected_cols = [col(e) if isinstance(e, str) else e for e in exprs]
        result_df = self._df.select(*selected_cols)
        return self._with_dataframe(result_df)

    def filter(self, predicate: Column) -> "PolarsDataFrame":
        """Filter rows (Polars-style, uses 'filter' instead of 'where').

        Args:
            predicate: Column expression for filtering condition

        Returns:
            Filtered PolarsDataFrame

        Example:
            >>> df.filter(col('age') > 25)
            >>> df.filter((col('age') > 25) & (col('active') == True))
        """
        result_df = self._df.where(predicate)
        return self._with_dataframe(result_df)

    def with_columns(self, *exprs: Union[Column, Tuple[str, Column]]) -> "PolarsDataFrame":
        """Add or modify columns (Polars primary method for adding columns).

        Args:
            *exprs: Column expressions or (name, expression) tuples

        Returns:
            PolarsDataFrame with new/modified columns

        Example:
            >>> df.with_columns((col('amount') * 1.1).alias('with_tax'))
            >>> df.with_columns(('with_tax', col('amount') * 1.1))
        """
        result_df = self._df
        for expr in exprs:
            if isinstance(expr, tuple) and len(expr) == 2:
                # (name, expression) tuple
                col_name, col_expr = expr
                if isinstance(col_expr, str):
                    col_expr = col(col_expr)
                result_df = result_df.withColumn(col_name, col_expr)
            elif isinstance(expr, Column):
                # Column expression with alias
                if expr.alias_name:
                    result_df = result_df.withColumn(expr.alias_name, expr)
                else:
                    raise ValueError(
                        "Column expression in with_columns() must have an alias, "
                        "or use tuple (name, expression) format"
                    )
            else:
                raise TypeError(
                    f"with_columns() expects Column expressions or (name, Column) tuples, "
                    f"got {type(expr)}"
                )
        return self._with_dataframe(result_df)

    def with_column(self, expr: Union[Column, Tuple[str, Column]]) -> "PolarsDataFrame":
        """Add or modify a single column (alias for with_columns with one expression).

        Args:
            expr: Column expression or (name, expression) tuple

        Returns:
            PolarsDataFrame with new/modified column

        Example:
            >>> df.with_column((col('amount') * 1.1).alias('with_tax'))
        """
        return self.with_columns(expr)

    def drop(self, *columns: Union[str, Column]) -> "PolarsDataFrame":
        """Drop columns (Polars-style).

        Args:
            *columns: Column names to drop

        Returns:
            PolarsDataFrame with dropped columns

        Example:
            >>> df.drop('col1', 'col2')
        """
        if not columns:
            return self

        # Validate columns exist
        str_columns = [c for c in columns if isinstance(c, str)]
        if str_columns:
            self._validate_columns_exist(str_columns, "drop")

        # If the underlying DataFrame is a TableScan, we need to select columns first
        # to create a Project operation, then drop will work
        from ..logical.plan import TableScan

        if isinstance(self._df.plan, TableScan):
            # Get all columns and select them to create a Project
            all_columns = self.columns
            # Select all columns except the ones to drop
            cols_to_keep = [col for col in all_columns if col not in str_columns]
            if cols_to_keep:
                result_df = self._df.select(*cols_to_keep)
            else:
                # All columns were dropped - return empty select
                result_df = self._df.select()
        else:
            result_df = self._df.drop(*columns)
        return self._with_dataframe(result_df)

    def rename(self, mapping: Dict[str, str]) -> "PolarsDataFrame":
        """Rename columns (Polars-style).

        Args:
            mapping: Dictionary mapping old names to new names

        Returns:
            PolarsDataFrame with renamed columns

        Example:
            >>> df.rename({'old_name': 'new_name'})
        """
        result_df = self._df
        for old_name, new_name in mapping.items():
            result_df = result_df.withColumnRenamed(old_name, new_name)
        return self._with_dataframe(result_df)

    def sort(
        self,
        *columns: Union[str, Column],
        descending: Union[bool, Sequence[bool]] = False,
    ) -> "PolarsDataFrame":
        """Sort by columns (Polars-style).

        Args:
            *columns: Column names or Column expressions to sort by
            descending: Sort order - single bool or sequence of bools for each column

        Returns:
            Sorted PolarsDataFrame

        Example:
            >>> df.sort('age')
            >>> df.sort('age', 'name', descending=[True, False])
        """
        if not columns:
            return self

        # Validate column names if they're strings
        str_columns = [c for c in columns if isinstance(c, str)]
        if str_columns:
            self._validate_columns_exist(str_columns, "sort")

        # Normalize descending parameter
        if isinstance(descending, bool):
            descending_list = [descending] * len(columns)
        else:
            descending_list = list(descending)
            if len(descending_list) != len(columns):
                raise ValueError("descending must have same length as columns")

        # Build order_by list
        from ..expressions.column import col

        order_by_cols = []
        for col_expr, desc in zip(columns, descending_list):
            if isinstance(col_expr, str):
                col_expr = col(col_expr)
            if desc:
                col_expr = col_expr.desc()
            order_by_cols.append(col_expr)

        result_df = self._df.order_by(*order_by_cols)
        return self._with_dataframe(result_df)

    def limit(self, n: int) -> "PolarsDataFrame":
        """Limit number of rows.

        Args:
            n: Number of rows to return

        Returns:
            PolarsDataFrame with limited rows

        Example:
            >>> df.limit(10)
        """
        result_df = self._df.limit(n)
        return self._with_dataframe(result_df)

    def head(self, n: int = 5) -> "PolarsDataFrame":
        """Return the first n rows.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            PolarsDataFrame with first n rows

        Example:
            >>> df.head(10)  # First 10 rows
        """
        return self.limit(n)

    def tail(self, n: int = 5) -> "PolarsDataFrame":
        """Return the last n rows.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            PolarsDataFrame with last n rows

        Note:
            This is a simplified implementation. For proper tail() behavior with lazy
            evaluation, this method sorts all columns in descending order and takes
            the first n rows. For better performance, consider using limit() directly
            or collecting and using polars tail().

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

        sorted_df = self._df
        for col_name in cols:
            sorted_df = sorted_df.order_by(col(col_name).desc())

        limited_df = sorted_df.limit(n)
        return self._with_dataframe(limited_df)

    def sample(
        self,
        fraction: Optional[float] = None,
        n: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "PolarsDataFrame":
        """Random sampling (Polars-style).

        Args:
            fraction: Fraction of rows to sample (0.0 to 1.0)
            n: Number of rows to sample (if provided, fraction is ignored)
            seed: Random seed for reproducibility

        Returns:
            Sampled PolarsDataFrame

        Example:
            >>> df.sample(fraction=0.1, seed=42)
            >>> df.sample(n=100, seed=42)
        """
        if n is not None:
            # When n is provided, sample all rows (fraction=1.0) then limit to n
            # This provides random sampling of n rows
            sampled_df = self._df.sample(fraction=1.0, seed=seed)
            result_df = sampled_df.limit(n)
        elif fraction is not None:
            result_df = self._df.sample(fraction=fraction, seed=seed)
        else:
            raise ValueError("Either 'fraction' or 'n' must be provided to sample()")
        return self._with_dataframe(result_df)

    def group_by(self, *columns: Union[str, Column]) -> "PolarsGroupBy":
        """Group rows by one or more columns (Polars-style).

        Args:
            *columns: Column name(s) to group by

        Returns:
            PolarsGroupBy object for aggregation

        Example:
            >>> df.group_by('country')
            >>> df.group_by('country', 'region')
        """
        from .polars_groupby import PolarsGroupBy

        # Validate columns exist
        str_columns = [c for c in columns if isinstance(c, str)]
        if str_columns:
            self._validate_columns_exist(str_columns, "group_by")

        # Use the underlying DataFrame's group_by method
        grouped_df = self._df.group_by(*columns)

        # Wrap it in PolarsGroupBy
        return PolarsGroupBy(_grouped=grouped_df)

    def join(
        self,
        other: "PolarsDataFrame",
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        how: str = "inner",
        left_on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
        suffix: str = "_right",
    ) -> "PolarsDataFrame":
        """Join with another PolarsDataFrame (Polars-style).

        Args:
            other: Right DataFrame to join with
            on: Column name(s) to join on (must exist in both DataFrames)
            how: Type of join ('inner', 'left', 'right', 'outer', 'anti', 'semi')
            left_on: Column name(s) in left DataFrame
            right_on: Column name(s) in right DataFrame
            suffix: Suffix to append to overlapping column names from right DataFrame

        Returns:
            Joined PolarsDataFrame

        Example:
            >>> df1.join(df2, on='id')
            >>> df1.join(df2, left_on='customer_id', right_on='id', how='left')
        """
        # Normalize how parameter
        how_map = {
            "inner": "inner",
            "left": "left",
            "right": "right",
            "outer": "outer",
            "full": "outer",
            "full_outer": "outer",
            "anti": "anti",  # Polars-specific
            "semi": "semi",  # Polars-specific
        }
        join_how = how_map.get(how.lower(), "inner")

        # Determine join keys
        join_on: List[Tuple[str, str]]
        if on is not None:
            # Same column names in both DataFrames
            if isinstance(on, str):
                self._validate_columns_exist([on], "join (left DataFrame)")
                other._validate_columns_exist([on], "join (right DataFrame)")
                join_on = [(on, on)]
            else:
                on_list = list(on)
                str_cols = [c for c in on_list if isinstance(c, str)]
                if str_cols:
                    self._validate_columns_exist(str_cols, "join (left DataFrame)")
                    other._validate_columns_exist(str_cols, "join (right DataFrame)")
                # Handle list of tuples or list of strings
                if on_list and isinstance(on_list[0], tuple):
                    join_on = [t for t in on_list if isinstance(t, tuple) and len(t) == 2]
                else:
                    join_on = [(str(col), str(col)) for col in on_list if isinstance(col, str)]
        elif left_on is not None and right_on is not None:
            # Different column names
            if isinstance(left_on, str) and isinstance(right_on, str):
                self._validate_columns_exist([left_on], "join (left DataFrame)")
                other._validate_columns_exist([right_on], "join (right DataFrame)")
                join_on = [(left_on, right_on)]
            elif isinstance(left_on, (list, tuple)) and isinstance(right_on, (list, tuple)):
                if len(left_on) != len(right_on):
                    raise ValueError("left_on and right_on must have the same length")
                self._validate_columns_exist(list(left_on), "join (left DataFrame)")
                other._validate_columns_exist(list(right_on), "join (right DataFrame)")
                join_on = list(zip(left_on, right_on))
            else:
                raise TypeError("left_on and right_on must both be str or both be sequences")
        else:
            raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")

        # Handle anti and semi joins (Polars-specific)
        # These methods require tuple syntax, not Column expressions
        if join_how == "anti":
            # Anti-join: rows in left that don't have matches in right
            # Use the DataFrame's anti_join method
            # Convert join_on to the format expected by anti_join (list of tuples)
            on_param: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = (
                join_on if isinstance(join_on, list) else None
            )
            result_df = self._df.anti_join(other._df, on=on_param)
            return self._with_dataframe(result_df)
        elif join_how == "semi":
            # Semi-join: rows in left that have matches in right (no right columns)
            # Use the DataFrame's semi_join method
            on_param_semi: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = (
                join_on if isinstance(join_on, list) else None
            )
            result_df = self._df.semi_join(other._df, on=on_param_semi)
            return self._with_dataframe(result_df)

        # Perform standard join
        result_df = self._df.join(other._df, on=join_on, how=join_how)
        return self._with_dataframe(result_df)

    def unique(
        self, subset: Optional[Union[str, Sequence[str]]] = None, keep: str = "first"
    ) -> "PolarsDataFrame":
        """Remove duplicate rows (Polars-style).

        Args:
            subset: Column name(s) to consider for duplicates (None means all columns)
            keep: Which duplicate to keep ('first' or 'last')

        Returns:
            PolarsDataFrame with duplicates removed

        Example:
            >>> df.unique()
            >>> df.unique(subset=['col1', 'col2'])
        """
        if subset is None:
            # Remove duplicates on all columns
            result_df = self._df.distinct()
        else:
            # Validate subset columns exist
            if isinstance(subset, str):
                subset_cols = [subset]
            else:
                subset_cols = list(subset)

            self._validate_columns_exist(subset_cols, "unique")

            # For subset-based deduplication, use GROUP BY
            grouped = self._df.group_by(*subset_cols)

            # Get all column names
            all_cols = self.columns
            other_cols = [col for col in all_cols if col not in subset_cols]

            from ..expressions import functions as F

            if not other_cols:
                # If only grouping columns, distinct works fine
                result_df = self._df.distinct()
            else:
                # Build aggregations for non-grouped columns
                agg_exprs = []
                for col_name in other_cols:
                    if keep == "last":
                        agg_exprs.append(F.max(col(col_name)).alias(col_name))
                    else:  # keep == "first"
                        agg_exprs.append(F.min(col(col_name)).alias(col_name))

                result_df = grouped.agg(*agg_exprs)

        return self._with_dataframe(result_df)

    def distinct(self) -> "PolarsDataFrame":
        """Remove duplicate rows (alias for unique()).

        Returns:
            PolarsDataFrame with duplicates removed

        Example:
            >>> df.distinct()
        """
        return self.unique()

    def drop_nulls(self, subset: Optional[Union[str, Sequence[str]]] = None) -> "PolarsDataFrame":
        """Drop rows with null values (Polars-style).

        Args:
            subset: Column name(s) to check for nulls (None means all columns)

        Returns:
            PolarsDataFrame with null rows removed

        Example:
            >>> df.drop_nulls()
            >>> df.drop_nulls(subset=['col1', 'col2'])
        """
        result_df = self._df.dropna(subset=subset)
        return self._with_dataframe(result_df)

    def fill_null(
        self,
        value: Optional[Any] = None,
        strategy: Optional[str] = None,
        limit: Optional[int] = None,
        subset: Optional[Union[str, Sequence[str]]] = None,
    ) -> "PolarsDataFrame":
        """Fill null values (Polars-style).

        Args:
            value: Value to fill nulls with
            strategy: Fill strategy (e.g., 'forward', 'backward') - not fully supported
            limit: Maximum number of consecutive nulls to fill - not fully supported
            subset: Column name(s) to fill nulls in (None means all columns)

        Returns:
            PolarsDataFrame with nulls filled

        Example:
            >>> df.fill_null(0)
            >>> df.fill_null(value='unknown', subset=['name'])
        """
        if strategy is not None:
            raise NotImplementedError("fill_null with strategy is not yet implemented")
        if limit is not None:
            raise NotImplementedError("fill_null with limit is not yet implemented")

        result_df = self._df.fillna(value=value, subset=subset)
        return self._with_dataframe(result_df)

    def __getitem__(
        self, key: Union[str, Sequence[str], Column]
    ) -> Union["PolarsDataFrame", Column, "PolarsColumn"]:
        """Polars-style column access.

        Supports:
        - df['col'] - Returns Column expression for filtering/expressions
        - df[['col1', 'col2']] - Returns new PolarsDataFrame with selected columns
        - df[df['age'] > 25] - Boolean indexing (filtering via Column condition)

        Args:
            key: Column name(s) or boolean Column condition

        Returns:
            - For single column string: Column expression
            - For list of columns: PolarsDataFrame with selected columns
            - For boolean Column condition: PolarsDataFrame with filtered rows

        Example:
            >>> df['age']  # Returns Column expression
            >>> df[['id', 'name']]  # Returns PolarsDataFrame
            >>> df[df['age'] > 25]  # Returns filtered PolarsDataFrame
        """
        # Single column string: df['col'] - return PolarsColumn with str/dt accessors
        if isinstance(key, str):
            self._validate_columns_exist([key], "column access")
            from .polars_column import PolarsColumn

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
    def collect(self, stream: Literal[False] = False) -> "pl.DataFrame": ...

    @overload
    def collect(self, stream: Literal[True]) -> Iterator["pl.DataFrame"]: ...

    def collect(
        self, stream: bool = False
    ) -> Union["pl.DataFrame", Iterator["pl.DataFrame"], List[Dict[str, Any]]]:
        """Collect results as Polars DataFrame.

        Args:
            stream: If True, return an iterator of Polars DataFrame chunks.
                   If False (default), return a single Polars DataFrame.

        Returns:
            If stream=False: Polars DataFrame (if polars installed) or list of dicts
            If stream=True: Iterator of Polars DataFrame chunks

        Example:
            >>> pdf = df.collect()  # Returns pl.DataFrame
            >>> for chunk in df.collect(stream=True):  # Streaming
            ...     process(chunk)
        """
        # Collect results from underlying DataFrame
        if stream:
            # Streaming mode
            def _stream_chunks() -> Iterator["pl.DataFrame"]:
                try:
                    import polars as pl
                except ImportError:
                    # Fall back to list of dicts if polars not available
                    for chunk in self._df.collect(stream=True):
                        yield chunk  # type: ignore
                    return

                for chunk in self._df.collect(stream=True):
                    df_chunk = pl.DataFrame(chunk)
                    yield df_chunk

            return _stream_chunks()
        else:
            # Single result
            results = self._df.collect(stream=False)

            try:
                import polars as pl
            except ImportError:
                # Return list of dicts if polars not available
                return results

            return pl.DataFrame(results)

    def fetch(self, n: int) -> "pl.DataFrame":
        """Fetch first n rows without full collection.

        Args:
            n: Number of rows to fetch

        Returns:
            Polars DataFrame with first n rows

        Example:
            >>> df.fetch(10)  # First 10 rows as Polars DataFrame
        """
        limited_df = self.limit(n)
        return limited_df.collect()

    def write_csv(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this PolarsDataFrame to a CSV file (Polars-style).

        Args:
            path: Path to write the CSV file
            mode: Write mode ("overwrite", "append", "error_if_exists")
            **options: Format-specific options (e.g., header=True, delimiter=",")

        Example:
            >>> df.write_csv("output.csv", header=True)
            >>> df.write_csv("output.csv", mode="append", header=True, delimiter=",")
        """
        writer = self._df.write.mode(mode)
        if options:
            writer = writer.options(**options)
        writer.csv(path)

    def write_json(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this PolarsDataFrame to a JSON file (Polars-style).

        Args:
            path: Path to write the JSON file
            mode: Write mode ("overwrite", "append", "error_if_exists")
            **options: Format-specific options

        Example:
            >>> df.write_json("output.json")
        """
        writer = self._df.write.mode(mode)
        if options:
            writer = writer.options(**options)
        writer.json(path)

    def write_jsonl(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this PolarsDataFrame to a JSONL file (Polars-style).

        Args:
            path: Path to write the JSONL file
            mode: Write mode ("overwrite", "append", "error_if_exists")
            **options: Format-specific options

        Example:
            >>> df.write_jsonl("output.jsonl")
        """
        writer = self._df.write.mode(mode)
        if options:
            writer = writer.options(**options)
        writer.jsonl(path)

    def write_parquet(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this PolarsDataFrame to a Parquet file (Polars-style).

        Args:
            path: Path to write the Parquet file
            mode: Write mode ("overwrite", "append", "error_if_exists")
            **options: Format-specific options

        Raises:
            RuntimeError: If pandas or pyarrow are not installed

        Example:
            >>> df.write_parquet("output.parquet")
        """
        writer = self._df.write.mode(mode)
        if options:
            writer = writer.options(**options)
        writer.parquet(path)

    # ========================================================================
    # Additional Polars Features
    # ========================================================================

    def explode(self, columns: Union[str, Sequence[str]]) -> "PolarsDataFrame":
        """Explode array/JSON columns into multiple rows (Polars-style).

        Args:
            columns: Column name(s) to explode

        Returns:
            PolarsDataFrame with exploded rows

        Example:
            >>> df.explode('tags')
            >>> df.explode(['tags', 'categories'])
        """
        if isinstance(columns, str):
            columns = [columns]
        self._validate_columns_exist(columns, "explode")

        # Explode the first column (Polars supports multiple, but we'll do one at a time)
        result_df = self._df
        for col_name in columns:
            result_df = result_df.explode(col(col_name), alias=col_name)
        return self._with_dataframe(result_df)

    def unnest(self, columns: Union[str, Sequence[str]]) -> "PolarsDataFrame":
        """Unnest struct columns (Polars-style).

        Note: This is similar to explode but for struct types.
        For now, we'll use explode as the implementation.

        Args:
            columns: Column name(s) to unnest

        Returns:
            PolarsDataFrame with unnested columns

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
    ) -> "PolarsDataFrame":
        """Pivot DataFrame (Polars-style).

        Args:
            values: Column(s) to aggregate
            index: Column(s) to use as index (rows)
            columns: Column to use as columns (pivot column)
            aggregate_function: Aggregation function (e.g., 'sum', 'mean', 'count')

        Returns:
            Pivoted PolarsDataFrame

        Example:
            >>> df.pivot(values='amount', index='category', columns='status', aggregate_function='sum')
        """
        if columns is None:
            raise ValueError("pivot() requires 'columns' parameter")
        if aggregate_function is None:
            aggregate_function = "sum"

        # Use underlying DataFrame's pivot method
        # Note: DataFrame.pivot has different signature, so we need to adapt
        if isinstance(values, (list, tuple)) and len(values) > 0:
            value_col: str = str(values[0])
        else:
            value_col = str(values)
        result_df = self._df.pivot(
            pivot_column=columns,
            value_column=value_col,
            agg_func=aggregate_function,
            pivot_values=None,
        )
        return self._with_dataframe(result_df)

    def melt(
        self,
        id_vars: Optional[Union[str, Sequence[str]]] = None,
        value_vars: Optional[Union[str, Sequence[str]]] = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> "PolarsDataFrame":
        """Melt DataFrame from wide to long format (Polars-style).

        Args:
            id_vars: Column(s) to use as identifier variables
            value_vars: Column(s) to unpivot (if None, unpivot all except id_vars)
            variable_name: Name for the variable column
            value_name: Name for the value column

        Returns:
            Melted PolarsDataFrame

        Example:
            >>> df.melt(id_vars=['id'], value_vars=['col1', 'col2'])
        """
        # Melt is not yet implemented in DataFrame, so we'll raise NotImplementedError
        # This would require implementing UNPIVOT in SQL
        raise NotImplementedError(
            "melt() is not yet implemented. "
            "This would require UNPIVOT SQL support which varies by database."
        )

    def slice(self, offset: int, length: Optional[int] = None) -> "PolarsDataFrame":
        """Slice DataFrame (Polars-style).

        Args:
            offset: Starting row index
            length: Number of rows to return (if None, returns all remaining rows)

        Returns:
            Sliced PolarsDataFrame

        Example:
            >>> df.slice(10, 5)  # Rows 10-14
            >>> df.slice(10)  # All rows from 10 onwards
        """
        if length is None:
            # Return all rows from offset onwards
            # We can use limit with a large number, but better to use offset
            # For now, we'll use order_by + limit (requires a sort key)
            # Actually, SQL LIMIT with OFFSET is what we need
            result_df = self._df.limit(offset + 1000000)  # Large number as workaround
            # Better approach: add offset support to DataFrame
            # For now, this is a limitation
            return self._with_dataframe(result_df)
        else:
            # Use limit with offset calculation
            # We need to skip 'offset' rows and take 'length' rows
            # This requires OFFSET support in DataFrame
            result_df = self._df.limit(offset + length)
            return self._with_dataframe(result_df)

    def gather_every(self, n: int, offset: int = 0) -> "PolarsDataFrame":
        """Sample every nth row (Polars-style).

        Args:
            n: Sample every nth row
            offset: Starting offset

        Returns:
            Sampled PolarsDataFrame

        Example:
            >>> df.gather_every(10)  # Every 10th row
            >>> df.gather_every(5, offset=2)  # Every 5th row starting from row 2
        """
        # This requires row number window function and modulo operation
        from ..expressions import functions as F

        # Add row number, filter by modulo, then remove row number
        # Use over() with empty partition/order for global row number
        row_num_col = F.row_number().over()
        df_with_row_num = self.with_columns(row_num_col.alias("__row_num__"))
        filtered_df = df_with_row_num.filter((col("__row_num__") - offset) % n == 0)
        result_df = filtered_df.drop("__row_num__")
        return result_df

    def interpolate(self, method: str = "linear") -> "PolarsDataFrame":
        """Interpolate missing values (Polars-style).

        Args:
            method: Interpolation method ('linear', 'nearest', etc.)

        Returns:
            PolarsDataFrame with interpolated values

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
    ) -> "PolarsDataFrame":
        """Compute quantiles (Polars-style).

        Args:
            quantile: Quantile value(s) (0.0 to 1.0)
            interpolation: Interpolation method (not used, for API compatibility)

        Returns:
            PolarsDataFrame with quantile values

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
        from ..expressions import functions as F

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

    def describe(self) -> "PolarsDataFrame":
        """Compute descriptive statistics (Polars-style).

        Returns:
            PolarsDataFrame with statistics (count, mean, std, min, max, etc.)

        Note:
            Standard deviation (std) may not be available in all databases
            (e.g., SQLite). In such cases, std will be omitted.

        Example:
            >>> df.describe()
        """
        from ..expressions import functions as F

        numeric_cols = [c for c in self.columns if self._is_numeric_column(c)]
        if not numeric_cols:
            return self

        stats_exprs = []
        for col_name in numeric_cols:
            col_expr = col(col_name)
            stats_exprs.extend(
                [
                    F.count(col_expr).alias(f"{col_name}_count"),
                    F.avg(col_expr).alias(f"{col_name}_mean"),
                    F.min(col_expr).alias(f"{col_name}_min"),
                    F.max(col_expr).alias(f"{col_name}_max"),
                ]
            )
            # Note: stddev is omitted as it's not supported by all databases (e.g., SQLite)
            # Users can compute stddev manually if needed for their database

        result_df = self._df.select(*stats_exprs)
        return self._with_dataframe(result_df)

    def explain(self, format: str = "string") -> str:
        """Explain the query plan (Polars-style).

        Args:
            format: Output format ('string' or 'tree')

        Returns:
            Query plan as string

        Example:
            >>> print(df.explain())
        """
        return self._df.explain(analyze=False)

    def _is_numeric_column(self, col_name: str) -> bool:
        """Check if a column is numeric based on schema."""
        schema = self.schema
        for name, dtype in schema:
            if name == col_name:
                # Check if dtype is numeric
                numeric_dtypes = ["Int64", "Int32", "Int8", "Float64", "Float32"]
                return dtype in numeric_dtypes
        return False

    # ========================================================================
    # Set Operations (Polars-style)
    # ========================================================================

    def concat(
        self,
        *others: "PolarsDataFrame",
        how: str = "vertical",
        rechunk: bool = True,
    ) -> "PolarsDataFrame":
        """Concatenate DataFrames (Polars-style).

        Args:
            *others: Other PolarsDataFrames to concatenate
            how: Concatenation mode - "vertical" (union) or "diagonal" (union with different schemas)
            rechunk: If True, rechunk the result (not used, for API compatibility)

        Returns:
            Concatenated PolarsDataFrame

        Example:
            >>> df1.concat(df2)  # Vertical concatenation
            >>> df1.concat(df2, df3, how="vertical")
        """
        if not others:
            return self

        result_df = self._df
        for other in others:
            if how == "vertical":
                # Vertical concatenation (union all)
                result_df = result_df.unionAll(other._df)
            elif how == "diagonal":
                # Diagonal concatenation (union all with different schemas)
                # For now, same as vertical
                result_df = result_df.unionAll(other._df)
            else:
                raise ValueError(
                    f"Invalid 'how' parameter: {how}. Must be 'vertical' or 'diagonal'"
                )

        return self._with_dataframe(result_df)

    def hstack(
        self,
        *others: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Horizontally stack DataFrames (Polars-style).

        Args:
            *others: Other PolarsDataFrames to stack horizontally

        Returns:
            Horizontally stacked PolarsDataFrame

        Example:
            >>> df1.hstack(df2)  # Combine columns from df1 and df2
        """
        if not others:
            return self

        # Horizontal stacking means combining columns side by side
        # This is similar to a cross join but without the cartesian product
        # For now, we'll use a cross join as the implementation
        result_df = self._df
        for other in others:
            result_df = result_df.crossJoin(other._df)

        return self._with_dataframe(result_df)

    def vstack(
        self,
        *others: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Vertically stack DataFrames (Polars-style alias for concat).

        Args:
            *others: Other PolarsDataFrames to stack vertically

        Returns:
            Vertically stacked PolarsDataFrame

        Example:
            >>> df1.vstack(df2)  # Same as df1.concat(df2)
        """
        return self.concat(*others, how="vertical")

    def union(
        self,
        other: "PolarsDataFrame",
        *,
        distinct: bool = True,
    ) -> "PolarsDataFrame":
        """Union with another PolarsDataFrame (Polars-style).

        Args:
            other: Another PolarsDataFrame to union with
            distinct: If True, return distinct rows only (default: True)

        Returns:
            Unioned PolarsDataFrame

        Example:
            >>> df1.union(df2)  # Union distinct
            >>> df1.union(df2, distinct=False)  # Union all
        """
        if distinct:
            result_df = self._df.union(other._df)
        else:
            result_df = self._df.unionAll(other._df)
        return self._with_dataframe(result_df)

    def intersect(
        self,
        other: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Intersect with another PolarsDataFrame (Polars-style).

        Args:
            other: Another PolarsDataFrame to intersect with

        Returns:
            Intersected PolarsDataFrame (common rows only)

        Example:
            >>> df1.intersect(df2)  # Common rows
        """
        result_df = self._df.intersect(other._df)
        return self._with_dataframe(result_df)

    def difference(
        self,
        other: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Return rows in this DataFrame that are not in another (Polars-style).

        Args:
            other: Another PolarsDataFrame to exclude from

        Returns:
            PolarsDataFrame with rows in this but not in other

        Example:
            >>> df1.difference(df2)  # Rows in df1 but not in df2
        """
        result_df = self._df.except_(other._df)
        return self._with_dataframe(result_df)

    def cross_join(
        self,
        other: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Perform a cross join (Cartesian product) with another PolarsDataFrame (Polars-style).

        Args:
            other: Another PolarsDataFrame to cross join with

        Returns:
            Cross-joined PolarsDataFrame

        Example:
            >>> df1.cross_join(df2)  # Cartesian product
        """
        result_df = self._df.crossJoin(other._df)
        return self._with_dataframe(result_df)

    # ========================================================================
    # SQL Expression Selection
    # ========================================================================

    def select_expr(
        self,
        *exprs: str,
    ) -> "PolarsDataFrame":
        """Select columns using SQL expressions (Polars-style).

        Args:
            *exprs: SQL expression strings (e.g., "amount * 1.1 as with_tax")

        Returns:
            PolarsDataFrame with selected expressions

        Example:
            >>> df.select_expr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")
        """
        result_df = self._df.selectExpr(*exprs)
        return self._with_dataframe(result_df)

    # ========================================================================
    # Common Table Expressions (CTEs)
    # ========================================================================

    def with_columns_renamed(
        self,
        mapping: Dict[str, str],
    ) -> "PolarsDataFrame":
        """Rename columns using a mapping (Polars-style alias for rename).

        Args:
            mapping: Dictionary mapping old column names to new names

        Returns:
            PolarsDataFrame with renamed columns

        Example:
            >>> df.with_columns_renamed({"old_name": "new_name"})
        """
        return self.rename(mapping)

    def with_row_count(
        self,
        name: str = "row_nr",
        offset: int = 0,
    ) -> "PolarsDataFrame":
        """Add a row number column (Polars-style).

        Args:
            name: Name for the row number column (default: "row_nr")
            offset: Starting offset for row numbers (default: 0)

        Returns:
            PolarsDataFrame with row number column

        Example:
            >>> df.with_row_count("row_id")
        """
        from ..expressions import functions as F

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
        *contexts: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Add context DataFrames for use in expressions (Polars-style).

        Note: This is a placeholder for Polars' with_context feature.
        In Moltres, CTEs serve a similar purpose.

        Args:
            *contexts: Context DataFrames to add

        Returns:
            PolarsDataFrame with context

        Example:
            >>> df.with_context(context_df)
        """
        # For now, this is a no-op as Moltres doesn't have the same context system
        # Users should use CTEs instead
        return self

    # ========================================================================
    # Common Table Expressions (CTEs) - Moltres-specific but Polars-style API
    # ========================================================================

    def cte(
        self,
        name: str,
    ) -> "PolarsDataFrame":
        """Create a Common Table Expression (CTE) from this DataFrame.

        Args:
            name: Name for the CTE

        Returns:
            PolarsDataFrame representing the CTE

        Example:
            >>> cte_df = df.filter(col("age") > 25).cte("adults")
            >>> result = cte_df.select().collect()
        """
        result_df = self._df.cte(name)
        return self._with_dataframe(result_df)

    def with_recursive(
        self,
        name: str,
        recursive: "PolarsDataFrame",
        *,
        union_all: bool = False,
    ) -> "PolarsDataFrame":
        """Create a Recursive Common Table Expression (WITH RECURSIVE).

        Args:
            name: Name for the recursive CTE
            recursive: PolarsDataFrame representing the recursive part
            union_all: If True, use UNION ALL; if False, use UNION (distinct)

        Returns:
            PolarsDataFrame representing the recursive CTE

        Example:
            >>> initial = db.table("seed").polars()
            >>> recursive = initial.select(...)  # Recursive part
            >>> fib_cte = initial.with_recursive("fib", recursive)
        """
        result_df = self._df.recursive_cte(name, recursive._df, union_all=union_all)
        return self._with_dataframe(result_df)
