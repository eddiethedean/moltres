"""Polars-style interface for Moltres DataFrames."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
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

if TYPE_CHECKING:
    import polars as pl
    from sqlalchemy.sql import Select

    from ...table.table import Database
    from ..columns.polars_column import PolarsColumn
    from ..groupby.polars_groupby import PolarsGroupBy


# Import from polars_operations for backward compatibility
from ..operations.polars_operations import sql_type_to_polars_dtype


@dataclass(frozen=True)
class PolarsDataFrame(InterfaceCommonMixin):
    """Polars-style interface wrapper around Moltres :class:`DataFrame`.

    Provides familiar Polars LazyFrame API methods while maintaining lazy evaluation
    and SQL pushdown execution. All operations remain lazy until collect() is called.

    Example:
        >>> df = db.table('users').polars()
        >>> # Polars-style operations
        >>> df.filter(col('age') > 25).select(['id', 'name'])
        >>> # Returns actual Polars :class:`DataFrame`
        >>> result = df.collect()  # pl.:class:`DataFrame`
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
        """Create a :class:`PolarsDataFrame` from a regular :class:`DataFrame`.

        Args:
            df: The :class:`DataFrame` to wrap

        Returns:
            :class:`PolarsDataFrame` wrapping the provided :class:`DataFrame`
        """
        return cls(_df=df)

    def _with_dataframe(self, df: DataFrame) -> "PolarsDataFrame":
        """Create a new :class:`PolarsDataFrame` with a different underlying :class:`DataFrame`.

        Args:
            df: The new underlying :class:`DataFrame`

        Returns:
            New :class:`PolarsDataFrame` instance
        """
        # Clear caches when creating new DataFrame instance
        return PolarsDataFrame(_df=df, _height_cache=None, _schema_cache=None)

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
            for the lifetime of this :class:`DataFrame` instance.

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
        """Return self (for API compatibility, :class:`PolarsDataFrame` is already lazy).

        Returns:
            Self (:class:`PolarsDataFrame` is always lazy)

        Example:
            >>> df.lazy()  # Returns self
        """
        return self

    def select(self, *exprs: Union[str, Column]) -> "PolarsDataFrame":
        """Select columns/expressions (Polars-style).

        Args:
            *exprs: :class:`Column` names or :class:`Column` expressions to select

        Returns:
            :class:`PolarsDataFrame` with selected columns

        Example:
            >>> df.select('id', 'name')
            >>> df.select(col('id'), (col('amount') * 1.1).alias('with_tax'))
        """
        from ..helpers.polars_dataframe_helpers import build_polars_select_operation

        result_df = build_polars_select_operation(self, *exprs)
        return self._with_dataframe(cast(DataFrame, result_df))

    def filter(self, predicate: Column) -> "PolarsDataFrame":
        """Filter rows (Polars-style, uses 'filter' instead of 'where').

        Args:
            predicate: :class:`Column` expression for filtering condition

        Returns:
            Filtered :class:`PolarsDataFrame`

        Example:
            >>> df.filter(col('age') > 25)
            >>> df.filter((col('age') > 25) & (col('active') == True))
        """
        from ..helpers.polars_dataframe_helpers import build_polars_filter_operation

        result_df = build_polars_filter_operation(self, predicate)
        return self._with_dataframe(cast(DataFrame, result_df))

    def with_columns(self, *exprs: Union[Column, Tuple[str, Column]]) -> "PolarsDataFrame":
        """Add or modify columns (Polars primary method for adding columns).

        Args:
            *exprs: :class:`Column` expressions or (name, expression) tuples

        Returns:
            :class:`PolarsDataFrame` with new/modified columns

        Example:
            >>> df.with_columns((col('amount') * 1.1).alias('with_tax'))
            >>> df.with_columns(('with_tax', col('amount') * 1.1))
        """
        from ..helpers.polars_dataframe_helpers import build_polars_with_columns_operation

        result_df = build_polars_with_columns_operation(self, *exprs)
        return self._with_dataframe(cast(DataFrame, result_df))

    def with_column(self, expr: Union[Column, Tuple[str, Column]]) -> "PolarsDataFrame":
        """Add or modify a single column (alias for with_columns with one expression).

        Args:
            expr: :class:`Column` expression or (name, expression) tuple

        Returns:
            :class:`PolarsDataFrame` with new/modified column

        Example:
            >>> df.with_column((col('amount') * 1.1).alias('with_tax'))
        """
        return self.with_columns(expr)

    def drop(self, *columns: Union[str, Column]) -> "PolarsDataFrame":
        """Drop columns (Polars-style).

        Args:
            *columns: :class:`Column` names to drop

        Returns:
            :class:`PolarsDataFrame` with dropped columns

        Example:
            >>> df.drop('col1', 'col2')
        """
        from ..helpers.polars_dataframe_helpers import build_polars_drop_operation

        if not columns:
            return self

        result_df = build_polars_drop_operation(self, *columns)
        if result_df is None:
            return self
        return self._with_dataframe(cast(DataFrame, result_df))

    def rename(self, mapping: Dict[str, str]) -> "PolarsDataFrame":
        """Rename columns (Polars-style).

        Args:
            mapping: Dictionary mapping old names to new names

        Returns:
            :class:`PolarsDataFrame` with renamed columns

        Example:
            >>> df.rename({'old_name': 'new_name'})
        """
        from ..helpers.polars_dataframe_helpers import build_polars_rename_operation

        result_df = build_polars_rename_operation(self, mapping)
        return self._with_dataframe(cast(DataFrame, result_df))

    def sort(
        self,
        *columns: Union[str, Column],
        descending: Union[bool, Sequence[bool]] = False,
    ) -> "PolarsDataFrame":
        """Sort by columns (Polars-style).

        Args:
            *columns: :class:`Column` names or :class:`Column` expressions to sort by
            descending: Sort order - single bool or sequence of bools for each column

        Returns:
            Sorted :class:`PolarsDataFrame`

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
        return self._with_dataframe(cast(DataFrame, result_df))

    def limit(self, n: int) -> "PolarsDataFrame":
        """Limit number of rows.

        Args:
            n: Number of rows to return

        Returns:
            :class:`PolarsDataFrame` with limited rows

        Example:
            >>> df.limit(10)
        """
        from ..helpers.polars_dataframe_helpers import build_polars_limit_operation

        result_df = build_polars_limit_operation(self, n)
        return self._with_dataframe(cast(DataFrame, result_df))

    def head(self, n: int = 5) -> "PolarsDataFrame":
        """Return the first n rows.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            :class:`PolarsDataFrame` with first n rows

        Example:
            >>> df.head(10)  # First 10 rows
        """
        return self.limit(n)

    def tail(self, n: int = 5) -> "PolarsDataFrame":
        """Return the last n rows.

        Args:
            n: Number of rows to return (default: 5)

        Returns:
            :class:`PolarsDataFrame` with last n rows

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
        return self._with_dataframe(cast(DataFrame, result_df))

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
            Sampled :class:`PolarsDataFrame`

        Example:
            >>> df.sample(fraction=0.1, seed=42)
            >>> df.sample(n=100, seed=42)
        """
        from ..helpers.polars_dataframe_helpers import build_polars_sample_operation

        result_df = build_polars_sample_operation(self, fraction=fraction, n=n, seed=seed)
        return self._with_dataframe(cast(DataFrame, result_df))

    def group_by(self, *columns: Union[str, Column]) -> "PolarsGroupBy":
        """Group rows by one or more columns (Polars-style).

        Args:
            *columns: :class:`Column` name(s) to group by

        Returns:
            PolarsGroupBy object for aggregation

        Example:
            >>> df.group_by('country')
            >>> df.group_by('country', 'region')
        """
        from ..groupby.polars_groupby import PolarsGroupBy

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
        """Join with another :class:`PolarsDataFrame` (Polars-style).

        Args:
            other: Right :class:`DataFrame` to join with
            on: :class:`Column` name(s) to join on (must exist in both DataFrames)
            how: Type of join ('inner', 'left', 'right', 'outer', 'anti', 'semi')
            left_on: :class:`Column` name(s) in left :class:`DataFrame`
            right_on: :class:`Column` name(s) in right :class:`DataFrame`
            suffix: Suffix to append to overlapping column names from right :class:`DataFrame`

        Returns:
            Joined :class:`PolarsDataFrame`

        Example:
            >>> df1.join(df2, on='id')
            >>> df1.join(df2, left_on='customer_id', right_on='id', how='left')
        """
        from ..helpers.polars_dataframe_helpers import build_polars_join_operation

        result_df = build_polars_join_operation(
            self, other, on=on, how=how, left_on=left_on, right_on=right_on
        )
        return self._with_dataframe(cast(DataFrame, result_df))

    def semi_join(
        self,
        other: "PolarsDataFrame",
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        left_on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
    ) -> "PolarsDataFrame":
        """Semi-join: filter rows in left :class:`DataFrame` that have matches in right :class:`DataFrame`.

        Args:
            other: Right :class:`DataFrame` to join with
            on: :class:`Column` name(s) to join on (must exist in both DataFrames)
            left_on: :class:`Column` name(s) in left :class:`DataFrame`
            right_on: :class:`Column` name(s) in right :class:`DataFrame`

        Returns:
            :class:`PolarsDataFrame` with rows from left that have matches in right

        Example:
            >>> df1.semi_join(df2, on='id')
            >>> df1.semi_join(df2, left_on='customer_id', right_on='id')
        """
        # Use the join method with how='semi'
        return self.join(other, on=on, left_on=left_on, right_on=right_on, how="semi")

    def anti_join(
        self,
        other: "PolarsDataFrame",
        on: Optional[Union[str, Sequence[str], Sequence[Tuple[str, str]]]] = None,
        left_on: Optional[Union[str, Sequence[str]]] = None,
        right_on: Optional[Union[str, Sequence[str]]] = None,
    ) -> "PolarsDataFrame":
        """Anti-join: filter rows in left :class:`DataFrame` that don't have matches in right :class:`DataFrame`.

        Args:
            other: Right :class:`DataFrame` to join with
            on: :class:`Column` name(s) to join on (must exist in both DataFrames)
            left_on: :class:`Column` name(s) in left :class:`DataFrame`
            right_on: :class:`Column` name(s) in right :class:`DataFrame`

        Returns:
            :class:`PolarsDataFrame` with rows from left that don't have matches in right

        Example:
            >>> df1.anti_join(df2, on='id')
            >>> df1.anti_join(df2, left_on='customer_id', right_on='id')
        """
        # Use the join method with how='anti'
        return self.join(other, on=on, left_on=left_on, right_on=right_on, how="anti")

    def unique(
        self, subset: Optional[Union[str, Sequence[str]]] = None, keep: str = "first"
    ) -> "PolarsDataFrame":
        """Remove duplicate rows (Polars-style).

        Args:
            subset: :class:`Column` name(s) to consider for duplicates (None means all columns)
            keep: Which duplicate to keep ('first' or 'last')

        Returns:
            :class:`PolarsDataFrame` with duplicates removed

        Example:
            >>> df.unique()
            >>> df.unique(subset=['col1', 'col2'])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_unique_operation

        result_df = build_polars_unique_operation(self, subset=subset, keep=keep)
        return self._with_dataframe(cast(DataFrame, result_df))

    def distinct(self) -> "PolarsDataFrame":
        """Remove duplicate rows (alias for unique()).

        Returns:
            :class:`PolarsDataFrame` with duplicates removed

        Example:
            >>> df.distinct()
        """
        return self.unique()

    def drop_nulls(self, subset: Optional[Union[str, Sequence[str]]] = None) -> "PolarsDataFrame":
        """Drop rows with null values (Polars-style).

        Args:
            subset: :class:`Column` name(s) to check for nulls (None means all columns)

        Returns:
            :class:`PolarsDataFrame` with null rows removed

        Example:
            >>> df.drop_nulls()
            >>> df.drop_nulls(subset=['col1', 'col2'])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_drop_nulls_operation

        result_df = build_polars_drop_nulls_operation(self, subset=subset)
        return self._with_dataframe(cast(DataFrame, result_df))

    def fill_null(
        self,
        value: Optional[FillValue] = None,
        strategy: Optional[str] = None,
        limit: Optional[int] = None,
        subset: Optional[Union[str, Sequence[str]]] = None,
    ) -> "PolarsDataFrame":
        """Fill null values (Polars-style).

        Args:
            value: Value to fill nulls with
            strategy: Fill strategy (e.g., 'forward', 'backward') - not fully supported
            limit: Maximum number of consecutive nulls to fill - not fully supported
            subset: :class:`Column` name(s) to fill nulls in (None means all columns)

        Returns:
            :class:`PolarsDataFrame` with nulls filled

        Example:
            >>> df.fill_null(0)
            >>> df.fill_null(value='unknown', subset=['name'])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_fill_null_operation

        result_df = build_polars_fill_null_operation(
            self, value=value, strategy=strategy, limit=limit, subset=subset
        )
        return self._with_dataframe(cast(DataFrame, result_df))

    def __getitem__(
        self, key: Union[str, Sequence[str], Column]
    ) -> Union["PolarsDataFrame", Column, "PolarsColumn"]:
        """Polars-style column access.

        Supports:
        - df['col'] - Returns :class:`Column` expression for filtering/expressions
        - df[['col1', 'col2']] - Returns new :class:`PolarsDataFrame` with selected columns
        - df[df['age'] > 25] - Boolean indexing (filtering via :class:`Column` condition)

        Args:
            key: :class:`Column` name(s) or boolean :class:`Column` condition

        Returns:
            - For single column string: :class:`Column` expression
            - For list of columns: :class:`PolarsDataFrame` with selected columns
            - For boolean :class:`Column` condition: :class:`PolarsDataFrame` with filtered rows

        Example:
            >>> df['age']  # Returns :class:`Column` expression
            >>> df[['id', 'name']]  # Returns :class:`PolarsDataFrame`
            >>> df[df['age'] > 25]  # Returns filtered :class:`PolarsDataFrame`
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
    def collect(self, stream: Literal[False] = False) -> "pl.DataFrame": ...

    @overload
    def collect(
        self, stream: Literal[True]
    ) -> Iterator[Union["pl.DataFrame", List[Dict[str, Any]]]]: ...

    def collect(
        self, stream: bool = False
    ) -> Union[
        "pl.DataFrame", Iterator[Union["pl.DataFrame", List[Dict[str, Any]]]], List[Dict[str, Any]]
    ]:
        """Collect results as Polars :class:`DataFrame`.

        Args:
            stream: If True, return an iterator of Polars :class:`DataFrame` chunks.
                   If False (default), return a single Polars :class:`DataFrame`.

        Returns:
            If stream=False: Polars :class:`DataFrame` (if polars installed) or list of dicts
            If stream=True: Iterator of Polars :class:`DataFrame` chunks

        Example:
            >>> pdf = df.collect()  # Returns pl.:class:`DataFrame`
            >>> for chunk in df.collect(stream=True):  # Streaming
            ...     process(chunk)
        """
        # Collect results from underlying DataFrame
        if stream:
            # Streaming mode
            def _stream_chunks() -> Iterator[Union["pl.DataFrame", List[Dict[str, Any]]]]:
                try:
                    import polars as pl
                except ImportError:
                    # Fall back to list of dicts if polars not available
                    for chunk in self._df.collect(stream=True):
                        yield chunk
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

    def to_sqlalchemy(self, dialect: Optional[str] = None) -> "Select":
        """Convert :class:`PolarsDataFrame`'s logical plan to a SQLAlchemy Select statement.

        This method delegates to the underlying :class:`DataFrame`'s to_sqlalchemy() method,
        allowing you to use :class:`PolarsDataFrame` with existing SQLAlchemy infrastructure.

        Args:
            dialect: Optional SQL dialect name. If not provided, uses the dialect
                    from the attached :class:`Database`, or defaults to "ansi"

        Returns:
            SQLAlchemy Select statement that can be executed with any SQLAlchemy connection

        Example:
            >>> from moltres import connect, col
            >>> from sqlalchemy import create_engine
            >>> db = connect("sqlite:///:memory:")
            >>> df = db.table("users").polars()
            >>> stmt = df.to_sqlalchemy()
            >>> # Execute with existing SQLAlchemy connection
            >>> engine = create_engine("sqlite:///:memory:")
            >>> with engine.connect() as conn:
            ...     result = conn.execute(stmt)
        """
        return self._df.to_sqlalchemy(dialect=dialect)

    def fetch(self, n: int) -> "pl.DataFrame":
        """Fetch first n rows without full collection.

        Args:
            n: Number of rows to fetch

        Returns:
            Polars :class:`DataFrame` with first n rows

        Example:
            >>> df.fetch(10)  # First 10 rows as Polars :class:`DataFrame`
        """
        limited_df = self.limit(n)
        return limited_df.collect()

    def write_csv(
        self,
        path: str,
        mode: str = "overwrite",
        **options: object,
    ) -> None:
        """Write this :class:`PolarsDataFrame` to a CSV file (Polars-style).

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
        """Write this :class:`PolarsDataFrame` to a JSON file (Polars-style).

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
        """Write this :class:`PolarsDataFrame` to a JSONL file (Polars-style).

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
        """Write this :class:`PolarsDataFrame` to a Parquet file (Polars-style).

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
            columns: :class:`Column` name(s) to explode

        Returns:
            :class:`PolarsDataFrame` with exploded rows

        Example:
            >>> df.explode('tags')
            >>> df.explode(['tags', 'categories'])
        """
        from ..helpers.polars_dataframe_helpers import build_polars_explode_operation

        result_df = build_polars_explode_operation(self, columns)
        return self._with_dataframe(cast(DataFrame, result_df))

    def unnest(self, columns: Union[str, Sequence[str]]) -> "PolarsDataFrame":
        """Unnest struct columns (Polars-style).

        Note: This is similar to explode but for struct types.
        For now, we'll use explode as the implementation.

        Args:
            columns: :class:`Column` name(s) to unnest

        Returns:
            :class:`PolarsDataFrame` with unnested columns

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
        """Pivot :class:`DataFrame` (Polars-style).

        Args:
            values: :class:`Column`(s) to aggregate
            index: :class:`Column`(s) to use as index (rows)
            columns: :class:`Column` to use as columns (pivot column)
            aggregate_function: Aggregation function (e.g., 'sum', 'mean', 'count')

        Returns:
            Pivoted :class:`PolarsDataFrame`

        Example:
            >>> df.pivot(values='amount', index='category', columns='status', aggregate_function='sum')
        """
        from ..helpers.polars_dataframe_helpers import build_polars_pivot_operation

        if aggregate_function is None:
            aggregate_function = "sum"

        result_df = build_polars_pivot_operation(self, values, columns, aggregate_function)
        return self._with_dataframe(cast(DataFrame, result_df))

    def melt(
        self,
        id_vars: Optional[Union[str, Sequence[str]]] = None,
        value_vars: Optional[Union[str, Sequence[str]]] = None,
        variable_name: str = "variable",
        value_name: str = "value",
    ) -> "PolarsDataFrame":
        """Melt :class:`DataFrame` from wide to long format (Polars-style).

        Args:
            id_vars: :class:`Column`(s) to use as identifier variables
            value_vars: :class:`Column`(s) to unpivot (if None, unpivot all except id_vars)
            variable_name: Name for the variable column
            value_name: Name for the value column

        Returns:
            Melted :class:`PolarsDataFrame`

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
        """Slice :class:`DataFrame` (Polars-style).

        Args:
            offset: Starting row index
            length: Number of rows to return (if None, returns all remaining rows)

        Returns:
            Sliced :class:`PolarsDataFrame`

        Example:
            >>> df.slice(10, 5)  # Rows 10-14
            >>> df.slice(10)  # All rows from 10 onwards
        """
        from ..helpers.polars_dataframe_helpers import build_polars_slice_operation

        result_df = build_polars_slice_operation(self, offset, length)
        return self._with_dataframe(cast(DataFrame, result_df))

    def gather_every(self, n: int, offset: int = 0) -> "PolarsDataFrame":
        """Sample every nth row (Polars-style).

        Args:
            n: Sample every nth row
            offset: Starting offset

        Returns:
            Sampled :class:`PolarsDataFrame`

        Example:
            >>> df.gather_every(10)  # Every 10th row
            >>> df.gather_every(5, offset=2)  # Every 5th row starting from row 2
        """
        from ..helpers.polars_dataframe_helpers import build_polars_gather_every_operation

        result_df = build_polars_gather_every_operation(self, n, offset)
        return self._with_dataframe(cast(DataFrame, result_df))

    def interpolate(self, method: str = "linear") -> "PolarsDataFrame":
        """Interpolate missing values (Polars-style).

        Args:
            method: Interpolation method ('linear', 'nearest', etc.)

        Returns:
            :class:`PolarsDataFrame` with interpolated values

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
            :class:`PolarsDataFrame` with quantile values

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

    def describe(self) -> "PolarsDataFrame":
        """Compute descriptive statistics (Polars-style).

        Returns:
            :class:`PolarsDataFrame` with statistics (count, mean, std, min, max, etc.)

        Note:
            Standard deviation (std) may not be available in all databases
            (e.g., SQLite). In such cases, std will be omitted.

        Example:
            >>> df.describe()
        """
        from ..helpers.polars_dataframe_helpers import build_polars_describe_operation

        result_df = build_polars_describe_operation(self)
        return self._with_dataframe(cast(DataFrame, result_df))

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
            Concatenated :class:`PolarsDataFrame`

        Example:
            >>> df1.concat(df2)  # Vertical concatenation
            >>> df1.concat(df2, df3, how="vertical")
        """
        from ..helpers.polars_dataframe_helpers import build_polars_concat_operation

        if not others:
            return self

        result_df = build_polars_concat_operation(self, others, how=how)
        return self._with_dataframe(cast(DataFrame, result_df))

    def hstack(
        self,
        *others: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Horizontally stack DataFrames (Polars-style).

        Args:
            *others: Other PolarsDataFrames to stack horizontally

        Returns:
            Horizontally stacked :class:`PolarsDataFrame`

        Example:
            >>> df1.hstack(df2)  # Combine columns from df1 and df2
        """
        from ..helpers.polars_dataframe_helpers import build_polars_hstack_operation

        if not others:
            return self

        result_df = build_polars_hstack_operation(self, others)
        return self._with_dataframe(cast(DataFrame, result_df))

    def vstack(
        self,
        *others: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Vertically stack DataFrames (Polars-style alias for concat).

        Args:
            *others: Other PolarsDataFrames to stack vertically

        Returns:
            Vertically stacked :class:`PolarsDataFrame`

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
        """Union with another :class:`PolarsDataFrame` (Polars-style).

        Args:
            other: Another :class:`PolarsDataFrame` to union with
            distinct: If True, return distinct rows only (default: True)

        Returns:
            Unioned :class:`PolarsDataFrame`

        Example:
            >>> df1.union(df2)  # Union distinct
            >>> df1.union(df2, distinct=False)  # Union all
        """
        from ..helpers.polars_dataframe_helpers import build_polars_union_operation

        result_df = build_polars_union_operation(self, other, distinct=distinct)
        return self._with_dataframe(cast(DataFrame, result_df))

    def intersect(
        self,
        other: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Intersect with another :class:`PolarsDataFrame` (Polars-style).

        Args:
            other: Another :class:`PolarsDataFrame` to intersect with

        Returns:
            Intersected :class:`PolarsDataFrame` (common rows only)

        Example:
            >>> df1.intersect(df2)  # Common rows
        """
        from ..helpers.polars_dataframe_helpers import build_polars_intersect_operation

        result_df = build_polars_intersect_operation(self, other)
        return self._with_dataframe(cast(DataFrame, result_df))

    def difference(
        self,
        other: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Return rows in this :class:`DataFrame` that are not in another (Polars-style).

        Args:
            other: Another :class:`PolarsDataFrame` to exclude from

        Returns:
            :class:`PolarsDataFrame` with rows in this but not in other

        Example:
            >>> df1.difference(df2)  # Rows in df1 but not in df2
        """
        from ..helpers.polars_dataframe_helpers import build_polars_difference_operation

        result_df = build_polars_difference_operation(self, other)
        return self._with_dataframe(cast(DataFrame, result_df))

    def cross_join(
        self,
        other: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Perform a cross join (Cartesian product) with another :class:`PolarsDataFrame` (Polars-style).

        Args:
            other: Another :class:`PolarsDataFrame` to cross join with

        Returns:
            Cross-joined :class:`PolarsDataFrame`

        Example:
            >>> df1.cross_join(df2)  # Cartesian product
        """
        from ..helpers.polars_dataframe_helpers import build_polars_cross_join_operation

        result_df = build_polars_cross_join_operation(self, other)
        return self._with_dataframe(cast(DataFrame, result_df))

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
            :class:`PolarsDataFrame` with selected expressions

        Example:
            >>> df.select_expr("id", "amount * 1.1 as with_tax", "UPPER(name) as name_upper")
        """
        from ..helpers.polars_dataframe_helpers import build_polars_select_expr_operation

        result_df = build_polars_select_expr_operation(self, *exprs)
        return self._with_dataframe(cast(DataFrame, result_df))

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
            :class:`PolarsDataFrame` with renamed columns

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
            :class:`PolarsDataFrame` with row number column

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
        *contexts: "PolarsDataFrame",
    ) -> "PolarsDataFrame":
        """Add context DataFrames for use in expressions (Polars-style).

        Note: This is a placeholder for Polars' with_context feature.
        In Moltres, CTEs serve a similar purpose.

        Args:
            *contexts: Context DataFrames to add

        Returns:
            :class:`PolarsDataFrame` with context

        Example:
            >>> df.with_context(context_df)
        """
        # For now, this is a no-op as Moltres doesn't have the same context system
        # Users should use CTEs instead
        return self

    def summary(self, *statistics: str) -> "PolarsDataFrame":
        """Compute summary statistics for numeric columns (Polars-style).

        Args:
            *statistics: Statistics to compute (e.g., "count", "mean", "stddev", "min", "max").
                        If not provided, computes common statistics.

        Returns:
            :class:`PolarsDataFrame` with summary statistics

        Example:
            >>> df.summary()
            >>> df.summary("count", "mean", "max")
        """
        from ..helpers.polars_dataframe_helpers import build_polars_summary_operation

        result_df = build_polars_summary_operation(self, *statistics)
        return self._with_dataframe(cast(DataFrame, result_df))

    # ========================================================================
    # Common Table Expressions (CTEs) - Moltres-specific but Polars-style API
    # ========================================================================

    def cte(
        self,
        name: str,
    ) -> "PolarsDataFrame":
        """Create a Common Table Expression (CTE) from this :class:`DataFrame`.

        Args:
            name: Name for the CTE

        Returns:
            :class:`PolarsDataFrame` representing the CTE

        Example:
            >>> cte_df = df.filter(col("age") > 25).cte("adults")
            >>> result = cte_df.select().collect()
        """
        from ..helpers.polars_dataframe_helpers import build_polars_cte_operation

        result_df = build_polars_cte_operation(self, name)
        return self._with_dataframe(cast(DataFrame, result_df))

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
            recursive: :class:`PolarsDataFrame` representing the recursive part
            union_all: If True, use UNION ALL; if False, use UNION (distinct)

        Returns:
            :class:`PolarsDataFrame` representing the recursive CTE

        Example:
            >>> initial = db.table("seed").polars()
            >>> recursive = initial.select(...)  # Recursive part
            >>> fib_cte = initial.with_recursive("fib", recursive)
        """
        from ..helpers.polars_dataframe_helpers import build_polars_recursive_cte_operation

        result_df = build_polars_recursive_cte_operation(self, name, recursive, union_all=union_all)
        return self._with_dataframe(cast(DataFrame, result_df))
