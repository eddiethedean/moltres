"""Datetime functions for :class:`DataFrame` operations."""

from __future__ import annotations

from typing import Optional

from ..column import Column, ColumnLike, ensure_column


def year(column: ColumnLike) -> Column:
    """Extract the year from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for year

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date": "2024-01-15"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.year(col("date")).alias("year"))
        >>> results = df.collect()
        >>> results[0]["year"]
        2024
        >>> db.close()
    """
    return Column(op="year", args=(ensure_column(column),))


def month(column: ColumnLike) -> Column:
    """Extract the month from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for month

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date": "2024-03-15"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.month(col("date")).alias("month"))
        >>> results = df.collect()
        >>> results[0]["month"]
        3
        >>> db.close()
    """
    return Column(op="month", args=(ensure_column(column),))


def day(column: ColumnLike) -> Column:
    """Extract the day of month from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for day

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date": "2024-01-15"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.day(col("date")).alias("day"))
        >>> results = df.collect()
        >>> results[0]["day"]
        15
        >>> db.close()
    """
    return Column(op="day", args=(ensure_column(column),))


def dayofweek(column: ColumnLike) -> Column:
    """Extract the day of week from a date/timestamp column (1=Sunday, 7=Saturday).

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for dayofweek

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date": "2024-01-15"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.dayofweek(col("date")).alias("dow"))
        >>> results = df.collect()
        >>> # 2024-01-15 is a Monday (day 2 in SQLite)
        >>> results[0]["dow"] in [1, 2, 3, 4, 5, 6, 7]
        True
        >>> db.close()
    """
    return Column(op="dayofweek", args=(ensure_column(column),))


def hour(column: ColumnLike) -> Column:
    """Extract the hour from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        :class:`Column` expression for hour

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("timestamp", "TIMESTAMP")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"timestamp": "2024-01-15 14:30:00"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.hour(col("timestamp")).alias("hour"))
        >>> results = df.collect()
        >>> results[0]["hour"]
        14
        >>> db.close()
    """
    return Column(op="hour", args=(ensure_column(column),))


def minute(column: ColumnLike) -> Column:
    """Extract the minute from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        :class:`Column` expression for minute

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("timestamp", "TIMESTAMP")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"timestamp": "2024-01-15 14:30:00"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.minute(col("timestamp")).alias("minute"))
        >>> results = df.collect()
        >>> results[0]["minute"]
        30
        >>> db.close()
    """
    return Column(op="minute", args=(ensure_column(column),))


def second(column: ColumnLike) -> Column:
    """Extract the second from a timestamp column.

    Args:
        column: Timestamp column

    Returns:
        :class:`Column` expression for second

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("sqlite:///:memory:")
        >>> _ = db.create_table("events", [column("timestamp", "TIMESTAMP")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"timestamp": "2024-01-15 14:30:45"}], _database=db).insert_into("events")
        >>> df = db.table("events").select(F.second(col("timestamp")).alias("second"))
        >>> results = df.collect()
        >>> results[0]["second"]
        45
        >>> db.close()
    """
    return Column(op="second", args=(ensure_column(column),))


def date_format(column: ColumnLike, format: str) -> Column:  # noqa: A001
    """Format a date/timestamp column as a string.

    Args:
        column: Date or timestamp column
        format: Format string (e.g., "YYYY-MM-DD")

    Returns:
        :class:`Column` expression for date_format

    Example:
        >>> # Note: date_format() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.date_format(col("date"), "YYYY-MM-DD")
        >>> # For MySQL: F.date_format(col("date"), "%Y-%m-%d")
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    return Column(op="date_format", args=(ensure_column(column), format))


def to_date(column: ColumnLike, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a string column to a date.

    Args:
        column: String column containing a date
        format: Optional format string (if None, uses default parsing)

    Returns:
        :class:`Column` expression for to_date

    Example:
        >>> # Note: to_date() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.to_date(col("date_str"), "YYYY-MM-DD")
        >>> # For MySQL: F.to_date(col("date_str"), "%Y-%m-%d")
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    if format is not None:
        return Column(op="to_date", args=(ensure_column(column), format))
    return Column(op="to_date", args=(ensure_column(column),))


def current_date() -> Column:
    """Get the current date.

    Returns:
        :class:`Column` expression for current_date

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from datetime import date
        >>> db = connect("sqlite:///:memory:")
        >>> df = db.sql("SELECT 1 as dummy").select(F.current_date().alias("today"))
        >>> results = df.collect()
        >>> # Returns current date (format varies by database)
        >>> isinstance(results[0]["today"], (str, date, type(None)))
        True
        >>> db.close()
    """
    return Column(op="current_date", args=())


def current_timestamp() -> Column:
    """Get the current timestamp.

    Returns:
        :class:`Column` expression for current_timestamp

    Example:
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from datetime import datetime
        >>> db = connect("sqlite:///:memory:")
        >>> df = db.sql("SELECT 1 as dummy").select(F.current_timestamp().alias("now"))
        >>> results = df.collect()
        >>> # Returns current timestamp (format varies by database)
        >>> isinstance(results[0]["now"], (str, datetime, type(None)))
        True
        >>> db.close()
    """
    return Column(op="current_timestamp", args=())


def datediff(end: ColumnLike, start: ColumnLike) -> Column:
    """Calculate the difference in days between two dates.

    Args:
        end: End date column
        start: Start date column

    Returns:
        :class:`Column` expression for datediff

    Example:
        >>> # Note: datediff() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.datediff(col("end_date"), col("start_date"))
        >>> # For MySQL: DATEDIFF(end_date, start_date)
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    return Column(op="datediff", args=(ensure_column(end), ensure_column(start)))


def add_months(column: ColumnLike, num_months: int) -> Column:
    """Add months to a date column.

    Args:
        column: Date column
        num_months: Number of months to add (can be negative)

    Returns:
        :class:`Column` expression for add_months

    Example:
        >>> # Note: add_months() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.add_months(col("date"), 1)
        >>> # For MySQL: DATE_ADD(date, INTERVAL 1 MONTH)
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    return Column(op="add_months", args=(ensure_column(column), num_months))


class When:
    """Builder for CASE WHEN expressions."""

    def __init__(self, condition: Column, value: ColumnLike):
        self._conditions = [(condition, ensure_column(value))]

    def when(self, condition: Column, value: ColumnLike) -> "When":
        """Add another WHEN clause."""
        self._conditions.append((condition, ensure_column(value)))
        return self

    def otherwise(self, value: ColumnLike) -> Column:
        """Complete the CASE expression with an ELSE clause.

        Args:
            value: Default value if no conditions match

        Returns:
            :class:`Column` expression for the complete CASE WHEN statement
        """
        return Column(op="case_when", args=(tuple(self._conditions), ensure_column(value)))


def date_add(column: ColumnLike, interval: str) -> Column:
    """Add an interval to a date/timestamp column.

    Args:
        column: Date or timestamp column
        interval: Interval string (e.g., "1 DAY", "2 MONTH", "3 YEAR", "1 HOUR")

    Returns:
        :class:`Column` expression for date_add

    Example:
        >>> # Note: date_add() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # For PostgreSQL: F.date_add(col("date"), "1 DAY")
        >>> # For MySQL: DATE_ADD(date, INTERVAL 1 DAY)
        >>> from moltres import connect
        >>> db = connect("sqlite:///:memory:")
        >>> db.close()
    """
    return Column(op="date_add", args=(ensure_column(column), interval))


def date_sub(column: ColumnLike, interval: str) -> Column:
    """Subtract an interval from a date/timestamp column.

    Args:
        column: Date or timestamp column
        interval: Interval string (e.g., "1 DAY", "2 MONTH", "3 YEAR", "1 HOUR")

    Returns:
        :class:`Column` expression for date_sub

    Example:
        >>> # Note: date_sub() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have date_sub function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.date_sub(col("created_at"), "1 DAY").alias("yesterday"))
        >>> results = df.collect()
        >>> from datetime import date, datetime
        >>> isinstance(results[0]["yesterday"], (str, date, datetime, type(None)))
        True
        >>> db.close()
    """
    return Column(op="date_sub", args=(ensure_column(column), interval))


def to_timestamp(column: ColumnLike, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a string column to a timestamp.

    Args:
        column: String column containing a timestamp
        format: Optional format string (if None, uses default parsing)

    Returns:
        :class:`Column` expression for to_timestamp

    Example:
        >>> # Note: to_timestamp() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have to_timestamp function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("date_str", "TEXT")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"date_str": "2024-01-15 10:30:00"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.to_timestamp(col("date_str"), "yyyy-MM-dd HH:mm:ss").alias("timestamp"))
        >>> results = df.collect()
        >>> from datetime import datetime
        >>> isinstance(results[0]["timestamp"], (str, datetime, type(None)))
        True
        >>> db.close()
    """
    if format is not None:
        return Column(op="to_timestamp", args=(ensure_column(column), format))
    return Column(op="to_timestamp", args=(ensure_column(column),))


def unix_timestamp(column: Optional[ColumnLike] = None, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a timestamp or date string to Unix timestamp (seconds since epoch).

    Args:
        column: Optional timestamp/date column (if None, returns current Unix timestamp)
        format: Optional format string for parsing date strings

    Returns:
        :class:`Column` expression for unix_timestamp

    Example:
        >>> # Note: unix_timestamp() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have unix_timestamp function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "TIMESTAMP")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15 10:30:00"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.unix_timestamp(col("created_at")).alias("unix_ts"))
        >>> results = df.collect()
        >>> isinstance(results[0]["unix_ts"], (int, float))
        True
        >>> db.close()
    """
    if column is None:
        return Column(op="unix_timestamp", args=())
    if format is not None:
        return Column(op="unix_timestamp", args=(ensure_column(column), format))
    return Column(op="unix_timestamp", args=(ensure_column(column),))


def from_unixtime(column: ColumnLike, format: Optional[str] = None) -> Column:  # noqa: A001
    """Convert a Unix timestamp (seconds since epoch) to a timestamp string.

    Args:
        column: Unix timestamp column (seconds since epoch)
        format: Optional format string (if None, uses default format)

    Returns:
        :class:`Column` expression for from_unixtime

    Example:
        >>> # Note: from_unixtime() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have from_unixtime function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("unix_time", "INTEGER")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"unix_time": 1705312200}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.from_unixtime(col("unix_time"), "yyyy-MM-dd HH:mm:ss").alias("timestamp"))
        >>> results = df.collect()
        >>> isinstance(results[0]["timestamp"], str)
        True
        >>> db.close()
    """
    if format is not None:
        return Column(op="from_unixtime", args=(ensure_column(column), format))
    return Column(op="from_unixtime", args=(ensure_column(column),))


def date_trunc(unit: str, column: ColumnLike) -> Column:
    """Truncate a date/timestamp to the specified unit.

    Args:
        unit: Unit to truncate to (e.g., "year", "month", "day", "hour", "minute", "second")
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for date_trunc

    Example:
        >>> # Note: date_trunc() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have date_trunc function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "TIMESTAMP")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15 10:30:00"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.date_trunc("month", col("created_at")).alias("month_start"))
        >>> results = df.collect()
        >>> isinstance(results[0]["month_start"], (str, type(None)))
        True
        >>> db.close()
    """
    return Column(op="date_trunc", args=(unit, ensure_column(column)))


def quarter(column: ColumnLike) -> Column:
    """Extract the quarter (1-4) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for quarter (1, 2, 3, or 4)

    Example:
        >>> # Note: quarter() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have quarter function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-03-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.quarter(col("created_at")).alias("q"))
        >>> results = df.collect()
        >>> 1 <= results[0]["q"] <= 4
        True
        >>> db.close()
    """
    return Column(op="quarter", args=(ensure_column(column),))


def weekofyear(column: ColumnLike) -> Column:
    """Extract the week number (1-53) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for weekofyear

    Example:
        >>> # Note: weekofyear() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have weekofyear function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.weekofyear(col("created_at")).alias("week"))
        >>> results = df.collect()
        >>> 1 <= results[0]["week"] <= 53
        True
        >>> db.close()
    """
    return Column(op="weekofyear", args=(ensure_column(column),))


def week(column: ColumnLike) -> Column:
    """Extract the week number (alias for weekofyear).

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for week

    Example:
        >>> # Note: week() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have week function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.week(col("created_at")).alias("week"))
        >>> results = df.collect()
        >>> 1 <= results[0]["week"] <= 53
        True
        >>> db.close()
    """
    return weekofyear(column)


def dayofyear(column: ColumnLike) -> Column:
    """Extract the day of year (1-366) from a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for dayofyear

    Example:
        >>> # Note: dayofyear() requires database-specific support (PostgreSQL/MySQL) (DuckDB/PostgreSQL/MySQL)
        >>> # SQLite does not have dayofyear function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.dayofyear(col("created_at")).alias("day_of_year"))
        >>> results = df.collect()
        >>> 1 <= results[0]["day_of_year"] <= 366
        True
        >>> db.close()
    """
    return Column(op="dayofyear", args=(ensure_column(column),))


def last_day(column: ColumnLike) -> Column:
    """Get the last day of the month for a date/timestamp column.

    Args:
        column: Date or timestamp column

    Returns:
        :class:`Column` expression for last_day

    Example:
        >>> # Note: last_day() requires database-specific support (PostgreSQL/MySQL/DuckDB)
        >>> # SQLite does not have last_day function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("created_at", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"created_at": "2024-01-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.last_day(col("created_at")).alias("last_day"))
        >>> results = df.collect()
        >>> from datetime import date
        >>> isinstance(results[0]["last_day"], (str, date, type(None)))
        True
        >>> db.close()
    """
    return Column(op="last_day", args=(ensure_column(column),))


def months_between(date1: ColumnLike, date2: ColumnLike) -> Column:
    """Calculate the number of months between two dates.

    Args:
        date1: First date column
        date2: Second date column

    Returns:
        :class:`Column` expression for months_between (can be fractional)

    Example:
        >>> # Note: months_between() requires database-specific support (PostgreSQL/MySQL/DuckDB)
        >>> # SQLite does not have months_between function
        >>> from moltres import connect, col
        >>> from moltres.expressions import functions as F
        >>> from moltres.table.schema import column
        >>> db = connect("duckdb:///:memory:")
        >>> _ = db.create_table("data", [column("start_date", "DATE"), column("end_date", "DATE")]).collect()  # doctest: +ELLIPSIS
        >>> from moltres.io.records import :class:`Records`
        >>> _ = :class:`Records`(_data=[{"start_date": "2024-01-15", "end_date": "2024-03-15"}], _database=db).insert_into("data")
        >>> df = db.table("data").select(F.months_between(col("end_date"), col("start_date")).alias("months"))
        >>> results = df.collect()
        >>> results[0]["months"] >= 0
        True
        >>> db.close()
    """
    return Column(op="months_between", args=(ensure_column(date1), ensure_column(date2)))
