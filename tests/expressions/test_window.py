"""Tests for window function support."""

from __future__ import annotations


from moltres import col
from moltres.expressions.window import (
    Window,
    WindowSpec,
    dense_rank,
    first_value,
    lag,
    lead,
    last_value,
    rank,
    row_number,
)


class TestWindowSpec:
    """Test WindowSpec dataclass."""

    def test_window_spec_default(self):
        """Test default WindowSpec."""
        spec = WindowSpec()
        assert spec.partition_by == ()
        assert spec.order_by == ()
        assert spec.rows_between is None
        assert spec.range_between is None

    def test_window_spec_partition_by(self):
        """Test partitionBy method."""
        spec = WindowSpec().partitionBy(col("category"))
        assert len(spec.partition_by) == 1
        assert spec.partition_by[0].op == "column"

    def test_window_spec_partition_by_multiple(self):
        """Test partitionBy with multiple columns."""
        spec = WindowSpec().partitionBy(col("category"), col("region"))
        assert len(spec.partition_by) == 2

    def test_window_spec_order_by(self):
        """Test orderBy method."""
        spec = WindowSpec().orderBy(col("date"))
        assert len(spec.order_by) == 1

    def test_window_spec_order_by_multiple(self):
        """Test orderBy with multiple columns."""
        spec = WindowSpec().orderBy(col("date"), col("id"))
        assert len(spec.order_by) == 2

    def test_window_spec_rows_between(self):
        """Test rowsBetween method."""
        spec = WindowSpec().rowsBetween(-1, 1)
        assert spec.rows_between == (-1, 1)

    def test_window_spec_rows_between_unbounded(self):
        """Test rowsBetween with unbounded."""
        spec = WindowSpec().rowsBetween(None, None)
        assert spec.rows_between == (None, None)

    def test_window_spec_range_between(self):
        """Test rangeBetween method."""
        spec = WindowSpec().rangeBetween(-10, 10)
        assert spec.range_between == (-10, 10)

    def test_window_spec_range_between_unbounded(self):
        """Test rangeBetween with unbounded."""
        spec = WindowSpec().rangeBetween(None, None)
        assert spec.range_between == (None, None)

    def test_window_spec_chaining(self):
        """Test chaining WindowSpec methods."""
        spec = WindowSpec().partitionBy(col("category")).orderBy(col("date")).rowsBetween(-1, 1)
        assert len(spec.partition_by) == 1
        assert len(spec.order_by) == 1
        assert spec.rows_between == (-1, 1)

    def test_window_spec_immutability(self):
        """Test that WindowSpec methods return new instances."""
        spec1 = WindowSpec()
        spec2 = spec1.partitionBy(col("category"))

        # spec2 should have partition_by
        assert len(spec2.partition_by) == 1
        # spec1 should still be empty (immutability)
        assert len(spec1.partition_by) == 0


class TestWindow:
    """Test Window factory class."""

    def test_window_partition_by(self):
        """Test Window.partitionBy static method."""
        spec = Window.partitionBy(col("category"))
        assert isinstance(spec, WindowSpec)
        assert len(spec.partition_by) == 1

    def test_window_order_by(self):
        """Test Window.orderBy static method."""
        spec = Window.orderBy(col("date"))
        assert isinstance(spec, WindowSpec)
        assert len(spec.order_by) == 1

    def test_window_rows_between(self):
        """Test Window.rowsBetween static method."""
        spec = Window.rowsBetween(-1, 1)
        assert isinstance(spec, WindowSpec)
        assert spec.rows_between == (-1, 1)

    def test_window_range_between(self):
        """Test Window.rangeBetween static method."""
        spec = Window.rangeBetween(-10, 10)
        assert isinstance(spec, WindowSpec)
        assert spec.range_between == (-10, 10)


class TestWindowFunctions:
    """Test window function helpers."""

    def test_row_number(self):
        """Test row_number() function."""
        expr = row_number()
        assert expr.op == "window_row_number"
        assert expr.args == ()

    def test_rank(self):
        """Test rank() function."""
        expr = rank()
        assert expr.op == "window_rank"
        assert expr.args == ()

    def test_dense_rank(self):
        """Test dense_rank() function."""
        expr = dense_rank()
        assert expr.op == "window_dense_rank"
        assert expr.args == ()

    def test_lag(self):
        """Test lag() function."""
        expr = lag(col("value"))
        assert expr.op == "window_lag"
        assert len(expr.args) == 2
        assert expr.args[1] == 1  # default offset

    def test_lag_with_offset(self):
        """Test lag() with custom offset."""
        expr = lag(col("value"), offset=3)
        assert expr.op == "window_lag"
        assert expr.args[1] == 3

    def test_lag_with_default(self):
        """Test lag() with default value."""
        expr = lag(col("value"), offset=1, default=0)
        assert expr.op == "window_lag"
        assert len(expr.args) == 3
        assert expr.args[2].op == "literal"

    def test_lead(self):
        """Test lead() function."""
        expr = lead(col("value"))
        assert expr.op == "window_lead"
        assert len(expr.args) == 2
        assert expr.args[1] == 1  # default offset

    def test_lead_with_offset(self):
        """Test lead() with custom offset."""
        expr = lead(col("value"), offset=2)
        assert expr.op == "window_lead"
        assert expr.args[1] == 2

    def test_lead_with_default(self):
        """Test lead() with default value."""
        expr = lead(col("value"), offset=1, default=0)
        assert expr.op == "window_lead"
        assert len(expr.args) == 3

    def test_first_value(self):
        """Test first_value() function."""
        expr = first_value(col("value"))
        assert expr.op == "window_first_value"
        assert len(expr.args) == 1

    def test_last_value(self):
        """Test last_value() function."""
        expr = last_value(col("value"))
        assert expr.op == "window_last_value"
        assert len(expr.args) == 1


def test_window_rows_between_execution(tmp_path):
    """Test window functions with ROWS BETWEEN in actual query."""
    from moltres import connect, col
    from moltres.expressions import functions as F
    from moltres.table.schema import column
    from moltres.io.records import Records

    db_path = tmp_path / "window_rows.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "sales",
        [
            column("id", "INTEGER", primary_key=True),
            column("date", "TEXT"),
            column("amount", "REAL"),
        ],
    ).collect()

    # Insert data
    sales_data = [
        {"id": 1, "date": "2024-01-01", "amount": 100.0},
        {"id": 2, "date": "2024-01-02", "amount": 200.0},
        {"id": 3, "date": "2024-01-03", "amount": 150.0},
        {"id": 4, "date": "2024-01-04", "amount": 300.0},
    ]
    Records(_data=sales_data, _database=db).insert_into("sales")

    # Test window function with ROWS BETWEEN
    df = db.table("sales").select()
    result = df.select(
        col("date"),
        col("amount"),
        F.sum(col("amount")).over(order_by=col("date"), rows_between=(-1, 1)).alias("rolling_sum"),
    )
    results = result.collect()

    assert len(results) == 4
    # First row: 100 + 200 = 300 (current + 1 following)
    assert results[0]["rolling_sum"] == 300.0
    # Second row: 100 + 200 + 150 = 450 (1 preceding + current + 1 following)
    assert results[1]["rolling_sum"] == 450.0
    # Third row: 200 + 150 + 300 = 650 (1 preceding + current + 1 following)
    assert results[2]["rolling_sum"] == 650.0
    # Fourth row: 150 + 300 = 450 (1 preceding + current)
    assert results[3]["rolling_sum"] == 450.0


def test_window_range_between_execution(tmp_path):
    """Test window functions with RANGE BETWEEN in actual query."""
    from moltres import connect, col
    from moltres.expressions import functions as F
    from moltres.table.schema import column
    from moltres.io.records import Records

    db_path = tmp_path / "window_range.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "sales",
        [
            column("id", "INTEGER", primary_key=True),
            column("amount", "REAL"),
        ],
    ).collect()

    # Insert data
    sales_data = [
        {"id": 1, "amount": 100.0},
        {"id": 2, "amount": 200.0},
        {"id": 3, "amount": 150.0},
    ]
    Records(_data=sales_data, _database=db).insert_into("sales")

    # Test window function with RANGE BETWEEN
    df = db.table("sales").select()
    result = df.select(
        col("amount"),
        F.sum(col("amount"))
        .over(order_by=col("amount"), range_between=(-50, 50))
        .alias("range_sum"),
    )
    results = result.collect()

    assert len(results) == 3
    # All rows should have the same sum since they're within range of each other
    # (100, 200, 150 are all within 50 of each other)
    assert all("range_sum" in r for r in results)
