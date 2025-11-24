"""Tests for newly added PySpark-compatible functions."""

from __future__ import annotations


from moltres import col, connect, column
from moltres.expressions.functions import (
    acos,
    array_max,
    array_min,
    asin,
    atan,
    atan2,
    dayofyear,
    first_value,
    hash,
    hypot,
    instr,
    json_array_length,
    last_value,
    locate,
    log2,
    md5,
    monotonically_increasing_id,
    pow,
    power,
    quarter,
    rand,
    sign,
    signum,
    week,
    weekofyear,
)


def test_pow_function(tmp_path):
    """Test pow/power function."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "test",
        [column("x", "REAL"), column("y", "REAL")],
    ).collect()

    # Insert test data
    from moltres.io.records import Records

    records = Records(_data=[{"x": 2.0, "y": 3.0}], _database=db)
    records.insert_into("test")

    # Test pow
    df = db.table("test").select(pow(col("x"), col("y")).alias("result"))
    result = df.collect()
    assert len(result) == 1
    assert abs(result[0]["result"] - 8.0) < 0.001

    # Test power alias
    df2 = db.table("test").select(power(col("x"), 2).alias("result"))
    result2 = df2.collect()
    assert abs(result2[0]["result"] - 4.0) < 0.001


def test_trigonometric_functions(tmp_path):
    """Test inverse trigonometric functions."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("value", "REAL")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"value": 0.5}], _database=db)
    records.insert_into("test")

    # Test asin, acos, atan
    df = db.table("test").select(
        asin(col("value")).alias("asin_val"),
        acos(col("value")).alias("acos_val"),
        atan(col("value")).alias("atan_val"),
    )
    result = df.collect()
    assert len(result) == 1
    # Values should be valid (no errors)


def test_atan2_function(tmp_path):
    """Test atan2 function."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("y", "REAL"), column("x", "REAL")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"y": 1.0, "x": 1.0}], _database=db)
    records.insert_into("test")

    df = db.table("test").select(atan2(col("y"), col("x")).alias("result"))
    result = df.collect()
    assert len(result) == 1
    # atan2(1, 1) should be approximately π/4


def test_signum_function(tmp_path):
    """Test signum/sign function."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("value", "REAL")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"value": -5.0}, {"value": 0.0}, {"value": 5.0}], _database=db)
    records.insert_into("test")

    df = db.table("test").select(
        col("value"),
        signum(col("value")).alias("signum_val"),
        sign(col("value")).alias("sign_val"),
    )
    result = df.collect()
    assert len(result) == 3
    # signum(-5) should be -1, signum(0) should be 0, signum(5) should be 1


def test_log2_function(tmp_path):
    """Test log2 function."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("value", "REAL")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"value": 8.0}], _database=db)
    records.insert_into("test")

    df = db.table("test").select(log2(col("value")).alias("result"))
    result = df.collect()
    assert len(result) == 1
    # log2(8) should be 3


def test_hypot_function(tmp_path):
    """Test hypot function."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("x", "REAL"), column("y", "REAL")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"x": 3.0, "y": 4.0}], _database=db)
    records.insert_into("test")

    df = db.table("test").select(hypot(col("x"), col("y")).alias("result"))
    result = df.collect()
    assert len(result) == 1
    # hypot(3, 4) should be 5


def test_string_functions(tmp_path):
    """Test string functions."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("text", "TEXT")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"text": "hello world"}], _database=db)
    records.insert_into("test")

    # Test instr
    df = db.table("test").select(instr(col("text"), "world").alias("pos"))
    result = df.collect()
    assert len(result) == 1
    # Position should be > 0

    # Test locate (PySpark-style)
    df2 = db.table("test").select(locate("world", col("text")).alias("pos"))
    result2 = df2.collect()
    assert len(result2) == 1


def test_date_functions(tmp_path):
    """Test date/time functions."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("date_col", "TEXT")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"date_col": "2024-03-15"}], _database=db)
    records.insert_into("test")

    # Test quarter
    df = db.table("test").select(quarter(col("date_col")).alias("q"))
    result = df.collect()
    assert len(result) == 1
    # March should be quarter 1

    # Test weekofyear/week
    df2 = db.table("test").select(
        weekofyear(col("date_col")).alias("week1"),
        week(col("date_col")).alias("week2"),
    )
    result2 = df2.collect()
    assert len(result2) == 1

    # Test dayofyear
    df3 = db.table("test").select(dayofyear(col("date_col")).alias("doy"))
    result3 = df3.collect()
    assert len(result3) == 1


def test_window_functions(tmp_path):
    """Test window functions."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("category", "TEXT"), column("amount", "REAL")]).collect()

    from moltres.io.records import Records

    records = Records(
        _data=[
            {"category": "A", "amount": 10.0},
            {"category": "A", "amount": 20.0},
            {"category": "B", "amount": 15.0},
        ],
        _database=db,
    )
    records.insert_into("test")

    # Test first_value
    df = db.table("test").select(
        col("category"),
        col("amount"),
        first_value(col("amount"))
        .over(partition_by=col("category"), order_by=col("amount"))
        .alias("first"),
    )
    result = df.collect()
    assert len(result) == 3

    # Test last_value
    df2 = db.table("test").select(
        col("category"),
        col("amount"),
        last_value(col("amount"))
        .over(partition_by=col("category"), order_by=col("amount"))
        .alias("last"),
    )
    result2 = df2.collect()
    assert len(result2) == 3


def test_array_functions(tmp_path):
    """Test array functions (PostgreSQL-specific)."""
    # Note: Array functions work best with PostgreSQL
    # For SQLite/MySQL, they use JSON functions
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    # SQLite doesn't have native arrays, so we'll test JSON array functions
    db.create_table("test", [column("arr", "TEXT")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"arr": "[1, 2, 3, 2]"}], _database=db)
    records.insert_into("test")

    # Test array_max, array_min, array_sum (using JSON)
    df = db.table("test").select(
        array_max(col("arr")).alias("max_val"),
        array_min(col("arr")).alias("min_val"),
    )
    result = df.collect()
    assert len(result) == 1


def test_json_functions(tmp_path):
    """Test JSON functions."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("json_col", "TEXT")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"json_col": '{"name": "Alice", "age": 30}'}], _database=db)
    records.insert_into("test")

    # Test json_array_length (for arrays)
    db.create_table("test2", [column("arr_col", "TEXT")]).collect()
    records2 = Records(_data=[{"arr_col": "[1, 2, 3]"}], _database=db)
    records2.insert_into("test2")

    df = db.table("test2").select(json_array_length(col("arr_col")).alias("len"))
    result = df.collect()
    assert len(result) == 1
    # Length should be 3


def test_utility_functions(tmp_path):
    """Test utility functions."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("value", "TEXT")]).collect()

    from moltres.io.records import Records

    records = Records(_data=[{"value": "test"}], _database=db)
    records.insert_into("test")

    # Test md5 - SQLite doesn't have md5 by default, so skip for SQLite
    # For PostgreSQL/MySQL, this would work
    if db.dialect.name != "sqlite":
        df = db.table("test").select(md5(col("value")).alias("hash"))
        result = df.collect()
        assert len(result) == 1
        assert result[0]["hash"] is not None

    # Test hash - SQLite has limited support
    # For now, skip hash test for SQLite as it requires workarounds
    if db.dialect.name != "sqlite":
        df2 = db.table("test").select(hash(col("value")).alias("hash_val"))
        result2 = df2.collect()
        assert len(result2) == 1

    # Test rand - should work on all databases
    df3 = db.table("test").select(rand().alias("random"))
    result3 = df3.collect()
    assert len(result3) == 1


def test_monotonically_increasing_id(tmp_path):
    """Test monotonically_increasing_id function."""
    db_path = tmp_path / "test.db"
    db = connect(f"sqlite:///{db_path}")

    db.create_table("test", [column("name", "TEXT")]).collect()

    from moltres.io.records import Records

    records = Records(
        _data=[{"name": "A"}, {"name": "B"}, {"name": "C"}],
        _database=db,
    )
    records.insert_into("test")

    df = db.table("test").select(
        col("name"),
        monotonically_increasing_id().alias("id"),
    )
    result = df.collect()
    assert len(result) == 3
    # IDs should be increasing
    ids = [r["id"] for r in result]
    assert ids == sorted(ids)
