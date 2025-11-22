"""Tests for table schema creation and DDL operations."""

from moltres import column, connect
from moltres.engine.dialects import get_dialect
from moltres.sql.ddl import compile_create_table, compile_drop_table
from moltres.table.schema import TableSchema


def test_column_def_creation():
    col_def = column("id", "INTEGER", nullable=False, primary_key=True)
    assert col_def.name == "id"
    assert col_def.type_name == "INTEGER"
    assert col_def.nullable is False
    assert col_def.primary_key is True


def test_compile_create_table_basic():
    schema = TableSchema(
        name="users",
        columns=[
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "TEXT", nullable=False),
            column("email", "TEXT", nullable=True),
        ],
    )
    dialect = get_dialect("sqlite")
    sql = compile_create_table(schema, dialect)

    assert "CREATE TABLE" in sql
    assert '"users"' in sql
    assert "IF NOT EXISTS" in sql
    assert '"id"' in sql
    assert '"name"' in sql
    assert '"email"' in sql
    assert "INTEGER" in sql
    assert "TEXT" in sql
    assert "NOT NULL" in sql
    assert "PRIMARY KEY" in sql


def test_compile_create_table_temporary():
    schema = TableSchema(
        name="temp_data",
        columns=[column("value", "TEXT")],
        temporary=True,
    )
    dialect = get_dialect("sqlite")
    sql = compile_create_table(schema, dialect)

    assert "CREATE TEMPORARY TABLE" in sql


def test_compile_create_table_with_defaults():
    schema = TableSchema(
        name="products",
        columns=[
            column("id", "INTEGER", primary_key=True),
            column("price", "REAL", default=0.0),
            column("active", "INTEGER", default=1),
        ],
    )
    dialect = get_dialect("sqlite")
    sql = compile_create_table(schema, dialect)

    assert "DEFAULT 0.0" in sql
    assert "DEFAULT 1" in sql


def test_compile_drop_table():
    dialect = get_dialect("sqlite")
    sql = compile_drop_table("users", dialect)

    assert sql == 'DROP TABLE IF EXISTS "users"'


def test_compile_drop_table_without_if_exists():
    dialect = get_dialect("sqlite")
    sql = compile_drop_table("users", dialect, if_exists=False)

    assert sql == 'DROP TABLE "users"'


def test_create_table_integration(tmp_path):
    db_path = tmp_path / "schema_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    table = db.create_table(
        "customers",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "TEXT", nullable=False),
            column("email", "TEXT", nullable=True),
        ],
    )

    assert table.name == "customers"

    table.insert([{"id": 1, "name": "Alice", "email": "alice@example.com"}])
    rows = table.select().collect()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_create_table_with_defaults_integration(tmp_path):
    db_path = tmp_path / "defaults_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    table = db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT", nullable=False),
            column("price", "REAL", default=0.0),
            column("active", "INTEGER", default=1),
        ],
    )

    table.insert([{"id": 1, "name": "Widget"}])
    rows = table.select().collect()
    assert rows[0]["price"] == 0.0
    assert rows[0]["active"] == 1


def test_decimal_type_with_precision_scale(tmp_path):
    """Test DECIMAL type with precision and scale."""
    from moltres.table.schema import decimal, column

    db_path = tmp_path / "decimal_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Test using decimal() helper
    table = db.create_table(
        "prices",
        [
            decimal("id", precision=10, scale=0, primary_key=True),
            decimal("price", precision=10, scale=2),
            decimal("discount", precision=5, scale=2),
        ],
    )

    # Insert test data
    table.insert(
        [
            {"id": 1, "price": 99.99, "discount": 0.10},
            {"id": 2, "price": 149.50, "discount": 0.15},
        ]
    )

    # Verify data
    result = table.select().collect()
    assert len(result) == 2
    assert result[0]["price"] == 99.99
    assert result[1]["price"] == 149.50

    # Test using column() with precision/scale
    table2 = db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("cost", "DECIMAL", precision=10, scale=2),
            column("tax_rate", "NUMERIC", precision=5, scale=4),
        ],
    )

    table2.insert(
        [
            {"id": 1, "cost": 100.50, "tax_rate": 0.0825},
        ]
    )

    result2 = table2.select().collect()
    assert len(result2) == 1
    assert result2[0]["cost"] == 100.50


def test_uuid_type(tmp_path):
    """Test UUID type support."""
    from moltres.table.schema import uuid, column

    db_path = tmp_path / "uuid_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Test using uuid() helper (SQLite will use TEXT)
    table = db.create_table(
        "users",
        [
            uuid("id", primary_key=True),
            column("name", "TEXT"),
        ],
    )

    # Insert test data with UUID strings
    import uuid as uuid_module

    user_id = str(uuid_module.uuid4())
    table.insert(
        [
            {"id": user_id, "name": "Alice"},
        ]
    )

    # Verify data
    result = table.select().collect()
    assert len(result) == 1
    assert result[0]["id"] == user_id
    assert result[0]["name"] == "Alice"

    # Test using column() with UUID type
    table2 = db.create_table(
        "sessions",
        [
            column("id", "UUID", primary_key=True),
            column("user_id", "UUID"),
        ],
    )

    session_id = str(uuid_module.uuid4())
    user_id2 = str(uuid_module.uuid4())
    table2.insert(
        [
            {"id": session_id, "user_id": user_id2},
        ]
    )

    result2 = table2.select().collect()
    assert len(result2) == 1
    assert result2[0]["id"] == session_id
    assert result2[0]["user_id"] == user_id2


def test_json_type(tmp_path):
    """Test JSON/JSONB type support."""
    from moltres.table.schema import json, column

    db_path = tmp_path / "json_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Test using json() helper (SQLite will use TEXT)
    table = db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            json("data"),
            json("metadata", jsonb=False),  # Explicit JSON
        ],
    )

    # Insert test data with JSON strings
    import json as json_module

    product_data = json_module.dumps({"name": "Widget", "price": 10.5})
    metadata = json_module.dumps({"category": "electronics"})
    table.insert(
        [
            {"id": 1, "data": product_data, "metadata": metadata},
        ]
    )

    # Verify data
    result = table.select().collect()
    assert len(result) == 1
    assert result[0]["id"] == 1
    # Data should be stored as text (SQLite)
    assert "Widget" in result[0]["data"]

    # Test using column() with JSON type
    table2 = db.create_table(
        "config",
        [
            column("id", "INTEGER", primary_key=True),
            column("settings", "JSON"),
        ],
    )

    settings = json_module.dumps({"theme": "dark", "lang": "en"})
    table2.insert(
        [
            {"id": 1, "settings": settings},
        ]
    )

    result2 = table2.select().collect()
    assert len(result2) == 1
    assert "theme" in result2[0]["settings"]


def test_interval_type(tmp_path):
    """Test INTERVAL type support."""
    from moltres.table.schema import column
    from moltres.expressions.functions import date_add, date_sub

    db_path = tmp_path / "interval_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Test INTERVAL type in schema (SQLite will use TEXT)
    table = db.create_table(
        "events",
        [
            column("id", "INTEGER", primary_key=True),
            column("start_time", "TIMESTAMP"),
            column("duration", "INTERVAL"),  # Duration as interval
        ],
    )

    # Insert test data
    import datetime

    start = datetime.datetime(2024, 1, 1, 10, 0, 0)
    table.insert(
        [
            {"id": 1, "start_time": start, "duration": "1 HOUR"},
        ]
    )

    # Verify data
    result = table.select().collect()
    assert len(result) == 1

    # Test date_add and date_sub functions
    from moltres import col

    df = (
        db.table("events")
        .select()
        .select(
            col("id"),
            date_add(col("start_time"), "1 DAY").alias("next_day"),
            date_sub(col("start_time"), "1 HOUR").alias("prev_hour"),
        )
    )

    result2 = df.collect()
    assert len(result2) == 1
    assert result2[0]["id"] == 1


def test_drop_table_integration(tmp_path):
    db_path = tmp_path / "drop_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    table = db.create_table("temp", [column("id", "INTEGER")])
    table.insert([{"id": 1}])
    rows = table.select().collect()
    assert len(rows) == 1

    db.drop_table("temp")
    result = db.execute_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='temp'")
    assert len(result.rows) == 0
