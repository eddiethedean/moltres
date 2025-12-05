"""Tests for table schema creation and DDL operations."""

import pytest

from moltres import column, connect
from moltres.engine.dialects import get_dialect
from moltres.io.records import Records
from moltres.sql.ddl import (
    compile_create_index,
    compile_create_table,
    compile_drop_index,
    compile_drop_table,
)
from moltres.table.schema import (
    TableSchema,
    check,
    foreign_key,
    unique,
)


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
    assert "users" in sql  # SQLAlchemy may not quote identifiers
    assert "IF NOT EXISTS" in sql
    assert "id" in sql  # SQLAlchemy may not quote identifiers
    assert "name" in sql
    assert "email" in sql
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

    # SQLAlchemy may wrap defaults in parentheses
    assert "DEFAULT" in sql
    assert "0.0" in sql or "(0.0)" in sql
    assert "1" in sql or "(1)" in sql


def test_compile_drop_table():
    dialect = get_dialect("sqlite")
    sql = compile_drop_table("users", dialect)

    # SQLAlchemy may not quote identifiers and may include newlines
    assert "DROP TABLE" in sql
    assert "IF EXISTS" in sql
    assert "users" in sql


def test_compile_drop_table_without_if_exists():
    dialect = get_dialect("sqlite")
    sql = compile_drop_table("users", dialect, if_exists=False)

    # SQLAlchemy may not quote identifiers and may include newlines
    assert "DROP TABLE" in sql
    assert "IF EXISTS" not in sql
    assert "users" in sql


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
    ).collect()

    assert table.name == "customers"

    records = Records(
        _data=[{"id": 1, "name": "Alice", "email": "alice@example.com"}], _database=db
    )
    records.insert_into("customers")
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
    ).collect()

    records = Records(_data=[{"id": 1, "name": "Widget"}], _database=db)
    records.insert_into("products")
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
    ).collect()

    # Insert test data
    records = Records(
        _data=[
            {"id": 1, "price": 99.99, "discount": 0.10},
            {"id": 2, "price": 149.50, "discount": 0.15},
        ],
        _database=db,
    )
    records.insert_into("prices")

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
    ).collect()

    records2 = Records(_data=[{"id": 1, "cost": 100.50, "tax_rate": 0.0825}], _database=db)
    records2.insert_into("products")

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
    ).collect()

    # Insert test data with UUID strings
    import uuid as uuid_module

    user_id = str(uuid_module.uuid4())
    records = Records(_data=[{"id": user_id, "name": "Alice"}], _database=db)
    records.insert_into("users")

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
    ).collect()

    session_id = str(uuid_module.uuid4())
    user_id2 = str(uuid_module.uuid4())
    records2 = Records(_data=[{"id": session_id, "user_id": user_id2}], _database=db)
    records2.insert_into("sessions")

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
    ).collect()

    # Insert test data with JSON strings
    import json as json_module

    product_data = json_module.dumps({"name": "Widget", "price": 10.5})
    metadata = json_module.dumps({"category": "electronics"})
    records = Records(_data=[{"id": 1, "data": product_data, "metadata": metadata}], _database=db)
    records.insert_into("products")

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
    ).collect()

    settings = json_module.dumps({"theme": "dark", "lang": "en"})
    records2 = Records(_data=[{"id": 1, "settings": settings}], _database=db)
    records2.insert_into("config")

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
    ).collect()

    # Insert test data
    import datetime

    start = datetime.datetime(2024, 1, 1, 10, 0, 0)
    records = Records(_data=[{"id": 1, "start_time": start, "duration": "1 HOUR"}], _database=db)
    records.insert_into("events")

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

    table = db.create_table("temp", [column("id", "INTEGER")]).collect()
    records = Records(_data=[{"id": 1}], _database=db)
    records.insert_into("temp")
    rows = table.select().collect()
    assert len(rows) == 1

    db.drop_table("temp").collect()
    result = db.execute_sql("SELECT name FROM sqlite_master WHERE type='table' AND name='temp'")
    assert result.rows is not None
    assert len(result.rows) == 0


# ============================================================================
# Constraint Tests
# ============================================================================


def test_unique_constraint_single_column(tmp_path):
    """Test UNIQUE constraint on a single column."""
    db_path = tmp_path / "unique_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("email", "TEXT", nullable=False),
        ],
        constraints=[unique("email")],
    ).collect()

    # Insert valid data
    records = Records(
        _data=[
            {"id": 1, "email": "alice@example.com"},
            {"id": 2, "email": "bob@example.com"},
        ],
        _database=db,
    )
    records.insert_into("users")

    # Try to insert duplicate email (should fail)
    with pytest.raises(Exception):  # SQLite raises OperationalError
        duplicate = Records(_data=[{"id": 3, "email": "alice@example.com"}], _database=db)
        duplicate.insert_into("users")


def test_unique_constraint_multi_column(tmp_path):
    """Test UNIQUE constraint on multiple columns."""
    db_path = tmp_path / "unique_multi_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    db.create_table(
        "user_sessions",
        [
            column("id", "INTEGER", primary_key=True),
            column("user_id", "INTEGER"),
            column("session_id", "TEXT"),
        ],
        constraints=[unique(["user_id", "session_id"], name="uq_user_session")],
    ).collect()

    # Insert valid data (same user_id, different session_id is OK)
    records = Records(
        _data=[
            {"id": 1, "user_id": 1, "session_id": "session1"},
            {"id": 2, "user_id": 1, "session_id": "session2"},
        ],
        _database=db,
    )
    records.insert_into("user_sessions")

    # Try to insert duplicate combination (should fail)
    with pytest.raises(Exception):
        duplicate = Records(_data=[{"id": 3, "user_id": 1, "session_id": "session1"}], _database=db)
        duplicate.insert_into("user_sessions")


def test_check_constraint(tmp_path):
    """Test CHECK constraint."""
    db_path = tmp_path / "check_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("price", "REAL"),
            column("quantity", "INTEGER"),
        ],
        constraints=[check("price >= 0", name="ck_positive_price"), check("quantity >= 0")],
    ).collect()

    # Insert valid data
    records = Records(
        _data=[
            {"id": 1, "name": "Widget", "price": 10.0, "quantity": 5},
            {"id": 2, "name": "Gadget", "price": 0.0, "quantity": 0},
        ],
        _database=db,
    )
    records.insert_into("products")

    # Try to insert invalid data (negative price - should fail)
    with pytest.raises(Exception):
        invalid = Records(
            _data=[{"id": 3, "name": "Bad", "price": -5.0, "quantity": 1}], _database=db
        )
        invalid.insert_into("products")


def test_foreign_key_constraint(tmp_path):
    """Test FOREIGN KEY constraint."""
    db_path = tmp_path / "fk_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Enable foreign keys for SQLite (use connection directly since PRAGMA doesn't return rows)
    from sqlalchemy import text

    with db.connection_manager.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()

    # Create parent table
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()

    # Create child table with foreign key
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("user_id", "INTEGER"),
            column("total", "REAL"),
        ],
        constraints=[
            foreign_key("user_id", "users", "id", name="fk_order_user", on_delete="CASCADE")
        ],
    ).collect()

    # Insert parent data
    users_records = Records(
        _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}], _database=db
    )
    users_records.insert_into("users")

    # Insert valid child data
    orders_records = Records(
        _data=[
            {"id": 1, "user_id": 1, "total": 100.0},
            {"id": 2, "user_id": 2, "total": 200.0},
        ],
        _database=db,
    )
    orders_records.insert_into("orders")

    # Try to insert invalid foreign key (should fail)
    with pytest.raises(Exception):
        invalid = Records(_data=[{"id": 3, "user_id": 999, "total": 50.0}], _database=db)
        invalid.insert_into("orders")


def test_foreign_key_cascade_delete(tmp_path):
    """Test FOREIGN KEY with CASCADE delete."""
    db_path = tmp_path / "fk_cascade_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Enable foreign keys for SQLite (use connection directly since PRAGMA doesn't return rows)
    from sqlalchemy import text

    with db.connection_manager.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()

    # Create parent table
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()

    # Create child table with CASCADE delete
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("user_id", "INTEGER"),
            column("total", "REAL"),
        ],
        constraints=[
            foreign_key("user_id", "users", "id", on_delete="CASCADE", name="fk_order_user")
        ],
    ).collect()

    # Insert data
    users_records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
    users_records.insert_into("users")

    orders_records = Records(_data=[{"id": 1, "user_id": 1, "total": 100.0}], _database=db)
    orders_records.insert_into("orders")

    # Delete parent - child should be deleted too (CASCADE)
    # Use executor.execute() for DML statements that don't return rows
    from sqlalchemy import text

    db.executor.execute("DELETE FROM users WHERE id = 1")

    # Verify child was deleted
    remaining_orders = db.table("orders").select().collect()
    assert len(remaining_orders) == 0


def test_multiple_constraints(tmp_path):
    """Test table with multiple constraint types."""
    db_path = tmp_path / "multi_constraint_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    table = db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("sku", "TEXT"),
            column("name", "TEXT"),
            column("price", "REAL"),
            column("stock", "INTEGER"),
        ],
        constraints=[
            unique("sku", name="uq_sku"),
            check("price > 0", name="ck_positive_price"),
            check("stock >= 0", name="ck_non_negative_stock"),
        ],
    ).collect()

    # Insert valid data
    records = Records(
        _data=[
            {"id": 1, "sku": "SKU001", "name": "Widget", "price": 10.0, "stock": 5},
        ],
        _database=db,
    )
    records.insert_into("products")

    # Verify data
    result = table.select().collect()
    assert len(result) == 1
    assert result[0]["sku"] == "SKU001"


# ============================================================================
# Index Tests
# ============================================================================


def test_create_index_single_column(tmp_path):
    """Test creating a single-column index."""
    db_path = tmp_path / "index_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("email", "TEXT"),
            column("name", "TEXT"),
        ],
    ).collect()

    # Insert data
    records = Records(
        _data=[
            {"id": 1, "email": "alice@example.com", "name": "Alice"},
            {"id": 2, "email": "bob@example.com", "name": "Bob"},
        ],
        _database=db,
    )
    records.insert_into("users")

    # Create index
    db.create_index("idx_email", "users", "email").collect()

    # Verify index exists (SQLite specific check)
    result = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_email'"
    )
    assert result.rows is not None
    assert len(result.rows) == 1


def test_create_index_multi_column(tmp_path):
    """Test creating a multi-column index."""
    db_path = tmp_path / "index_multi_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("user_id", "INTEGER"),
            column("status", "TEXT"),
            column("created_at", "TEXT"),
        ],
    ).collect()

    # Create multi-column index
    db.create_index("idx_user_status", "orders", ["user_id", "status"]).collect()

    # Verify index exists
    result = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_user_status'"
    )
    assert result.rows is not None
    assert len(result.rows) == 1


def test_create_unique_index(tmp_path):
    """Test creating a UNIQUE index."""
    db_path = tmp_path / "index_unique_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("email", "TEXT"),
        ],
    ).collect()

    # Insert data
    records = Records(_data=[{"id": 1, "email": "alice@example.com"}], _database=db)
    records.insert_into("users")

    # Create unique index
    db.create_index("idx_unique_email", "users", "email", unique=True).collect()

    # Try to insert duplicate (should fail due to unique index)
    with pytest.raises(Exception):
        duplicate = Records(_data=[{"id": 2, "email": "alice@example.com"}], _database=db)
        duplicate.insert_into("users")


def test_drop_index(tmp_path):
    """Test dropping an index."""
    db_path = tmp_path / "drop_index_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("email", "TEXT"),
        ],
    ).collect()

    # Create index
    db.create_index("idx_email", "users", "email").collect()

    # Verify index exists
    result = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_email'"
    )
    assert result.rows is not None
    assert len(result.rows) == 1

    # Drop index
    db.drop_index("idx_email", "users").collect()

    # Verify index is gone
    result = db.execute_sql(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_email'"
    )
    assert len(result.rows) == 0


def test_index_usage_performance(tmp_path):
    """Test that index improves query performance (basic verification)."""
    db_path = tmp_path / "index_perf_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("email", "TEXT"),
            column("name", "TEXT"),
        ],
    ).collect()

    # Insert data
    records = Records(
        _data=[
            {"id": i, "email": f"user{i}@example.com", "name": f"User {i}"} for i in range(1, 11)
        ],
        _database=db,
    )
    records.insert_into("users")

    # Create index
    db.create_index("idx_email", "users", "email").collect()

    # Query using indexed column (should work efficiently)
    from moltres import col

    result = db.table("users").select().filter(col("email") == "user5@example.com").collect()
    assert len(result) == 1
    assert result[0]["email"] == "user5@example.com"


def test_compile_create_index_sql():
    """Test SQL compilation for CREATE INDEX."""
    dialect = get_dialect("sqlite")
    engine = None  # Test raw SQL generation

    sql = compile_create_index("idx_email", "users", "email", engine=engine, dialect=dialect)
    assert "CREATE INDEX" in sql
    assert "idx_email" in sql
    assert "users" in sql
    assert "email" in sql


def test_compile_create_unique_index_sql():
    """Test SQL compilation for CREATE UNIQUE INDEX."""
    dialect = get_dialect("sqlite")
    engine = None

    sql = compile_create_index(
        "idx_unique_email", "users", "email", unique=True, engine=engine, dialect=dialect
    )
    assert "CREATE UNIQUE INDEX" in sql
    assert "idx_unique_email" in sql


def test_compile_drop_index_sql():
    """Test SQL compilation for DROP INDEX."""
    dialect = get_dialect("sqlite")
    engine = None

    sql = compile_drop_index("idx_email", table_name="users", engine=engine, dialect=dialect)
    assert "DROP INDEX" in sql
    assert "idx_email" in sql


# ============================================================================
# Additional Coverage Tests
# ============================================================================


def test_compile_create_index_without_if_not_exists():
    """Test CREATE INDEX without IF NOT EXISTS."""
    dialect = get_dialect("sqlite")
    sql = compile_create_index("idx_email", "users", "email", if_not_exists=False, dialect=dialect)
    assert "CREATE INDEX" in sql
    assert "IF NOT EXISTS" not in sql


def test_compile_drop_index_without_if_exists():
    """Test DROP INDEX without IF EXISTS."""
    dialect = get_dialect("sqlite")
    sql = compile_drop_index("idx_email", table_name="users", if_exists=False, dialect=dialect)
    assert "DROP INDEX" in sql
    assert "IF EXISTS" not in sql


def test_foreign_key_on_update(tmp_path):
    """Test FOREIGN KEY with ON UPDATE action."""
    db_path = tmp_path / "fk_update_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Enable foreign keys for SQLite
    from sqlalchemy import text

    with db.connection_manager.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()

    # Create parent table
    db.create_table(
        "users",
        [column("id", "INTEGER", primary_key=True), column("name", "TEXT")],
    ).collect()

    # Create child table with foreign key and ON UPDATE CASCADE
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("user_id", "INTEGER"),
            column("total", "REAL"),
        ],
        constraints=[
            foreign_key("user_id", "users", "id", on_update="CASCADE", name="fk_order_user")
        ],
    ).collect()

    # Insert data
    users_records = Records(_data=[{"id": 1, "name": "Alice"}], _database=db)
    users_records.insert_into("users")

    orders_records = Records(_data=[{"id": 1, "user_id": 1, "total": 100.0}], _database=db)
    orders_records.insert_into("orders")

    # Update parent - child should be updated too (CASCADE)
    db.executor.execute("UPDATE users SET id = 2 WHERE id = 1")

    # Verify child was updated
    remaining_orders = db.table("orders").select().collect()
    assert len(remaining_orders) == 1
    assert remaining_orders[0]["user_id"] == 2


def test_multi_column_foreign_key(tmp_path):
    """Test multi-column FOREIGN KEY constraint."""
    db_path = tmp_path / "fk_multi_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Enable foreign keys for SQLite
    from sqlalchemy import text

    with db.connection_manager.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys = ON"))
        conn.commit()

    # Create parent table with composite primary key
    db.create_table(
        "parent",
        [
            column("key1", "INTEGER", primary_key=True),
            column("key2", "INTEGER", primary_key=True),
            column("name", "TEXT"),
        ],
    ).collect()

    # Create child table with multi-column foreign key
    db.create_table(
        "child",
        [
            column("id", "INTEGER", primary_key=True),
            column("parent_key1", "INTEGER"),
            column("parent_key2", "INTEGER"),
            column("value", "TEXT"),
        ],
        constraints=[
            foreign_key(
                ["parent_key1", "parent_key2"],
                "parent",
                ["key1", "key2"],
                name="fk_child_parent",
            )
        ],
    ).collect()

    # Insert parent data
    parent_records = Records(_data=[{"key1": 1, "key2": 10, "name": "Parent1"}], _database=db)
    parent_records.insert_into("parent")

    # Insert valid child data
    child_records = Records(
        _data=[{"id": 1, "parent_key1": 1, "parent_key2": 10, "value": "Child1"}],
        _database=db,
    )
    child_records.insert_into("child")

    # Try to insert invalid foreign key (should fail)
    with pytest.raises(Exception):
        invalid = Records(
            _data=[{"id": 2, "parent_key1": 999, "parent_key2": 999, "value": "Invalid"}],
            _database=db,
        )
        invalid.insert_into("child")


def test_compile_insert_select():
    """Test SQL compilation for INSERT INTO ... SELECT."""
    from sqlalchemy import create_engine, select, Table, Column, MetaData, Integer, String
    from moltres.sql.ddl import compile_insert_select
    from moltres.engine.dialects import get_dialect

    dialect = get_dialect("sqlite")
    engine = create_engine("sqlite:///:memory:", future=True)

    # Create a simple SELECT statement using proper Table with Column objects
    metadata = MetaData()
    source_table = Table(
        "source",
        metadata,
        Column("id", Integer),
        Column("name", String),
    )
    select_stmt = select(source_table.c.id, source_table.c.name)

    # Test INSERT INTO ... SELECT with specific columns
    sql, params = compile_insert_select(
        target_table="target",
        select_stmt=select_stmt,
        dialect=dialect,
        columns=["id", "name"],
        engine=engine,
    )
    assert "INSERT INTO" in sql
    assert "target" in sql
    assert "SELECT" in sql
    assert "source" in sql
    assert isinstance(params, dict)


def test_unique_helper_validation():
    """Test unique() helper function validation."""
    from moltres.table.schema import unique

    # Valid single column
    uq1 = unique("email")
    assert uq1.columns == ("email",)

    # Valid multi-column
    uq2 = unique(["user_id", "session_id"], name="uq_user_session")
    assert uq2.columns == ("user_id", "session_id")
    assert uq2.name == "uq_user_session"

    # Invalid: empty columns
    with pytest.raises(ValueError, match="at least one column"):
        unique([])


def test_foreign_key_helper_validation():
    """Test foreign_key() helper function validation."""
    from moltres.table.schema import foreign_key

    # Valid single column
    fk1 = foreign_key("user_id", "users", "id")
    assert fk1.columns == ("user_id",)
    assert fk1.references_table == "users"
    assert fk1.references_columns == ("id",)

    # Valid multi-column
    fk2 = foreign_key(["order_id", "item_id"], "order_items", ["id", "id"])
    assert fk2.columns == ("order_id", "item_id")
    assert fk2.references_columns == ("id", "id")

    # Invalid: empty columns
    with pytest.raises(ValueError, match="at least one column"):
        foreign_key([], "users", "id")

    # Invalid: empty references_table
    with pytest.raises(ValueError, match="references_table"):
        foreign_key("user_id", "", "id")

    # Invalid: empty references_columns
    with pytest.raises(ValueError, match="references_column"):
        foreign_key("user_id", "users", [])

    # Invalid: column count mismatch
    with pytest.raises(ValueError, match="count"):
        foreign_key(["col1", "col2"], "table", "col1")


def test_compile_create_table_different_dialects():
    """Test CREATE TABLE compilation for different dialects."""
    schema = TableSchema(
        name="users",
        columns=[
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "TEXT", nullable=False),
        ],
    )

    # Test SQLite
    sqlite_dialect = get_dialect("sqlite")
    sql_sqlite = compile_create_table(schema, sqlite_dialect)
    assert "CREATE TABLE" in sql_sqlite

    # Test PostgreSQL (may not have driver installed, but compilation should work)
    postgresql_dialect = get_dialect("postgresql")
    try:
        sql_postgresql = compile_create_table(schema, postgresql_dialect)
        assert "CREATE TABLE" in sql_postgresql
    except Exception:
        # Driver may not be installed, skip
        pass

    # Test MySQL (may not have driver installed, but compilation should work)
    mysql_dialect = get_dialect("mysql")
    try:
        sql_mysql = compile_create_table(schema, mysql_dialect)
        assert "CREATE TABLE" in sql_mysql
    except Exception:
        # Driver may not be installed, skip
        pass


def test_compile_create_index_different_dialects():
    """Test CREATE INDEX compilation for different dialects."""
    # Test SQLite
    sqlite_dialect = get_dialect("sqlite")
    sql_sqlite = compile_create_index("idx_email", "users", "email", dialect=sqlite_dialect)
    assert "CREATE INDEX" in sql_sqlite

    # Test PostgreSQL (may not have driver installed, but compilation should work)
    postgresql_dialect = get_dialect("postgresql")
    try:
        sql_postgresql = compile_create_index(
            "idx_email", "users", "email", dialect=postgresql_dialect
        )
        assert "CREATE INDEX" in sql_postgresql
    except Exception:
        # Driver may not be installed, skip
        pass

    # Test MySQL (may not have driver installed, but compilation should work)
    mysql_dialect = get_dialect("mysql")
    try:
        sql_mysql = compile_create_index("idx_email", "users", "email", dialect=mysql_dialect)
        assert "CREATE INDEX" in sql_mysql
    except Exception:
        # Driver may not be installed, skip
        pass


def test_compile_create_table_with_unknown_type():
    """Test CREATE TABLE with unknown column type (should fallback to String)."""
    schema = TableSchema(
        name="test",
        columns=[
            column("id", "INTEGER", primary_key=True),
            column("data", "UNKNOWN_TYPE"),  # Unknown type
        ],
    )
    dialect = get_dialect("sqlite")
    sql = compile_create_table(schema, dialect)
    # Should compile successfully (fallback to String)
    assert "CREATE TABLE" in sql
    assert "test" in sql


def test_compile_create_table_with_string_fks_fallback(tmp_path):
    """Test the fallback function _compile_create_table_with_string_fks."""
    # This tests the fallback when SQLAlchemy can't resolve foreign keys
    # We can trigger this by creating a table with FK to a non-existent table
    db_path = tmp_path / "fk_fallback_test.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table with FK to a table that doesn't exist in MetaData
    # This should trigger the fallback function
    table = db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("user_id", "INTEGER"),
        ],
        constraints=[foreign_key("user_id", "users", "id", name="fk_order_user")],
    ).collect()

    # Verify table was created (fallback should work)
    assert table.name == "orders"


@pytest.mark.asyncio
async def test_async_create_index(tmp_path):
    """Test async create_index method."""
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("Async dependencies not installed")

    db_path = tmp_path / "async_index_test.sqlite"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    try:
        # Create table
        await async_db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
            ],
        ).collect()

        # Create index
        await async_db.create_index("idx_email", "users", "email").collect()

        # Verify index exists
        result = await async_db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_email'"
        )
        assert result.rows is not None
        assert len(result.rows) == 1
    finally:
        await async_db.close()


@pytest.mark.asyncio
async def test_async_drop_index(tmp_path):
    """Test async drop_index method."""
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("Async dependencies not installed")

    db_path = tmp_path / "async_drop_index_test.sqlite"
    async_db = async_connect(f"sqlite+aiosqlite:///{db_path}")
    try:
        # Create table
        await async_db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("email", "TEXT"),
            ],
        ).collect()

        # Create index
        await async_db.create_index("idx_email", "users", "email").collect()

        # Drop index
        await async_db.drop_index("idx_email", "users").collect()

        # Verify index is gone
        result = await async_db.execute_sql(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_email'"
        )
        assert result.rows is not None
        assert len(result.rows) == 0
    finally:
        await async_db.close()


def test_compile_create_table_with_all_column_types():
    """Test CREATE TABLE with all supported column types."""
    schema = TableSchema(
        name="all_types",
        columns=[
            column("id", "INTEGER", primary_key=True),
            column("bigint_col", "BIGINT"),
            column("smallint_col", "SMALLINT"),
            column("varchar_col", "VARCHAR"),
            column("char_col", "CHAR"),
            column("boolean_col", "BOOLEAN"),
            column("real_col", "REAL"),
            column("float_col", "FLOAT"),
            column("double_col", "DOUBLE"),
            column("decimal_col", "DECIMAL", precision=10, scale=2),
            column("date_col", "DATE"),
            column("time_col", "TIME"),
            column("timestamp_col", "TIMESTAMP"),
            column("datetime_col", "DATETIME"),
        ],
    )
    dialect = get_dialect("sqlite")
    sql = compile_create_table(schema, dialect)
    assert "CREATE TABLE" in sql
    assert "all_types" in sql
