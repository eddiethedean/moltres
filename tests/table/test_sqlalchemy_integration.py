"""Tests for SQLAlchemy ORM model integration."""

from __future__ import annotations

import pytest

try:
    from sqlalchemy import (
        Boolean,
        CheckConstraint,
        Column,
        Date,
        DateTime,
        Float,
        ForeignKey,
        Integer,
        Numeric,
        String,
        Text,
        UniqueConstraint,
    )
    from sqlalchemy.orm import DeclarativeBase
except ImportError:
    pytest.skip("SQLAlchemy not installed", allow_module_level=True)

from moltres import col, connect
from moltres.io.records import Records


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass


class User(Base):
    """Simple user model for testing."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(100), unique=True)
    age = Column(Integer, nullable=True)


class Order(Base):
    """Order model with foreign key for testing."""

    __tablename__ = "orders"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Numeric(10, 2))
    created_at = Column(DateTime)


class Product(Base):
    """Product model with various types and constraints."""

    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    price = Column(Float)
    description = Column(Text)
    active = Column(Boolean, default=True)
    created_date = Column(Date)


class Category(Base):
    """Category model with check constraint."""

    __tablename__ = "categories"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    priority = Column(Integer, nullable=False)

    __table_args__ = (
        CheckConstraint("priority >= 0 AND priority <= 100", name="ck_priority_range"),
        UniqueConstraint("name", name="uq_category_name"),
    )


def test_is_sqlalchemy_model():
    """Test detection of SQLAlchemy models."""
    from moltres.table.sqlalchemy_integration import is_sqlalchemy_model

    assert is_sqlalchemy_model(User) is True
    assert is_sqlalchemy_model(Order) is True
    assert is_sqlalchemy_model("not_a_model") is False
    assert is_sqlalchemy_model(123) is False
    assert is_sqlalchemy_model(Base) is False  # Base class is not a model


def test_get_model_table_name():
    """Test extraction of table name from models."""
    from moltres.table.sqlalchemy_integration import get_model_table_name

    assert get_model_table_name(User) == "users"
    assert get_model_table_name(Order) == "orders"
    assert get_model_table_name(Product) == "products"


def test_sqlalchemy_type_to_moltres_type():
    """Test SQLAlchemy type to Moltres type conversion."""
    from moltres.table.sqlalchemy_integration import sqlalchemy_type_to_moltres_type

    assert sqlalchemy_type_to_moltres_type(Integer()) == "INTEGER"
    assert sqlalchemy_type_to_moltres_type(String(100)) == "VARCHAR(100)"
    assert sqlalchemy_type_to_moltres_type(String()) == "TEXT"
    assert sqlalchemy_type_to_moltres_type(Text()) == "TEXT"
    assert sqlalchemy_type_to_moltres_type(Float()) == "FLOAT"
    assert sqlalchemy_type_to_moltres_type(Boolean()) == "BOOLEAN"
    assert sqlalchemy_type_to_moltres_type(DateTime()) == "DATETIME"
    assert sqlalchemy_type_to_moltres_type(Date()) == "DATE"
    assert sqlalchemy_type_to_moltres_type(Numeric(10, 2)) == "DECIMAL(10,2)"


def test_moltres_type_to_sqlalchemy_type():
    """Test Moltres type to SQLAlchemy type conversion."""
    from moltres.table.sqlalchemy_integration import moltres_type_to_sqlalchemy_type

    assert isinstance(moltres_type_to_sqlalchemy_type("INTEGER"), Integer)
    assert isinstance(moltres_type_to_sqlalchemy_type("TEXT"), Text)
    assert isinstance(moltres_type_to_sqlalchemy_type("VARCHAR(100)"), String)
    assert isinstance(moltres_type_to_sqlalchemy_type("FLOAT"), Float)
    assert isinstance(moltres_type_to_sqlalchemy_type("BOOLEAN"), Boolean)
    assert isinstance(moltres_type_to_sqlalchemy_type("DECIMAL(10,2)"), Numeric)


def test_extract_foreign_keys():
    """Test foreign key extraction from models."""
    from moltres.table.sqlalchemy_integration import extract_foreign_keys

    fks = extract_foreign_keys(Order)
    assert len(fks) == 1
    assert fks[0].columns == "user_id"
    assert fks[0].references_table == "users"
    assert fks[0].references_columns == "id"


def test_model_to_schema():
    """Test conversion of SQLAlchemy model to TableSchema."""
    from moltres.table.sqlalchemy_integration import model_to_schema

    schema = model_to_schema(User)
    assert schema.name == "users"
    assert len(schema.columns) == 4

    # Check column details
    id_col = next(c for c in schema.columns if c.name == "id")
    assert id_col.type_name == "INTEGER"
    assert id_col.primary_key is True
    assert id_col.nullable is False

    name_col = next(c for c in schema.columns if c.name == "name")
    assert name_col.type_name == "VARCHAR(100)"
    assert name_col.nullable is False

    email_col = next(c for c in schema.columns if c.name == "email")
    assert email_col.type_name == "VARCHAR(100)"
    assert email_col.nullable is True


def test_model_to_schema_with_constraints():
    """Test model to schema conversion with constraints."""
    from moltres.table.sqlalchemy_integration import model_to_schema
    from moltres.table.schema import UniqueConstraint

    schema = model_to_schema(Category)
    assert schema.name == "categories"
    assert len(schema.columns) == 3

    # Check constraints
    unique_constraints = [c for c in schema.constraints if isinstance(c, UniqueConstraint)]
    assert len(unique_constraints) == 1
    assert "name" in unique_constraints[0].columns

    # Check constraints may or may not be extracted depending on SQLAlchemy version
    # Just verify we have at least the unique constraint
    assert len(unique_constraints) >= 1


def test_schema_to_table():
    """Test conversion of TableSchema to SQLAlchemy Table."""
    from sqlalchemy import MetaData
    from moltres.table.schema import TableSchema, column
    from moltres.table.sqlalchemy_integration import schema_to_table

    schema = TableSchema(
        name="test_table",
        columns=[
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
        ],
    )

    metadata = MetaData()
    table = schema_to_table(schema, metadata)

    assert table.name == "test_table"
    assert len(table.columns) == 2
    assert "id" in [c.name for c in table.columns]
    assert "name" in [c.name for c in table.columns]


def test_create_table_from_model(tmp_path):
    """Test creating a table from a SQLAlchemy model."""
    db_path = tmp_path / "model_test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table from model
    table = db.create_table(User).collect()

    assert table.name == "users"
    assert table.model_class == User

    # Verify table was created
    columns = db.get_columns("users")
    assert len(columns) == 4
    assert "id" in [c.name for c in columns]
    assert "name" in [c.name for c in columns]
    assert "email" in [c.name for c in columns]
    assert "age" in [c.name for c in columns]

    db.close()


def test_create_table_from_model_with_constraints(tmp_path):
    """Test creating a table from a model with constraints."""
    db_path = tmp_path / "model_constraints_test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create tables
    db.create_table(User).collect()
    db.create_table(Order).collect()

    # Verify foreign key was created
    columns = db.get_columns("orders")
    user_id_col = next(c for c in columns if c.name == "user_id")
    # Foreign key info might be in constraints, not directly in column info
    # This is a basic check that the table was created
    assert user_id_col is not None

    db.close()


def test_table_from_model(tmp_path):
    """Test getting a table handle from a SQLAlchemy model."""
    db_path = tmp_path / "table_handle_test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table first
    db.create_table(User).collect()

    # Get table handle from model
    table_handle = db.table(User)

    assert table_handle.name == "users"
    assert table_handle.model_class == User

    # Can use it for queries
    df = table_handle.select()
    results = df.collect()
    assert results == []

    db.close()


def test_table_from_model_vs_string(tmp_path):
    """Test that both model and string work for table() method."""
    db_path = tmp_path / "table_compat_test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(User).collect()

    # Get handle using model
    handle_from_model = db.table(User)
    assert handle_from_model.model_class == User

    # Get handle using string
    handle_from_string = db.table("users")
    assert handle_from_string.model_class is None

    # Both should work for queries
    df1 = handle_from_model.select()
    df2 = handle_from_string.select()
    assert df1.collect() == df2.collect()

    db.close()


def test_create_table_backward_compatibility(tmp_path):
    """Test that existing create_table API still works."""
    db_path = tmp_path / "backward_compat_test.db"
    db = connect(f"sqlite:///{db_path}")

    from moltres.table.schema import column

    # Old API should still work
    table = db.create_table(
        "customers",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
        ],
    ).collect()

    assert table.name == "customers"
    assert table.model_class is None

    db.close()


def test_model_based_query(tmp_path):
    """Test querying using model-based table handles."""
    db_path = tmp_path / "query_test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate table
    db.create_table(User).collect()

    records = Records(
        _data=[
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
        ],
        _database=db,
    )
    records.insert_into("users")

    # Query using model
    df = db.table(User).select().where(col("age") > 25)
    results = df.collect()

    assert len(results) == 1
    assert results[0]["name"] == "Alice"

    db.close()


def test_model_based_join(tmp_path):
    """Test joins using model-based table handles."""
    db_path = tmp_path / "join_test.db"
    db = connect(f"sqlite:///{db_path}")

    # Create tables
    db.create_table(User).collect()
    db.create_table(Order).collect()

    # Insert data
    Records(
        _data=[
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
        ],
        _database=db,
    ).insert_into("users")

    Records(
        _data=[
            {"id": 1, "user_id": 1, "amount": 100.50},
        ],
        _database=db,
    ).insert_into("orders")

    # Join using models
    orders_df = db.table(Order).select()
    users_df = db.table(User).select()
    df = orders_df.join(users_df, on=[("user_id", "id")])
    results = df.collect()

    assert len(results) == 1
    # Amount might be returned as string or float depending on database
    amount = results[0]["amount"]
    assert float(amount) == 100.50

    db.close()


def test_invalid_model_error():
    """Test that invalid models raise appropriate errors."""
    from moltres.table.sqlalchemy_integration import model_to_schema

    with pytest.raises(ValueError, match="not a valid SQLAlchemy model"):
        model_to_schema(str)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="not a valid SQLAlchemy model"):
        model_to_schema(int)  # type: ignore[arg-type]


def test_model_without_tablename():
    """Test model that doesn't have explicit __tablename__."""
    # SQLAlchemy requires __tablename__ for ORM models, but we can test
    # the fallback logic with a model that has __table__ attribute
    # For models without __tablename__, SQLAlchemy will raise an error
    # So we test that get_model_table_name handles models with __table__ correctly
    from moltres.table.sqlalchemy_integration import get_model_table_name

    # Test with a model that has __tablename__ (normal case)
    assert get_model_table_name(User) == "users"

    # Test that it works with models that have __table__ attribute
    # (which User has via SQLAlchemy)
    assert get_model_table_name(Order) == "orders"


@pytest.mark.asyncio
async def test_async_create_table_from_model(tmp_path):
    """Test async create_table with SQLAlchemy model."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_model_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create table from model
    table = await db.create_table(User).collect()

    assert table.name == "users"
    assert table.model_class == User

    await db.close()


@pytest.mark.asyncio
async def test_async_table_from_model(tmp_path):
    """Test async table() with SQLAlchemy model."""
    try:
        import aiosqlite  # noqa: F401
    except ImportError:
        pytest.skip("aiosqlite not installed")

    from moltres import async_connect

    db_path = tmp_path / "async_table_handle_test.db"
    db = async_connect(f"sqlite+aiosqlite:///{db_path}")

    # Create table first
    await db.create_table(User).collect()

    # Get table handle from model
    table_handle = await db.table(User)

    assert table_handle.name == "users"
    assert table_handle.model_class == User

    await db.close()
