"""Tests for session support in connect() and async_connect().

This module tests the ability to pass SQLAlchemy/SQLModel sessions
directly to connect() and async_connect() instead of engines.
"""

from __future__ import annotations

import uuid
from typing import Union

import pytest

from moltres import connect, async_connect, col
from moltres.table.schema import column


# Test SQLAlchemy Sync Session
# =============================


def test_connect_with_sqlalchemy_session(tmp_path):
    """Test connect() with SQLAlchemy Session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test_session.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        db = connect(session=session)

        # Verify dialect is detected correctly
        assert db.dialect.name == "sqlite"

        # Create table and insert data
        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        from moltres.io.records import Records

        Records(
            _data=[
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
            ],
            _database=db,
        ).insert_into("users")

        # Query using Moltres
        df = db.table("users").select().where(col("age") > 25)
        results = df.collect()

        assert len(results) == 1
        assert results[0]["name"] == "Alice"
        assert results[0]["age"] == 30


def test_connect_with_sqlalchemy_session_operations(tmp_path):
    """Test various operations with SQLAlchemy Session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test_session_ops.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        db = connect(session=session)

        # Create table
        db.create_table(
            "products",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("price", "REAL"),
            ],
        ).collect()

        # Insert data
        from moltres.io.records import Records

        Records(
            _data=[
                {"id": 1, "name": "Laptop", "price": 999.99},
                {"id": 2, "name": "Mouse", "price": 29.99},
            ],
            _database=db,
        ).insert_into("products")

        # Test aggregations (global aggregation - no group_by needed)
        from moltres.expressions import functions as F

        stats_df = db.table("products").select(
            F.count(col("id")).alias("count"),
            F.sum(col("price")).alias("total"),
        )
        stats = stats_df.collect()[0]
        assert stats["count"] == 2
        assert stats["total"] == 1029.98


# Test SQLModel Sync Session
# ===========================


@pytest.mark.skipif(
    not pytest.importorskip("sqlmodel", reason="SQLModel not installed"),
    reason="SQLModel not installed",
)
def test_connect_with_sqlmodel_session(tmp_path):
    """Test connect() with SQLModel Session."""
    from sqlmodel import SQLModel, Field, create_engine
    from sqlalchemy.orm import sessionmaker

    # Define SQLModel with unique class name to avoid conflicts when running all tests
    class SessionUser(SQLModel, table=True):
        __tablename__ = "session_users"
        id: Union[int, None] = Field(default=None, primary_key=True)
        name: str
        age: int

    db_path = tmp_path / "test_sqlmodel_session.db"
    engine = create_engine(f"sqlite:///{db_path}")

    # Create tables
    SQLModel.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        db = connect(session=session)

        # Verify dialect
        assert db.dialect.name == "sqlite"

        # Insert data using SQLModel session
        user1 = SessionUser(name="Alice", age=30)
        user2 = SessionUser(name="Bob", age=25)
        session.add(user1)
        session.add(user2)
        session.commit()

        # Query using Moltres with SQLModel
        df = db.table(SessionUser).select()
        results = df.collect()

        # Results should be SessionUser instances (SQLModel)
        assert len(results) == 2
        assert isinstance(results[0], SessionUser)
        assert results[0].name in ("Alice", "Bob")
        assert results[0].age in (30, 25)


@pytest.mark.skipif(
    not pytest.importorskip("sqlmodel", reason="SQLModel not installed"),
    reason="SQLModel not installed",
)
def test_connect_with_sqlmodel_session_filtering(tmp_path):
    """Test filtering with SQLModel Session."""
    from sqlmodel import SQLModel, Field, create_engine
    from sqlalchemy.orm import sessionmaker

    table_name = f"products_{uuid.uuid4().hex[:8]}"

    class Product(SQLModel, table=True):
        __tablename__ = table_name
        id: Union[int, None] = Field(default=None, primary_key=True)
        name: str
        price: float

    db_path = tmp_path / "test_sqlmodel_filter.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        # Insert data
        session.add(Product(name="Laptop", price=999.99))
        session.add(Product(name="Mouse", price=29.99))
        session.commit()

        db = connect(session=session)

        # Filter using Moltres
        df = db.table(Product).select().where(col("price") > 100)
        results = df.collect()

        assert len(results) == 1
        assert isinstance(results[0], Product)
        assert results[0].name == "Laptop"
        assert results[0].price == 999.99


# Test SQLAlchemy Async Session
# ==============================


@pytest.mark.asyncio
async def test_async_connect_with_sqlalchemy_async_session(tmp_path):
    """Test async_connect() with SQLAlchemy AsyncSession."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test_async_session.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        db = async_connect(session=session)

        # Verify dialect
        assert db.dialect.name == "sqlite"

        # Create table
        await db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("age", "INTEGER"),
            ],
        ).collect()

        # Insert data
        from moltres.io.records import AsyncRecords

        records = AsyncRecords(
            _data=[
                {"id": 1, "name": "Alice", "age": 30},
                {"id": 2, "name": "Bob", "age": 25},
            ],
            _database=db,
        )
        await records.insert_into("users")

        # Query using Moltres
        df = (await db.table("users")).select().where(col("age") > 25)
        results = await df.collect()

        assert len(results) == 1
        assert results[0]["name"] == "Alice"
        assert results[0]["age"] == 30

        await db.close()


@pytest.mark.asyncio
async def test_async_connect_with_sqlalchemy_async_session_operations(tmp_path):
    """Test various operations with SQLAlchemy AsyncSession."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test_async_ops.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        db = async_connect(session=session)

        # Create table
        await db.create_table(
            "products",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("price", "REAL"),
            ],
        ).collect()

        # Insert data
        from moltres.io.records import AsyncRecords

        records = AsyncRecords(
            _data=[
                {"id": 1, "name": "Laptop", "price": 999.99},
                {"id": 2, "name": "Mouse", "price": 29.99},
            ],
            _database=db,
        )
        await records.insert_into("products")

        # Test aggregations (global aggregation - no group_by needed)
        from moltres.expressions import functions as F

        stats_df = (await db.table("products")).select(
            F.count(col("id")).alias("count"),
            F.sum(col("price")).alias("total"),
        )
        stats = (await stats_df.collect())[0]
        assert stats["count"] == 2
        assert stats["total"] == 1029.98

        await db.close()


# Test SQLModel Async Session
# ============================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not pytest.importorskip("sqlmodel", reason="SQLModel not installed"),
    reason="SQLModel not installed",
)
async def test_async_connect_with_sqlmodel_async_session(tmp_path):
    """Test async_connect() with SQLModel AsyncSession."""
    from sqlmodel import SQLModel, Field
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    # Define SQLModel
    class User(SQLModel, table=True):
        __tablename__ = "async_users"
        id: Union[int, None] = Field(default=None, primary_key=True)
        name: str
        age: int

    db_path = tmp_path / "test_sqlmodel_async_session.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        # Insert data using SQLModel session
        user1 = User(name="Alice", age=30)
        user2 = User(name="Bob", age=25)
        session.add(user1)
        session.add(user2)
        await session.commit()

        db = async_connect(session=session)

        # Verify dialect
        assert db.dialect.name == "sqlite"

        # Query using Moltres with SQLModel
        df = (await db.table(User)).select()
        results = await df.collect()

        # Results should be User instances (SQLModel)
        assert len(results) == 2
        assert isinstance(results[0], User)
        assert results[0].name in ("Alice", "Bob")
        assert results[0].age in (30, 25)

        await db.close()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not pytest.importorskip("sqlmodel", reason="SQLModel not installed"),
    reason="SQLModel not installed",
)
async def test_async_connect_with_sqlmodel_async_session_filtering(tmp_path):
    """Test filtering with SQLModel AsyncSession."""
    from sqlmodel import SQLModel, Field
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    class Product(SQLModel, table=True):
        __tablename__ = "async_products"
        id: Union[int, None] = Field(default=None, primary_key=True)
        name: str
        price: float

    db_path = tmp_path / "test_sqlmodel_async_filter.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        # Insert data
        session.add(Product(name="Laptop", price=999.99))
        session.add(Product(name="Mouse", price=29.99))
        await session.commit()

        db = async_connect(session=session)

        # Filter using Moltres
        df = (await db.table(Product)).select().where(col("price") > 100)
        results = await df.collect()

        assert len(results) == 1
        assert isinstance(results[0], Product)
        assert results[0].name == "Laptop"
        assert results[0].price == 999.99

        await db.close()


# Test Error Cases
# ================


def test_connect_with_invalid_session():
    """Test that connect() raises error for invalid session type."""
    with pytest.raises(TypeError, match="session must be a SQLAlchemy Session"):
        connect(session="not a session")


def test_connect_with_multiple_params():
    """Test that connect() raises error when multiple connection params provided."""
    from sqlalchemy import create_engine

    engine = create_engine("sqlite:///:memory:")
    with pytest.raises(ValueError, match="Cannot provide"):
        connect(dsn="sqlite:///:memory:", engine=engine)


def test_connect_with_session_and_engine():
    """Test that connect() raises error when both session and engine provided."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    with pytest.raises(ValueError, match="Cannot provide"):
        connect(session=session, engine=engine)

    session.close()


@pytest.mark.asyncio
async def test_async_connect_with_invalid_session():
    """Test that async_connect() raises error for invalid session type."""
    with pytest.raises(TypeError, match="session must be a SQLAlchemy AsyncSession"):
        await async_connect(session="not a session")


def test_async_connect_with_sync_session():
    """Test that async_connect() raises error for sync session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    db = async_connect(session=session)
    # Error is raised when engine is accessed, not when async_connect() is called
    with pytest.raises(TypeError, match="Session's bind must be an AsyncEngine"):
        _ = db.connection_manager.engine

    session.close()


# Test Dialect Detection
# ======================


def test_session_dialect_detection_sqlite(tmp_path):
    """Test dialect detection from session for SQLite."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test_dialect.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        db = connect(session=session)
        assert db.dialect.name == "sqlite"


def test_session_dialect_detection_postgresql(postgresql_connection):
    """Test dialect detection from session for PostgreSQL."""
    from sqlalchemy.orm import sessionmaker

    engine = postgresql_connection.connection_manager.engine
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        db = connect(session=session)
        assert db.dialect.name == "postgresql"


@pytest.mark.asyncio
async def test_async_session_dialect_detection_sqlite(tmp_path):
    """Test dialect detection from async session for SQLite."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test_async_dialect.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    SessionLocal = sessionmaker(engine, class_=AsyncSession)

    async with SessionLocal() as session:
        db = async_connect(session=session)
        assert db.dialect.name == "sqlite"
        await db.close()


# Test Complex Operations with Sessions
# ======================================


def test_session_with_joins(tmp_path):
    """Test joins using session."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test_session_joins.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SessionLocal = sessionmaker(bind=engine)

    with SessionLocal() as session:
        db = connect(session=session)

        # Create tables
        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        db.create_table(
            "orders",
            [
                column("id", "INTEGER", primary_key=True),
                column("user_id", "INTEGER"),
                column("amount", "REAL"),
            ],
        ).collect()

        # Insert data
        from moltres.io.records import Records

        Records(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        ).insert_into("users")

        Records(
            _data=[
                {"id": 1, "user_id": 1, "amount": 100.0},
                {"id": 2, "user_id": 1, "amount": 50.0},
            ],
            _database=db,
        ).insert_into("orders")

        # Join using Moltres
        users_df = db.table("users").select()
        orders_df = db.table("orders").select()

        # After join, reference columns without table prefix
        result_df = users_df.join(orders_df, on=[col("users.id") == col("orders.user_id")]).select(
            col("name"), col("amount")
        )

        results = result_df.collect()
        assert len(results) == 2
        assert all(r["name"] == "Alice" for r in results)


@pytest.mark.asyncio
async def test_async_session_with_joins(tmp_path):
    """Test joins using async session."""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    db_path = tmp_path / "test_async_session_joins.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionLocal() as session:
        db = async_connect(session=session)

        # Create tables
        await db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        ).collect()

        await db.create_table(
            "orders",
            [
                column("id", "INTEGER", primary_key=True),
                column("user_id", "INTEGER"),
                column("amount", "REAL"),
            ],
        ).collect()

        # Insert data
        from moltres.io.records import AsyncRecords

        records1 = AsyncRecords(
            _data=[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            _database=db,
        )
        await records1.insert_into("users")

        records2 = AsyncRecords(
            _data=[
                {"id": 1, "user_id": 1, "amount": 100.0},
                {"id": 2, "user_id": 1, "amount": 50.0},
            ],
            _database=db,
        )
        await records2.insert_into("orders")

        # Join using Moltres
        users_df = (await db.table("users")).select()
        orders_df = (await db.table("orders")).select()

        result_df = users_df.join(orders_df, on=[col("users.id") == col("orders.user_id")]).select(
            col("name"), col("amount")
        )

        results = await result_df.collect()
        assert len(results) == 2
        assert all(r["name"] == "Alice" for r in results)

        await db.close()


# Test with_model() with Sessions
# ===============================


@pytest.mark.skipif(
    not pytest.importorskip("sqlmodel", reason="SQLModel not installed"),
    reason="SQLModel not installed",
)
def test_session_with_model_attachment(tmp_path):
    """Test with_model() with session."""
    from sqlmodel import SQLModel, Field, create_engine
    from sqlalchemy.orm import sessionmaker

    class User(SQLModel, table=True):
        __tablename__ = "model_users"
        id: Union[int, None] = Field(default=None, primary_key=True)
        name: str
        age: int

    db_path = tmp_path / "test_with_model_session.db"
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine)
    with SessionLocal() as session:
        # Insert data
        session.add(User(name="Alice", age=30))
        session.add(User(name="Bob", age=25))
        session.commit()

        db = connect(session=session)

        # Start without model, then attach
        df = db.table("model_users").select()
        df_with_model = df.with_model(User)
        results = df_with_model.collect()

        assert len(results) == 2
        assert isinstance(results[0], User)
        assert results[0].name in ("Alice", "Bob")


@pytest.mark.asyncio
@pytest.mark.skipif(
    not pytest.importorskip("sqlmodel", reason="SQLModel not installed"),
    reason="SQLModel not installed",
)
async def test_async_session_with_model_attachment(tmp_path):
    """Test with_model() with async session."""
    from sqlmodel import SQLModel, Field
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker

    class User(SQLModel, table=True):
        __tablename__ = "async_model_users"
        id: Union[int, None] = Field(default=None, primary_key=True)
        name: str
        age: int

    db_path = tmp_path / "test_async_with_model_session.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    SessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        # Insert data
        session.add(User(name="Alice", age=30))
        session.add(User(name="Bob", age=25))
        await session.commit()

        db = async_connect(session=session)

        # Start without model, then attach
        df = (await db.table("async_model_users")).select()
        df_with_model = df.with_model(User)
        results = await df_with_model.collect()

        assert len(results) == 2
        assert isinstance(results[0], User)
        assert results[0].name in ("Alice", "Bob")

        await db.close()
