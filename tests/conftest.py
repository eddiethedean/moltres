"""Pytest configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sqlite_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    from moltres import connect

    db_path = tmp_path / "test.db"
    return connect(f"sqlite:///{db_path}")


@pytest.fixture
def sample_table(sqlite_db):
    """Create a sample table with test data."""
    from moltres.table.schema import column

    sqlite_db.create_table(
        "users",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "TEXT", nullable=False),
            column("email", "TEXT", nullable=True),
            column("age", "INTEGER", nullable=True),
        ],
    )

    table = sqlite_db.table("users")
    table.insert(
        [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
            {"id": 3, "name": "Charlie", "email": None, "age": 35},
        ]
    )

    return table


def create_sample_table(db, table_name: str = "users"):
    """Helper function to create a sample table in any database."""
    from moltres.table.schema import column

    # Use appropriate types for each database
    # For PostgreSQL/MySQL, use VARCHAR instead of TEXT
    db.create_table(
        table_name,
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "VARCHAR(255)", nullable=False),
            column("email", "VARCHAR(255)", nullable=True),
            column("age", "INTEGER", nullable=True),
        ],
    )

    table = db.table(table_name)
    table.insert(
        [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
            {"id": 3, "name": "Charlie", "email": None, "age": 35},
        ]
    )

    return table


@pytest.fixture(scope="function")
def postgresql_db() -> Generator:
    """Create an ephemeral PostgreSQL database for testing."""
    try:
        from testing.postgresql import Postgresql
    except ImportError:
        pytest.skip("testing.postgresql not available")

    postgres = Postgresql()
    yield postgres
    postgres.stop()


@pytest.fixture(scope="function")
def postgresql_connection(postgresql_db) -> Generator:
    """Create a Moltres Database connection to ephemeral PostgreSQL."""
    from moltres import connect

    # Extract connection info from postgresql_db
    dsn = postgresql_db.url()
    # Convert to SQLAlchemy format: postgresql://user:pass@host:port/dbname
    db = connect(dsn)
    try:
        yield db
    finally:
        # Close all connections and dispose engine before database stops
        db.close()


@pytest.fixture(scope="function")
def mysql_db() -> Generator:
    """Create an ephemeral MySQL database for testing."""
    try:
        from testing.mysqld import Mysqld
    except ImportError:
        pytest.skip("testing.mysqld not available")

    mysql = Mysqld()
    yield mysql
    mysql.stop()


@pytest.fixture(scope="function")
def mysql_connection(mysql_db) -> Generator:
    """Create a Moltres Database connection to ephemeral MySQL."""
    from moltres import connect

    # Extract connection info from mysql_db
    dsn = mysql_db.url()
    # Convert to SQLAlchemy format: mysql://user:pass@host:port/dbname
    db = connect(dsn)
    try:
        yield db
    finally:
        # Close all connections and dispose engine before database stops
        db.close()


def seed_customers_orders(db):
    """Helper function to seed customers and orders tables in any database."""
    from moltres.table.schema import column

    # Create customers table
    db.create_table(
        "customers",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "VARCHAR(255)", nullable=False),
            column("active", "INTEGER", nullable=True),
        ],
    )

    # Create orders table
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("customer_id", "INTEGER", nullable=True),
            column("amount", "INTEGER", nullable=True),
        ],
    )

    # Insert test data
    customers_table = db.table("customers")
    customers_table.insert(
        [
            {"id": 1, "name": "Alice", "active": 1},
            {"id": 2, "name": "Bob", "active": 0},
        ]
    )

    orders_table = db.table("orders")
    orders_table.insert(
        [
            {"id": 100, "customer_id": 1, "amount": 50},
            {"id": 101, "customer_id": 2, "amount": 75},
        ]
    )


async def seed_customers_orders_async(db):
    """Helper function to seed customers and orders tables in any async database."""
    from moltres.table.schema import column

    # Create customers table
    await db.create_table(
        "customers",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("name", "VARCHAR(255)", nullable=False),
            column("active", "INTEGER", nullable=True),
        ],
    )

    # Create orders table
    await db.create_table(
        "orders",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("customer_id", "INTEGER", nullable=True),
            column("amount", "INTEGER", nullable=True),
        ],
    )

    # Insert test data
    customers_table = await db.table("customers")
    await customers_table.insert(
        [
            {"id": 1, "name": "Alice", "active": 1},
            {"id": 2, "name": "Bob", "active": 0},
        ]
    )

    orders_table = await db.table("orders")
    await orders_table.insert(
        [
            {"id": 100, "customer_id": 1, "amount": 50},
            {"id": 101, "customer_id": 2, "amount": 75},
        ]
    )


@pytest.fixture(scope="function")
async def postgresql_async_connection(postgresql_db):
    """Create an async Moltres Database connection to ephemeral PostgreSQL."""
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("async_connect not available (async dependencies not installed)")

    # Extract connection info from postgresql_db
    dsn = postgresql_db.url()
    # async_connect will automatically convert postgresql:// to postgresql+asyncpg://
    db = async_connect(dsn)
    try:
        yield db
    finally:
        # Close all connections and dispose engine before database stops
        if hasattr(db, "close"):
            await db.close()


@pytest.fixture(scope="function")
async def mysql_async_connection(mysql_db):
    """Create an async Moltres Database connection to ephemeral MySQL."""
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("async_connect not available (async dependencies not installed)")

    # Extract connection info from mysql_db
    dsn = mysql_db.url()
    # async_connect will automatically convert mysql:// to mysql+aiomysql://
    db = async_connect(dsn)
    try:
        yield db
    finally:
        # Close all connections and dispose engine before database stops
        if hasattr(db, "close"):
            await db.close()
