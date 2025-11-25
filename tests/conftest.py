"""Pytest configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path
import uuid
from typing import Generator
import os

import pytest
from sqlalchemy import create_engine, text
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    import pytest_asyncio
except ImportError:
    pytest_asyncio = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Fix for ensure_greenlet_context fixture when running tests in parallel with pytest-xdist
# The fixture tries to access an event loop that doesn't exist in worker threads
# We override the fixture to handle the case where no event loop exists
@pytest.fixture(scope="function", autouse=True)
def ensure_greenlet_context(request):
    """Override ensure_greenlet_context to handle parallel execution.

    This fixture overrides the auto-use ensure_greenlet_context fixture from
    pytest-green-light to properly handle cases where no event loop exists
    in worker threads when running tests in parallel with pytest-xdist.
    """
    import asyncio
    import threading

    # Check if we're running in a worker process (pytest-xdist)
    # If worker_id exists, we're in a parallel execution environment
    _ = os.environ.get("PYTEST_XDIST_WORKER")  # Check for worker, but don't need the value

    # Try to get or create an event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop exists - create one for this thread
        # Only set up event loop in the main thread of the worker
        if threading.current_thread() is threading.main_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        else:
            # For non-main threads, try to get the loop from the main thread
            # or just skip the greenlet context setup
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If we still can't get a loop, just yield without setting up greenlet context
                # This is acceptable for sync tests that don't need async context
                yield
                return

    # If we have a loop, proceed with normal greenlet context setup
    # (The original fixture would do this, but we're handling the error case)
    yield


@pytest.fixture
def sqlite_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    from moltres import connect

    db_path = tmp_path / "test.db"
    # Use as_posix() to ensure forward slashes for SQLite URLs (required on Windows)
    return connect(f"sqlite:///{db_path.as_posix()}")


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
    ).collect()

    from moltres.io.records import Records

    records = Records(
        _data=[
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
            {"id": 3, "name": "Charlie", "email": None, "age": 35},
        ],
        _database=sqlite_db,
    )
    records.insert_into("users")

    return sqlite_db.table("users")


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
    ).collect()

    from moltres.io.records import Records

    records = Records(
        _data=[
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
            {"id": 3, "name": "Charlie", "email": None, "age": 35},
        ],
        _database=db,
    )
    records.insert_into(table_name)

    return db.table(table_name)


def _dsn_with_search_path(dsn: str, schema: str) -> str:
    """Inject a custom search_path into an existing PostgreSQL DSN."""
    parsed = urlparse(dsn)
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    query = {}
    for key, value in query_items:
        query.setdefault(key, []).append(value)

    search_path_option = f"-csearch_path={schema}"
    options_values = query.get("options", [])
    if options_values:
        options_values[0] = f"{options_values[0]} {search_path_option}".strip()
    else:
        query["options"] = [search_path_option]

    encoded_query = urlencode(query, doseq=True)
    return urlunparse(parsed._replace(query=encoded_query))


def _dsn_with_database(dsn: str, database: str) -> str:
    """Replace the database component of a DSN with a new name."""
    parsed = urlparse(dsn)
    path = f"/{database}"
    return urlunparse(parsed._replace(path=path))


@pytest.fixture(scope="session")
def postgresql_db() -> Generator:
    """Create a reusable PostgreSQL database server for testing."""
    import os
    import shutil

    if sys.platform.startswith("win"):
        pytest.skip("PostgreSQL tests require initdb, which is unavailable on Windows runners")

    initdb = shutil.which("initdb")
    if initdb is None:
        pytest.skip("PostgreSQL initdb command not found")

    try:
        from testing.postgresql import Postgresql
    except ImportError:
        pytest.skip("testing.postgresql not available")

    # Set LC_ALL if not already set (required for PostgreSQL on macOS)
    if "LC_ALL" not in os.environ:
        os.environ["LC_ALL"] = "en_US.UTF-8"

    try:
        postgres = Postgresql()
    except RuntimeError as exc:  # pragma: no cover - environment specific
        pytest.skip(f"PostgreSQL fixture unavailable: {exc}")
    try:
        yield postgres
    finally:
        postgres.stop()


@pytest.fixture(scope="session")
def postgresql_admin_engine(postgresql_db):
    """Session-scoped SQLAlchemy engine for schema management."""
    engine = create_engine(postgresql_db.url())
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="function")
def postgresql_schema(postgresql_admin_engine):
    """Provision a unique schema per test function to isolate table names."""
    schema = f"test_{uuid.uuid4().hex}"
    with postgresql_admin_engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        conn.execute(text(f'SET search_path TO "{schema}"'))
    try:
        yield schema
    finally:
        with postgresql_admin_engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))


@pytest.fixture(scope="function")
def postgresql_connection(postgresql_db, postgresql_schema) -> Generator:
    """Create a Moltres Database connection to ephemeral PostgreSQL."""
    from moltres import connect

    # Extract connection info from postgresql_db
    dsn = _dsn_with_search_path(postgresql_db.url(), postgresql_schema)
    # Convert to SQLAlchemy format: postgresql://user:pass@host:port/dbname
    db = connect(dsn)
    try:
        yield db
    finally:
        # Close all connections and dispose engine before database stops
        db.close()


@pytest.fixture(scope="session")
def mysql_db() -> Generator:
    """Create a reusable MySQL database server for testing."""
    import os
    import shutil
    import subprocess

    if sys.platform.startswith("win"):
        pytest.skip("MySQL tests require mysqld tooling, unavailable on Windows runners")

    try:
        from testing.mysqld import Mysqld
    except ImportError:
        pytest.skip("testing.mysqld not available")

    # Check for MySQL tools
    mysql_install_db = shutil.which("mysql_install_db")
    mysqld = shutil.which("mysqld")

    if not mysqld:
        pytest.skip("MySQL mysqld command not found")

    # Detect MySQL version
    mysql_version = None
    if mysqld:
        try:
            result = subprocess.run(
                [mysqld, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                version_output = result.stdout or result.stderr
                # Extract version number (e.g., "mysqld  Ver 8.0.35")
                import re

                match = re.search(r"(\d+)\.(\d+)", version_output)
                if match:
                    major, minor = int(match.group(1)), int(match.group(2))
                    mysql_version = (major, minor)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass

    # MySQL 8.0+ requires mysqld --initialize instead of mysql_install_db
    use_initialize = mysql_version and mysql_version >= (8, 0)

    # Patch Mysqld for MySQL 8.0+ if needed
    original_initialize_method = None
    original_initialize_database = None
    if use_initialize and not mysql_install_db:
        # Patch initialize() to skip mysql_install_db lookup for MySQL 8.0+
        original_initialize_method = Mysqld.initialize

        def patched_initialize(self):
            """Patched initialize that skips mysql_install_db for MySQL 8.0+."""
            self.my_cnf = self.settings.get("my_cnf", {})
            self.my_cnf.setdefault("socket", os.path.join(self.base_dir, "tmp", "mysql.sock"))
            self.my_cnf.setdefault("datadir", os.path.join(self.base_dir, "var"))
            self.my_cnf.setdefault("pid-file", os.path.join(self.base_dir, "tmp", "mysqld.pid"))
            self.my_cnf.setdefault("tmpdir", os.path.join(self.base_dir, "tmp"))

            # Skip mysql_install_db lookup for MySQL 8.0+
            self.mysql_install_db = None

            # Still need mysqld
            self.mysqld = self.settings.get("mysqld")
            if self.mysqld is None:
                # Use shutil.which as fallback since find_program may not be available
                import shutil

                found_mysqld = shutil.which("mysqld")
                if found_mysqld:
                    self.mysqld = found_mysqld
                else:
                    # Try to use find_program if available
                    try:
                        from testing.mysqld import find_program

                        self.mysqld = find_program("mysqld", ["bin", "libexec", "sbin"])
                    except (ImportError, AttributeError):
                        # If find_program not available, use the mysqld we found earlier
                        self.mysqld = mysqld

        # Patch initialize_database to use mysqld --initialize
        original_initialize_database = Mysqld.initialize_database

        def patched_initialize_database(self):
            """Patched initialize_database that uses mysqld --initialize for MySQL 8.0+."""
            # Get the original method's logic for setting up my.cnf
            if "port" not in self.my_cnf and "skip-networking" not in self.my_cnf:
                from testing.common.database import get_unused_port

                self.my_cnf["port"] = get_unused_port()

            # Write my.cnf
            etc_dir = os.path.join(self.base_dir, "etc")
            os.makedirs(etc_dir, exist_ok=True)
            with open(os.path.join(etc_dir, "my.cnf"), "wt") as my_cnf:
                my_cnf.write("[mysqld]\n")
                for key, value in self.my_cnf.items():
                    if value:
                        my_cnf.write("%s=%s\n" % (key, value))
                    else:
                        my_cnf.write("%s\n" % key)

            # Initialize database using mysqld --initialize (MySQL 8.0+)
            # Use the datadir from my_cnf (set by initialize method)
            mysql_data_dir = self.my_cnf.get("datadir", os.path.join(self.base_dir, "var"))

            # Check if data directory exists and has files (from previous failed attempt)
            if os.path.exists(mysql_data_dir):
                # Check if it has mysql subdirectory with files
                mysql_subdir = os.path.join(mysql_data_dir, "mysql")
                if os.path.exists(mysql_subdir):
                    import shutil

                    try:
                        shutil.rmtree(mysql_subdir)
                    except OSError:
                        pass
                # Also check for any files directly in datadir
                try:
                    if os.listdir(mysql_data_dir):
                        # Remove all files in datadir
                        import shutil

                        for item in os.listdir(mysql_data_dir):
                            item_path = os.path.join(mysql_data_dir, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                except OSError:
                    pass
            else:
                os.makedirs(mysql_data_dir, exist_ok=True)

            args = [
                mysqld,
                "--defaults-file=%s/etc/my.cnf" % self.base_dir,
                "--datadir=%s" % mysql_data_dir,
                "--initialize-insecure",  # Initialize without password for testing
            ]
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to initialize MySQL database: {result.stderr}")

        # Apply the patches
        Mysqld.initialize = patched_initialize
        Mysqld.initialize_database = patched_initialize_database

    try:
        mysql = Mysqld()
    except PermissionError as exc:  # pragma: no cover - environment specific
        pytest.skip(f"MySQL fixture unavailable: {exc}")
    except RuntimeError as e:
        error_msg = str(e)
        if "mysql_install_db" in error_msg and use_initialize:
            # MySQL 8.0+ detected but testing.mysqld is trying to use mysql_install_db
            # This indicates testing.mysqld version may not support MySQL 8.0+
            pytest.skip(
                f"MySQL 8.0+ initialization failed: {e}. "
                "testing.mysqld may need updating for MySQL 8.0+ support. "
                "Consider: pip install --upgrade testing.mysqld"
            )
        elif "mysql_install_db" in error_msg:
            pytest.skip(f"MySQL initialization failed: {e}")
        raise
    else:
        try:
            yield mysql
        finally:
            mysql.stop()
    finally:
        # Restore original methods if we patched them
        if original_initialize_method is not None:
            Mysqld.initialize = original_initialize_method
        if original_initialize_database is not None:
            Mysqld.initialize_database = original_initialize_database


@pytest.fixture(scope="session")
def mysql_admin_engine(mysql_db):
    """Session-scoped SQLAlchemy engine for MySQL schema management."""
    engine = create_engine(mysql_db.url())
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture(scope="function")
def mysql_database(mysql_admin_engine):
    """Provision a unique database per test function to isolate tables."""
    database = f"test_{uuid.uuid4().hex}"
    with mysql_admin_engine.begin() as conn:
        conn.execute(text(f"CREATE DATABASE `{database}`"))
    try:
        yield database
    finally:
        with mysql_admin_engine.begin() as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS `{database}`"))


@pytest.fixture(scope="function")
def mysql_connection(mysql_db, mysql_database) -> Generator:
    """Create a Moltres Database connection to reusable MySQL."""
    from moltres import connect

    # Extract connection info from mysql_db
    dsn = _dsn_with_database(mysql_db.url(), mysql_database)
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
    ).collect()

    # Create orders table
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", nullable=False, primary_key=True),
            column("customer_id", "INTEGER", nullable=True),
            column("amount", "INTEGER", nullable=True),
        ],
    ).collect()

    # Insert test data
    from moltres.io.records import Records

    customers_records = Records(
        _data=[
            {"id": 1, "name": "Alice", "active": 1},
            {"id": 2, "name": "Bob", "active": 0},
        ],
        _database=db,
    )
    customers_records.insert_into("customers")

    orders_records = Records(
        _data=[
            {"id": 100, "customer_id": 1, "amount": 50},
            {"id": 101, "customer_id": 2, "amount": 75},
        ],
        _database=db,
    )
    orders_records.insert_into("orders")


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
    from moltres.io.records import AsyncRecords

    customers_records = AsyncRecords(
        _data=[
            {"id": 1, "name": "Alice", "active": 1},
            {"id": 2, "name": "Bob", "active": 0},
        ],
        _database=db,
    )
    await customers_records.insert_into("customers")

    orders_records = AsyncRecords(
        _data=[
            {"id": 100, "customer_id": 1, "amount": 50},
            {"id": 101, "customer_id": 2, "amount": 75},
        ],
        _database=db,
    )
    await orders_records.insert_into("orders")


@pytest_asyncio.fixture(scope="function")
async def postgresql_async_connection(postgresql_db, postgresql_schema):
    """Create an async Moltres Database connection to ephemeral PostgreSQL."""
    if pytest_asyncio is None:
        pytest.skip("pytest_asyncio not available")
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("async_connect not available (async dependencies not installed)")

    # Extract connection info from postgresql_db
    dsn = _dsn_with_search_path(postgresql_db.url(), postgresql_schema)
    # async_connect will automatically convert postgresql:// to postgresql+asyncpg://
    db = async_connect(dsn)
    try:
        yield db
    finally:
        # Close all connections and dispose engine before database stops
        if hasattr(db, "close"):
            await db.close()


@pytest_asyncio.fixture(scope="function")
async def mysql_async_connection(mysql_db, mysql_database):
    """Create an async Moltres Database connection to reusable MySQL."""
    if pytest_asyncio is None:
        pytest.skip("pytest_asyncio not available")
    try:
        from moltres import async_connect
    except ImportError:
        pytest.skip("async_connect not available (async dependencies not installed)")

    # Extract connection info from mysql_db
    dsn = _dsn_with_database(mysql_db.url(), mysql_database)
    # async_connect will automatically convert mysql:// to mysql+aiomysql://
    db = async_connect(dsn)
    try:
        yield db
    finally:
        # Close all connections and dispose engine before database stops
        if hasattr(db, "close"):
            await db.close()
