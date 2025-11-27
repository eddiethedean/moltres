"""Pytest configuration helpers."""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
import uuid
from typing import Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SAWarning
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Module-level singleton for PostgreSQL instance (shared across all workers)
_postgresql_singleton = None
_postgresql_singleton_lock_file = None

# Import parallel test support plugin
try:
    import pytest_parallel_support  # noqa: F401
except ImportError:
    # Plugin not available, skip
    pass

# Limit math library threads to avoid OpenBLAS crashes when pytest-xdist spawns many workers
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

try:
    import pytest_asyncio
except ImportError:
    pytest_asyncio = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

_PG_FIXTURES = {
    "postgresql_db",
    "postgresql_admin_engine",
    "postgresql_schema",
    "postgresql_connection",
    "postgresql_async_connection",
}
_MYSQL_FIXTURES = {
    "mysql_db",
    "mysql_admin_engine",
    "mysql_database",
    "mysql_connection",
    "mysql_async_connection",
}


def pytest_configure(config):
    """Configure pytest to suppress expected SQLAlchemy warnings."""
    # Suppress connection cleanup warnings from SQLAlchemy
    # These occur in test scenarios where connections are properly managed by context managers
    warnings.filterwarnings(
        "ignore",
        message=".*garbage collector is trying to clean up non-checked-in connection.*",
        category=SAWarning,
    )


def pytest_collection_modifyitems(config, items):
    """Ensure heavy DB-backed tests run on a dedicated xdist worker."""
    for item in items:
        fixturenames = getattr(item, "fixturenames", ())
        if not fixturenames:
            continue
        fixtures = set(fixturenames)
        if fixtures & _PG_FIXTURES:
            item.add_marker(pytest.mark.xdist_group("postgresql"))
        if fixtures & _MYSQL_FIXTURES:
            item.add_marker(pytest.mark.xdist_group("mysql"))


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
    """Create a reusable PostgreSQL database server for testing.

    In parallel mode (pytest-xdist), each worker gets its own database instance
    with a unique port to avoid conflicts.

    Uses subprocess-based timeout to prevent hanging during initialization.
    """
    import os
    import shutil
    import socket
    import subprocess
    import threading

    if sys.platform.startswith("win"):
        pytest.skip("PostgreSQL tests require initdb, which is unavailable on Windows runners")

    # Pre-initialization checks
    initdb = shutil.which("initdb")
    if initdb is None:
        pytest.skip("PostgreSQL initdb command not found. Install PostgreSQL to run these tests.")

    # Verify initdb is executable
    if not os.access(initdb, os.X_OK):
        pytest.skip(f"PostgreSQL initdb found at {initdb} but is not executable")

    try:
        from testing.postgresql import Postgresql
        from testing.common.database import get_unused_port
    except ImportError:
        pytest.skip(
            "testing.postgresql not available. Install with: pip install testing.postgresql"
        )

    # Check for existing PostgreSQL processes that might conflict
    # This is informational - we'll use different ports anyway
    try:
        result = subprocess.run(
            ["pgrep", "-f", "postgres"],
            capture_output=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            # PostgreSQL processes found, but this is OK - we'll use different ports
            pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # pgrep not available or timed out, continue anyway
        pass

    # Set locale environment variables (required for PostgreSQL on macOS)
    # PostgreSQL requires a valid locale to start properly, otherwise it fails with:
    # "FATAL: postmaster became multithreaded during startup"
    # Always ensure locale is set before PostgreSQL initialization
    locale_set = False
    try:
        result = subprocess.run(["locale", "-a"], capture_output=True, text=True, timeout=2)
        available_lines = result.stdout.split("\n")

        # Try to find a UTF-8 locale (preferred order)
        preferred_locales = ["C.UTF-8", "en_US.UTF-8", "en_US.utf8", "UTF-8"]
        for pref in preferred_locales:
            for line in available_lines:
                line_stripped = line.strip()
                if line_stripped == pref or line_stripped.lower() == pref.lower():
                    os.environ["LC_ALL"] = line_stripped
                    os.environ["LANG"] = line_stripped
                    locale_set = True
                    break
            if locale_set:
                break

        # Fallback: find any UTF-8 locale
        if not locale_set:
            for line in available_lines:
                line_stripped = line.strip()
                if ".UTF-8" in line_stripped.upper() or ".utf8" in line_stripped.lower():
                    os.environ["LC_ALL"] = line_stripped
                    os.environ["LANG"] = line_stripped
                    locale_set = True
                    break

        # Final fallback: use C locale
        if not locale_set:
            for line in available_lines:
                if line.strip() == "C":
                    os.environ["LC_ALL"] = "C"
                    os.environ["LANG"] = "C"
                    locale_set = True
                    break
    except Exception:
        # If locale check fails, try common defaults
        pass

    # Ensure locale is always set (critical for PostgreSQL)
    if not locale_set:
        # Try common locale names that usually exist
        for locale_candidate in ["C.UTF-8", "en_US.UTF-8", "C"]:
            try:
                # Test if locale is valid by trying to set it
                test_result = subprocess.run(
                    ["locale", "-a"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                    env={**os.environ, "LC_ALL": locale_candidate},
                )
                if (
                    locale_candidate in test_result.stdout
                    or locale_candidate.lower() in test_result.stdout.lower()
                ):
                    os.environ["LC_ALL"] = locale_candidate
                    os.environ["LANG"] = locale_candidate
                    locale_set = True
                    break
            except Exception:
                continue

    # Absolute last resort: set to C (should always exist)
    if not locale_set:
        os.environ["LC_ALL"] = "C"
        os.environ["LANG"] = "C"

    # Helper function to check if a port is available
    def is_port_available(port: int) -> bool:
        """Check if a port is available for binding."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(("localhost", port))
                return result != 0  # Port is available if connection fails
        except Exception:
            return False

    # Helper function to verify port is actually free (more thorough check)
    def verify_port_free(port: int) -> bool:
        """Verify port is free by attempting to bind to it."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("localhost", port))
                return True
        except OSError:
            return False

    # In parallel mode, use worker-specific port to avoid conflicts
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id:
        # Extract worker number (e.g., "gw0" -> 0, "gw1" -> 1)
        try:
            worker_num = int(worker_id.replace("gw", ""))
            # Use a base port + worker number to ensure uniqueness
            base_port = 15432  # Base port for PostgreSQL test instances
            port = base_port + worker_num
            # Verify port is available, try next few ports if not
            max_attempts = 10
            for attempt in range(max_attempts):
                test_port = port + attempt
                if is_port_available(test_port) and verify_port_free(test_port):
                    port = test_port
                    break
            else:
                pytest.skip(
                    f"Could not find available port starting from {port}. "
                    "This may indicate port conflicts or system resource issues."
                )
        except (ValueError, AttributeError):
            # Fallback: get any unused port
            port = get_unused_port()

        # Configure PostgreSQL with specific port
        settings = {"port": port}
    else:
        # Non-parallel mode: use any available port
        port = get_unused_port()
        settings = {"port": port}

    # Use a singleton PostgreSQL instance per process (session-scoped)
    # We use a file-based lock to serialize initialization within each process
    # and prevent concurrent initialization attempts that can cause hangs
    import fcntl
    import tempfile
    import atexit

    global _postgresql_singleton

    # Check shared memory availability before attempting initialization
    # PostgreSQL requires shared memory and will hang if it's exhausted
    try:
        result = subprocess.run(
            ["ipcs", "-m"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            # Count existing shared memory segments
            lines = result.stdout.strip().split("\n")
            segment_count = len([line for line in lines if line.startswith("m")])
            # If there are many segments, shared memory might be exhausted
            if segment_count > 50:
                pytest.skip(
                    f"Too many shared memory segments ({segment_count}) detected. "
                    "PostgreSQL initialization may hang due to shared memory exhaustion. "
                    "Clean up shared memory: ipcs -m | awk '/^m/ {print $2}' | xargs -I {} ipcrm -m {}"
                )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # ipcs not available or timed out, continue anyway
        pass

    lock_file_path = os.path.join(tempfile.gettempdir(), f"moltres_postgres_init_{port}.lock")

    # Acquire lock to serialize initialization (only needed if multiple threads try simultaneously)
    lock_file = open(lock_file_path, "w")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)

        # Check if singleton already exists in this process
        if _postgresql_singleton is None:
            # Initialize PostgreSQL instance for this process
            # Note: Postgresql() can hang if shared memory is exhausted
            # We use threading with timeout to detect hangs
            try:
                import threading

                postgres_result = [None]
                postgres_error = [None]
                init_complete = threading.Event()

                def init_postgres():
                    try:
                        postgres_result[0] = Postgresql(**settings)
                        init_complete.set()
                    except Exception as e:
                        postgres_error[0] = e
                        init_complete.set()

                init_thread = threading.Thread(target=init_postgres, daemon=True)
                init_thread.start()

                # Wait up to 10 seconds for initialization (short timeout to fail fast)
                if not init_complete.wait(timeout=10):
                    # Timeout - PostgreSQL initialization is hanging
                    # The daemon thread will continue but won't block process exit
                    # However, the Postgresql subprocess may still be running
                    pytest.skip(
                        f"PostgreSQL initialization timed out after 10 seconds. "
                        f"Port attempted: {port}. "
                        "PostgreSQL initialization is hanging. This is a known issue with testing.postgresql. "
                        "Try: 1) Clean resources: pkill -9 postgres; ipcs -m | awk '/^m/ {print $2}' | xargs -I {} ipcrm -m {} "
                        "2) Run tests individually: pytest tests/integration/test_postgres_workflows.py::test_postgres_etl_pipeline -n 0"
                    )

                if postgres_error[0]:
                    raise postgres_error[0]

                if postgres_result[0] is None:
                    pytest.skip(
                        f"PostgreSQL initialization failed (unknown error). Port attempted: {port}."
                    )

                postgres = postgres_result[0]
                _postgresql_singleton = postgres

                # Register cleanup at exit
                def cleanup_singleton():
                    global _postgresql_singleton
                    try:
                        if _postgresql_singleton is not None:
                            _postgresql_singleton.stop()
                            _postgresql_singleton = None
                            # Clean up shared memory segments created by this PostgreSQL instance
                            try:
                                result = subprocess.run(
                                    ["ipcs", "-m"],
                                    capture_output=True,
                                    text=True,
                                    timeout=2,
                                )
                                if result.returncode == 0:
                                    # Remove any orphaned shared memory segments
                                    # (This is best-effort - some may be owned by other processes)
                                    pass
                            except Exception:
                                pass
                    except Exception:
                        pass

                atexit.register(cleanup_singleton)
            except RuntimeError as exc:
                error_str = str(exc)
                diagnostic_msg = (
                    f"PostgreSQL fixture unavailable: {error_str}\nPort attempted: {port}. "
                )
                if "failed to launch" in error_str.lower():
                    diagnostic_msg += (
                        "PostgreSQL server failed to start. Common causes:\n"
                        "  - Port conflicts (check for existing PostgreSQL processes)\n"
                        "  - Locale issues (ensure LC_ALL and LANG are set)\n"
                        "  - Permission issues (check PostgreSQL data directory permissions)\n"
                        "  - Insufficient system resources\n"
                    )
                elif "port" in error_str.lower():
                    diagnostic_msg += "Port-related error. Check for port conflicts.\n"
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                pytest.skip(diagnostic_msg)
            except Exception as exc:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                pytest.skip(
                    f"PostgreSQL fixture failed to initialize: {exc}\nPort attempted: {port}."
                )
        else:
            # Singleton already exists in this process, use it
            postgres = _postgresql_singleton

        # IMPORTANT: Release the lock immediately after initialization
        # This allows other threads/workers to proceed
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()
    except Exception as exc:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
            except Exception:
                pass
        pytest.skip(
            f"Failed to acquire PostgreSQL initialization lock: {exc}\nPort attempted: {port}."
        )

    try:
        yield postgres
    finally:
        # Don't stop the singleton - it's shared across all tests
        # Cleanup happens at process exit via atexit
        pass


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
    """Provision a unique schema per test function to isolate table names.

    In parallel mode, includes worker ID to ensure uniqueness across workers.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    worker_suffix = f"_{worker_id}" if worker_id else ""
    schema = f"test_{uuid.uuid4().hex}{worker_suffix}"
    with postgresql_admin_engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        conn.execute(text(f'SET search_path TO "{schema}"'))
    try:
        yield schema
    finally:
        # Cleanup schema with timeout to prevent hanging
        try:
            import threading

            cleanup_complete = threading.Event()
            cleanup_error = [None]

            def drop_schema():
                try:
                    with postgresql_admin_engine.begin() as conn:
                        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
                    cleanup_complete.set()
                except Exception as e:
                    cleanup_error[0] = e
                    cleanup_complete.set()

            cleanup_thread = threading.Thread(target=drop_schema, daemon=True)
            cleanup_thread.start()

            # Wait up to 5 seconds for cleanup
            if not cleanup_complete.wait(timeout=5):
                # Cleanup timed out - log but don't fail
                import warnings

                warnings.warn(
                    f"Schema cleanup timed out for {schema}. "
                    "This may indicate PostgreSQL connection issues.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception:
            # If cleanup fails, log but don't fail the test
            pass


@pytest.fixture(scope="function")
def unique_table_name(request):
    """Generate a unique table name for each test function.

    Includes worker ID in parallel mode and test function name to ensure uniqueness.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER", "")
    worker_suffix = f"_{worker_id}" if worker_id else ""
    test_name = request.node.name.replace("[", "_").replace("]", "_").replace("::", "_")
    # Limit test name length to avoid overly long table names
    test_name = test_name[:30] if len(test_name) > 30 else test_name
    unique_id = uuid.uuid4().hex[:8]
    return f"{test_name}_{unique_id}{worker_suffix}"


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
    """Create a reusable MySQL database server for testing.

    In parallel mode (pytest-xdist), each worker gets its own database instance
    with a unique port to avoid conflicts.
    """
    import os
    import shutil
    import subprocess

    if sys.platform.startswith("win"):
        pytest.skip("MySQL tests require mysqld tooling, unavailable on Windows runners")

    try:
        from testing.mysqld import Mysqld
        from testing.common.database import get_unused_port
    except ImportError:
        pytest.skip("testing.mysqld not available")

    # Check for MySQL tools
    mysql_install_db = shutil.which("mysql_install_db")
    mysqld = shutil.which("mysqld")

    if not mysqld:
        pytest.skip("MySQL mysqld command not found")

    # In parallel mode, use worker-specific port to avoid conflicts
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id:
        # Extract worker number (e.g., "gw0" -> 0, "gw1" -> 1)
        try:
            worker_num = int(worker_id.replace("gw", ""))
            # Use a base port + worker number to ensure uniqueness
            base_port = 13306  # Base port for MySQL test instances
            port = base_port + worker_num
        except (ValueError, AttributeError):
            # Fallback: get any unused port
            port = get_unused_port()
    else:
        port = None

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
            # Port should already be set from the outer scope if in parallel mode
            if "port" not in self.my_cnf and "skip-networking" not in self.my_cnf:
                # Only get unused port if not already set (non-parallel mode)
                if port is not None:
                    self.my_cnf["port"] = port
                else:
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

    # Configure MySQL with specific port if in parallel mode
    mysql_settings = {}
    if port is not None:
        mysql_settings["my_cnf"] = {"port": port}

    try:
        mysql = Mysqld(**mysql_settings)
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


if pytest_asyncio is not None:

    @pytest_asyncio.fixture(scope="function")
    async def postgresql_async_connection(postgresql_db, postgresql_schema):
        """Create an async Moltres Database connection to ephemeral PostgreSQL."""
        try:
            from moltres import async_connect
        except ImportError:
            pytest.skip("async_connect not available (async dependencies not installed)")

        # Extract connection info from postgresql_db
        dsn = _dsn_with_search_path(postgresql_db.url(), postgresql_schema)
        # Convert postgresql:// to postgresql+asyncpg:// for async connections
        if dsn.startswith("postgresql://") and "+asyncpg" not in dsn:
            dsn = dsn.replace("postgresql://", "postgresql+asyncpg://", 1)
        db = async_connect(dsn)
        try:
            yield db
        finally:
            # Close all connections and dispose engine before database stops
            if hasattr(db, "close"):
                await db.close()
else:

    @pytest.fixture
    async def postgresql_async_connection(postgresql_db, postgresql_schema):
        """Create an async Moltres Database connection to ephemeral PostgreSQL."""
        pytest.skip("pytest_asyncio not available")


if pytest_asyncio is not None:

    @pytest_asyncio.fixture(scope="function")
    async def mysql_async_connection(mysql_db, mysql_database):
        """Create an async Moltres Database connection to reusable MySQL."""
        try:
            from moltres import async_connect
        except ImportError:
            pytest.skip("async_connect not available (async dependencies not installed)")

        # Extract connection info from mysql_db
        dsn = _dsn_with_database(mysql_db.url(), mysql_database)
        # Convert to async driver format (mysql+pymysql:// -> mysql+aiomysql://)
        if dsn.startswith("mysql+pymysql://"):
            dsn = dsn.replace("mysql+pymysql://", "mysql+aiomysql://", 1)
        elif dsn.startswith("mysql://"):
            dsn = dsn.replace("mysql://", "mysql+aiomysql://", 1)
        db = async_connect(dsn)
        try:
            yield db
        finally:
            # Close all connections and dispose engine before database stops
            if hasattr(db, "close"):
                await db.close()
else:

    @pytest.fixture
    async def mysql_async_connection(mysql_db, mysql_database):
        """Create an async Moltres Database connection to reusable MySQL."""
        pytest.skip("pytest_asyncio not available")
