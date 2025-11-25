"""Shared fixtures for integration tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from moltres import connect
from moltres.io.records import Records
from moltres.table.schema import column


@pytest.fixture
def sample_database(tmp_path):
    """Create a pre-populated database with customers, orders, and products tables."""
    db_path = tmp_path / "sample.db"
    db = connect(f"sqlite:///{db_path.as_posix()}")

    # Create customers table
    db.create_table(
        "customers",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("country", "TEXT"),
            column("active", "INTEGER"),
        ],
    ).collect()

    # Create orders table
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("customer_id", "INTEGER"),
            column("amount", "REAL"),
            column("status", "TEXT"),
            column("order_date", "TEXT"),
        ],
    ).collect()

    # Create products table
    db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("price", "REAL"),
            column("category", "TEXT"),
        ],
    ).collect()

    # Insert sample data
    Records.from_list(
        [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "country": "USA", "active": 1},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "country": "UK", "active": 1},
            {
                "id": 3,
                "name": "Charlie",
                "email": "charlie@example.com",
                "country": "USA",
                "active": 0,
            },
            {"id": 4, "name": "Diana", "email": "diana@example.com", "country": "CA", "active": 1},
        ],
        database=db,
    ).insert_into("customers")

    Records.from_list(
        [
            {
                "id": 101,
                "customer_id": 1,
                "amount": 100.0,
                "status": "completed",
                "order_date": "2024-01-15",
            },
            {
                "id": 102,
                "customer_id": 1,
                "amount": 50.0,
                "status": "pending",
                "order_date": "2024-01-20",
            },
            {
                "id": 103,
                "customer_id": 2,
                "amount": 200.0,
                "status": "completed",
                "order_date": "2024-01-18",
            },
            {
                "id": 104,
                "customer_id": 3,
                "amount": 75.0,
                "status": "completed",
                "order_date": "2024-01-19",
            },
            {
                "id": 105,
                "customer_id": 4,
                "amount": 150.0,
                "status": "pending",
                "order_date": "2024-01-21",
            },
        ],
        database=db,
    ).insert_into("orders")

    Records.from_list(
        [
            {"id": 1, "name": "Widget", "price": 10.0, "category": "Electronics"},
            {"id": 2, "name": "Gadget", "price": 20.0, "category": "Electronics"},
            {"id": 3, "name": "Tool", "price": 15.0, "category": "Hardware"},
            {"id": 4, "name": "Book", "price": 5.0, "category": "Education"},
        ],
        database=db,
    ).insert_into("products")

    yield db

    db.close()


@pytest.fixture
def large_dataset(tmp_path):
    """Create a large dataset for performance testing."""
    db_path = tmp_path / "large.db"
    db = connect(f"sqlite:///{db_path.as_posix()}")

    # Create table
    db.create_table(
        "large_table",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("value", "REAL"),
            column("category", "TEXT"),
        ],
    ).collect()

    # Insert 10,000 rows
    data = [
        {"id": i, "name": f"item_{i}", "value": i * 1.5, "category": f"cat_{i % 10}"}
        for i in range(10000)
    ]

    # Insert in batches for efficiency
    batch_size = 1000
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        Records.from_list(batch, database=db).insert_into("large_table")

    yield db

    db.close()


@pytest.fixture
def temp_file_factory(tmp_path):
    """Factory for creating temporary test files."""

    def _create_file(extension: str, content: str | list[dict] | None = None) -> Path:
        """Create a temporary file with optional content.

        Args:
            extension: File extension (e.g., '.csv', '.json')
            content: Optional content. For CSV, pass list of dicts. For JSON, pass string or list.

        Returns:
            Path to the created file.
        """
        file_path = tmp_path / f"test_{len(list(tmp_path.glob('*')))}.{extension}"

        if content is not None:
            if extension == ".csv" and isinstance(content, list):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "w", newline="", encoding="utf-8") as f:
                    if content and len(content) > 0:
                        writer = csv.DictWriter(f, fieldnames=content[0].keys())
                        writer.writeheader()
                        writer.writerows(content)
                        f.flush()  # Ensure data is written
                    else:
                        # Empty list - create empty file with header
                        writer = csv.writer(f)
                        writer.writerow(["id", "value"])  # Default header
                        f.flush()
                # Verify file was created and has content
                if (
                    file_path.exists()
                    and file_path.stat().st_size == 0
                    and content
                    and len(content) > 0
                ):
                    # Retry with explicit flush
                    with open(file_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=content[0].keys())
                        writer.writeheader()
                        writer.writerows(content)
            elif extension == ".json":
                with open(file_path, "w") as f:
                    if isinstance(content, list):
                        json.dump(content, f)
                    else:
                        f.write(content)
            elif extension == ".jsonl":
                with open(file_path, "w") as f:
                    if isinstance(content, list):
                        for item in content:
                            json.dump(item, f)
                            f.write("\n")
                    else:
                        f.write(content)
            else:
                with open(file_path, "w") as f:
                    f.write(str(content))

        return file_path

    return _create_file


@pytest.fixture
def transaction_db(tmp_path):
    """Create a database with transaction support configured."""
    db_path = tmp_path / "transaction.db"
    db = connect(f"sqlite:///{db_path.as_posix()}")

    # Create a simple table for transaction tests
    db.create_table(
        "test_table",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "TEXT"),
        ],
    ).collect()

    yield db

    db.close()


@pytest.fixture
def empty_database(tmp_path):
    """Create an empty database for testing."""
    db_path = tmp_path / "empty.db"
    db = connect(f"sqlite:///{db_path.as_posix()}")

    yield db

    db.close()
