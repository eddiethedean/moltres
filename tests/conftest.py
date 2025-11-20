"""Pytest configuration helpers."""

from __future__ import annotations

import sys
from pathlib import Path

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
