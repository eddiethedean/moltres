"""Pytest fixtures for example testing."""

import pytest

from moltres import connect


@pytest.fixture
def temp_db_path(tmp_path):
    """Create a temporary database path."""
    return tmp_path / "example.db"


@pytest.fixture
def example_db(temp_db_path):
    """Create a database connection for examples."""
    db = connect(f"sqlite:///{temp_db_path}")
    yield db
    # Cleanup handled by tmp_path


@pytest.fixture
def temp_file_dir(tmp_path):
    """Create a temporary directory for example files."""
    return tmp_path


@pytest.fixture
def sample_csv_file(temp_file_dir):
    """Create a sample CSV file for examples."""
    csv_path = temp_file_dir / "data.csv"
    csv_path.write_text("id,name,score\n1,Alice,95.5\n2,Bob,87.0\n")
    return csv_path


@pytest.fixture
def sample_json_file(temp_file_dir):
    """Create a sample JSON file for examples."""
    import json

    json_path = temp_file_dir / "data.json"
    data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
    json_path.write_text(json.dumps(data))
    return json_path


@pytest.fixture
def sample_table(temp_db_path):
    """Create a sample table with data for examples."""
    from moltres import connect
    from moltres.io.records import Records
    from moltres.table.schema import ColumnDef

    db = connect(f"sqlite:///{temp_db_path}")
    db.create_table(
        "users",
        [
            ColumnDef(name="id", type_name="INTEGER"),
            ColumnDef(name="name", type_name="TEXT"),
            ColumnDef(name="age", type_name="INTEGER"),
        ],
    ).collect()
    records = Records(
        _data=[
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
            {"id": 3, "name": "Charlie", "age": 35},
        ],
        _database=db,
    )
    records.insert_into("users")
    return db.table("users")
