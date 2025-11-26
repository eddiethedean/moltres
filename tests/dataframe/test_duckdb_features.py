"""Tests for DuckDB-specific features and compatibility."""

import pytest

from moltres import col, connect
from moltres.expressions import functions as F


@pytest.fixture
def duckdb_db():
    """Create a DuckDB in-memory database connection."""
    try:
        db = connect("duckdb:///:memory:")
        yield db
        db.close()
    except Exception:
        pytest.skip("DuckDB not available. Install with: pip install duckdb-engine")


def test_basic_connection(duckdb_db):
    """Test basic DuckDB connection."""
    assert duckdb_db is not None


def test_create_table_and_insert(duckdb_db):
    """Test creating a table and inserting data in DuckDB."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "test_users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "VARCHAR"),
                column("age", "INTEGER"),
            ],
        ).collect()

    # Insert data
    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ],
        database=db,
    ).insert_into("test_users")

    # Query data
    result = db.table("test_users").select().collect()
    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"


def test_filter_clause_support(duckdb_db):
    """Test FILTER clause support in DuckDB (key differentiator from SQLite)."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "test_orders",
            [
                column("id", "INTEGER", primary_key=True),
                column("amount", "REAL"),
                column("status", "VARCHAR"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "amount": 100.0, "status": "active"},
            {"id": 2, "amount": 200.0, "status": "inactive"},
            {"id": 3, "amount": 150.0, "status": "active"},
        ],
        database=db,
    ).insert_into("test_orders")

    # Test FILTER clause with aggregation
    result = (
        db.table("test_orders")
        .select()
        .group_by("status")
        .agg(
            F.sum(col("amount")).filter(col("amount") > 100).alias("high_amount_sum"),
            F.count("*").alias("total_count"),
        )
        .collect()
    )

    assert len(result) == 2
    # Find active status row
    active_row = next(r for r in result if r["status"] == "active")
    assert active_row["total_count"] == 2
    # High amount sum should only include amounts > 100 (150.0)
    assert active_row["high_amount_sum"] == 150.0


def test_quote_characters(duckdb_db):
    """Test that DuckDB uses double quotes for identifiers."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        # Create table with a column name that might need quoting
        db.create_table(
            "test_table",
            [
                column("id", "INTEGER", primary_key=True),
                column("user_name", "VARCHAR"),  # Contains underscore
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [{"id": 1, "user_name": "Alice"}],
        database=db,
    ).insert_into("test_table")

    result = db.table("test_table").select().collect()
    assert len(result) == 1
    assert result[0]["user_name"] == "Alice"


def test_basic_dataframe_operations(duckdb_db):
    """Test basic DataFrame operations with DuckDB."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "test_data",
            [
                column("id", "INTEGER", primary_key=True),
                column("value", "INTEGER"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
            {"id": 3, "value": 30},
        ],
        database=db,
    ).insert_into("test_data")

    # Test select, filter, order_by
    result = (
        db.table("test_data").select().where(col("value") > 15).order_by(col("value")).collect()
    )

    assert len(result) == 2
    assert result[0]["value"] == 20
    assert result[1]["value"] == 30


def test_join_operations(duckdb_db):
    """Test join operations with DuckDB."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "VARCHAR"),
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

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ],
        database=db,
    ).insert_into("users")

    Records.from_list(
        [
            {"id": 1, "user_id": 1, "amount": 100.0},
            {"id": 2, "user_id": 2, "amount": 200.0},
        ],
        database=db,
    ).insert_into("orders")

    # Test inner join
    result = (
        db.table("orders")
        .select()
        .join(db.table("users").select(), on=[("user_id", "id")])
        .collect()
    )

    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"


def test_groupby_aggregation(duckdb_db):
    """Test groupBy and aggregation with DuckDB."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "sales",
            [
                column("id", "INTEGER", primary_key=True),
                column("category", "VARCHAR"),
                column("amount", "REAL"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "category": "A", "amount": 100.0},
            {"id": 2, "category": "A", "amount": 150.0},
            {"id": 3, "category": "B", "amount": 200.0},
        ],
        database=db,
    ).insert_into("sales")

    result = (
        db.table("sales")
        .select()
        .group_by("category")
        .agg(F.sum(col("amount")).alias("total"))
        .collect()
    )

    assert len(result) == 2
    totals = {r["category"]: r["total"] for r in result}
    assert totals["A"] == 250.0
    assert totals["B"] == 200.0


def test_cte_support(duckdb_db):
    """Test CTE (Common Table Expression) support in DuckDB."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "numbers",
            [
                column("id", "INTEGER", primary_key=True),
                column("value", "INTEGER"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "value": 10},
            {"id": 2, "value": 20},
        ],
        database=db,
    ).insert_into("numbers")

    # Test raw SQL with CTE
    result = db.sql(
        """
        WITH doubled AS (
            SELECT id, value * 2 AS doubled_value
            FROM numbers
        )
        SELECT * FROM doubled
        """
    ).collect()

    assert len(result) == 2
    assert result[0]["doubled_value"] == 20
    assert result[1]["doubled_value"] == 40


def test_array_agg_collect_list(duckdb_db):
    """Test collect_list() uses ARRAY_AGG in DuckDB."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "test_array",
            [
                column("id", "INTEGER", primary_key=True),
                column("value", "VARCHAR"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "a"},
        ],
        database=db,
    ).insert_into("test_array")

    result = (
        db.table("test_array")
        .select()
        .group_by("value")
        .agg(F.collect_list(col("id")).alias("ids"))
        .collect()
    )

    assert len(result) == 2
    # Find value "a" row
    a_row = next(r for r in result if r["value"] == "a")
    # DuckDB returns arrays as lists in Python
    assert isinstance(a_row["ids"], (list, tuple))
    assert set(a_row["ids"]) == {1, 3}


def test_merge_upsert(duckdb_db):
    """Test MERGE/UPSERT operations with DuckDB (uses ON CONFLICT like PostgreSQL)."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "test_merge",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "VARCHAR"),
                column("status", "VARCHAR"),
            ],
        ).collect()

    from moltres.io.records import Records

    # Insert initial data
    Records.from_list(
        [{"id": 1, "name": "Alice", "status": "active"}],
        database=db,
    ).insert_into("test_merge")

    # Merge/upsert (executes immediately, returns rowcount)
    # Note: DuckDB may return -1 for rowcount if it can't be determined
    result = db.merge(
        "test_merge",
        [{"id": 1, "name": "Alice Updated", "status": "inactive"}],
        on=["id"],
        when_matched={"name": "Alice Updated", "status": "inactive"},
    )

    # DuckDB may return -1 for rowcount, but operation should still succeed
    assert result >= -1

    # Verify update actually occurred
    updated = db.table("test_merge").select().where(col("id") == 1).collect()
    assert len(updated) == 1
    assert updated[0]["name"] == "Alice Updated"
    assert updated[0]["status"] == "inactive"


def test_explain_analyze(duckdb_db):
    """Test EXPLAIN ANALYZE support in DuckDB (PostgreSQL-style)."""
    from moltres.table.schema import column

    db = duckdb_db
    with db.batch():
        db.create_table(
            "test_explain",
            [
                column("id", "INTEGER", primary_key=True),
                column("value", "INTEGER"),
            ],
        ).collect()

    from moltres.io.records import Records

    Records.from_list(
        [{"id": 1, "value": 10}, {"id": 2, "value": 20}],
        database=db,
    ).insert_into("test_explain")

    df = db.table("test_explain").select().where(col("value") > 15)
    plan = df.explain(analyze=True)

    # DuckDB should support EXPLAIN ANALYZE
    assert plan is not None
    assert len(plan) > 0
    # Plan should contain SQL-related text
    assert "EXPLAIN" in plan.upper() or "SELECT" in plan.upper()
