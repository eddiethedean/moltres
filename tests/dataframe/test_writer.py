"""Tests for DataFrame write operations."""

from moltres import col, column, connect
from moltres.table.schema import ColumnDef


def test_write_append_mode(tmp_path):
    """Test writing DataFrame in append mode."""
    db_path = tmp_path / "write_append.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source table
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
        ],
    ).collect()
    source.insert(
        [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
    ).collect()

    # Write to new table
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Verify data was written
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 2
    assert rows[0]["name"] == "Alice"
    assert rows[1]["name"] == "Bob"

    # Append more data
    source.insert([{"id": 3, "name": "Charlie"}]).collect()
    df2 = db.table("source").select().where(col("id") == 3)
    df2.write.mode("append").save_as_table("target")

    # Verify append worked
    rows = target.select().collect()
    assert len(rows) == 3


def test_write_overwrite_mode(tmp_path):
    """Test writing DataFrame in overwrite mode."""
    db_path = tmp_path / "write_overwrite.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create and populate initial table
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("value", "INTEGER"),
        ],
    ).collect()
    source.insert([{"id": 1, "value": 100}]).collect()

    # Write initial data
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Verify initial write
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert rows[0]["value"] == 100

    # Overwrite with new data
    source.insert([{"id": 2, "value": 200}]).collect()
    df2 = db.table("source").select()
    df2.write.mode("overwrite").save_as_table("target")

    # Verify overwrite (should only have new data)
    rows = target.select().collect()
    assert len(rows) == 2
    assert rows[0]["value"] == 100
    assert rows[1]["value"] == 200


def test_write_error_if_exists_mode(tmp_path):
    """Test error_if_exists mode raises error when table exists."""
    db_path = tmp_path / "write_error.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table first
    source = db.create_table(
        "source",
        [column("id", "INTEGER")],
    ).collect()
    source.insert([{"id": 1}]).collect()

    # Write once (creates table)
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Try to write again with error_if_exists
    import pytest

    with pytest.raises(ValueError, match="already exists"):
        df.write.mode("error_if_exists").save_as_table("target")


def test_write_with_explicit_schema(tmp_path):
    """Test writing with explicit schema."""
    db_path = tmp_path / "write_schema.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source with different types
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("score", "REAL"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice", "score": 95.5}]).collect()

    # Write with explicit schema
    explicit_schema = [
        ColumnDef(name="id", type_name="INTEGER", nullable=False),
        ColumnDef(name="name", type_name="TEXT", nullable=False),
        ColumnDef(name="score", type_name="REAL", nullable=True),
    ]

    df = db.table("source").select()
    df.write.schema(explicit_schema).save_as_table("target")

    # Verify table was created with correct schema
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["name"] == "Alice"
    assert rows[0]["score"] == 95.5


def test_write_empty_dataframe(tmp_path):
    """Test writing empty DataFrame creates table but inserts nothing."""
    db_path = tmp_path / "write_empty.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source table
    db.create_table(
        "source",
        [column("id", "INTEGER")],
    ).collect()

    # Create empty DataFrame
    df = db.table("source").select().where(col("id") == 999)

    # Write empty DataFrame with explicit schema (required for empty DataFrames)
    explicit_schema = [ColumnDef(name="id", type_name="INTEGER", nullable=True)]
    df.write.schema(explicit_schema).save_as_table("target")

    # Verify table exists but is empty
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 0


def test_write_with_transformed_columns(tmp_path):
    """Test writing DataFrame with transformed/aliased columns."""
    db_path = tmp_path / "write_transformed.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("first_name", "TEXT"),
            column("last_name", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "first_name": "John", "last_name": "Doe"}]).collect()

    # Create DataFrame with transformed columns
    df = db.table("source").select(
        col("id"),
        (col("first_name") + " " + col("last_name")).alias("full_name"),
    )

    # Write transformed DataFrame
    df.write.save_as_table("target")

    # Verify data
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert "id" in rows[0]
    assert "full_name" in rows[0]


def test_write_chained_api(tmp_path):
    """Test chained write API similar to PySpark."""
    db_path = tmp_path / "write_chained.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER")],
    ).collect()
    source.insert([{"id": 1}]).collect()

    df = db.table("source").select()
    df.write.mode("append").option("test", "value").save_as_table("target")

    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1


def test_write_with_primary_key_chaining(tmp_path):
    """Test specifying primary key using chaining method."""
    db_path = tmp_path / "write_pk_chaining.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source table
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]).collect()

    # Write with primary key specified via chaining
    df = db.table("source").select()
    df.write.primaryKey("id").save_as_table("target")

    # Verify table was created with primary key
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 2

    # Verify primary key constraint exists by trying to insert duplicate
    import pytest

    with pytest.raises(Exception):  # Should fail due to primary key constraint
        target.insert([{"id": 1, "name": "Duplicate"}]).collect()


def test_write_with_primary_key_parameter(tmp_path):
    """Test specifying primary key using direct parameter."""
    db_path = tmp_path / "write_pk_param.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}]).collect()

    # Write with primary key specified via parameter
    df = db.table("source").select()
    df.write.save_as_table("target", primary_key=["id"])

    # Verify table was created
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1


def test_write_with_composite_primary_key(tmp_path):
    """Test specifying composite primary key."""
    db_path = tmp_path / "write_composite_pk.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [
            column("user_id", "INTEGER"),
            column("order_id", "INTEGER"),
            column("amount", "REAL"),
        ],
    ).collect()
    source.insert([{"user_id": 1, "order_id": 100, "amount": 50.0}]).collect()

    # Write with composite primary key
    df = db.table("source").select()
    df.write.primaryKey("user_id", "order_id").save_as_table("target")

    # Verify table was created
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1


def test_write_primary_key_with_explicit_schema(tmp_path):
    """Test primary key with explicit schema."""
    db_path = tmp_path / "write_pk_schema.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}]).collect()

    # Write with explicit schema and primary key
    explicit_schema = [
        ColumnDef(name="id", type_name="INTEGER", nullable=False),
        ColumnDef(name="name", type_name="TEXT", nullable=True),
    ]

    df = db.table("source").select()
    df.write.schema(explicit_schema).primaryKey("id").save_as_table("target")

    # Verify table was created
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1


def test_write_primary_key_validation_error(tmp_path):
    """Test that specifying non-existent column as primary key raises error."""
    db_path = tmp_path / "write_pk_validation.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}]).collect()

    # Try to use non-existent column as primary key
    df = db.table("source").select()
    import pytest

    with pytest.raises(ValueError, match="do not exist in schema"):
        df.write.primaryKey("nonexistent").save_as_table("target")


def test_write_primary_key_parameter_overrides_chaining(tmp_path):
    """Test that primary_key parameter overrides chained method."""
    db_path = tmp_path / "write_pk_override.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("email", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice", "email": "alice@example.com"}]).collect()

    # Chain primaryKey but override with parameter
    df = db.table("source").select()
    df.write.primaryKey("name").save_as_table("target", primary_key=["id"])

    # Verify table was created (should use "id" as primary key, not "name")
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1


def test_write_primary_key_with_filtered_query(tmp_path):
    """Test primary key when SELECT excludes original primary key."""
    db_path = tmp_path / "write_pk_filtered.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source with primary key
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("status", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice", "status": "active"}]).collect()

    # Select only name and status (excluding id)
    df = db.table("source").select("name", "status")

    # Write with name as primary key (since id is not in the result)
    df.write.primaryKey("name").save_as_table("target")

    # Verify table was created
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert "name" in rows[0]
    assert "status" in rows[0]
    assert "id" not in rows[0]


def test_write_optimized_simple_select(tmp_path):
    """Test optimized INSERT INTO ... SELECT for simple table scan."""
    db_path = tmp_path / "write_optimized_simple.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create source table
    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]).collect()

    # Write using optimization (should use INSERT INTO ... SELECT)
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Verify data was written
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"Alice", "Bob"}


def test_write_optimized_with_filter(tmp_path):
    """Test optimized INSERT INTO ... SELECT with filtered query."""
    db_path = tmp_path / "write_optimized_filter.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("status", "TEXT"),
        ],
    ).collect()
    source.insert(
        [
            {"id": 1, "status": "active"},
            {"id": 2, "status": "inactive"},
            {"id": 3, "status": "active"},
        ]
    ).collect()

    # Write filtered data using optimization
    df = db.table("source").select().where(col("status") == "active")
    df.write.save_as_table("target")

    # Verify only filtered rows were written
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 2
    assert all(row["status"] == "active" for row in rows)


def test_write_optimized_with_project(tmp_path):
    """Test optimized INSERT INTO ... SELECT with projected columns."""
    db_path = tmp_path / "write_optimized_project.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("email", "TEXT"),
        ],
    ).collect()
    source.insert([{"id": 1, "name": "Alice", "email": "alice@example.com"}]).collect()

    # Write with selected columns only
    df = db.table("source").select(col("name"), col("email"))
    df.write.save_as_table("target")

    # Verify only selected columns were written
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert "name" in rows[0]
    assert "email" in rows[0]
    assert "id" not in rows[0]


def test_write_optimized_with_join(tmp_path):
    """Test optimized INSERT INTO ... SELECT with join."""
    db_path = tmp_path / "write_optimized_join.sqlite"
    db = connect(f"sqlite:///{db_path}")

    customers = db.create_table(
        "customers",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
        ],
    ).collect()
    orders = db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("customer_id", "INTEGER"),
            column("amount", "REAL"),
        ],
    ).collect()

    customers.insert([{"id": 1, "name": "Alice"}]).collect()
    orders.insert([{"id": 100, "customer_id": 1, "amount": 50.0}]).collect()

    # Write joined data using optimization
    # Select with aliases before join to avoid column qualification issues
    orders_df = db.table("orders").select(col("id").alias("order_id"), col("customer_id"))
    customers_df = db.table("customers").select(col("id").alias("customer_id"), col("name"))
    df = orders_df.join(customers_df, on=[("customer_id", "customer_id")]).select(
        col("order_id"), col("name").alias("customer")
    )
    df.write.save_as_table("target")

    # Verify joined data was written
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert rows[0]["order_id"] == 100
    assert rows[0]["customer"] == "Alice"


def test_write_optimized_with_aggregate(tmp_path):
    """Test optimized INSERT INTO ... SELECT with aggregation."""
    db_path = tmp_path / "write_optimized_aggregate.sqlite"
    db = connect(f"sqlite:///{db_path}")

    from moltres.expressions.functions import sum as sum_

    orders = db.create_table(
        "orders",
        [
            column("customer_id", "INTEGER"),
            column("amount", "REAL"),
        ],
    ).collect()
    orders.insert(
        [
            {"customer_id": 1, "amount": 10.0},
            {"customer_id": 1, "amount": 20.0},
            {"customer_id": 2, "amount": 15.0},
        ]
    ).collect()

    # Write aggregated data using optimization
    df = (
        db.table("orders")
        .select()
        .group_by(col("customer_id"))
        .agg(sum_(col("amount")).alias("total"))
    )
    df.write.save_as_table("target")

    # Verify aggregated data was written
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 2
    totals = {row["customer_id"]: row["total"] for row in rows}
    assert totals[1] == 30.0
    assert totals[2] == 15.0


def test_write_optimized_overwrite_mode(tmp_path):
    """Test optimized INSERT INTO ... SELECT with overwrite mode."""
    db_path = tmp_path / "write_optimized_overwrite.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("value", "INTEGER")],
    ).collect()
    source.insert([{"id": 1, "value": 100}]).collect()

    # Write initial data
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Overwrite with new data using optimization
    source.insert([{"id": 2, "value": 200}]).collect()
    df2 = db.table("source").select()
    df2.write.mode("overwrite").save_as_table("target")

    # Verify overwrite (should have both rows)
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 2


def test_write_optimized_append_mode(tmp_path):
    """Test optimized INSERT INTO ... SELECT with append mode."""
    db_path = tmp_path / "write_optimized_append.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}]).collect()

    # Write initial data
    df = db.table("source").select()
    df.write.save_as_table("target")

    # Append more data using optimization
    source.insert([{"id": 2, "name": "Bob"}]).collect()
    df2 = db.table("source").select().where(col("id") == 2)
    df2.write.mode("append").save_as_table("target")

    # Verify append worked
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 2


def test_write_optimized_with_explicit_schema(tmp_path):
    """Test optimized INSERT INTO ... SELECT with explicit schema."""
    db_path = tmp_path / "write_optimized_schema.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}]).collect()

    # Write with explicit schema using optimization
    schema = [ColumnDef("id", "INTEGER"), ColumnDef("name", "TEXT")]
    df = db.table("source").select()
    df.write.schema(schema).save_as_table("target")

    # Verify data was written
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_write_optimized_with_primary_key(tmp_path):
    """Test optimized INSERT INTO ... SELECT with primary key."""
    db_path = tmp_path / "write_optimized_pk.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}]).collect()

    # Write with primary key using optimization
    df = db.table("source").select()
    df.write.primaryKey("id").save_as_table("target")

    # Verify table was created with primary key
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    # Primary key constraint should be enforced (we can't easily test this without trying to insert duplicates)


def test_write_fallback_to_materialization(tmp_path):
    """Test that write falls back to materialization when optimization not possible."""
    db_path = tmp_path / "write_fallback.sqlite"
    db = connect(f"sqlite:///{db_path}")

    source = db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()
    source.insert([{"id": 1, "name": "Alice"}]).collect()

    # Use streaming mode (should fall back to materialization)
    df = db.table("source").select()
    df.write.stream(True).save_as_table("target")

    # Verify data was still written (using materialization)
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 1
    assert rows[0]["name"] == "Alice"


def test_write_optimized_empty_result_set(tmp_path):
    """Test optimized INSERT INTO ... SELECT with empty result set."""
    db_path = tmp_path / "write_optimized_empty.sqlite"
    db = connect(f"sqlite:///{db_path}")

    db.create_table(
        "source",
        [column("id", "INTEGER"), column("name", "TEXT")],
    ).collect()
    # Don't insert any data

    # Write empty result set with explicit schema
    schema = [ColumnDef("id", "INTEGER"), ColumnDef("name", "TEXT")]
    df = db.table("source").select()
    df.write.schema(schema).save_as_table("target")

    # Verify table was created (even though empty)
    target = db.table("target")
    rows = target.select().collect()
    assert len(rows) == 0
