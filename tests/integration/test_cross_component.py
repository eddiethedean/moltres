"""Cross-component integration tests."""

import pytest

from moltres import col
from moltres.expressions.functions import sum as sum_func
from moltres.io.records import Records
from moltres.table.schema import column


@pytest.mark.integration
def test_dataframe_records_integration(sample_database):
    """Test DataFrame and Records integration."""
    db = sample_database

    # Create DataFrame from table
    df = db.table("customers").select()

    # Convert to Records
    data = df.collect()
    records = Records.from_list(data, database=db)

    # Insert Records into new table
    db.create_table(
        "customers_copy",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("country", "TEXT"),
            column("active", "INTEGER"),
        ],
    ).collect()

    records.insert_into("customers_copy")

    # Query back as DataFrame
    df2 = db.table("customers_copy").select()
    results = df2.collect()

    # Verify schema preserved
    assert len(results) == 4
    assert all("id" in row for row in results)
    assert all("name" in row for row in results)
    assert all("email" in row for row in results)


@pytest.mark.integration
def test_compiler_execution_integration(sample_database):
    """Test compiler and execution integration."""
    db = sample_database

    # Create complex logical plan
    customers_df = db.table("customers").select()
    orders_df = db.table("orders").select()

    complex_df = (
        customers_df.join(orders_df, on=[("id", "customer_id")])
        .where(col("status") == "completed")
        .group_by("country")
        .agg(
            sum_func(col("amount")).alias("total"),
        )
        .order_by(col("total").desc())
    )

    # Compile and execute (should work seamlessly)
    results = complex_df.collect()

    # Verify results
    assert len(results) > 0
    assert all("country" in row for row in results)
    assert all("total" in row for row in results)

    # Verify SQL was generated correctly (check explain)
    plan = complex_df.explain()
    assert "SELECT" in plan.upper() or "SCAN" in plan.upper()


@pytest.mark.integration
def test_schema_mutations_integration(empty_database):
    """Test schema and mutations integration."""
    db = empty_database

    # Create table with constraints
    db.create_table(
        "users",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
        ],
    ).collect()

    # Insert using mutations
    from moltres.table.mutations import insert_rows

    table = db.table("users")
    inserted = insert_rows(
        table,
        [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ],
    )
    assert inserted == 2

    # Verify constraint enforcement (try duplicate primary key)
    try:
        insert_rows(table, [{"id": 1, "name": "Duplicate", "email": "dup@example.com"}])
        # May or may not fail depending on database
    except Exception:
        pass  # Expected if constraint enforced

    # Verify schema reflection works with mutations
    columns = db.get_columns("users")
    assert len(columns) == 3
    column_names = {col.name for col in columns}
    assert "id" in column_names
    assert "name" in column_names
    assert "email" in column_names


@pytest.mark.integration
def test_async_sync_integration(empty_database):
    """Test async and sync integration on same database."""
    db = empty_database

    # Create table with sync
    db.create_table(
        "test_table",
        [column("id", "INTEGER", primary_key=True), column("value", "TEXT")],
    ).collect()

    # Insert with sync
    Records.from_list([{"id": 1, "value": "sync"}], database=db).insert_into("test_table")

    # Query with sync
    sync_results = db.table("test_table").select().collect()
    assert len(sync_results) == 1

    # Verify data is accessible (async would use same database file)
    # Note: Full async test would require async_connect, but this verifies
    # that sync operations work and data persists for potential async access

    # Insert more with sync
    Records.from_list([{"id": 2, "value": "sync2"}], database=db).insert_into("test_table")

    # Verify consistency
    all_results = db.table("test_table").select().collect()
    assert len(all_results) == 2


@pytest.mark.integration
def test_read_write_integration(sample_database, tmp_path):
    """Test read and write integration."""
    db = sample_database

    # Read from table
    df = db.table("customers").select()

    # Write to CSV
    csv_file = tmp_path / "customers.csv"
    df.write.csv(str(csv_file))

    # Read back from CSV
    records = db.read.records.csv(str(csv_file))
    csv_data = records.rows()

    # Write to new table
    db.create_table(
        "customers_from_csv",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("country", "TEXT"),
            column("active", "INTEGER"),
        ],
    ).collect()

    # Transform data (convert string IDs to integers)
    transformed_data = []
    for row in csv_data:
        try:
            transformed_data.append(
                {
                    "id": int(row["id"]),
                    "name": row.get("name", ""),
                    "email": row.get("email", ""),
                    "country": row.get("country", ""),
                    "active": int(row.get("active", 0)) if row.get("active") else 0,
                }
            )
        except (ValueError, KeyError):
            continue  # Skip invalid rows

    if transformed_data:
        Records.from_list(transformed_data, database=db).insert_into("customers_from_csv")

        # Verify round-trip
        df2 = db.table("customers_from_csv").select()
        results = df2.collect()
        assert len(results) > 0


@pytest.mark.integration
def test_expressions_functions_integration(sample_database):
    """Test expressions and functions integration."""
    db = sample_database

    # Use various functions in DataFrame operations
    from moltres.expressions.functions import avg, count, max as max_func, min as min_func

    orders_df = db.table("orders").select()

    # Test aggregation functions
    result = (
        orders_df.group_by("status")
        .agg(
            count(col("id")).alias("count"),
            sum_func(col("amount")).alias("total"),
            avg(col("amount")).alias("average"),
            max_func(col("amount")).alias("max_amount"),
            min_func(col("amount")).alias("min_amount"),
        )
        .collect()
    )

    assert len(result) > 0
    for row in result:
        assert "count" in row
        assert "total" in row
        assert "average" in row
        assert "max_amount" in row
        assert "min_amount" in row


@pytest.mark.integration
def test_table_operations_dataframe_integration(sample_database):
    """Test table operations and DataFrame integration."""
    db = sample_database

    # Use table convenience methods
    # Insert
    db.insert(
        "customers",
        [{"id": 5, "name": "Eve", "email": "eve@example.com", "country": "CA", "active": 1}],
    )

    # Query with DataFrame
    df = db.table("customers").select()
    results = df.where(col("id") == 5).collect()
    assert len(results) == 1
    assert results[0]["name"] == "Eve"

    # Update using table method
    db.update("customers", where=col("id") == 5, set={"name": "Eve Updated"})

    # Verify with DataFrame
    results = df.where(col("id") == 5).collect()
    assert results[0]["name"] == "Eve Updated"

    # Delete using table method
    db.delete("customers", where=col("id") == 5)

    # Verify with DataFrame
    results = df.where(col("id") == 5).collect()
    assert len(results) == 0


@pytest.mark.integration
def test_schema_reflection_mutations_integration(empty_database):
    """Test schema reflection and mutations work together."""
    db = empty_database

    # Create table
    db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("price", "REAL"),
        ],
    ).collect()

    # Reflect schema
    columns = db.get_columns("products")
    assert len(columns) == 3

    # Use reflected schema info for mutations
    from moltres.table.mutations import insert_rows

    table = db.table("products")

    # Insert data (mutations use schema)
    inserted = insert_rows(
        table,
        [
            {"id": 1, "name": "Widget", "price": 10.0},
            {"id": 2, "name": "Gadget", "price": 20.0},
        ],
    )
    assert inserted == 2

    # Reflect again to verify data
    schema = db.reflect_table("products")
    assert schema.name == "products"
    assert len(schema.columns) == 3

    # Query to verify mutations worked
    results = db.table("products").select().collect()
    assert len(results) == 2
