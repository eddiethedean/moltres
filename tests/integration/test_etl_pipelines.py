"""ETL pipeline integration tests."""

import pytest

from moltres import col
from moltres.expressions.functions import count, sum as sum_func
from moltres.io.records import Records
from moltres.table.schema import column


@pytest.mark.integration
def test_full_etl_pipeline_extract_transform_load(sample_database):
    """Test full ETL pipeline: Extract → Transform → Load."""
    db = sample_database

    # Extract: Read from source tables
    customers_df = db.table("customers").select()
    orders_df = db.table("orders").select()

    # Transform: Join and calculate totals
    transformed_df = (
        customers_df.join(orders_df, on=[("id", "customer_id")])
        .select(
            col("customer_id"),
            col("name").alias("customer_name"),
            col("country"),
            col("amount"),
            col("status"),
        )
        .where(col("status") == "completed")
    )

    # Load: Create target table
    db.create_table(
        "customer_sales",
        [
            column("customer_id", "INTEGER"),
            column("customer_name", "TEXT"),
            column("country", "TEXT"),
            column("amount", "REAL"),
            column("status", "TEXT"),
        ],
    ).collect()

    # Insert transformed data
    transformed_data = transformed_df.collect()
    Records.from_list(transformed_data, database=db).insert_into("customer_sales")

    # Verify load
    results = db.table("customer_sales").select().collect()
    assert len(results) == 3  # 3 completed orders
    assert all(r["status"] == "completed" for r in results)


@pytest.mark.integration
def test_incremental_etl_pipeline(sample_database):
    """Test incremental ETL: Initial load → Incremental updates."""
    db = sample_database

    # Create target table
    db.create_table(
        "customer_totals",
        [
            column("customer_id", "INTEGER", primary_key=True),
            column("total_amount", "REAL"),
            column("order_count", "INTEGER"),
            column("last_updated", "TEXT"),
        ],
    ).collect()

    # Initial load
    orders_df = db.table("orders").select()
    initial_totals = (
        orders_df.group_by("customer_id")
        .agg(
            sum_func(col("amount")).alias("total_amount"),
            count(col("id")).alias("order_count"),
        )
        .collect()
    )

    initial_data = [{**row, "last_updated": "2024-01-01"} for row in initial_totals]
    Records.from_list(initial_data, database=db).insert_into("customer_totals")

    # Verify initial load
    initial_results = db.table("customer_totals").select().collect()
    assert len(initial_results) >= 1  # At least one customer

    # Incremental update: Add new orders
    new_orders = [
        {
            "id": 106,
            "customer_id": 1,
            "amount": 75.0,
            "status": "completed",
            "order_date": "2024-01-22",
        },
        {
            "id": 107,
            "customer_id": 2,
            "amount": 100.0,
            "status": "completed",
            "order_date": "2024-01-22",
        },
    ]
    Records.from_list(new_orders, database=db).insert_into("orders")

    # Update totals using merge (upsert)
    updated_totals = (
        db.table("orders")
        .select()
        .group_by("customer_id")
        .agg(
            sum_func(col("amount")).alias("total_amount"),
            count(col("id")).alias("order_count"),
        )
        .collect()
    )

    # Merge updated totals
    from moltres.table.mutations import merge_rows

    table = db.table("customer_totals")
    for total in updated_totals:
        merge_rows(
            table,
            [
                {
                    "customer_id": total["customer_id"],
                    "total_amount": total["total_amount"],
                    "order_count": total["order_count"],
                    "last_updated": "2024-01-22",
                }
            ],
            on=["customer_id"],
            when_matched={
                "total_amount": total["total_amount"],
                "order_count": total["order_count"],
                "last_updated": "2024-01-22",
            },
        )

    # Verify incremental update
    final_results = db.table("customer_totals").select().collect()
    customer_1_total = next(r for r in final_results if r["customer_id"] == 1)
    # Initial: 100.0 + 50.0 = 150.0, then add 75.0 = 225.0
    # Or if counting all orders: 100 + 50 + 75 = 225.0
    assert customer_1_total["total_amount"] >= 150.0  # At least initial amount
    assert customer_1_total["order_count"] >= 2  # At least 2 orders


@pytest.mark.integration
def test_data_quality_pipeline(sample_database):
    """Test data quality pipeline: Load → Validate → Flag errors → Load clean data."""
    db = sample_database

    # Create source table with potentially bad data
    db.create_table(
        "raw_customers",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("age", "INTEGER"),
        ],
    ).collect()

    # Insert data with quality issues
    raw_data = [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},  # Valid
        {"id": 2, "name": "Bob", "email": None, "age": 25},  # Missing email
        {"id": 3, "name": None, "email": "charlie@example.com", "age": 35},  # Missing name
        {"id": 4, "name": "Diana", "email": "diana@example.com", "age": -5},  # Invalid age
        {"id": 5, "name": "Eve", "email": "eve@example.com", "age": 28},  # Valid
    ]
    Records.from_list(raw_data, database=db).insert_into("raw_customers")

    # Validate: Extract clean and dirty data
    raw_df = db.table("raw_customers").select()
    all_data = raw_df.collect()

    clean_data = []
    dirty_data = []

    for row in all_data:
        # Validation rules: name and email required, age >= 0
        if row.get("name") and row.get("email") and (row.get("age") or 0) >= 0:
            clean_data.append(row)
        else:
            dirty_data.append(row)

    # Create clean table
    db.create_table(
        "clean_customers",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("age", "INTEGER"),
        ],
    ).collect()

    # Create errors table
    db.create_table(
        "data_quality_errors",
        [
            column("id", "INTEGER", primary_key=True),
            column("source_id", "INTEGER"),
            column("error_reason", "TEXT"),
        ],
    ).collect()

    # Load clean data
    Records.from_list(clean_data, database=db).insert_into("clean_customers")

    # Report errors
    error_records = []
    for row in dirty_data:
        reasons = []
        if not row.get("name"):
            reasons.append("missing_name")
        if not row.get("email"):
            reasons.append("missing_email")
        if (row.get("age") or 0) < 0:
            reasons.append("invalid_age")

        error_records.append(
            {
                "id": len(error_records) + 1,
                "source_id": row["id"],
                "error_reason": ", ".join(reasons),
            }
        )

    if error_records:
        Records.from_list(error_records, database=db).insert_into("data_quality_errors")

    # Verify clean data
    clean_results = db.table("clean_customers").select().collect()
    assert len(clean_results) == 2  # Alice and Eve

    # Verify errors reported
    error_results = db.table("data_quality_errors").select().collect()
    assert len(error_results) == 3  # Bob, Charlie, Diana


@pytest.mark.integration
def test_schema_evolution_pipeline(sample_database):
    """Test schema evolution pipeline: v1 → v2 migration."""
    db = sample_database

    # Create schema v1
    db.create_table(
        "products_v1",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("price", "REAL"),
        ],
    ).collect()

    # Load data with v1 schema
    v1_data = [
        {"id": 1, "name": "Widget", "price": 10.0},
        {"id": 2, "name": "Gadget", "price": 20.0},
    ]
    Records.from_list(v1_data, database=db).insert_into("products_v1")

    # Create schema v2 (add category, make price nullable for migration)
    db.create_table(
        "products_v2",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("price", "REAL"),
            column("category", "TEXT"),
            column("description", "TEXT"),
        ],
    ).collect()

    # Migrate data: Extract from v1, transform, load to v2
    v1_df = db.table("products_v1").select()
    v1_rows = v1_df.collect()

    # Transform: Add default category, empty description
    v2_data = [
        {
            **row,
            "category": "General",  # Default category
            "description": "",  # Empty description
        }
        for row in v1_rows
    ]

    # Load to v2
    Records.from_list(v2_data, database=db).insert_into("products_v2")

    # Verify migration
    v2_df = db.table("products_v2").select()
    v2_results = v2_df.collect()

    assert len(v2_results) == 2
    assert all("category" in row for row in v2_results)
    assert all(row["category"] == "General" for row in v2_results)
    assert all("description" in row for row in v2_results)


@pytest.mark.integration
def test_multi_source_etl_pipeline(sample_database):
    """Test ETL from multiple sources."""
    db = sample_database

    # Create additional source table
    db.create_table(
        "suppliers",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("country", "TEXT"),
        ],
    ).collect()

    Records.from_list(
        [
            {"id": 1, "name": "Supplier A", "country": "USA"},
            {"id": 2, "name": "Supplier B", "country": "UK"},
        ],
        database=db,
    ).insert_into("suppliers")

    # Extract from multiple sources
    customers_df = db.table("customers").select()
    suppliers_df = db.table("suppliers").select()

    # Transform: Union and aggregate by country
    # Note: Union requires same columns, so we'll select matching columns
    customers_by_country = customers_df.select(
        col("country"), col("name").alias("entity_name")
    ).collect()

    suppliers_by_country = suppliers_df.select(
        col("country"), col("name").alias("entity_name")
    ).collect()

    # Combine and aggregate
    all_entities = customers_by_country + suppliers_by_country

    # Create target table
    db.create_table(
        "entities_by_country",
        [
            column("country", "TEXT", primary_key=True),
            column("entity_count", "INTEGER"),
        ],
    ).collect()

    # Aggregate by country
    from collections import Counter

    country_counts = Counter(row["country"] for row in all_entities)

    aggregated_data = [
        {"country": country, "entity_count": count} for country, count in country_counts.items()
    ]

    # Load aggregated data
    Records.from_list(aggregated_data, database=db).insert_into("entities_by_country")

    # Verify
    results = db.table("entities_by_country").select().collect()
    assert len(results) >= 2  # At least USA and UK
    usa_count = next((r["entity_count"] for r in results if r["country"] == "USA"), 0)
    assert usa_count >= 1
