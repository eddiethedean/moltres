"""Real-world use case scenario tests."""

import pytest

from moltres import col
from moltres.expressions.functions import count, sum as sum_func
from moltres.io.records import Records
from moltres.table.schema import column, foreign_key


@pytest.mark.integration
def test_ecommerce_analytics_scenario(empty_database):
    """Test e-commerce analytics: Orders, customers, products → Sales reports → Customer segmentation."""
    db = empty_database

    # Create schema
    db.create_table(
        "customers",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("country", "TEXT"),
            column("registration_date", "TEXT"),
        ],
    ).collect()

    db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("price", "REAL"),
            column("category", "TEXT"),
        ],
    ).collect()

    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("customer_id", "INTEGER"),
            column("order_date", "TEXT"),
            column("status", "TEXT"),
        ],
        constraints=[foreign_key("customer_id", "customers", "id")],
    ).collect()

    db.create_table(
        "order_items",
        [
            column("id", "INTEGER", primary_key=True),
            column("order_id", "INTEGER"),
            column("product_id", "INTEGER"),
            column("quantity", "INTEGER"),
            column("unit_price", "REAL"),
        ],
        constraints=[
            foreign_key("order_id", "orders", "id"),
            foreign_key("product_id", "products", "id"),
        ],
    ).collect()

    # Load data
    Records.from_list(
        [
            {
                "id": 1,
                "name": "Alice",
                "email": "alice@example.com",
                "country": "USA",
                "registration_date": "2024-01-01",
            },
            {
                "id": 2,
                "name": "Bob",
                "email": "bob@example.com",
                "country": "UK",
                "registration_date": "2024-01-02",
            },
            {
                "id": 3,
                "name": "Charlie",
                "email": "charlie@example.com",
                "country": "USA",
                "registration_date": "2024-01-03",
            },
        ],
        database=db,
    ).insert_into("customers")

    Records.from_list(
        [
            {"id": 1, "name": "Widget", "price": 10.0, "category": "Electronics"},
            {"id": 2, "name": "Gadget", "price": 20.0, "category": "Electronics"},
            {"id": 3, "name": "Book", "price": 5.0, "category": "Education"},
        ],
        database=db,
    ).insert_into("products")

    Records.from_list(
        [
            {"id": 101, "customer_id": 1, "order_date": "2024-01-15", "status": "completed"},
            {"id": 102, "customer_id": 1, "order_date": "2024-01-20", "status": "completed"},
            {"id": 103, "customer_id": 2, "order_date": "2024-01-18", "status": "completed"},
        ],
        database=db,
    ).insert_into("orders")

    Records.from_list(
        [
            {"id": 1, "order_id": 101, "product_id": 1, "quantity": 2, "unit_price": 10.0},
            {"id": 2, "order_id": 101, "product_id": 2, "quantity": 1, "unit_price": 20.0},
            {"id": 3, "order_id": 102, "product_id": 3, "quantity": 3, "unit_price": 5.0},
            {"id": 4, "order_id": 103, "product_id": 1, "quantity": 5, "unit_price": 10.0},
        ],
        database=db,
    ).insert_into("order_items")

    # Generate sales report: Total sales by country
    customers_df = db.table("customers").select()
    orders_df = db.table("orders").select()
    items_df = db.table("order_items").select()

    # Join customers with orders, then with order_items
    # The issue: After first join, both customers.id and orders.id exist,
    # so the second join with ("id", "order_id") is ambiguous.
    # Workaround: Join orders directly with order_items first, then join with customers
    # This avoids the ambiguity since we're joining on order_id which is unique to order_items
    sales_report = (
        orders_df.join(items_df, on=[("id", "order_id")], how="inner")
        .join(customers_df, on=[("customer_id", "id")], how="inner")
        .where(col("status") == "completed")
        .group_by("country")
        .agg(
            sum_func(col("quantity") * col("unit_price")).alias("total_sales"),
            count(col("id")).alias("order_count"),
        )
        .order_by(col("total_sales").desc())
        .collect()
    )

    # Verify we got results
    assert len(sales_report) > 0, "Join should produce results"

    # Verify sales data
    usa_sales = next((r for r in sales_report if r["country"] == "USA"), None)
    assert usa_sales is not None, "Should have USA sales"
    assert usa_sales["total_sales"] > 0, "USA sales should be positive"

    # Verify calculation: USA should have orders from customer 1 (Alice) and 3 (Charlie)
    # Order 101: (2*10 + 1*20) = 40, Order 102: (3*5) = 15, Order 103: (5*10) = 50
    # Total for USA: 40 + 15 = 55 (customer 1) + 50 (customer 3) = 105
    # But since we're grouping by country, we need to sum all USA orders
    assert usa_sales["total_sales"] >= 55.0, (
        f"USA sales should be at least 55, got {usa_sales['total_sales']}"
    )

    # Customer segmentation: High-value customers
    # Use the same join order to avoid ambiguity
    customer_value = (
        orders_df.join(items_df, on=[("id", "order_id")], how="inner")
        .join(customers_df, on=[("customer_id", "id")], how="inner")
        .where(col("status") == "completed")
        .group_by("customer_id", "name")
        .agg(
            sum_func(col("quantity") * col("unit_price")).alias("lifetime_value"),
            count(col("id")).alias("order_count"),
        )
        .collect()
    )

    high_value_customers = [c for c in customer_value if c["lifetime_value"] >= 50.0]
    assert len(high_value_customers) >= 1


@pytest.mark.integration
def test_log_analysis_pipeline_scenario(empty_database, tmp_path):
    """Test log analysis pipeline: Read log files → Parse → Aggregate by time windows → Store summaries."""
    db = empty_database

    # Create log file
    import json

    log_file = tmp_path / "app.log"
    log_entries = [
        {
            "timestamp": "2024-01-15 10:00:00",
            "level": "INFO",
            "message": "User login",
            "user_id": 1,
        },
        {
            "timestamp": "2024-01-15 10:01:00",
            "level": "ERROR",
            "message": "Database error",
            "user_id": None,
        },
        {
            "timestamp": "2024-01-15 10:02:00",
            "level": "INFO",
            "message": "User logout",
            "user_id": 1,
        },
        {
            "timestamp": "2024-01-15 11:00:00",
            "level": "INFO",
            "message": "User login",
            "user_id": 2,
        },
        {
            "timestamp": "2024-01-15 11:01:00",
            "level": "WARNING",
            "message": "Slow query",
            "user_id": 2,
        },
    ]

    with open(log_file, "w") as f:
        for entry in log_entries:
            json.dump(entry, f)
            f.write("\n")

    # Create raw logs table (include hour column for transformed data)
    db.create_table(
        "raw_logs",
        [
            column("id", "INTEGER", primary_key=True),
            column("timestamp", "TEXT"),
            column("level", "TEXT"),
            column("message", "TEXT"),
            column("user_id", "INTEGER"),
            column("hour", "TEXT"),
        ],
    ).collect()

    # Read and parse logs
    records = db.read.records.jsonl(str(log_file))
    log_data = records.rows()

    # Transform: Extract hour from timestamp
    transformed_data = []
    for row in log_data:
        timestamp = row.get("timestamp", "")
        hour = timestamp.split(":")[0] if ":" in timestamp else "unknown"
        transformed_data.append(
            {
                "id": len(transformed_data) + 1,
                "timestamp": row.get("timestamp"),
                "level": row.get("level"),
                "message": row.get("message"),
                "user_id": row.get("user_id"),
                "hour": hour,
            }
        )

    Records.from_list(transformed_data, database=db).insert_into("raw_logs")

    # Aggregate by time window (hour) and level
    logs_df = db.table("raw_logs").select()
    summary = (
        logs_df.group_by("hour", "level")
        .agg(
            count(col("id")).alias("count"),
        )
        .order_by(col("hour"), col("level"))
        .collect()
    )

    assert len(summary) >= 3
    info_count = sum(r["count"] for r in summary if r["level"] == "INFO")
    assert info_count == 3


@pytest.mark.integration
def test_data_warehouse_etl_scenario(empty_database):
    """Test data warehouse ETL: Staging tables → Fact/dimension loads → Aggregations."""
    db = empty_database

    # Create staging table
    db.create_table(
        "staging_sales",
        [
            column("id", "INTEGER", primary_key=True),
            column("customer_name", "TEXT"),
            column("product_name", "TEXT"),
            column("sale_date", "TEXT"),
            column("amount", "REAL"),
        ],
    ).collect()

    # Load staging data
    Records.from_list(
        [
            {
                "id": 1,
                "customer_name": "Alice",
                "product_name": "Widget",
                "sale_date": "2024-01-15",
                "amount": 100.0,
            },
            {
                "id": 2,
                "customer_name": "Bob",
                "product_name": "Gadget",
                "sale_date": "2024-01-15",
                "amount": 200.0,
            },
            {
                "id": 3,
                "customer_name": "Alice",
                "product_name": "Widget",
                "sale_date": "2024-01-16",
                "amount": 150.0,
            },
        ],
        database=db,
    ).insert_into("staging_sales")

    # Create dimension tables
    db.create_table(
        "dim_customers",
        [
            column("customer_id", "INTEGER", primary_key=True),
            column("customer_name", "TEXT"),
        ],
    ).collect()

    db.create_table(
        "dim_products",
        [
            column("product_id", "INTEGER", primary_key=True),
            column("product_name", "TEXT"),
        ],
    ).collect()

    # Extract unique customers and products
    staging_df = db.table("staging_sales").select()
    customers = staging_df.select("customer_name").distinct().collect()
    products = staging_df.select("product_name").distinct().collect()

    # Load dimensions
    customer_data = [
        {"customer_id": i + 1, "customer_name": c["customer_name"]} for i, c in enumerate(customers)
    ]
    product_data = [
        {"product_id": i + 1, "product_name": p["product_name"]} for i, p in enumerate(products)
    ]

    Records.from_list(customer_data, database=db).insert_into("dim_customers")
    Records.from_list(product_data, database=db).insert_into("dim_products")

    # Create fact table
    db.create_table(
        "fact_sales",
        [
            column("sale_id", "INTEGER", primary_key=True),
            column("customer_id", "INTEGER"),
            column("product_id", "INTEGER"),
            column("sale_date", "TEXT"),
            column("amount", "REAL"),
        ],
    ).collect()

    # Transform staging to fact (lookup dimension keys)
    staging_data = staging_df.collect()
    customer_map = {c["customer_name"]: c["customer_id"] for c in customer_data}
    product_map = {p["product_name"]: p["product_id"] for p in product_data}

    fact_data = [
        {
            "sale_id": row["id"],
            "customer_id": customer_map[row["customer_name"]],
            "product_id": product_map[row["product_name"]],
            "sale_date": row["sale_date"],
            "amount": row["amount"],
        }
        for row in staging_data
    ]

    Records.from_list(fact_data, database=db).insert_into("fact_sales")

    # Create aggregation (materialized view pattern)
    fact_df = db.table("fact_sales").select()
    dim_customers_df = db.table("dim_customers").select()
    dim_products_df = db.table("dim_products").select()

    sales_summary = (
        fact_df.join(dim_customers_df, on=[("customer_id", "customer_id")])
        .join(dim_products_df, on=[("product_id", "product_id")])
        .group_by("customer_name", "product_name")
        .agg(
            sum_func(col("amount")).alias("total_sales"),
            count(col("sale_id")).alias("sale_count"),
        )
        .collect()
    )

    assert len(sales_summary) == 2
    alice_widget = next(
        r for r in sales_summary if r["customer_name"] == "Alice" and r["product_name"] == "Widget"
    )
    assert alice_widget["total_sales"] == 250.0  # 100 + 150


@pytest.mark.integration
def test_api_data_sync_scenario(empty_database):
    """Test API data sync: Fetch data → Transform → Load → Update existing records."""
    db = empty_database

    # Create target table
    db.create_table(
        "api_users",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("last_sync", "TEXT"),
        ],
    ).collect()

    # Initial load
    initial_data = [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "last_sync": "2024-01-01"},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "last_sync": "2024-01-01"},
    ]
    Records.from_list(initial_data, database=db).insert_into("api_users")

    # Simulate API fetch (new data with updates)
    api_data = [
        {"id": 1, "name": "Alice Updated", "email": "alice.new@example.com"},  # Updated
        {"id": 2, "name": "Bob", "email": "bob@example.com"},  # Unchanged
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"},  # New
    ]

    # Transform: Add sync timestamp
    transformed_data = [{**row, "last_sync": "2024-01-15"} for row in api_data]

    # Merge (upsert) data
    from moltres.table.mutations import merge_rows

    table = db.table("api_users")

    for row in transformed_data:
        merge_rows(
            table,
            [row],
            on=["id"],
            when_matched={
                "name": row["name"],
                "email": row["email"],
                "last_sync": row["last_sync"],
            },
        )

    # Verify sync results
    results = db.table("api_users").select().order_by(col("id")).collect()
    assert len(results) == 3

    # Verify updates
    alice = next(r for r in results if r["id"] == 1)
    assert alice["name"] == "Alice Updated"
    assert alice["email"] == "alice.new@example.com"
    assert alice["last_sync"] == "2024-01-15"

    # Verify new record
    charlie = next(r for r in results if r["id"] == 3)
    assert charlie["name"] == "Charlie"
