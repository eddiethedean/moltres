"""Full workflow integration tests for PostgreSQL.

These tests run by default but will skip gracefully if PostgreSQL is unavailable.
For reliable execution, especially in parallel mode, consider running sequentially:
    pytest tests/integration/test_postgres_workflows.py -n 0

These tests require:
    - The testing.postgresql package
    - A working PostgreSQL installation with initdb available
    - Available ports for test database instances

Note: PostgreSQL initialization can be slow in parallel mode. If tests hang,
try running with fewer workers or sequentially.
"""

import csv
import json

import pytest

from moltres import col
from moltres.expressions import functions as F
from moltres.io.records import Records
from moltres.table.schema import column, decimal


@pytest.mark.integration
@pytest.mark.postgres
def test_postgres_etl_pipeline(postgresql_connection, unique_table_name, tmp_path):
    """Test complete ETL pipeline on PostgreSQL: Extract → Transform → Load."""
    db = postgresql_connection

    # Use unique table names to avoid conflicts in parallel mode
    source_customers = f"{unique_table_name}_source_customers"
    source_orders = f"{unique_table_name}_source_orders"
    customer_summary = f"{unique_table_name}_customer_summary"

    # Step 1: Extract - Create source tables
    db.create_table(
        source_customers,
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "VARCHAR(100)"),
            column("email", "VARCHAR(255)"),
            column("country", "VARCHAR(50)"),
        ],
    ).collect()

    db.create_table(
        source_orders,
        [
            column("id", "INTEGER", primary_key=True),
            column("customer_id", "INTEGER"),
            decimal("amount", precision=10, scale=2),
            column("order_date", "DATE"),
        ],
    ).collect()

    # Load source data
    Records.from_list(
        [
            {"id": 1, "name": "Alice", "email": "alice@example.com", "country": "USA"},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "country": "UK"},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com", "country": "USA"},
        ],
        database=db,
    ).insert_into(source_customers)

    Records.from_list(
        [
            {"id": 101, "customer_id": 1, "amount": 100.50, "order_date": "2024-01-15"},
            {"id": 102, "customer_id": 1, "amount": 75.25, "order_date": "2024-01-20"},
            {"id": 103, "customer_id": 2, "amount": 200.00, "order_date": "2024-01-18"},
            {"id": 104, "customer_id": 3, "amount": 50.75, "order_date": "2024-01-22"},
        ],
        database=db,
    ).insert_into(source_orders)

    # Step 2: Transform - Join and aggregate
    # Select specific columns to avoid duplicate 'id' column issue
    customers_df = db.table(source_customers).select("id", "name", "country")
    orders_df = db.table(source_orders).select("id", "customer_id", "amount")

    # Join and aggregate - use orders.id for count to avoid ambiguity
    joined = customers_df.join(orders_df, on=[("id", "customer_id")])
    transformed = (
        joined.group_by("country", "name")
        .agg(
            F.sum(col("amount")).alias("total_spent"),
            F.count(col("customer_id")).alias(
                "order_count"
            ),  # Use customer_id which is unique to orders
        )
        .collect()
    )

    # Step 3: Load - Create target table
    db.create_table(
        customer_summary,
        [
            column("country", "VARCHAR(50)"),
            column("name", "VARCHAR(100)"),
            decimal("total_spent", precision=10, scale=2),
            column("order_count", "INTEGER"),
        ],
    ).collect()

    Records.from_list(transformed, database=db).insert_into(customer_summary)

    # Verify results
    results = db.table(customer_summary).select().order_by(col("total_spent").desc()).collect()
    assert len(results) == 3

    # Verify Bob has highest total
    bob = next(r for r in results if r["name"] == "Bob")
    assert bob["total_spent"] == 200.00
    assert bob["order_count"] == 1


@pytest.mark.integration
@pytest.mark.postgres
def test_postgres_file_import_export(postgresql_connection, unique_table_name, tmp_path):
    """Test file import and export workflow on PostgreSQL."""
    db = postgresql_connection

    products = f"{unique_table_name}_products"

    # Step 1: Create CSV input
    csv_input = tmp_path / "products.csv"
    with open(csv_input, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "price", "category"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "1", "name": "Widget", "price": "10.50", "category": "Electronics"},
                {"id": "2", "name": "Gadget", "price": "25.75", "category": "Electronics"},
                {"id": "3", "name": "Book", "price": "15.00", "category": "Education"},
            ]
        )

    # Step 2: Import CSV
    records = db.read.records.csv(str(csv_input))
    data = records.rows()

    # Step 3: Create and populate table
    db.create_table(
        products,
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "VARCHAR(100)"),
            decimal("price", precision=10, scale=2),
            column("category", "VARCHAR(50)"),
        ],
    ).collect()

    transformed = [
        {
            "id": int(row["id"]),
            "name": row["name"],
            "price": float(row["price"]),
            "category": row["category"],
        }
        for row in data
    ]
    Records.from_list(transformed, database=db).insert_into(products)

    # Step 4: Query and filter
    df = db.table(products).select()
    electronics = df.where(col("category") == "Electronics").collect()
    assert len(electronics) == 2

    # Step 5: Export to JSON
    json_output = tmp_path / "electronics.json"
    df.where(col("category") == "Electronics").write.json(str(json_output))

    # Verify export
    assert json_output.exists()
    with open(json_output) as f:
        exported = json.load(f)
        assert len(exported) == 2
        assert all(item["category"] == "Electronics" for item in exported)
        # Convert Decimal to float for comparison (PostgreSQL returns Decimal)
        for item in exported:
            if "price" in item and isinstance(item["price"], (str, int)):
                item["price"] = float(item["price"])


@pytest.mark.integration
@pytest.mark.postgres
def test_postgres_transaction_workflow(postgresql_connection, unique_table_name):
    """Test transaction workflow on PostgreSQL."""
    db = postgresql_connection

    accounts = f"{unique_table_name}_accounts"

    # Create table
    db.create_table(
        accounts,
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "VARCHAR(100)"),
            decimal("balance", precision=10, scale=2),
        ],
    ).collect()

    # Insert initial data
    Records.from_list(
        [
            {"id": 1, "name": "Alice", "balance": 1000.00},
            {"id": 2, "name": "Bob", "balance": 500.00},
        ],
        database=db,
    ).insert_into(accounts)

    # Transaction: Transfer money
    # Use raw SQL for expressions since update_rows expects literal values
    with db.transaction() as txn:
        from sqlalchemy import text

        # Debit from Alice
        conn = txn.connection
        conn.execute(text(f"UPDATE {accounts} SET balance = balance - 100.00 WHERE id = 1"))

        # Credit to Bob
        conn.execute(text(f"UPDATE {accounts} SET balance = balance + 100.00 WHERE id = 2"))
        # Transaction commits automatically

    # Verify transfer
    results = db.table(accounts).select().order_by(col("id")).collect()
    assert results[0]["balance"] == 900.00
    assert results[1]["balance"] == 600.00


@pytest.mark.integration
@pytest.mark.postgres
def test_postgres_analytics_workflow(postgresql_connection, unique_table_name):
    """Test analytics workflow with complex queries on PostgreSQL."""
    db = postgresql_connection

    sales = f"{unique_table_name}_sales"

    # Create schema
    db.create_table(
        sales,
        [
            column("id", "INTEGER", primary_key=True),
            column("product_id", "INTEGER"),
            column("quantity", "INTEGER"),
            decimal("unit_price", precision=10, scale=2),
            column("sale_date", "DATE"),
            column("region", "VARCHAR(50)"),
        ],
    ).collect()

    # Load data
    Records.from_list(
        [
            {
                "id": 1,
                "product_id": 1,
                "quantity": 10,
                "unit_price": 10.00,
                "sale_date": "2024-01-15",
                "region": "North",
            },
            {
                "id": 2,
                "product_id": 1,
                "quantity": 5,
                "unit_price": 10.00,
                "sale_date": "2024-01-16",
                "region": "South",
            },
            {
                "id": 3,
                "product_id": 2,
                "quantity": 8,
                "unit_price": 20.00,
                "sale_date": "2024-01-15",
                "region": "North",
            },
            {
                "id": 4,
                "product_id": 2,
                "quantity": 12,
                "unit_price": 20.00,
                "sale_date": "2024-01-17",
                "region": "South",
            },
        ],
        database=db,
    ).insert_into(sales)

    # Analytics: Revenue by region
    df = db.table(sales).select()
    analytics = (
        df.group_by("region")
        .agg(
            F.sum(col("quantity") * col("unit_price")).alias("total_revenue"),
            F.sum(col("quantity")).alias("total_quantity"),
            F.count(col("id")).alias("transaction_count"),
        )
        .order_by(col("total_revenue").desc())
        .collect()
    )

    assert len(analytics) == 2
    south = next(r for r in analytics if r["region"] == "South")
    # (5*10) + (12*20) = 50 + 240 = 290
    # Convert Decimal to float for comparison
    south_revenue = (
        float(south["total_revenue"])
        if hasattr(south["total_revenue"], "__float__")
        else south["total_revenue"]
    )
    assert south_revenue == 290.00
    assert south["total_quantity"] == 17


@pytest.mark.integration
@pytest.mark.postgres
def test_postgres_schema_evolution(postgresql_connection, unique_table_name):
    """Test schema evolution workflow on PostgreSQL."""
    db = postgresql_connection

    users = f"{unique_table_name}_users"

    # Create initial table
    db.create_table(
        users,
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "VARCHAR(100)"),
            column("email", "VARCHAR(255)"),
        ],
    ).collect()

    # Insert data
    Records.from_list(
        [
            {"id": 1, "name": "Alice", "email": "alice@example.com"},
            {"id": 2, "name": "Bob", "email": "bob@example.com"},
        ],
        database=db,
    ).insert_into(users)

    # Add new column (PostgreSQL supports ALTER TABLE)
    # Note: This tests that we can work with evolving schemas
    from sqlalchemy import text

    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.execute(text(f"ALTER TABLE {users} ADD COLUMN phone VARCHAR(20)"))

    # Insert new data with phone
    Records.from_list(
        [{"id": 3, "name": "Charlie", "email": "charlie@example.com", "phone": "555-1234"}],
        database=db,
    ).insert_into(users)

    # Query with new column
    results = db.table(users).select().where(col("id") == 3).collect()
    assert len(results) == 1
    assert results[0]["phone"] == "555-1234"
