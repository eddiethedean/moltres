from moltres import col, connect
from moltres.expressions.functions import sum as sum_  # noqa: A001
from moltres.io.read import read_table


def _seed_customers(db):
    engine = db.connection_manager.engine
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, active INTEGER)"
        )
        conn.exec_driver_sql(
            "INSERT INTO customers (id, name, active) VALUES (1, 'Alice', 1), (2, 'Bob', 0)"
        )
    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount INTEGER)"
        )
        conn.exec_driver_sql(
            "INSERT INTO orders (id, customer_id, amount) VALUES (100, 1, 50), (101, 2, 75)"
        )


def test_dataframe_collects_rows(tmp_path):
    db_path = tmp_path / "collect.sqlite"
    db = connect(f"sqlite:///{db_path}")
    _seed_customers(db)

    df = db.table("customers").select("id", "name").where(col("active") == True)  # noqa: E712
    result = df.collect()

    assert result == [{"id": 1, "name": "Alice"}]


def test_read_table_helper(tmp_path):
    db_path = tmp_path / "read.sqlite"
    db = connect(f"sqlite:///{db_path}")
    _seed_customers(db)

    rows = read_table(db, "customers", columns=["name"])
    assert {row["name"] for row in rows} == {"Alice", "Bob"}


def test_join_and_groupby_flow(tmp_path):
    db_path = tmp_path / "join.sqlite"
    db = connect(f"sqlite:///{db_path}")
    _seed_customers(db)

    orders_df = db.table("orders").select()
    customers_df = db.table("customers").select()
    df = (
        orders_df.join(customers_df, on=[("customer_id", "id")])
        .select(col("orders.id").alias("order_id"), col("customers.name").alias("customer"))
        .order_by(col("order_id"))
    )
    rows = df.collect()
    assert rows == [
        {"order_id": 100, "customer": "Alice"},
        {"order_id": 101, "customer": "Bob"},
    ]

    totals = (
        db.table("orders").select()
        .group_by("customer_id")
        .agg(sum_(col("amount")).alias("total_amount"))
        .order_by(col("customer_id"))
        .collect()
    )
    assert totals == [
        {"customer_id": 1, "total_amount": 50},
        {"customer_id": 2, "total_amount": 75},
    ]
