from moltres.expressions import col
from moltres.expressions.functions import sum as sum_  # noqa: A001
from moltres.logical import operators
from moltres.sql.compiler import compile_plan


def test_compile_project_filter_limit():
    scan = operators.scan("customers")
    proj = operators.project(scan, (col("id"), col("name")))
    predicate = (col("active") == True) & ~(col("country") == "US")  # noqa: E712
    filt = operators.filter(proj, predicate)
    limited = operators.limit(filt, 10)

    stmt = compile_plan(limited)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # Check key parts of the SQL (SQLAlchemy formatting may vary)
    assert "SELECT" in sql
    assert "id" in sql
    assert "name" in sql
    assert "FROM" in sql
    assert "customers" in sql
    assert "WHERE" in sql
    assert "active" in sql
    assert "country" in sql
    assert "US" in sql
    assert "LIMIT 10" in sql


def test_compile_order_and_aliases():
    scan = operators.scan("orders")
    expr = (col("spend") * 1.1).alias("adj_spend")
    proj = operators.project(scan, (col("id"), expr))
    ordered = operators.order_by(proj, [operators.sort_order(col("created_at"), descending=True)])
    stmt = compile_plan(ordered)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # Check key parts of the SQL
    assert "SELECT" in sql
    assert "id" in sql
    assert "adj_spend" in sql
    assert "spend" in sql
    assert "1.1" in sql
    assert "FROM" in sql
    assert "orders" in sql
    assert "ORDER BY" in sql
    assert "created_at" in sql
    assert "DESC" in sql


def test_compile_join_on_columns():
    orders = operators.scan("orders")
    customers = operators.scan("customers")
    joined = operators.join(orders, customers, how="inner", on=[("customer_id", "id")])
    proj = operators.project(joined, (col("orders.id").alias("order_id"), col("customers.name")))
    stmt = compile_plan(proj)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # Check key parts of the SQL
    assert "JOIN" in sql
    assert "customer_id" in sql
    assert "id" in sql
    assert "order_id" in sql
    assert "name" in sql


def test_compile_groupby_aggregate():
    scan = operators.scan("orders")
    total = sum_(col("amount")).alias("total_amount")
    grouped = operators.aggregate(scan, (col("country"),), (total,))
    stmt = compile_plan(grouped)
    sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))

    # Check key parts of the SQL
    assert "GROUP BY" in sql
    assert "country" in sql
    assert "SUM" in sql or "sum" in sql
    assert "amount" in sql
    assert "total_amount" in sql
