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

    sql = compile_plan(limited)

    assert (
        sql
        == 'SELECT "id", "name" FROM "customers" WHERE (("active" = TRUE) AND (NOT ("country" = \'US\'))) LIMIT 10'
    )


def test_compile_order_and_aliases():
    scan = operators.scan("orders")
    expr = (col("spend") * 1.1).alias("adj_spend")
    proj = operators.project(scan, (col("id"), expr))
    ordered = operators.order_by(proj, [operators.sort_order(col("created_at"), descending=True)])
    sql = compile_plan(ordered)

    assert sql == (
        'SELECT "id", ("spend" * 1.1) AS "adj_spend" FROM "orders" ORDER BY "created_at" DESC'
    )


def test_compile_join_on_columns():
    orders = operators.scan("orders")
    customers = operators.scan("customers")
    joined = operators.join(orders, customers, how="inner", on=[("customer_id", "id")])
    proj = operators.project(joined, (col("orders.id").alias("order_id"), col("customers.name")))
    sql = compile_plan(proj)

    assert "JOIN" in sql
    assert 'ON ("orders"."customer_id" = "customers"."id")' in sql
    assert '"orders"."id" AS "order_id"' in sql


def test_compile_groupby_aggregate():
    scan = operators.scan("orders")
    total = sum_(col("amount")).alias("total_amount")
    grouped = operators.aggregate(scan, (col("country"),), (total,))
    sql = compile_plan(grouped)

    assert 'GROUP BY "country"' in sql
    assert 'SUM("amount") AS "total_amount"' in sql
