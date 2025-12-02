import moltres.expressions.functions as F
from moltres.expressions import col, lit


def test_column_repr():
    """Test user-friendly __repr__ for Column expressions."""
    # Simple column
    assert repr(col("age")) == "col('age')"

    # Column with alias
    assert repr(col("age").alias("user_age")) == "col('age').alias('user_age')"

    # Comparison
    assert repr(col("age") > 25) == "col('age') > lit(25)"
    assert repr(col("age") < 30) == "col('age') < lit(30)"
    assert repr(col("age") == 25) == "col('age') == lit(25)"
    assert repr(col("age") != 25) == "col('age') != lit(25)"
    assert repr(col("age") >= 25) == "col('age') >= lit(25)"
    assert repr(col("age") <= 30) == "col('age') <= lit(30)"

    # Arithmetic operations
    assert repr(col("price") * 1.1) == "col('price') * lit(1.1)"
    assert repr(col("a") + col("b")) == "col('a') + col('b')"
    assert repr(col("total") - col("discount")) == "col('total') - col('discount')"
    assert repr(col("a") / col("b")) == "col('a') / col('b')"
    assert repr(col("a") % col("b")) == "col('a') % col('b')"
    assert repr(col("a") ** 2) == "col('a') ** lit(2)"

    # Complex arithmetic expressions
    assert (
        repr((col("price") * 1.1).alias("with_tax")) == "col('price') * lit(1.1).alias('with_tax')"
    )
    assert repr((col("age") + 1) * 2) == "(col('age') + lit(1)) * lit(2)"
    assert (
        repr(col("total") - (col("discount") + col("tax")))
        == "col('total') - (col('discount') + col('tax'))"
    )

    # Logical operations
    expr = (col("age") > 25) & (col("active") == True)  # noqa: E712
    assert "col('age')" in repr(expr)
    assert "col('active')" in repr(expr)
    assert "&" in repr(expr) or "and" in repr(expr).lower()

    expr_or = (col("age") < 18) | (col("age") > 65)
    assert "col('age')" in repr(expr_or)
    assert "|" in repr(expr_or) or "or" in repr(expr_or).lower()

    expr_not = ~(col("active") == True)  # noqa: E712
    assert "col('active')" in repr(expr_not)

    # Null checks
    assert repr(col("name").is_null()) == "col('name').is_null()"
    assert repr(col("name").is_not_null()) == "col('name').is_not_null()"

    # String operations
    assert repr(col("name").contains("test")) == "col('name').contains('test')"
    assert repr(col("email").like("%@example.com")) == "col('email').like('%@example.com')"
    assert repr(col("email").ilike("%@EXAMPLE.COM")) == "col('email').ilike('%@EXAMPLE.COM')"
    assert repr(col("name").startswith("A")) == "col('name').startswith('A')"
    assert repr(col("name").endswith("e")) == "col('name').endswith('e')"

    # Between
    assert "between" in repr(col("age").between(18, 65)).lower()

    # In
    in_expr = col("id").isin([1, 2, 3])
    assert "isin" in repr(in_expr).lower()
    assert "1" in repr(in_expr) or "col('id')" in repr(in_expr)

    # Cast
    cast_expr = col("price").cast("DECIMAL", precision=10, scale=2)
    assert "cast" in repr(cast_expr).lower()
    assert "DECIMAL" in repr(cast_expr) or "decimal" in repr(cast_expr).lower()

    # Sort operations
    assert repr(col("age").asc()) == "col('age').asc()"
    assert repr(col("age").desc()) == "col('age').desc()"

    # Functions
    func_repr = repr(F.sum(col("amount")))
    assert "sum" in func_repr
    assert "amount" in func_repr or "col('amount')" in func_repr


def test_column_repr_edge_cases():
    """Test Column __repr__ with edge cases."""
    # Qualified column names
    assert "users" in repr(col("users.age")) or "age" in repr(col("users.age"))

    # Very long column names
    long_name = "a" * 50
    assert long_name[:20] in repr(col(long_name)) or long_name in repr(col(long_name))

    # Column with special characters in name
    assert "col" in repr(col("user_name")).lower()

    # Nested complex expressions
    complex_expr = ((col("a") + col("b")) * (col("c") - col("d"))).alias("result")
    repr_str = repr(complex_expr)
    assert "col" in repr_str.lower()
    assert ".alias('result')" in repr_str

    # Window functions
    window_expr = F.sum(col("amount")).over(partition_by=col("category"))
    repr_str = repr(window_expr)
    assert "over" in repr_str.lower() or "window" in repr_str.lower()

    # Filter clause on aggregation
    filtered_agg = F.sum(col("amount")).filter(col("status") == "active")
    repr_str = repr(filtered_agg)
    assert "sum" in repr_str.lower()

    # Multiple aliases (should only show the final one)
    multi_alias = col("age").alias("user_age").alias("final_age")
    repr_str = repr(multi_alias)
    assert "alias" in repr_str.lower()

    assert "avg" in repr(F.avg(col("age"))).lower()
    assert "count" in repr(F.count(col("id"))).lower()
    assert "max" in repr(F.max(col("age"))).lower()
    assert "min" in repr(F.min(col("age"))).lower()

    # Function with alias
    func_alias = F.sum(col("amount")).alias("total")
    assert "sum" in repr(func_alias)
    assert ".alias('total')" in repr(func_alias)

    # Literal values
    assert "lit" in repr(lit(25)).lower()
    assert "lit" in repr(lit("test")).lower()
    assert "lit" in repr(lit(True)).lower()
    assert "lit" in repr(lit(None)).lower() or "None" in repr(lit(None))

    # Complex nested expressions
    complex_expr = ((col("price") * 1.1) + col("tax")).alias("total")
    assert "col('price')" in repr(complex_expr)
    assert ".alias('total')" in repr(complex_expr)

    # Unary operations
    assert "-" in repr(-col("amount")) or "neg" in repr(-col("amount")).lower()


def test_column_arithmetic_chain():
    expr = col("spend") * 1.1 - col("discount")
    assert expr.op == "sub"
    left = expr.args[0]
    assert left.op == "mul"
    assert left.args[0].source == "spend"
    assert left.args[1].op == "literal"


def test_column_comparisons_and_boolean_ops():
    expr = (col("active") == True) & ~(col("country") == "US")  # noqa: E712
    assert expr.op == "and"
    left = expr.args[0]
    assert left.op == "eq"
    right = expr.args[1]
    assert right.op == "not"


def test_literal_helper_creates_literal_expression():
    literal_expr = lit("value")
    assert literal_expr.op == "literal"
    assert literal_expr.args == ("value",)


def test_aggregate_helpers():
    assert callable(F.sum)
    agg = F.sum(col("amount")).alias("total")
    assert agg.op == "agg_sum"
    assert agg.alias_name == "total"
    assert F.avg("amount").op == "agg_avg"


def test_count_helpers():
    assert F.count().op == "agg_count_star"
    single = F.count(col("id"))
    assert single.op == "agg_count"
    distinct = F.count_distinct(col("country"), col("state"))
    assert distinct.op == "agg_count_distinct"
    assert len(distinct.args) == 2


def test_variadic_string_helpers():
    expr = F.concat(col("first"), " ", col("last"))
    assert expr.op == "concat"
    assert len(expr.args) == 3
    nz = F.coalesce(col("nickname"), col("first"))
    assert nz.op == "coalesce"


def test_isnull_isnotnull_aliases():
    """Test isnull() and isnotnull() aliases for is_null() and is_not_null()."""
    from moltres.expressions.functions import isnull, isnotnull

    # Test isnull() creates is_null operation
    isnull_expr = isnull(col("email"))
    assert isnull_expr.op == "is_null"
    assert isnull_expr.args[0].source == "email"

    # Test isnotnull() creates is_not_null operation
    isnotnull_expr = isnotnull(col("email"))
    assert isnotnull_expr.op == "is_not_null"
    assert isnotnull_expr.args[0].source == "email"

    # Verify they work the same as is_null() and is_not_null() methods
    assert isnull_expr.op == col("email").is_null().op
    assert isnotnull_expr.op == col("email").is_not_null().op


def test_cast_decimal_with_precision_scale_execution(tmp_path):
    """Test cast() to DECIMAL with precision and scale in actual query."""
    from moltres import connect, col

    db_path = tmp_path / "cast_decimal.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE prices (id INTEGER PRIMARY KEY, amount REAL)")
        conn.exec_driver_sql("INSERT INTO prices (id, amount) VALUES (1, 99.999), (2, 149.50)")

    # Test casting to DECIMAL with precision and scale
    df = db.table("prices").select(
        col("id"), col("amount").cast("DECIMAL", precision=10, scale=2).alias("amount_decimal")
    )

    result = df.collect()
    assert len(result) == 2
    # The cast should work (SQLite will handle it)
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2


def test_case_when_expression(tmp_path):
    """Test CASE WHEN expressions work correctly."""
    from moltres import connect, col
    from moltres.table.schema import column
    from moltres.io.records import Records

    db_path = tmp_path / "case_when.sqlite"
    db = connect(f"sqlite:///{db_path}")

    # Create table
    db.create_table(
        "employees",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("salary", "REAL"),
        ],
    ).collect()

    # Insert data
    employees_data = [
        {"id": 1, "name": "Alice", "salary": 80000.0},
        {"id": 2, "name": "Bob", "salary": 50000.0},
        {"id": 3, "name": "Charlie", "salary": 60000.0},
    ]
    Records(_data=employees_data, _database=db).insert_into("employees")

    # Test CASE WHEN with single condition
    df = db.table("employees").select()
    result = df.select(
        col("name"),
        col("salary"),
        F.when(col("salary") > 75000, lit("High")).otherwise(lit("Low")).alias("tier"),
    )
    results = result.collect()

    assert len(results) == 3
    assert results[0]["tier"] == "High"  # Alice: 80000 > 75000
    assert results[1]["tier"] == "Low"  # Bob: 50000 <= 75000
    assert results[2]["tier"] == "Low"  # Charlie: 60000 <= 75000

    # Test CASE WHEN with multiple conditions
    result2 = df.select(
        col("name"),
        F.when(col("salary") > 75000, lit("High"))
        .when(col("salary") > 55000, lit("Medium"))
        .otherwise(lit("Low"))
        .alias("tier"),
    )
    results2 = result2.collect()

    assert results2[0]["tier"] == "High"  # Alice: 80000 > 75000
    assert results2[1]["tier"] == "Low"  # Bob: 50000 <= 55000
    assert results2[2]["tier"] == "Medium"  # Charlie: 55000 < 60000 <= 75000
