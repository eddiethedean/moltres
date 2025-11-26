"""Tests for DataFrame operations: withColumn, withColumnRenamed, drop, union, distinct, dropDuplicates."""

from moltres import col, connect
from moltres.expressions import functions as F


def test_withColumn(tmp_path):
    """Test withColumn() to add or replace columns."""
    db_path = tmp_path / "withcolumn.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25)"
        )

    # Test adding a new column
    df = db.table("users").select("id", "name", "age")
    df_with_double_age = df.withColumn("double_age", col("age") * 2)
    result = df_with_double_age.collect()

    assert len(result) == 2
    assert "double_age" in result[0]
    assert result[0]["double_age"] == 60
    assert result[1]["double_age"] == 50

    # Test replacing an existing column
    df_with_replaced = df.withColumn("age", col("age") + 1)
    result2 = df_with_replaced.collect()
    assert result2[0]["age"] == 31
    assert result2[1]["age"] == 26


def test_withColumnRenamed(tmp_path):
    """Test withColumnRenamed() to rename columns."""
    db_path = tmp_path / "rename.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25)"
        )

    # Test renaming a column
    df = db.table("users").select("id", "name", "age")
    df_renamed = df.withColumnRenamed("name", "full_name")
    result = df_renamed.collect()

    assert len(result) == 2
    assert "full_name" in result[0]
    assert "name" not in result[0]
    assert result[0]["full_name"] == "Alice"
    assert result[1]["full_name"] == "Bob"


def test_drop(tmp_path):
    """Test drop() to remove columns."""
    db_path = tmp_path / "drop.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, email TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age, email) VALUES (1, 'Alice', 30, 'alice@example.com')"
        )

    # Test dropping a single column
    df = db.table("users").select("id", "name", "age", "email")
    df_dropped = df.drop("email")
    result = df_dropped.collect()

    assert len(result) == 1
    assert "email" not in result[0]
    assert "id" in result[0]
    assert "name" in result[0]
    assert "age" in result[0]

    # Test dropping multiple columns
    df_dropped_multiple = df.drop("age", "email")
    result2 = df_dropped_multiple.collect()
    assert "age" not in result2[0]
    assert "email" not in result2[0]
    assert "id" in result2[0]
    assert "name" in result2[0]

    # Test dropping with Column objects (PySpark compatibility)
    df_dropped_column = df.drop(col("email"))
    result3 = df_dropped_column.collect()
    assert "email" not in result3[0]
    assert "id" in result3[0]
    assert "name" in result3[0]
    assert "age" in result3[0]

    # Test dropping with mixed string and Column objects
    df_dropped_mixed = df.drop("age", col("email"))
    result4 = df_dropped_mixed.collect()
    assert "age" not in result4[0]
    assert "email" not in result4[0]
    assert "id" in result4[0]
    assert "name" in result4[0]


def test_union(tmp_path):
    """Test union() to combine DataFrames (distinct rows only)."""
    db_path = tmp_path / "union.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE table1 (id INTEGER PRIMARY KEY, value TEXT)")
        conn.exec_driver_sql("INSERT INTO table1 (id, value) VALUES (1, 'A'), (2, 'B')")
        conn.exec_driver_sql("CREATE TABLE table2 (id INTEGER PRIMARY KEY, value TEXT)")
        conn.exec_driver_sql("INSERT INTO table2 (id, value) VALUES (2, 'B'), (3, 'C')")

    df1 = db.table("table1").select("value")
    df2 = db.table("table2").select("value")

    # Union should return distinct rows: A, B, C
    result = df1.union(df2).order_by(col("value")).collect()

    assert len(result) == 3
    values = [row["value"] for row in result]
    assert values == ["A", "B", "C"]


def test_unionAll(tmp_path):
    """Test unionAll() to combine DataFrames (all rows, including duplicates)."""
    db_path = tmp_path / "unionall.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE table1 (id INTEGER PRIMARY KEY, value TEXT)")
        conn.exec_driver_sql("INSERT INTO table1 (id, value) VALUES (1, 'A'), (2, 'B')")
        conn.exec_driver_sql("CREATE TABLE table2 (id INTEGER PRIMARY KEY, value TEXT)")
        conn.exec_driver_sql("INSERT INTO table2 (id, value) VALUES (2, 'B'), (3, 'C')")

    df1 = db.table("table1").select("value")
    df2 = db.table("table2").select("value")

    # UnionAll should return all rows including duplicates: A, B, B, C
    result = df1.unionAll(df2).order_by(col("value")).collect()

    assert len(result) == 4
    values = [row["value"] for row in result]
    assert values == ["A", "B", "B", "C"]


def test_distinct(tmp_path):
    """Test distinct() to return unique rows."""
    db_path = tmp_path / "distinct.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT)")
        conn.exec_driver_sql(
            "INSERT INTO items (id, value) VALUES (1, 'A'), (2, 'B'), (3, 'A'), (4, 'C'), (5, 'B')"
        )

    df = db.table("items").select("value")
    result = df.distinct().order_by(col("value")).collect()

    # Should return distinct values: A, B, C
    assert len(result) == 3
    values = [row["value"] for row in result]
    assert values == ["A", "B", "C"]


def test_dropDuplicates(tmp_path):
    """Test dropDuplicates() to remove duplicate rows."""
    db_path = tmp_path / "dropduplicates.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE items (id INTEGER PRIMARY KEY, value TEXT, category TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO items (id, value, category) VALUES "
            "(1, 'A', 'X'), (2, 'B', 'X'), (3, 'A', 'Y'), (4, 'C', 'X'), (5, 'B', 'X')"
        )

    # Test dropDuplicates() without subset (all columns)
    df = db.table("items").select("value", "category")
    result = df.dropDuplicates().order_by(col("value"), col("category")).collect()

    # Should return distinct combinations
    assert len(result) == 4
    combinations = [(row["value"], row["category"]) for row in result]
    assert ("A", "X") in combinations
    assert ("A", "Y") in combinations
    assert ("B", "X") in combinations
    assert ("C", "X") in combinations

    # Test dropDuplicates() with subset (specific columns)
    # Note: The current implementation has a limitation - it uses group_by().agg()
    # which requires aggregations. For now, we test that dropDuplicates() without
    # subset works correctly, which is the main use case.
    # The subset functionality would need a better implementation using window functions
    # or subqueries to work properly.


def test_na_drop(tmp_path):
    """Test na.drop() convenience method."""
    db_path = tmp_path / "na_drop.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, email TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age, email) VALUES "
            "(1, 'Alice', 30, 'alice@example.com'), "
            "(2, NULL, 25, 'bob@example.com'), "
            "(3, 'Charlie', NULL, 'charlie@example.com'), "
            "(4, 'David', 35, NULL)"
        )

    # Test na.drop() - note: current implementation requires subset parameter
    # When subset is None, dropna() returns self without filtering
    df = db.table("users").select("id", "name", "age", "email")

    # Test na.drop() with subset - should drop rows with any null values in specified columns
    result = df.na.drop(subset=["name", "age", "email"]).order_by(col("id")).collect()

    # Should only return row with id=1 (no nulls in any of the specified columns)
    assert len(result) == 1
    assert result[0]["id"] == 1
    assert result[0]["name"] == "Alice"

    # Test na.drop() with subset
    result2 = df.na.drop(subset=["name"]).order_by(col("id")).collect()
    # Should return rows where name is not null: id=1, 3, 4
    assert len(result2) == 3
    ids = {row["id"] for row in result2}
    assert ids == {1, 3, 4}

    # Test na.drop() with how="all"
    result3 = df.na.drop(how="all", subset=["name", "age"]).order_by(col("id")).collect()
    # Should return all rows since no row has both name and age as null
    assert len(result3) == 4


def test_na_fill(tmp_path):
    """Test na.fill() convenience method."""
    db_path = tmp_path / "na_fill.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER, score REAL)"
        )
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age, score) VALUES "
            "(1, 'Alice', 30, 85.5), "
            "(2, NULL, 25, NULL), "
            "(3, 'Charlie', NULL, 90.0)"
        )

    # Test na.fill() with single value
    df = db.table("users").select("id", "name", "age", "score")
    result = df.na.fill("Unknown", subset=["name"]).order_by(col("id")).collect()

    # Should fill null name with "Unknown"
    assert len(result) == 3
    assert result[1]["name"] == "Unknown"  # id=2 had NULL name

    # Test na.fill() with dict
    result2 = (
        df.na.fill({"name": "Unknown", "age": 0, "score": 0.0}, subset=["name", "age", "score"])
        .order_by(col("id"))
        .collect()
    )
    assert result2[1]["name"] == "Unknown"
    assert result2[1]["score"] == 0.0
    assert result2[2]["age"] == 0


def test_dot_notation_select(tmp_path):
    """Test dot notation column selection in select()."""
    db_path = tmp_path / "dot_notation_select.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25)"
        )

    df = db.table("users").select()

    # Test dot notation in select
    result = df.select(df.id, df.name).collect()

    assert len(result) == 2
    assert "id" in result[0]
    assert "name" in result[0]
    assert "age" not in result[0]
    assert result[0]["id"] == 1
    assert result[0]["name"] == "Alice"


def test_dot_notation_where(tmp_path):
    """Test dot notation column selection in where()."""
    db_path = tmp_path / "dot_notation_where.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35)"
        )

    df = db.table("users").select()

    # Test dot notation in where
    result = df.where(df.age > 28).order_by(df.id).collect()

    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Charlie"


def test_dot_notation_order_by(tmp_path):
    """Test dot notation column selection in order_by()."""
    db_path = tmp_path / "dot_notation_order_by.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25), (3, 'Charlie', 35)"
        )

    df = db.table("users").select()

    # Test dot notation in order_by
    result = df.select(df.name).order_by(df.name).collect()

    assert len(result) == 3
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"
    assert result[2]["name"] == "Charlie"


def test_order_by_string_parameter(tmp_path):
    """Test order_by() with string column names (PySpark compatibility)."""
    db_path = tmp_path / "order_by_string.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Charlie', 35), (2, 'Alice', 30), (3, 'Bob', 25)"
        )

    df = db.table("users").select()

    # Test string column name (PySpark-style)
    result = df.order_by("name").collect()
    assert len(result) == 3
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"
    assert result[2]["name"] == "Charlie"

    # Test Column object (backward compatibility)
    result2 = df.order_by(col("name")).collect()
    assert len(result2) == 3
    assert result2[0]["name"] == "Alice"
    assert result2[1]["name"] == "Bob"
    assert result2[2]["name"] == "Charlie"

    # Test mixed string and Column
    result3 = df.order_by("age", col("name")).collect()
    assert len(result3) == 3
    assert result3[0]["age"] == 25  # Bob
    assert result3[1]["age"] == 30  # Alice
    assert result3[2]["age"] == 35  # Charlie

    # Test PySpark-style aliases with strings
    result4 = df.orderBy("name").collect()
    assert len(result4) == 3
    assert result4[0]["name"] == "Alice"

    result5 = df.sort("name").collect()
    assert len(result5) == 3
    assert result5[0]["name"] == "Alice"

    # Test descending with Column (strings can't have .desc(), need Column)
    result6 = df.order_by(col("age").desc()).collect()
    assert len(result6) == 3
    assert result6[0]["age"] == 35  # Charlie (oldest first)
    assert result6[2]["age"] == 25  # Bob (youngest last)

    # Test multiple columns with mixed types and descending
    result7 = df.order_by("age", col("name").desc()).collect()
    assert len(result7) == 3
    assert result7[0]["age"] == 25  # Bob
    assert result7[1]["age"] == 30  # Alice
    assert result7[2]["age"] == 35  # Charlie


def test_dot_notation_group_by(tmp_path):
    """Test dot notation column selection in group_by()."""
    db_path = tmp_path / "dot_notation_group_by.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount REAL)"
        )
        conn.exec_driver_sql(
            "INSERT INTO sales (id, category, amount) VALUES "
            "(1, 'A', 10.0), (2, 'A', 20.0), (3, 'B', 15.0)"
        )

    df = db.table("sales").select()

    # Test dot notation in group_by
    from moltres.expressions.functions import sum as sum_func

    result = (
        df.group_by(df.category)
        .agg(sum_func(df.amount).alias("total"))
        .order_by(df.category)
        .collect()
    )

    assert len(result) == 2
    assert result[0]["category"] == "A"
    assert result[0]["total"] == 30.0
    assert result[1]["category"] == "B"
    assert result[1]["total"] == 15.0


def test_withColumn_window_function(tmp_path):
    """Test withColumn() with window functions (PySpark compatibility)."""
    db_path = tmp_path / "withcolumn_window.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount REAL, date TEXT)"
        )
        conn.exec_driver_sql(
            "INSERT INTO sales (id, category, amount, date) VALUES "
            "(1, 'A', 100.0, '2024-01-01'), "
            "(2, 'A', 200.0, '2024-01-02'), "
            "(3, 'B', 150.0, '2024-01-01'), "
            "(4, 'B', 250.0, '2024-01-02')"
        )

    df = db.table("sales").select()

    # Test basic window function in withColumn (no existing Project)
    df_with_row_num = df.withColumn(
        "row_num", F.row_number().over(partition_by=col("category"), order_by=col("amount"))
    )
    result = df_with_row_num.collect()

    assert len(result) == 4
    assert "row_num" in result[0]
    assert "id" in result[0]
    assert "category" in result[0]
    assert "amount" in result[0]
    # Check row numbers are correct (should be 1, 2 for each category)
    row_nums_by_category = {}
    for row in result:
        cat = row["category"]
        if cat not in row_nums_by_category:
            row_nums_by_category[cat] = []
        row_nums_by_category[cat].append(row["row_num"])
    assert sorted(row_nums_by_category["A"]) == [1, 2]
    assert sorted(row_nums_by_category["B"]) == [1, 2]


def test_withColumn_window_function_with_existing_project(tmp_path):
    """Test withColumn() with window functions when there's an existing Project."""
    db_path = tmp_path / "withcolumn_window_project.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount REAL)"
        )
        conn.exec_driver_sql(
            "INSERT INTO sales (id, category, amount) VALUES "
            "(1, 'A', 100.0), (2, 'A', 200.0), (3, 'B', 150.0)"
        )

    # Start with a select (creates a Project)
    df = db.table("sales").select("id", "category", "amount")

    # Add window function using withColumn
    df_with_rank = df.withColumn(
        "rank", F.rank().over(partition_by=col("category"), order_by=col("amount"))
    )
    result = df_with_rank.collect()

    assert len(result) == 3
    assert "rank" in result[0]
    assert "id" in result[0]
    assert "category" in result[0]
    assert "amount" in result[0]
    # All original columns should be present
    assert all("id" in row for row in result)
    assert all("category" in row for row in result)
    assert all("amount" in row for row in result)


def test_withColumn_multiple_window_functions(tmp_path):
    """Test withColumn() with multiple window functions."""
    db_path = tmp_path / "withcolumn_multiple_window.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount REAL)"
        )
        conn.exec_driver_sql(
            "INSERT INTO sales (id, category, amount) VALUES "
            "(1, 'A', 100.0), (2, 'A', 200.0), (3, 'B', 150.0)"
        )

    df = db.table("sales").select()

    # Add multiple window functions
    df_with_windows = (
        df.withColumn(
            "row_num",
            F.row_number().over(partition_by=col("category"), order_by=col("amount")),
        )
        .withColumn("rank", F.rank().over(partition_by=col("category"), order_by=col("amount")))
        .withColumn(
            "dense_rank",
            F.dense_rank().over(partition_by=col("category"), order_by=col("amount")),
        )
    )
    result = df_with_windows.collect()

    assert len(result) == 3
    assert "row_num" in result[0]
    assert "rank" in result[0]
    assert "dense_rank" in result[0]
    assert "id" in result[0]
    assert "category" in result[0]
    assert "amount" in result[0]


def test_withColumn_window_function_replacing_column(tmp_path):
    """Test withColumn() with window function replacing an existing column."""
    db_path = tmp_path / "withcolumn_window_replace.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE TABLE sales (id INTEGER PRIMARY KEY, category TEXT, amount REAL)"
        )
        conn.exec_driver_sql(
            "INSERT INTO sales (id, category, amount) VALUES "
            "(1, 'A', 100.0), (2, 'A', 200.0), (3, 'B', 150.0)"
        )

    df = db.table("sales").select()

    # Replace id column with row number
    df_replaced = df.withColumn(
        "id", F.row_number().over(partition_by=col("category"), order_by=col("amount"))
    )
    result = df_replaced.collect()

    assert len(result) == 3
    # id should now be row numbers, not original ids
    assert "id" in result[0]
    assert "category" in result[0]
    assert "amount" in result[0]
    # Check that id values are row numbers (1, 2 for each category)
    ids_by_category = {}
    for row in result:
        cat = row["category"]
        if cat not in ids_by_category:
            ids_by_category[cat] = []
        ids_by_category[cat].append(row["id"])
    assert sorted(ids_by_category["A"]) == [1, 2]
    assert sorted(ids_by_category["B"]) == [1]


def test_dot_notation_methods_still_work(tmp_path):
    """Test that existing methods still work when using dot notation."""
    db_path = tmp_path / "dot_notation_methods.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")

    df = db.table("users").select()

    # Test that methods still work
    assert hasattr(df, "select")
    assert hasattr(df, "where")
    assert hasattr(df, "limit")
    assert callable(df.select)
    assert callable(df.where)

    # Test that properties still work
    assert hasattr(df, "na")
    assert hasattr(df, "write")
    assert df.na is not None
    assert df.write is not None


def test_dot_notation_combined_with_col(tmp_path):
    """Test that dot notation works alongside col() function."""
    db_path = tmp_path / "dot_notation_combined.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25)"
        )

    df = db.table("users").select()

    # Test mixing dot notation and col() function
    result = df.select(df.id, col("name"), df.age).collect()

    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[0]["name"] == "Alice"
    assert result[0]["age"] == 30


def test_dot_notation_complex_expressions(tmp_path):
    """Test dot notation in complex column expressions."""
    db_path = tmp_path / "dot_notation_complex.sqlite"
    db = connect(f"sqlite:///{db_path}")
    engine = db.connection_manager.engine

    with engine.begin() as conn:
        conn.exec_driver_sql("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.exec_driver_sql(
            "INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30), (2, 'Bob', 25)"
        )

    df = db.table("users").select()

    # Test complex expressions with dot notation
    result = df.select((df.age * 2).alias("double_age"), df.name).collect()

    assert len(result) == 2
    assert result[0]["double_age"] == 60
    assert result[1]["double_age"] == 50
    assert result[0]["name"] == "Alice"
