"""Example: Column expressions and functions.

This example demonstrates using column expressions, functions, and operators.
"""

from moltres import connect, col, lit
from moltres.expressions import functions as F

db = connect("sqlite:///example.db")

# Clean up any existing tables
db.drop_table("employees", if_exists=True).collect()

# Create table
from moltres.table.schema import column

db.create_table(
    "employees",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("salary", "REAL"),
        column("department", "TEXT"),
        column("hire_date", "TEXT"),
    ],
).collect()

# Insert data
from moltres.io.records import Records

employees_data = [
    {
        "id": 1,
        "name": "Alice Smith",
        "salary": 75000.0,
        "department": "Engineering",
        "hire_date": "2020-01-15",
    },
    {
        "id": 2,
        "name": "Bob Jones",
        "salary": 80000.0,
        "department": "Engineering",
        "hire_date": "2019-06-01",
    },
    {
        "id": 3,
        "name": "Charlie Brown",
        "salary": 70000.0,
        "department": "Sales",
        "hire_date": "2021-03-10",
    },
]

Records(_data=employees_data, _database=db).insert_into("employees")

df = db.table("employees").select()

# Arithmetic operations
result = df.select(
    col("name"),
    col("salary"),
    (col("salary") * 1.1).alias("salary_plus_10_percent"),
)
results = result.collect()
print(f"Salary calculations: {results}")
# Output: Salary calculations: [{'name': 'Alice Smith', 'salary': 75000.0, 'salary_plus_10_percent': 82500.0}, {'name': 'Bob Jones', 'salary': 80000.0, 'salary_plus_10_percent': 88000.0}, {'name': 'Charlie Brown', 'salary': 70000.0, 'salary_plus_10_percent': 77000.0}]

# String functions
result = df.select(
    col("name"),
    F.upper(col("name")).alias("name_upper"),
    F.lower(col("name")).alias("name_lower"),
    F.substring(col("name"), 1, 5).alias("name_prefix"),
)
results = result.collect()
print(f"String operations: {results}")
# Output: String operations: [{'name': 'Alice Smith', 'name_upper': 'ALICE SMITH', 'name_lower': 'alice smith', 'name_prefix': 'Alice'}, {'name': 'Bob Jones', 'name_upper': 'BOB JONES', 'name_lower': 'bob jones', 'name_prefix': 'Bob J'}, {'name': 'Charlie Brown', 'name_upper': 'CHARLIE BROWN', 'name_lower': 'charlie brown', 'name_prefix': 'Charl'}]

# Conditional expressions (CASE WHEN)
result = df.select(
    col("name"),
    col("salary"),
    F.when(col("salary") > 75000, lit("High")).otherwise(lit("Low")).alias("salary_tier"),
)
results = result.collect()
print(f"Conditional expressions: {results}")
# Output: Conditional expressions: [{'name': 'Alice Smith', 'salary': 75000.0, 'salary_tier': 'Low'}, {'name': 'Bob Jones', 'salary': 80000.0, 'salary_tier': 'High'}, {'name': 'Charlie Brown', 'salary': 70000.0, 'salary_tier': 'Low'}]

# Using simple comparison as alternative
result = df.select(
    col("name"),
    col("salary"),
    (col("salary") > 75000).alias("is_high_salary"),
)
results = result.collect()
print(f"Salary comparison: {results}")
# Output: Salary comparison: [{'name': 'Alice Smith', 'salary': 75000.0, 'is_high_salary': False}, {'name': 'Bob Jones', 'salary': 80000.0, 'is_high_salary': True}, {'name': 'Charlie Brown', 'salary': 70000.0, 'is_high_salary': False}]

# Aggregate functions
grouped = df.group_by("department")
result = grouped.agg(
    F.count(col("id")).alias("count"),
    F.avg(col("salary")).alias("avg_salary"),
    F.min(col("salary")).alias("min_salary"),
    F.max(col("salary")).alias("max_salary"),
)
results = result.collect()
print(f"Aggregations by department: {results}")
# Output: Aggregations by department: [{'department': 'Engineering', 'count': 2, 'avg_salary': 77500.0, 'min_salary': 75000.0, 'max_salary': 80000.0}, {'department': 'Sales', 'count': 1, 'avg_salary': 70000.0, 'min_salary': 70000.0, 'max_salary': 70000.0}]

# Complex expressions
result = df.select(
    col("name"),
    col("salary"),
    (col("salary") / lit(12)).alias("monthly_salary"),
    F.concat(col("department"), lit(" - "), col("name")).alias("dept_name"),
)
results = result.collect()
print(f"Complex expressions: {results}")
# Output: Complex expressions: [{'name': 'Alice Smith', 'salary': 75000.0, 'monthly_salary': 6250.0, 'dept_name': 'Engineering - Alice Smith'}, {'name': 'Bob Jones', 'salary': 80000.0, 'monthly_salary': 6666.666666666667, 'dept_name': 'Engineering - Bob Jones'}, {'name': 'Charlie Brown', 'salary': 70000.0, 'monthly_salary': 5833.333333333333, 'dept_name': 'Sales - Charlie Brown'}]

db.close()
