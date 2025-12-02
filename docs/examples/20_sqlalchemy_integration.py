"""Example: SQLAlchemy Integration.

This example demonstrates how to integrate Moltres with existing SQLAlchemy projects,
including using existing engines, connections, sessions, and converting between
Moltres DataFrames and SQLAlchemy statements.
"""

from moltres import connect, col
from moltres.dataframe.dataframe import DataFrame
from moltres.table.schema import column
from moltres.integrations.sqlalchemy import (
    execute_with_connection,
    execute_with_session,
    to_sqlalchemy_select,
    from_sqlalchemy_select,
)
from sqlalchemy import create_engine, select, table, column as sa_column, text
from sqlalchemy.orm import sessionmaker

# ============================================================================
# Example 1: Using existing SQLAlchemy Engine
# ============================================================================

print("Example 1: Using existing SQLAlchemy Engine")

# Create an existing SQLAlchemy engine
engine = create_engine("sqlite:///:memory:")

# Create Database from existing engine
from moltres import Database

db = Database.from_engine(engine)

# Use Moltres with your existing engine
db.create_table(
    "users",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("age", "INTEGER"),
    ],
).collect()

from moltres.io.records import Records

Records(
    _data=[{"id": 1, "name": "Alice", "age": 30}, {"id": 2, "name": "Bob", "age": 25}], _database=db
).insert_into("users")

# Query with Moltres
df = db.table("users").select().where(col("age") >= 25)
results = df.collect()
print(f"Results: {results}")

db.close()

# ============================================================================
# Example 2: Using existing SQLAlchemy Connection
# ============================================================================

print("\nExample 2: Using existing SQLAlchemy Connection")

engine = create_engine("sqlite:///:memory:")

# Create Database from connection
with engine.connect() as conn:
    db = Database.from_connection(conn)

    # Use Moltres within the connection's transaction
    db.create_table(
        "orders",
        [
            column("id", "INTEGER", primary_key=True),
            column("amount", "REAL"),
            column("status", "TEXT"),
        ],
    ).collect()

    Records(_data=[{"id": 1, "amount": 100.0, "status": "active"}], _database=db).insert_into(
        "orders"
    )

    # Query with Moltres
    df = db.table("orders").select().where(col("status") == "active")
    results = df.collect()
    print(f"Results: {results}")

# ============================================================================
# Example 3: Using existing SQLAlchemy Session
# ============================================================================

print("\nExample 3: Using existing SQLAlchemy Session")

engine = create_engine("sqlite:///:memory:")
Session = sessionmaker(bind=engine)

with Session() as session:
    # Note: Database.from_session() requires Engine to be imported at runtime
    # For now, use connect() with the engine directly
    db = connect(engine=engine)

    # Use Moltres with your existing session
    db.create_table(
        "products",
        [
            column("id", "INTEGER", primary_key=True),
            column("name", "TEXT"),
            column("price", "REAL"),
        ],
    ).collect()

    Records(_data=[{"id": 1, "name": "Widget", "price": 19.99}], _database=db).insert_into(
        "products"
    )

    # Query with Moltres
    df = db.table("products").select().where(col("price") > 10.0)
    results = df.collect()
    print(f"Results: {results}")

# ============================================================================
# Example 4: Convert DataFrame to SQLAlchemy Statement
# ============================================================================

print("\nExample 4: Convert DataFrame to SQLAlchemy Statement")

db = connect("sqlite:///:memory:")

db.create_table(
    "customers",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("email", "TEXT"),
    ],
).collect()

Records(_data=[{"id": 1, "name": "Alice", "email": "alice@example.com"}], _database=db).insert_into(
    "customers"
)

# Create a Moltres DataFrame
df = db.table("customers").select().where(col("id") == 1)

# Convert to SQLAlchemy Select statement
stmt = df.to_sqlalchemy()
print(f"SQLAlchemy statement: {stmt}")

# Execute with the same engine (since we're using in-memory database)
# Note: In a real scenario with persistent databases, you could use a different engine
# For in-memory databases, we need to use the same connection

result = db.execute_sql(str(stmt.compile(compile_kwargs={"literal_binds": True})))
print(f"Executed statement: {result}")

# Or use the convenience function
stmt2 = to_sqlalchemy_select(df)
print(f"Using convenience function: {stmt2}")

db.close()

# ============================================================================
# Example 5: Create DataFrame from SQLAlchemy Statement
# ============================================================================

print("\nExample 5: Create DataFrame from SQLAlchemy Statement")

# Create a SQLAlchemy Select statement
users_table = table("users", sa_column("id"), sa_column("name"), sa_column("age"))
sa_stmt = select(users_table.c.id, users_table.c.name).where(users_table.c.age > 25)

# Convert to Moltres DataFrame
df = DataFrame.from_sqlalchemy(sa_stmt)

# Can now chain Moltres operations
df2 = df.select("name")
print(f"DataFrame from SQLAlchemy: {df2}")

# Or use the convenience function
df3 = from_sqlalchemy_select(sa_stmt)
print(f"Using convenience function: {df3}")

# ============================================================================
# Example 6: Execute DataFrame with existing Connection
# ============================================================================

print("\nExample 6: Execute DataFrame with existing Connection")

db = connect("sqlite:///:memory:")

db.create_table(
    "items",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
        column("quantity", "INTEGER"),
    ],
).collect()

Records(_data=[{"id": 1, "name": "Item A", "quantity": 10}], _database=db).insert_into("items")

# Create DataFrame
df = db.table("items").select().where(col("quantity") > 5)

# Execute with existing connection (using the same engine since we're using in-memory database)
# Note: In a real scenario with persistent databases, you could use a different connection
with db.connection_manager.connect() as conn:
    results = execute_with_connection(df, conn)
    print(f"Results from connection: {results}")

db.close()

# ============================================================================
# Example 7: Execute DataFrame with existing Session
# ============================================================================

print("\nExample 7: Execute DataFrame with existing Session")

db = connect("sqlite:///:memory:")

db.create_table(
    "inventory",
    [
        column("id", "INTEGER", primary_key=True),
        column("product", "TEXT"),
        column("stock", "INTEGER"),
    ],
).collect()

Records(_data=[{"id": 1, "product": "Product X", "stock": 50}], _database=db).insert_into(
    "inventory"
)

# Create DataFrame
df = db.table("inventory").select().where(col("stock") > 20)

# Execute with existing session (using the same engine since we're using in-memory database)
# Note: In a real scenario with persistent databases, you could use a different session
engine = create_engine("sqlite:///:memory:")
# Create the same table in the new engine for demonstration
with engine.connect() as conn:
    conn.execute(
        text("CREATE TABLE inventory (id INTEGER PRIMARY KEY, product TEXT, stock INTEGER)")
    )
    conn.execute(text("INSERT INTO inventory (id, product, stock) VALUES (1, 'Product X', 50)"))
    conn.commit()

Session = sessionmaker(bind=engine)
with Session() as session:
    results = execute_with_session(df, session)
    print(f"Results from session: {results}")

db.close()

# ============================================================================
# Example 8: Using within transactions
# ============================================================================

print("\nExample 8: Using within transactions")

engine = create_engine("sqlite:///:memory:")
db = Database.from_engine(engine)

db.create_table(
    "accounts",
    [
        column("id", "INTEGER", primary_key=True),
        column("balance", "REAL"),
    ],
).collect()

Records(_data=[{"id": 1, "balance": 100.0}], _database=db).insert_into("accounts")

# Use Moltres within a transaction
with engine.begin() as conn:
    # Create DataFrame
    df = db.table("accounts").select().where(col("id") == 1)

    # Convert to SQLAlchemy statement and execute within transaction
    stmt = df.to_sqlalchemy()
    result = conn.execute(stmt)  # type: ignore[assignment]
    rows = result.fetchall()  # type: ignore[attr-defined]
    print(f"Results within transaction: {rows}")

db.close()

print("\nAll examples completed!")
