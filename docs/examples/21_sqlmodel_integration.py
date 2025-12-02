"""SQLModel Integration Examples

This example demonstrates how to use Moltres with SQLModel models.
SQLModel combines SQLAlchemy and Pydantic, providing type-safe database models.

When a SQLModel is attached to a DataFrame, `collect()` will return
SQLModel instances instead of dictionaries.
"""

# Example 1: Basic SQLModel Integration
# =====================================

try:
    from sqlmodel import SQLModel, Field, create_engine

    # Define a SQLModel
    class User(SQLModel, table=True):
        __tablename__ = "users"

        id: int = Field(primary_key=True)
        name: str
        email: str
        age: int

    # Create database and table
    from moltres import connect
    from moltres.table.schema import column

    db = connect("sqlite:///:memory:")
    db.create_table(
        "users",
        [
            column("id", "INTEGER"),
            column("name", "TEXT"),
            column("email", "TEXT"),
            column("age", "INTEGER"),
        ],
    ).collect()

    # Insert some data
    from moltres.io.records import Records

    Records(
        _data=[
            {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30},
            {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25},
            {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35},
        ],
        _database=db,
    ).insert_into("users")

    # Method 1: Attach model when creating DataFrame from table
    # ---------------------------------------------------------
    print("=== Method 1: Using table() with SQLModel ===")
    users_table = db.table(User)  # type: ignore[arg-type]  # Pass SQLModel class directly
    df = users_table.select()
    results = df.collect()

    # Results are SQLModel instances
    print(f"Type of first result: {type(results[0])}")
    print(f"First user: {results[0].name} ({results[0].email})")  # type: ignore[attr-defined]
    print(f"First user age: {results[0].age}")  # type: ignore[attr-defined]
    print()

    # Method 2: Attach model using with_model()
    # ------------------------------------------
    print("=== Method 2: Using with_model() ===")
    df2 = db.table("users").select()
    df2_with_model = df2.with_model(User)
    results2 = df2_with_model.collect()

    print(f"Type of first result: {type(results2[0])}")
    print(f"First user: {results2[0].name}")  # type: ignore[attr-defined]
    print()

    # Method 3: Using integration helpers
    # ------------------------------------
    print("=== Method 3: Using integration helpers ===")
    from moltres.integrations.sqlalchemy import with_sqlmodel
    from moltres import col

    df3 = db.table("users").select().where(col("age") > 28)
    df3_with_model = with_sqlmodel(df3, User)
    results3 = df3_with_model.collect()

    print(f"Users over 28: {len(results3)}")
    for user in results3:
        print(f"  - {user.name} ({user.age})")  # type: ignore[attr-defined]
    print()

    # Method 4: Chaining operations with model attached
    # --------------------------------------------------
    print("=== Method 4: Chaining operations ===")
    df4 = db.table(User).select().where(col("age") > 25).order_by("age")  # type: ignore[arg-type]
    results4 = df4.collect()

    print("Users over 25 (sorted by age):")
    for user in results4:
        print(f"  - {user.name} ({user.age})")  # type: ignore[attr-defined]
    print()

    # Method 5: Streaming with SQLModel
    # ----------------------------------
    print("=== Method 5: Streaming with SQLModel ===")
    df5 = db.table(User).select()  # type: ignore[arg-type]
    stream_results = df5.collect(stream=True)

    print("Streaming results:")
    for chunk in stream_results:
        for user in chunk:
            print(f"  - {user.name}")  # type: ignore[attr-defined]
        break  # Just show first chunk
    print()

    # Method 6: Using with existing SQLAlchemy infrastructure
    # --------------------------------------------------------
    print("=== Method 6: With SQLAlchemy Session ===")
    from moltres.integrations.sqlalchemy import execute_with_session_model
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(bind=engine)

    # Create table using SQLAlchemy
    User.__table__.create(engine, checkfirst=True)  # type: ignore[attr-defined]

    # Insert data
    with SessionLocal() as session:
        session.add(User(id=1, name="Alice", email="alice@example.com", age=30))
        session.add(User(id=2, name="Bob", email="bob@example.com", age=25))
        session.commit()

    # Use Moltres with existing session
    db2 = connect(engine=engine)
    df6 = db2.table("users").select()
    with SessionLocal() as session:
        results6 = execute_with_session_model(df6, session, User)

    print(f"Results from SQLAlchemy session: {len(results6)}")
    for user in results6:
        print(f"  - {user.name}")  # type: ignore[attr-defined]
    print()

    db.close()
    print("✅ All SQLModel integration examples completed successfully!")

except ImportError:
    print("SQLModel is not installed. Install with: pip install sqlmodel")
    print("Skipping SQLModel integration examples.")
