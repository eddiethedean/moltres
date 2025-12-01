"""FastAPI Integration Examples with SQLModel and Moltres

This example demonstrates how to use Moltres with FastAPI following
the standard FastAPI + SQLModel pattern, then leveraging Moltres's
SQLModel integration features for advanced DataFrame operations.

Key features:
- Standard FastAPI + SQLModel setup (engine, session dependency)
- SQLModel models for type safety
- Moltres DataFrame operations with SQLModel integration
- Automatic SQLModel .exec() usage - returns SQLModel instances directly!
- Complex queries with Moltres while maintaining SQLModel types
- Sync and async examples

IMPORTANT: When you pass a SQLModel session to connect() or async_connect(),
Moltres automatically detects it and uses SQLModel's .exec() method instead of
.execute(). This means collect() returns SQLModel instances directly - no
conversion needed! The .exec() method is used automatically when:
1. You use connect(session=sqlmodel_session) or async_connect(session=sqlmodel_async_session)
2. Your DataFrame has a model attached (from db.table(Model) or .with_model(Model))
3. You call collect() - results are SQLModel instances, not dicts!
"""

# Example 1: Standard FastAPI + SQLModel with Moltres Integration (Async)
# =======================================================================

try:
    from fastapi import FastAPI, HTTPException, Depends, Query
    from sqlmodel import SQLModel, Field
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from typing import Any, Generator, List, Optional

    from moltres import async_connect, col
    from moltres.expressions import functions as F
    from moltres.table.async_table import AsyncDatabase
    from moltres.integrations.fastapi import (
        register_exception_handlers,
        create_async_db_dependency,
        handle_moltres_errors,
    )

    # SQLModel Definitions
    # --------------------
    class UserBase(SQLModel):
        """Base model for User (shared fields)."""

        name: str
        email: str
        age: int

    class User(UserBase, table=True):
        """SQLModel for users table."""

        __tablename__ = "users"
        id: Optional[int] = Field(default=None, primary_key=True)

    class UserCreate(UserBase):
        """Request model for creating a user."""

        pass

    class UserRead(UserBase):
        """Response model for reading a user."""

        id: int

    # Database Setup (Standard FastAPI + SQLModel Pattern with Async)
    # ----------------------------------------------------------------
    sqlite_url = "sqlite+aiosqlite:///./example.db"
    engine = create_async_engine(sqlite_url, echo=True)

    async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def create_db_and_tables():
        """Create database tables from SQLModel definitions."""
        async with engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def get_session():
        """Dependency to get async SQLModel session (standard FastAPI pattern)."""
        async with async_session_maker() as session:
            yield session

    # Initialize FastAPI app
    app = FastAPI(title="Moltres + SQLModel FastAPI Example (Async)", version="1.0.0")

    # Register Moltres exception handlers for better error responses
    # This automatically converts Moltres errors to appropriate HTTP responses
    register_exception_handlers(app)

    # Create Moltres database dependency using the integration helper
    # This makes it easy to use Moltres with FastAPI's dependency injection
    get_db = create_async_db_dependency(get_session)

    @app.on_event("startup")
    async def on_startup():
        """Create tables on startup."""
        await create_db_and_tables()

    # Standard FastAPI + SQLModel Endpoints (Async)
    # ---------------------------------------------
    @app.post("/users", response_model=UserRead, status_code=201)
    async def create_user(user: UserCreate, session: AsyncSession = Depends(get_session)):
        """Create a user using standard async SQLModel session."""
        db_user = User(**user.model_dump())
        session.add(db_user)
        await session.commit()
        await session.refresh(db_user)
        return db_user

    @app.get("/users/{user_id}", response_model=UserRead)
    async def get_user(user_id: int, session: AsyncSession = Depends(get_session)):
        """Get a user using standard async SQLModel session."""
        user = await session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    # Moltres Integration Endpoints (Async)
    # -------------------------------------
    @app.get("/users", response_model=List[UserRead])
    @handle_moltres_errors  # Automatically converts Moltres errors to HTTPException
    async def get_users(
        db: AsyncDatabase = Depends(get_db),  # Use the dependency helper
        min_age: Optional[int] = Query(None, description="Minimum age filter"),
        max_age: Optional[int] = Query(None, description="Maximum age filter"),
        search: Optional[str] = Query(None, description="Search in name or email"),
    ):
        """Get users using async Moltres DataFrame with SQLModel integration.

        This endpoint demonstrates:
        - Automatic SQLModel .exec() usage (returns User instances directly)
        - FastAPI dependency injection with create_async_db_dependency()
        - Automatic error handling with @handle_moltres_errors decorator
        - Exception handlers registered globally for all Moltres errors
        """
        # Use SQLModel directly with async Moltres
        table_handle = await db.table(User)
        df = table_handle.select()

        # Apply filters using Moltres expressions
        if min_age is not None:
            df = df.where(col("age") >= min_age)
        if max_age is not None:
            df = df.where(col("age") <= max_age)
        if search:
            df = df.where((col("name").like(f"%{search}%")) | (col("email").like(f"%{search}%")))

        # Collect returns SQLModel instances directly!
        # Moltres automatically uses SQLModel's .exec() when a SQLModel session
        # is detected, so results are User instances (not dicts) - no conversion needed!
        results = await df.collect()
        return results  # Already User instances, not dicts!

    @app.get("/users/stats/summary")
    @handle_moltres_errors  # Automatic error handling
    async def get_user_stats(db: AsyncDatabase = Depends(get_db)):
        """Get user statistics using async Moltres aggregations.

        Note: Aggregations return dicts (not SQLModel instances) since
        they don't map to a single model row.

        Any Moltres errors (e.g., table not found, column not found) will be
        automatically converted to appropriate HTTP responses by the exception handlers.
        """
        table_handle = await db.table(User)
        df = table_handle.select()

        # Calculate statistics with Moltres
        # For aggregations, we use select() with aggregation functions
        stats_df = df.select(
            F.count(col("id")).alias("total_users"),
            F.avg(col("age")).alias("avg_age"),
            F.min(col("age")).alias("min_age"),
            F.max(col("age")).alias("max_age"),
        )

        result = (await stats_df.collect())[0]
        return {
            "total_users": result["total_users"],
            "average_age": round(result["avg_age"], 2),
            "min_age": result["min_age"],
            "max_age": result["max_age"],
        }

    @app.put("/users/{user_id}", response_model=UserRead)
    async def update_user(
        user_id: int,
        user_update: UserCreate,
        session: AsyncSession = Depends(get_session),
    ):
        """Update a user - hybrid approach: SQLModel for update, Moltres for query."""
        # Use standard async SQLModel for update
        user = await session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Update fields
        user_data = user_update.model_dump(exclude_unset=True)
        for field, value in user_data.items():
            setattr(user, field, value)

        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user

    @app.delete("/users/{user_id}", status_code=204)
    async def delete_user(user_id: int, session: AsyncSession = Depends(get_session)):
        """Delete a user using standard async SQLModel."""
        user = await session.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        await session.delete(user)
        await session.commit()
        return None

    # Advanced Moltres Features with SQLModel (Async)
    # -----------------------------------------------
    @app.get("/users/advanced/filtered", response_model=List[UserRead])
    @handle_moltres_errors
    async def get_filtered_users_advanced(
        db: AsyncDatabase = Depends(get_db),
        min_age: Optional[int] = Query(None),
        order_by: str = Query("age", description="Column to sort by"),
    ):
        """Advanced filtering and sorting with async Moltres, returning SQLModel instances.

        Demonstrates:
        - Dynamic column ordering
        - Automatic error handling (invalid column names will return 400 with suggestions)
        """
        table_handle = await db.table(User)
        df = table_handle.select()

        if min_age is not None:
            df = df.where(col("age") >= min_age)

        # Order by any column
        df = df.order_by(col(order_by).asc())

        # Results are User instances (SQLModel) - automatically returned via .exec()
        return await df.collect()

    @app.get("/users/advanced/grouped")
    @handle_moltres_errors
    async def get_users_by_age_group(db: AsyncDatabase = Depends(get_db)):
        """Group users by age ranges using async Moltres.

        Demonstrates complex aggregations with CASE expressions.
        """
        table_handle = await db.table(User)
        df = table_handle.select()

        # Create age groups using CASE expression
        from moltres.expressions.functions import when

        age_group = (
            when(col("age") < 25, "Young")
            .when(col("age") < 40, "Middle")
            .otherwise("Senior")
            .alias("age_group")
        )

        grouped_df = (
            df.select(col("*"), age_group)
            .group_by("age_group")
            .agg(F.count(col("id")).alias("count"))
        )

        return await grouped_df.collect()

    # Run with: uvicorn examples.22_fastapi_integration:app --reload

except ImportError as e:
    print(f"Required dependencies not installed: {e}")
    print("Install with: pip install fastapi uvicorn sqlmodel")


# Example 2: Async FastAPI + SQLModel with Moltres
# ================================================

try:
    from fastapi import FastAPI as AsyncFastAPI
    from fastapi import HTTPException, Depends, Query
    from sqlmodel import SQLModel as AsyncSQLModel, Field as AsyncField
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
    from sqlalchemy.orm import sessionmaker
    from typing import Any, Generator, List, Optional

    from moltres import async_connect, col
    from moltres.table.async_table import AsyncDatabase

    # SQLModel Definitions
    class AsyncUserBase(AsyncSQLModel):
        name: str
        email: str
        age: int

    class AsyncUser(AsyncUserBase, table=True):
        __tablename__ = "async_users"
        id: Optional[int] = AsyncField(default=None, primary_key=True)

    class AsyncUserRead(AsyncUserBase):
        id: int

    # Async Database Setup
    async_sqlite_url = "sqlite+aiosqlite:///./example_async.db"
    async_engine = create_async_engine(async_sqlite_url, echo=True)

    async_session_maker = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

    async def get_async_session():
        """Dependency to get async SQLModel session."""
        async with async_session_maker() as session:
            yield session

    async def create_async_db_and_tables():
        """Create database tables."""
        async with async_engine.begin() as conn:
            await conn.run_sync(AsyncSQLModel.metadata.create_all)

    # Initialize async FastAPI app
    async_app = AsyncFastAPI(title="Moltres + SQLModel Async FastAPI Example", version="1.0.0")

    @async_app.on_event("startup")
    async def async_on_startup():
        """Create tables on startup."""
        await create_async_db_and_tables()

    # Async Endpoints with Moltres
    @async_app.get("/users", response_model=List[AsyncUserRead])
    async def get_users_async(
        session: AsyncSession = Depends(get_async_session),
        min_age: Optional[int] = Query(None),
    ):
        """Get users using async Moltres with SQLModel."""
        # Create async Moltres database connection from session
        db = async_connect(session=session)

        # Use SQLModel with async Moltres
        df = (await db.table(AsyncUser)).select()

        if min_age is not None:
            df = df.where(col("age") >= min_age)

        # Collect returns SQLModel instances directly via automatic .exec() usage
        # When using async_connect(session=session) with a SQLModel session,
        # Moltres automatically uses session.exec() which returns model instances
        results = await df.collect()
        return results  # Already AsyncUser instances!

    @async_app.post("/users", response_model=AsyncUserRead, status_code=201)
    async def create_user_async(
        user_data: AsyncUserBase, session: AsyncSession = Depends(get_async_session)
    ):
        """Create user using standard async SQLModel session."""
        db_user = AsyncUser(**user_data.model_dump())
        session.add(db_user)
        await session.commit()
        await session.refresh(db_user)
        return db_user

    # Run with: uvicorn examples.22_fastapi_integration:async_app --reload

except ImportError as e:
    print(f"Async dependencies not installed: {e}")
    print("Install with: pip install fastapi uvicorn sqlmodel aiosqlite")


# Example 3: Sync FastAPI + SQLModel with Moltres Integration Utilities
# ======================================================================

try:
    from fastapi import FastAPI, HTTPException, Depends, Query
    from sqlmodel import SQLModel, Field, create_engine
    from sqlalchemy.orm import sessionmaker
    from typing import Any, Generator, List, Optional

    from moltres import connect, col
    from moltres.expressions import functions as F
    from moltres.integrations.fastapi import (
        register_exception_handlers,
        create_db_dependency,
        handle_moltres_errors,
    )

    # SQLModel Definitions
    class SyncUserBase(SQLModel):
        """Base model for User (shared fields)."""

        name: str
        email: str
        age: int

    class SyncUser(SyncUserBase, table=True):
        """SQLModel for users table."""

        __tablename__ = "sync_users"
        id: Optional[int] = Field(default=None, primary_key=True)

    class SyncUserRead(SyncUserBase):
        """Response model for reading a user."""

        id: int

    # Database Setup (Sync)
    sqlite_url = "sqlite:///./example_sync.db"
    engine = create_engine(sqlite_url, echo=True)

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    def get_sync_session():
        """Dependency to get SQLModel session."""
        with SessionLocal() as session:
            yield session

    # Create Moltres database dependency using the sync helper
    get_sync_db = create_db_dependency(get_sync_session)

    # Initialize FastAPI app
    sync_app = FastAPI(
        title="Moltres FastAPI Integration Utilities Example (Sync)", version="1.0.0"
    )

    # Register exception handlers - automatically converts Moltres errors to HTTP responses
    register_exception_handlers(sync_app)

    def create_sync_db_and_tables():
        """Create database tables from SQLModel definitions."""
        SQLModel.metadata.create_all(engine)

    @sync_app.on_event("startup")
    def sync_on_startup():
        """Create tables on startup."""
        create_sync_db_and_tables()

    # Example: Using dependency injection helper (sync)
    @sync_app.get("/users", response_model=List[SyncUserRead])
    def get_sync_users(
        db=Depends(get_sync_db),  # Use the dependency helper
        min_age: Optional[int] = Query(None, description="Minimum age filter"),
    ):
        """Get users using Moltres with dependency injection helper.

        The get_sync_db dependency automatically creates a Moltres Database instance
        from the FastAPI session dependency.
        """
        df = db.table(SyncUser).select()

        if min_age is not None:
            df = df.where(col("age") >= min_age)

        # Results are automatically SyncUser instances (SQLModel) when using .exec()
        return df.collect()

    # Example: Using error handling decorator
    @sync_app.get("/users/{user_id}", response_model=SyncUserRead)
    @handle_moltres_errors  # Automatically converts Moltres errors to HTTPException
    def get_sync_user(user_id: int, db=Depends(get_sync_db)):
        """Get a user by ID with automatic error handling.

        The @handle_moltres_errors decorator catches Moltres exceptions and
        converts them to appropriate HTTPException responses.
        """
        df = db.table(SyncUser).select().where(col("id") == user_id)
        results = df.collect()

        if not results:
            raise HTTPException(status_code=404, detail="User not found")

        return results[0]

    # Example: Exception handlers handle errors automatically
    @sync_app.get("/users/stats/summary")
    def get_sync_user_stats(db=Depends(get_sync_db)):
        """Get user statistics.

        If any Moltres error occurs (e.g., table not found, column not found),
        the registered exception handlers will automatically convert it to an
        appropriate HTTP response with helpful error messages and suggestions.
        """
        df = db.table(SyncUser).select()

        stats_df = df.select(
            F.count(col("id")).alias("total_users"),
            F.avg(col("age")).alias("avg_age"),
            F.min(col("age")).alias("min_age"),
            F.max(col("age")).alias("max_age"),
        )

        result = stats_df.collect()[0]
        return {
            "total_users": result["total_users"],
            "average_age": round(result["avg_age"], 2),
            "min_age": result["min_age"],
            "max_age": result["max_age"],
        }

    # Run with: uvicorn examples.22_fastapi_integration:sync_app --reload

except ImportError as e:
    print(f"Required dependencies not installed: {e}")
    print("Install with: pip install fastapi uvicorn sqlmodel")


# Example 3: Using with_model() for Dynamic Model Attachment
# ===========================================================

try:
    from fastapi import FastAPI as ModelFastAPI
    from fastapi import Depends, Query
    from sqlmodel import SQLModel as ModelSQLModel, Field as ModelField, Session
    from sqlalchemy import create_engine
    from typing import Any, Generator, List, Optional

    from moltres import connect, col
    from moltres.expressions import functions as F

    # SQLModel Definitions
    class ProductBase(ModelSQLModel):
        name: str
        price: float
        category: str

    class Product(ProductBase, table=True):
        __tablename__ = "products"
        id: Optional[int] = ModelField(default=None, primary_key=True)

    class ProductRead(ProductBase):
        id: int

    # Database Setup
    model_engine = create_engine("sqlite:///./example_model.db", echo=True)

    def create_model_db_and_tables():
        ModelSQLModel.metadata.create_all(model_engine)

    def get_model_session():
        with Session(model_engine) as session:
            yield session

    model_app = ModelFastAPI(title="Moltres with_model() Example", version="1.0.0")

    @model_app.on_event("startup")
    def model_on_startup():
        create_model_db_and_tables()

    @model_app.get("/products", response_model=List[ProductRead])
    def get_products(
        session: Session = Depends(get_model_session),
        min_price: Optional[float] = Query(None),
    ):
        """Get products using with_model() to attach SQLModel dynamically."""
        db = connect(session=session)

        # Start with table name (no model attached)
        df = db.table("products").select()

        # Apply filters
        if min_price is not None:
            df = df.where(col("price") >= min_price)

        # Attach SQLModel using with_model()
        df_with_model = df.with_model(Product)

        # Now collect() returns Product instances directly!
        # Since we're using connect(session=session) with a SQLModel session,
        # Moltres automatically uses session.exec() which returns Product instances
        results = df_with_model.collect()
        return results  # Already Product instances, not dicts!

    @model_app.get("/products/stats")
    def get_product_stats(session: Session = Depends(get_model_session)):
        """Get product statistics - no model needed for aggregations."""
        db = connect(session=session)
        df = db.table("products").select()

        # Aggregations return dicts (no model needed)
        stats = (
            df.group_by("category")
            .agg(
                F.count(col("id")).alias("count"),
                F.avg(col("price")).alias("avg_price"),
                F.sum(col("price")).alias("total_value"),
            )
            .collect()
        )

        return stats

    # Run with: uvicorn examples.22_fastapi_integration:model_app --reload

except ImportError as e:
    print(f"Model dependencies not installed: {e}")
    print("Install with: pip install fastapi uvicorn sqlmodel")


# Example 4: Complex Queries with Joins and SQLModel
# ===================================================

try:
    from fastapi import FastAPI as JoinFastAPI
    from fastapi import Depends
    from sqlmodel import SQLModel as JoinSQLModel, Field as JoinField, Session
    from sqlalchemy import create_engine
    from typing import Any, Generator, List, Optional

    from moltres import connect, col

    # SQLModel Definitions with Relationships
    class OrderBase(JoinSQLModel):
        user_id: int
        product_name: str
        amount: float

    class Order(OrderBase, table=True):
        __tablename__ = "orders"
        id: Optional[int] = JoinField(default=None, primary_key=True)

    class UserBase(JoinSQLModel):
        name: str
        email: str

    class User(UserBase, table=True):
        __tablename__ = "join_users"
        id: Optional[int] = JoinField(default=None, primary_key=True)

    class UserWithOrders(UserBase):
        id: int
        total_orders: float

    # Database Setup
    join_engine = create_engine("sqlite:///./example_join.db", echo=True)

    def create_join_db_and_tables() -> None:
        JoinSQLModel.metadata.create_all(join_engine)

    def get_join_session() -> Generator[Session, None, None]:
        with Session(join_engine) as session:
            yield session

    join_app = JoinFastAPI(title="Moltres Joins with SQLModel Example", version="1.0.0")

    @join_app.on_event("startup")
    def join_on_startup() -> None:
        create_join_db_and_tables()

    @join_app.get("/users/{user_id}/orders")
    def get_user_orders(user_id: int, session: Session = Depends(get_join_session)) -> list[dict[str, Any]]:
        """Get user orders using Moltres join with SQLModel."""
        db = connect(session=session)

        # Join users and orders using Moltres
        users_df = db.table(User).select()  # type: ignore[arg-type]
        orders_df = db.table(Order).select()

        result_df = (
            users_df.join(orders_df, on=[col("users.id") == col("orders.user_id")])
            .where(col("users.id") == user_id)
            .select(
                col("users.id"),
                col("users.name"),
                col("orders.product_name"),
                col("orders.amount"),
            )
        )

        # Returns list of dicts (mixed columns from join)
        return result_df.collect()

    @join_app.get("/users/{user_id}/summary")
    def get_user_summary(user_id: int, session: Session = Depends(get_join_session)) -> Any:
        """Get user summary with order totals using Moltres aggregations."""
        db = connect(session=session)

        users_df = db.table(User).select()  # type: ignore[arg-type]  # type: ignore[arg-type]
        orders_df = db.table(Order).select()  # type: ignore[arg-type]

        # Join and aggregate
        summary_df = (
            users_df.join(orders_df, on=[col("users.id") == col("orders.user_id")])
            .where(col("users.id") == user_id)
            .group_by("users.id", "users.name")
            .agg(F.sum(col("orders.amount")).alias("total_orders"))
        )

        results = summary_df.collect()
        if not results:
            return {"user_id": user_id, "total_orders": 0.0}

        return results[0]

    # Run with: uvicorn examples.22_fastapi_integration:join_app --reload

except ImportError as e:
    print(f"Join dependencies not installed: {e}")
    print("Install with: pip install fastapi uvicorn sqlmodel")
    print("Or: pip install moltres[sqlmodel]")


if __name__ == "__main__":
    print("=" * 70)
    print("FastAPI Integration Examples")
    print("=" * 70)
    print("\nThis file contains FastAPI integration examples.")
    print("To run these examples, you need to install dependencies:")
    print("  pip install fastapi uvicorn sqlmodel")
    print("  Or: pip install moltres[sqlmodel]")
    print("\nThen run with uvicorn:")
    print("  uvicorn docs.examples.22_fastapi_integration:app --reload")
    print("  uvicorn docs.examples.22_fastapi_integration:sync_app --reload")
    print("  uvicorn docs.examples.22_fastapi_integration:model_app --reload")
    print("  uvicorn docs.examples.22_fastapi_integration:join_app --reload")
    print("\nEach example defines a separate FastAPI app instance.")
