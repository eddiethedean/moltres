"""FastAPI Integration Utilities Example

This example demonstrates how to use Moltres's FastAPI integration utilities
to make your FastAPI application more robust and user-friendly.

Key features demonstrated:
- Automatic exception handling with register_exception_handlers()
- Dependency injection helpers for database connections
- Error handling decorator for route handlers
"""

try:
    from fastapi import FastAPI, HTTPException, Depends, Query
    from sqlmodel import SQLModel, Field, create_engine, Session
    from sqlalchemy.orm import sessionmaker
    from typing import List, Optional

    from moltres import connect, col
    from moltres.expressions import functions as F
    from moltres.integrations.fastapi import (
        register_exception_handlers,
        create_db_dependency,
        handle_moltres_errors,
    )

    # SQLModel Definitions
    class UserBase(SQLModel):
        """Base model for User (shared fields)."""
        name: str
        email: str
        age: int

    class User(UserBase, table=True):
        """SQLModel for users table."""
        __tablename__ = "users"
        id: Optional[int] = Field(default=None, primary_key=True)

    class UserRead(UserBase):
        """Response model for reading a user."""
        id: int

    # Database Setup
    sqlite_url = "sqlite:///./example_utilities.db"
    engine = create_engine(sqlite_url, echo=True)

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

    def get_session():
        """Dependency to get SQLModel session."""
        with SessionLocal() as session:
            yield session

    # Create Moltres database dependency using the helper
    get_db = create_db_dependency(get_session)

    # Initialize FastAPI app
    app = FastAPI(
        title="Moltres FastAPI Integration Utilities Example", version="1.0.0"
    )

    # Register exception handlers - this automatically converts Moltres errors
    # to appropriate HTTP responses (400, 404, 500, 503, 504, etc.)
    register_exception_handlers(app)

    def create_db_and_tables():
        """Create database tables from SQLModel definitions."""
        SQLModel.metadata.create_all(engine)

    @app.on_event("startup")
    def on_startup():
        """Create tables on startup."""
        create_db_and_tables()

    # Example 1: Using dependency injection helper
    # -------------------------------------------
    @app.get("/users", response_model=List[UserRead])
    def get_users(
        db=Depends(get_db),  # Use the dependency helper
        min_age: Optional[int] = Query(None, description="Minimum age filter"),
    ):
        """Get users using Moltres with dependency injection helper.
        
        The get_db dependency automatically creates a Moltres Database instance
        from the FastAPI session dependency.
        """
        df = db.table(User).select()

        if min_age is not None:
            df = df.where(col("age") >= min_age)

        # Results are automatically User instances (SQLModel) when using .exec()
        return df.collect()

    # Example 2: Using error handling decorator
    # -------------------------------------------
    @app.get("/users/{user_id}", response_model=UserRead)
    @handle_moltres_errors  # Automatically converts Moltres errors to HTTPException
    def get_user(user_id: int, db=Depends(get_db)):
        """Get a user by ID with automatic error handling.
        
        The @handle_moltres_errors decorator catches Moltres exceptions and
        converts them to appropriate HTTPException responses.
        """
        df = db.table(User).select().where(col("id") == user_id)
        results = df.collect()

        if not results:
            raise HTTPException(status_code=404, detail="User not found")

        return results[0]

    # Example 3: Exception handlers handle errors automatically
    # ----------------------------------------------------------
    @app.get("/users/stats/summary")
    def get_user_stats(db=Depends(get_db)):
        """Get user statistics.
        
        If any Moltres error occurs (e.g., table not found, column not found),
        the registered exception handlers will automatically convert it to an
        appropriate HTTP response with helpful error messages and suggestions.
        """
        df = db.table(User).select()

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

    # Example 4: Error handling works even without decorator
    # -------------------------------------------------------
    @app.get("/users/invalid-column")
    def get_users_invalid_column(db=Depends(get_db)):
        """This will trigger a ColumnNotFoundError.
        
        The exception handler registered with register_exception_handlers()
        will automatically catch this and return a 400 Bad Request with
        helpful suggestions about available columns.
        """
        # This will raise ColumnNotFoundError
        df = db.table(User).select(col("nonexistent_column"))
        return df.collect()

    # Run with: uvicorn examples.23_fastapi_integration_utilities:app --reload

except ImportError as e:
    print(f"Required dependencies not installed: {e}")
    print("Install with: pip install fastapi uvicorn sqlmodel")

