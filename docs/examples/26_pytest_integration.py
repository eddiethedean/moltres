"""Example: Using Moltres with Pytest for Testing

This example demonstrates how to use Moltres's Pytest integration for
comprehensive testing of database operations and DataFrame queries.

Features shown:
- Database fixtures for isolated test databases
- Test data fixtures and helpers
- Custom assertions for DataFrame comparisons
- Query logging for debugging
- Database-specific test markers
"""

# Example 1: Basic Database Fixture Usage
# =======================================


def example_basic_fixture():
    """Example of using moltres_db fixture."""
    # In a test file:
    """
    def test_user_operations(moltres_db):
        # Each test gets an isolated database
        db = moltres_db
        
        # Create a table
        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("email", "TEXT"),
            ],
        ).collect()
        
        # Insert test data
        Records(
            _data=[
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"},
            ],
            _database=db,
        ).insert_into("users")
        
        # Query and verify
        df = db.table("users").select()
        results = df.collect()
        assert len(results) == 2
        assert results[0]["name"] == "Alice"
    """
    pass


# Example 2: Async Database Fixture
# ==================================


def example_async_fixture():
    """Example of using moltres_async_db fixture."""
    # In a test file:
    """
    import pytest
    
    @pytest.mark.asyncio
    async def test_async_operations(moltres_async_db):
        # Get async database
        db = await moltres_async_db
        
        # Create table asynchronously
        await db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
            ],
        )
        
        # Query asynchronously
        table = await db.table("users")
        df = table.select()
        results = await df.collect()
        
        assert len(results) == 0
    """
    pass


# Example 3: Test Data Fixtures
# ==============================


def example_test_data_fixture():
    """Example of using test_data fixture."""
    # Create a test_data/ directory with CSV/JSON files:
    # test_data/users.csv:
    #   id,name,email
    #   1,Alice,alice@example.com
    #   2,Bob,bob@example.com

    # In a test file:
    """
    def test_with_file_data(moltres_db, test_data):
        db = moltres_db
        
        # Load schema and data from files
        # test_data automatically loads CSV/JSON files
        users_data = test_data["users"]  # Loaded from users.csv
        
        # Create table from data
        db.create_table(
            "users",
            [
                column("id", "INTEGER", primary_key=True),
                column("name", "TEXT"),
                column("email", "TEXT"),
            ],
        ).collect()
        
        # Insert loaded data
        Records(_data=users_data, _database=db).insert_into("users")
        
        # Verify
        df = db.table("users").select()
        results = df.collect()
        assert len(results) == len(users_data)
    """
    pass


# Example 4: Custom Assertions
# =============================


def example_custom_assertions():
    """Example of using custom DataFrame assertions."""
    # In a test file:
    """
    def test_dataframe_comparison(moltres_db):
        db = moltres_db
        
        # Create and populate two tables
        db.create_table("source", [...]).collect()
        Records(_data=[...], _database=db).insert_into("source")
        
        db.create_table("target", [...]).collect()
        Records(_data=[...], _database=db).insert_into("target")
        
        # Compare DataFrames
        df1 = db.table("source").select()
        df2 = db.table("target").select()
        
        # Assert they're equal (schema and data)
        assert_dataframe_equal(df1, df2)
        
        # Or compare with options
        assert_dataframe_equal(df1, df2, ignore_order=True)
        
        # Compare schemas only
        assert_schema_equal(df1.schema, df2.schema)
        
        # Assert query results
        assert_query_results(df1, expected_count=10)
        assert_query_results(df1, min_count=5, max_count=20)
    """
    pass


# Example 5: Query Logging
# =========================


def example_query_logging():
    """Example of using query logging for debugging."""
    # In a test file:
    """
    def test_query_debugging(moltres_db, query_logger):
        db = moltres_db
        
        # Create table
        db.create_table("users", [...]).collect()
        
        # Execute queries (they'll be logged automatically)
        df = db.table("users").select()
        df.collect()
        
        # Check query count
        assert query_logger.count == 1
        assert "SELECT" in query_logger.queries[0]
        
        # Check performance
        assert query_logger.get_average_time() < 0.1  # Should be fast
        
        # Clear logs
        query_logger.clear()
    """
    pass


# Example 6: Database-Specific Tests
# ===================================


def example_database_markers():
    """Example of using database-specific test markers."""
    # In a test file:
    """
    import pytest
    
    @pytest.mark.moltres_db("postgresql")
    def test_postgresql_specific_feature(moltres_db):
        # This test only runs if PostgreSQL is available
        # Configure via environment variables:
        # TEST_POSTGRES_HOST, TEST_POSTGRES_PORT, etc.
        db = moltres_db
        # Test PostgreSQL-specific features
        pass
    
    @pytest.mark.moltres_db("mysql")
    def test_mysql_specific_feature(moltres_db):
        # This test only runs if MySQL is available
        pass
    
    @pytest.mark.moltres_performance
    def test_query_performance(moltres_db):
        # Marked as performance test
        # Can use pytest-benchmark or similar
        import time
        
        start = time.time()
        # Run query
        elapsed = time.time() - start
        assert elapsed < 1.0
    """
    pass


# Example 7: Creating Test DataFrames
# ====================================


def example_create_test_df():
    """Example of creating test DataFrames."""
    # In a test file:
    """
    def test_with_test_df(moltres_db):
        db = moltres_db
        
        # Create DataFrame from test data
        data = [
            {"id": 1, "name": "Alice", "age": 30},
            {"id": 2, "name": "Bob", "age": 25},
        ]
        
        df = create_test_df(data, database=db)
        
        # Use the DataFrame
        results = df.collect()
        assert len(results) == 2
        assert results[0]["name"] == "Alice"
    """
    pass


if __name__ == "__main__":
    print("Pytest Integration Examples")
    print("=" * 50)
    print("\nThis file contains examples of using Moltres with Pytest.")
    print("See the function docstrings for code examples.")
    print("\nRequired dependencies:")
    print("  pip install pytest")
    print("  (pytest is typically already installed for testing)")
    print("\nTo use these in your tests:")
    print("1. Import the fixtures: from moltres.integrations.pytest import moltres_db")
    print("2. Use fixtures in your test functions")
    print("3. Use custom assertions for DataFrame comparisons")
    print("\nFor more details, see: guides/15-pytest-integration.md")
