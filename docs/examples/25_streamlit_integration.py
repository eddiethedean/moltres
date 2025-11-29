"""Streamlit Integration Examples with Moltres

This example demonstrates how to use Moltres with Streamlit to build
interactive data applications with DataFrame operations, query visualization,
caching, and session state management.

Key features:
- Display DataFrames with query information
- Interactive query builder
- Query result caching
- Session state management for database connections
- Query visualization (SQL, plan, metrics)
- Error handling

To run this example:
1. Install Streamlit: pip install streamlit
2. Install Moltres with Streamlit: pip install moltres[streamlit]
3. Run: streamlit run docs/examples/25_streamlit_integration.py

Note: This is a demonstration file. In a real Streamlit app, you would
organize this code into separate functions and use Streamlit's app structure.
"""

# Example 1: Basic DataFrame Display
# ===================================

try:
    import time

    import streamlit as st
    from moltres import connect, col
    from moltres.table.schema import column
    from moltres.integrations.streamlit import (
        moltres_dataframe,
        query_builder,
        cached_query,
        visualize_query,
        display_moltres_error,
        clear_moltres_cache,
    )

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    print("Streamlit is required for this example. Install with: pip install streamlit")

if STREAMLIT_AVAILABLE:
    # Page configuration
    st.set_page_config(
        page_title="Moltres Streamlit Integration",
        page_icon="🔥",
        layout="wide",
    )

    st.title("🔥 Moltres Streamlit Integration")
    st.markdown(
        """
        This app demonstrates Moltres integration with Streamlit, including:
        - DataFrame display with query information
        - Interactive query builder
        - Query result caching
        - Session state management
        - Query visualization
        - Error handling
        """
    )

    # Initialize database connection
    # ==============================
    st.header("1. Database Connection Management")

    # Option 1: Use get_db_from_session (automatic connection management)
    st.subheader("Option 1: Automatic Connection Management")
    st.code(
        """
        from moltres.integrations.streamlit import get_db_from_session

        # Automatically manages connection in session state
        db = get_db_from_session()
        """,
        language="python",
    )

    # For demo purposes, we'll use an in-memory database
    if "demo_db" not in st.session_state:
        demo_db = connect("sqlite:///:memory:")

        # Create tables and insert sample data
        demo_db.create_table(
            "users",
            [
                column("id", "INTEGER"),
                column("name", "TEXT"),
                column("email", "TEXT"),
                column("age", "INTEGER"),
                column("city", "TEXT"),
            ],
        ).collect()

        demo_db.create_table(
            "orders",
            [
                column("id", "INTEGER"),
                column("user_id", "INTEGER"),
                column("product", "TEXT"),
                column("amount", "REAL"),
                column("order_date", "TEXT"),
            ],
        ).collect()

        # Insert sample data
        from moltres.io.records import Records

        Records(
            _data=[
                {
                    "id": 1,
                    "name": "Alice",
                    "email": "alice@example.com",
                    "age": 30,
                    "city": "New York",
                },
                {
                    "id": 2,
                    "name": "Bob",
                    "email": "bob@example.com",
                    "age": 25,
                    "city": "San Francisco",
                },
                {
                    "id": 3,
                    "name": "Charlie",
                    "email": "charlie@example.com",
                    "age": 35,
                    "city": "Chicago",
                },
                {
                    "id": 4,
                    "name": "Diana",
                    "email": "diana@example.com",
                    "age": 28,
                    "city": "New York",
                },
            ],
            _database=demo_db,
        ).insert_into("users")

        Records(
            _data=[
                {
                    "id": 1,
                    "user_id": 1,
                    "product": "Laptop",
                    "amount": 999.99,
                    "order_date": "2024-01-15",
                },
                {
                    "id": 2,
                    "user_id": 1,
                    "product": "Mouse",
                    "amount": 29.99,
                    "order_date": "2024-01-16",
                },
                {
                    "id": 3,
                    "user_id": 2,
                    "product": "Keyboard",
                    "amount": 79.99,
                    "order_date": "2024-01-17",
                },
                {
                    "id": 4,
                    "user_id": 3,
                    "product": "Monitor",
                    "amount": 299.99,
                    "order_date": "2024-01-18",
                },
                {
                    "id": 5,
                    "user_id": 4,
                    "product": "Headphones",
                    "amount": 149.99,
                    "order_date": "2024-01-19",
                },
            ],
            _database=demo_db,
        ).insert_into("orders")

        st.session_state["demo_db"] = demo_db

    db = st.session_state["demo_db"]

    # Option 2: Manual connection management
    st.subheader("Option 2: Manual Connection Management")
    st.code(
        """
        from moltres.integrations.streamlit import init_db_connection, close_db_connection

        # Initialize connection
        db = init_db_connection("sqlite:///example.db")

        # Close connection when done
        close_db_connection()
        """,
        language="python",
    )

    # DataFrame Display
    # ==================
    st.header("2. DataFrame Display")

    st.subheader("Basic Display")
    st.code(
        """
        from moltres.integrations.streamlit import moltres_dataframe

        df = db.table("users").select()
        moltres_dataframe(df)
        """,
        language="python",
    )

    df_users = db.table("users").select()
    moltres_dataframe(df_users, show_query_info=True)

    st.subheader("Display with Custom Options")
    st.code(
        """
        moltres_dataframe(df, show_query_info=True, height=400, use_container_width=True)
        """,
        language="python",
    )

    df_filtered = db.table("users").select().where(col("age") > 25)
    moltres_dataframe(df_filtered, show_query_info=True, height=300)

    # Query Builder
    # =============
    st.header("3. Interactive Query Builder")

    st.code(
        """
        from moltres.integrations.streamlit import query_builder

        df = query_builder(db)
        if df:
            results = df.collect()
            st.dataframe(results)
        """,
        language="python",
    )

    st.info("Query builder widget - Select a table to build a query interactively")
    built_df = query_builder(db)
    if built_df:
        st.success("Query built successfully!")
        moltres_dataframe(built_df)

    # Query Visualization
    # ====================
    st.header("4. Query Visualization")

    st.code(
        """
        from moltres.integrations.streamlit import visualize_query

        df = db.table("users").select().where(col("age") > 25)
        visualize_query(df, show_sql=True, show_plan=True, show_metrics=True)
        """,
        language="python",
    )

    complex_df = (
        db.table("users")
        .select("name", "email", "age")
        .where(col("age") > 25)
        .order_by(col("age").desc())
        .limit(10)
    )

    visualize_query(complex_df, show_sql=True, show_plan=True, show_metrics=True)

    # Caching
    # =======
    st.header("5. Query Result Caching")

    st.code(
        """
        from moltres.integrations.streamlit import cached_query

        @cached_query(ttl=3600)  # Cache for 1 hour
        def get_user_stats():
            return db.table("users").select().agg(...).collect()
        """,
        language="python",
    )

    @cached_query(ttl=60)  # Cache for 60 seconds
    def get_expensive_query():
        """Simulate an expensive query."""
        import time

        time.sleep(0.1)  # Simulate processing time
        return db.table("users").select().where(col("age") > 25).collect()

    st.subheader("Cached Query Example")
    st.write("This query is cached for 60 seconds. Run it multiple times to see caching in action.")

    if st.button("Run Cached Query"):
        start_time = time.time()
        results = get_expensive_query()
        execution_time = time.time() - start_time

        st.metric("Execution Time", f"{execution_time:.3f}s")
        st.metric("Rows Returned", len(results))
        st.dataframe(results)

    st.subheader("Cache Management")
    st.code(
        """
        from moltres.integrations.streamlit import clear_moltres_cache

        if st.button("Clear Cache"):
            clear_moltres_cache()
            st.success("Cache cleared!")
        """,
        language="python",
    )

    if st.button("Clear All Caches"):
        clear_moltres_cache()
        st.success("Cache cleared!")

    # Error Handling
    # ===============
    st.header("6. Error Handling")

    st.code(
        """
        from moltres.integrations.streamlit import display_moltres_error

        try:
            df = db.table("nonexistent").select()
            df.collect()
        except Exception as e:
            display_moltres_error(e)
        """,
        language="python",
    )

    st.subheader("Error Handling Example")
    if st.button("Trigger Error"):
        try:
            df_error = db.table("nonexistent_table").select()
            df_error.collect()
        except Exception as e:
            display_moltres_error(e)

    # Advanced Example: Data Analysis Dashboard
    # ==========================================
    st.header("7. Advanced Example: Data Analysis Dashboard")

    st.subheader("User Statistics")
    col1, col2, col3 = st.columns(3)

    with col1:
        total_users = len(db.table("users").select().collect())
        st.metric("Total Users", total_users)

    with col2:
        avg_age = (
            db.table("users").select().agg({"age": "avg"}).collect()[0]
            if db.table("users").select().collect()
            else 0
        )
        st.metric("Average Age", f"{avg_age:.1f}" if isinstance(avg_age, (int, float)) else "N/A")

    with col3:
        cities = db.table("users").select("city").distinct().collect()
        st.metric("Unique Cities", len(cities))

    st.subheader("Orders by User")
    orders_by_user = (
        db.table("orders").select("user_id", "product", "amount").order_by(col("user_id"))
    )
    moltres_dataframe(orders_by_user, show_query_info=False)

    st.subheader("Users with Orders (Join Example)")
    users_with_orders = (
        db.table("users")
        .join(db.table("orders"), on="id", how="inner")
        .select("users.name", "users.email", "orders.product", "orders.amount")
    )
    visualize_query(users_with_orders, show_sql=True, show_plan=False, show_metrics=False)
    moltres_dataframe(users_with_orders, show_query_info=False)

    # Footer
    # ======
    st.markdown("---")
    st.markdown(
        """
        **Moltres Streamlit Integration** - For more information, see the
        [Moltres documentation](https://github.com/eddiethedean/moltres).
        """
    )

else:
    print("This example requires Streamlit. Install with: pip install streamlit")
