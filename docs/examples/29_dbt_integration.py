"""Example: Using Moltres with dbt (data build tool)

This example demonstrates how to use Moltres DataFrames in dbt Python models.

Required dependencies:
- moltres (required)
- dbt-core (optional): pip install dbt-core or pip install moltres[dbt]
"""

# Example 1: Basic dbt Python Model with Moltres
# ===============================================

try:
    from moltres.integrations.dbt import get_moltres_connection, moltres_ref  # noqa: F401

    # In a dbt Python model file (models/my_model.py):
    """
    def model(dbt, session):
        from moltres.integrations.dbt import get_moltres_connection, moltres_ref
        
        # Get database connection from dbt config
        db = get_moltres_connection(dbt.config)
        
        # Reference other dbt models as Moltres DataFrames
        users = moltres_ref(dbt, "users", db)
        orders = moltres_ref(dbt, "orders", db)
        
        # Use Moltres DataFrame API
        df = (
            users
            .join(orders, on="user_id")
            .group_by("user_id")
            .agg(F.sum(col("amount")).alias("total_amount"))
        )
        
        # Return as list of dicts for dbt
        return df.collect()
    """

    print("Example 1: Basic dbt model created successfully!")

except ImportError:
    print("dbt-core not installed. Install with: pip install dbt-core")
    print("Or: pip install moltres[dbt]")

# Example 2: Using dbt Sources
# =============================

try:
    from moltres.integrations.dbt import moltres_source  # noqa: F401

    # In a dbt Python model:
    """
    def model(dbt, session):
        from moltres.integrations.dbt import get_moltres_connection, moltres_source
        
        db = get_moltres_connection(dbt.config)
        
        # Reference dbt sources
        raw_users = moltres_source(dbt, "raw", "users", db)
        
        # Transform using Moltres
        df = raw_users.select().where(col("active") == True)
        return df.collect()
    """

    print("Example 2: Using dbt sources with Moltres!")

except ImportError:
    pass

# Example 3: Using dbt Variables
# ===============================

try:
    from moltres.integrations.dbt import moltres_var  # noqa: F401

    # In a dbt Python model:
    """
    def model(dbt, session):
        from moltres.integrations.dbt import get_moltres_connection, moltres_var
        
        # Get dbt variable
        min_age = moltres_var(dbt, "min_age", default=18)
        
        db = get_moltres_connection(dbt.config)
        df = db.table("users").select().where(col("age") > min_age)
        return df.collect()
    """

    print("Example 3: Using dbt variables with Moltres!")

except ImportError:
    pass

if __name__ == "__main__":
    print("dbt Integration Examples")
    print("=" * 50)
    print("\nThis file contains examples of using Moltres with dbt.")
    print("See the function docstrings for code examples.")
    print("\nRequired dependencies:")
    print("  pip install dbt-core")
    print("  Or: pip install moltres[dbt]")
    print("\nFor more details, see: guides/17-dbt-integration.md")
