"""Example dbt Python model using Moltres.

This is an example dbt Python model that demonstrates how to use
Moltres DataFrames in dbt.
"""


def model(dbt, session):
    """Example dbt Python model using Moltres."""
    from moltres.integrations.dbt import get_moltres_connection  # noqa: F401

    # Get database connection from dbt config
    db = get_moltres_connection(dbt.config)

    # Reference other dbt models (if they exist)
    # users = moltres_ref(dbt, "users", db)

    # Create a simple query
    # In a real scenario, you would reference actual tables/models
    df = db.table("example_table").select()

    # Return results as list of dicts for dbt
    return df.collect()
