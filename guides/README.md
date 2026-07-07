# Moltres User Guides

Welcome to the Moltres user guides. These guides provide practical, step-by-step instructions for using Moltres effectively.

> **Contributors:** Edit guides in this directory (`guides/NN-*.md`) only. Files under `docs/guides/` are Sphinx include stubs for Read the Docs — do not edit them directly.

## Guide Index

### Start here
- **[00 - Quick Start (5 min)](https://moltres.readthedocs.io/en/latest/guides/quick-start.html)** - Install, connect, query, and CRUD in five minutes.
- **[01 - Complete Tutorial](https://moltres.readthedocs.io/en/latest/guides/getting-started.html)** - Full step-by-step introduction (~30 minutes).

### Migration Guides
- **[02 - Migrating from Pandas](https://moltres.readthedocs.io/en/latest/guides/migrating-from-pandas.html)** - Transition from Pandas to Moltres with side-by-side comparisons.
- **[03 - Migrating from PySpark](https://moltres.readthedocs.io/en/latest/guides/migrating-from-pyspark.html)** - Migrate from PySpark; includes migration footguns (union, file I/O, writes).

### Optimization & Best Practices
- **[04 - Performance Optimization](https://moltres.readthedocs.io/en/latest/guides/performance-optimization.html)** - Write efficient queries and optimize performance.
- **[08 - Best Practices](https://moltres.readthedocs.io/en/latest/guides/best-practices.html)** - Essential best practices for maintainable code.

### Practical Guides
- **[05 - Common Patterns](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html)** - Real-world patterns and use cases.
- **[06 - Error Handling](https://moltres.readthedocs.io/en/latest/guides/error-handling.html)** - Debug issues and handle errors effectively.
- **[07 - Advanced Topics](https://moltres.readthedocs.io/en/latest/guides/advanced-topics.html)** - Advanced features for power users.
- **[09 - Pandas Interface](https://moltres.readthedocs.io/en/latest/guides/pandas-interface.html)** - Pandas-style API with string accessor and inspection methods.
- **[10 - Polars Interface](https://moltres.readthedocs.io/en/latest/guides/polars-interface.html)** - Polars-style API.
- **[18 - SQL Approaches Comparison](https://moltres.readthedocs.io/en/latest/guides/sql-approaches-comparison.html)** - Compare raw SQL, SQLAlchemy Core, and Moltres.

### Transactions
- **[19 - Transaction Control](https://moltres.readthedocs.io/en/latest/guides/transaction-control.html)** - Savepoints, isolation levels, locking.
- **[20 - Transaction Utilities](https://moltres.readthedocs.io/en/latest/guides/transaction-utilities.html)** - Decorators, hooks, retries, metrics.

### Framework & Tooling Integrations
- **[11 - SQLAlchemy Integration](https://moltres.readthedocs.io/en/latest/guides/sqlalchemy-integration.html)** - SQLAlchemy engine and session patterns.
- **[12 - SQLModel Integration](https://moltres.readthedocs.io/en/latest/guides/sqlmodel-integration.html)** - SQLModel and Pydantic models.
- **[13 - Django Integration](https://moltres.readthedocs.io/en/latest/guides/django-integration.html)** - Django views, middleware, template tags.
- **[14 - Streamlit Integration](https://moltres.readthedocs.io/en/latest/guides/streamlit-integration.html)** - Streamlit dashboards.
- **[15 - pytest Integration](https://moltres.readthedocs.io/en/latest/guides/pytest-integration.html)** - pytest fixtures and test patterns.
- **[16 - Workflow Integration](https://moltres.readthedocs.io/en/latest/guides/workflow-integration.html)** - Airflow and Prefect orchestration.
- **[17 - dbt Integration](https://moltres.readthedocs.io/en/latest/guides/dbt-integration.html)** - dbt Python models with Moltres.

## Quick Navigation

**New to Moltres?**
1. Start with [Quick Start (5 min)](https://moltres.readthedocs.io/en/latest/guides/quick-start.html)
2. Run `docs/examples/01_connecting.py` and `02_dataframe_basics.py`
3. Read [Common Patterns](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html) for real-world examples
4. Review [Best Practices](https://moltres.readthedocs.io/en/latest/guides/best-practices.html) as you write code

**Coming from Pandas?**
1. Read [Migrating from Pandas](https://moltres.readthedocs.io/en/latest/guides/migrating-from-pandas.html)
2. Learn the [Pandas Interface](https://moltres.readthedocs.io/en/latest/guides/pandas-interface.html)
3. Check [Common Patterns](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html)

**Coming from PySpark?**
1. Read [Migrating from PySpark](https://moltres.readthedocs.io/en/latest/guides/migrating-from-pyspark.html) — especially [migration footguns](https://moltres.readthedocs.io/en/latest/guides/migrating-from-pyspark.html#migration-footguns)
2. Review [PySpark migration footguns](https://moltres.readthedocs.io/en/latest/PYSPARK_MIGRATION_INCONSISTENCIES.html) before production ports

**Having Issues?**
1. Check [Error Handling](https://moltres.readthedocs.io/en/latest/guides/error-handling.html)
2. Review [Troubleshooting](https://moltres.readthedocs.io/en/latest/TROUBLESHOOTING.html)
3. See [FAQ](https://moltres.readthedocs.io/en/latest/FAQ.html)

## Additional Resources

- **Runnable scripts:** [Example scripts index](https://moltres.readthedocs.io/en/latest/EXAMPLE_SCRIPTS.html) (`docs/examples/*.py`)
- **Narrative patterns:** [Common patterns guide](https://moltres.readthedocs.io/en/latest/guides/common-patterns.html) (prefer over archived `EXAMPLES.md`)
- **API surface:** [Public API](https://moltres.readthedocs.io/en/latest/PUBLIC_API.html) and [API reference](https://moltres.readthedocs.io/en/latest/api/dataframe.html)
- **Changelog:** [CHANGELOG](https://moltres.readthedocs.io/en/latest/CHANGELOG.html) on Read the Docs
- **Main README:** [GitHub README](https://github.com/eddiethedean/moltres) for repository overview

## Contributing

Found an issue with a guide? Edit the numbered file in `guides/` and open a pull request on GitHub.
