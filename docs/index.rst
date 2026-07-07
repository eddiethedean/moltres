Moltres Documentation
=====================

Moltres is **the missing DataFrame layer for SQL in Python**.
It provides a PySpark-style DataFrame API that compiles to SQL and executes directly
in your database with full SQL CRUD support and optional pandas/polars result formats.

Use these docs to:

- Understand Moltres concepts and architecture
- Follow step‑by‑step guides and recipes
- Explore framework and tooling integrations
- Look up the full, generated API reference

.. note::

   **New to Moltres?** Start with :doc:`guides/quick-start` (5 minutes), then :doc:`guides/getting-started` for the full tutorial,
   then :doc:`PUBLIC_API`, then :doc:`EXAMPLE_SCRIPTS` for runnable scripts.


Start here
==========

.. toctree::
   :maxdepth: 1
   :caption: Start here

   guides/quick-start
   guides/getting-started
   PUBLIC_API
   EXAMPLE_SCRIPTS
   CHANGELOG
   FAQ
   ROADMAP


Getting started & migration
===========================

.. toctree::
   :maxdepth: 2
   :caption: Getting started & migration

   guides/migrating-from-pandas
   guides/migrating-from-pyspark


Guides & how-to
===============

.. toctree::
   :maxdepth: 2
   :caption: Guides & how-to

   guides/common-patterns
   guides/performance-optimization
   guides/error-handling
   guides/advanced-topics
   guides/best-practices
   guides/pandas-interface
   guides/polars-interface
   guides/sql-approaches-comparison
   guides/transaction-control
   guides/transaction-utilities


Framework & tooling integrations
================================

.. toctree::
   :maxdepth: 2
   :caption: Integrations

   guides/sqlalchemy-integration
   guides/sqlmodel-integration
   guides/django-integration
   guides/streamlit-integration
   guides/pytest-integration
   guides/workflow-integration
   guides/dbt-integration


Concepts, operations, and internals
===================================

.. toctree::
   :maxdepth: 2
   :caption: Concepts & operations

   WHY_MOLTRES
   moltres-design-notes
   PERFORMANCE
   RUNTIME_SUPPORT
   SECURITY
   TESTING
   DEBUGGING
   DEPLOYMENT
   TROUBLESHOOTING


Comparisons (reference)
=======================

Historical comparison reports. For current migration guidance, prefer :doc:`guides/migrating-from-pyspark` and :doc:`PYSPARK_MIGRATION_INCONSISTENCIES`.

.. toctree::
   :maxdepth: 1
   :caption: Comparisons (reference)

   MOLTRES_VS_PYSPARK_COMPARISON
   MOLTRES_VS_SQLFRAME_COMPARISON
   MOLTRES_VS_IBIS_COMPARISON
   PYSPARK_MIGRATION_INCONSISTENCIES


API reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/moltres
   api/records
   api/dataframe
   api/expressions
   api/table
   api/engine


Indices and tables
==================

.. toctree::
   :hidden:

   CONTRIBUTING
   EXAMPLES
   PYDANTABLE_ENGINE
   RELEASE_PROCESS

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
