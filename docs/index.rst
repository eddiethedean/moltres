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

   **New to Moltres?** Start with :doc:`guides/getting-started`, then explore the
   “Guides & How‑To” and “Integrations” sections below. The API reference is designed
   for day‑to‑day lookups once you are familiar with the basics.


Quick navigation
----------------

- :doc:`guides/getting-started`
- :doc:`EXAMPLES`
- :doc:`api/dataframe`
- :doc:`guides/performance-optimization`
- :doc:`WHY_MOLTRES`


Getting started & migration
===========================

.. toctree::
   :maxdepth: 2
   :caption: Getting started & migration

   guides/getting-started
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
   EXAMPLES
   FAQ


Internal and archive docs
=========================

.. toctree::
   :maxdepth: 1
   :caption: Internal & archive

   BUSINESS_CASE
   PROJECT_CHARTER
   PROJECT_SCOPE_STATEMENT
   moltres_plan
   moltres_advocacy
   integration_features_plan
   PRIORITY_IMPLEMENTATION
   MIGRATION_GUIDE
   MIGRATION_SPARK
   MOLTRES_VS_PYSPARK_COMPARISON
   PYSPARK_FEATURE_COMPARISON
   PYSPARK_INTERFACE_AUDIT
   PYSPARK_MIGRATION_INCONSISTENCIES
   PERFORMANCE_SLA
   RUNTIME_MATRIX
   TEST_HARNESSES
   OPS_RUNBOOKS
   IMPROVE_PYTEST_GREEN_LIGHT
   RELEASE_PROCESS


API reference
=============

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/dataframe
   api/expressions
   api/table
   api/engine


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
