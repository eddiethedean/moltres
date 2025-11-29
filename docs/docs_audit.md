## Moltres Docs Inventory and Audit

This file tracks the main documentation sources for Moltres and their audit status.

### Status Legend

- `current` – Accurate and up to date.
- `needs-update` – Mostly correct but needs edits (APIs, links, or narrative).
- `archive` – Historical or internal; kept but clearly marked as such.
- `remove` – To be deleted once any unique content is migrated.

### Core entrypoints

| Path | Category | Status | Notes |
| --- | --- | --- | --- |
| `README.md` | Getting started / overview | current | GitHub landing page, synced with RTD. |
| `docs/index.rst` | RTD home | current | New landing page + navigation. |
| `docs/README.md` | Docs contributor guide | current | How to build and work on docs. |

### Guides (user-facing)

RTD-rendered guides in `docs/guides/` correspond 1:1 with the numbered guides in `guides/`.

| Path | Category | Status | Notes |
| --- | --- | --- | --- |
| `docs/guides/getting-started.md` | Getting started | current | Primary intro; matches core API. |
| `docs/guides/migrating-from-pandas.md` | Migration | current | Pandas → Moltres mapping. |
| `docs/guides/migrating-from-pyspark.md` | Migration | current | PySpark → Moltres mapping. |
| `docs/guides/common-patterns.md` | Patterns | current | Pipelines, analytics, CRUD patterns. |
| `docs/guides/performance-optimization.md` | Performance | current | Performance best practices. |
| `docs/guides/error-handling.md` | DX / debugging | current | Error handling and debugging. |
| `docs/guides/advanced-topics.md` | Advanced | current | Async, streaming, advanced flows. |
| `docs/guides/best-practices.md` | Best practices | current | General usage guidance. |
| `docs/guides/pandas-interface.md` | Interface | current | Pandas-style API. |
| `docs/guides/polars-interface.md` | Interface | current | Polars-style API. |
| `docs/guides/sqlalchemy-integration.md` | Integration | current | SQLAlchemy ORM / models. |
| `docs/guides/sqlmodel-integration.md` | Integration | current | SQLModel / Pydantic. |
| `docs/guides/django-integration.md` | Integration | current | Django views, middleware, tags. |
| `docs/guides/streamlit-integration.md` | Integration | current | Streamlit dashboards. |
| `docs/guides/pytest-integration.md` | Integration | current | pytest fixtures, green-light. |
| `docs/guides/workflow-integration.md` | Integration | current | Airflow, Prefect, dbt workflows. |
| `docs/guides/dbt-integration.md` | Integration | current | dbt adapter usage. |

### Examples

| Path | Category | Status | Notes |
| --- | --- | --- | --- |
| `docs/EXAMPLES.md` | Examples index | current | Narrative index for examples. |
| `docs/examples/*.py` | Executable examples | current | Synced with scripts/validation harnesses. |
| `docs/examples/README.md` | Examples README | current | How to run examples. |

### Concepts, operations, and internals

| Path | Category | Status | Notes |
| --- | --- | --- | --- |
| `docs/WHY_MOLTRES.md` | Concept / positioning | current | High-level rationale. |
| `docs/PERFORMANCE.md` | Performance internals | needs-update | Cross-check vs performance guide. |
| `docs/PERFORMANCE_SLA.md` | Performance SLA | archive | Historical SLA notes. |
| `docs/RUNTIME_SUPPORT.md` | Runtime support | current | Supported Python / DBs. |
| `docs/RUNTIME_MATRIX.md` | Runtime matrix | needs-update | Keep in sync with pyproject + CI. |
| `docs/SECURITY.md` | Security | current | Threat model and best practices. |
| `docs/TESTING.md` | Testing | current | How tests are structured. |
| `docs/DEBUGGING.md` | Debugging | current | Deeper debugging flows. |
| `docs/DEPLOYMENT.md` | Deployment | current | Deploying Moltres-based apps. |
| `docs/TROUBLESHOOTING.md` | Troubleshooting | current | FAQ-style operational issues. |
| `docs/FAQ.md` | FAQ | current | User-facing questions. |
| `docs/TEST_HARNESSES.md` | Internal testing | archive | For maintainers only. |
| `docs/OPS_RUNBOOKS.md` | Ops runbooks | archive | Operational guidance, internal. |

### Project / planning / comparison docs

| Path | Category | Status | Notes |
| --- | --- | --- | --- |
| `docs/BUSINESS_CASE.md` | Project meta | archive | Historical; not user-facing. |
| `docs/PROJECT_CHARTER.md` | Project meta | archive | Historical charter. |
| `docs/PROJECT_SCOPE_STATEMENT.md` | Project meta | archive | Historical scope. |
| `docs/moltres_plan.md` | Roadmap | archive | Old planning document. |
| `docs/moltres_advocacy.md` | Advocacy | archive | Internal positioning notes. |
| `docs/integration_features_plan.md` | Planning | archive | Integration roadmap details. |
| `docs/PRIORITY_IMPLEMENTATION.md` | Planning | archive | Legacy priorities. |
| `docs/MIGRATION_GUIDE.md` | Migration | archive | Superseded by structured guides. |
| `docs/MIGRATION_SPARK.md` | Migration | archive | Superseded by guides. |
| `docs/MOLTRES_VS_PYSPARK_COMPARISON.md` | Comparison | archive | Historical comparison. |
| `docs/PYSPARK_FEATURE_COMPARISON.md` | Comparison | archive | Feature matrix. |
| `docs/PYSPARK_INTERFACE_AUDIT.md` | Comparison | archive | Audit notes. |
| `docs/PYSPARK_MIGRATION_INCONSISTENCIES.md` | Comparison | archive | Edge cases. |
| `docs/IMPROVE_PYTEST_GREEN_LIGHT.md` | Internal testing | archive | pytest-green-light notes. |
| `docs/RELEASE_PROCESS.md` | Process | current | Release checklist. |

### API reference helpers

| Path | Category | Status | Notes |
| --- | --- | --- | --- |
| `docs/api/dataframe.rst` | API reference | current | DataFrame + AsyncDataFrame. |
| `docs/api/expressions.rst` | API reference | current | Column + functions + window. |
| `docs/api/table.rst` | API reference | current | Table / AsyncTable / schema. |
| `docs/api/engine.rst` | API reference | current | Engine + connection. |

### Miscellaneous / other

| Path | Category | Status | Notes |
| --- | --- | --- | --- |
| `docs/CONTRIBUTING.md` | Contributing | current | Docs-specific contribution notes. |
| `CONTRIBUTING.md` | Contributing | current | Project-level contribution guide. |
| `benchmarks/README.md` | Benchmarks | current | Benchmark usage. |
| `notebooks/README.md` | Notebooks | current | How to use example notebook. |
| `notebooks/KERNEL_SETUP.md` | Notebooks | current | Kernel setup instructions. |


