# Moltres Roadmap

This document outlines **future 1.x release phases** to close competitive gaps identified in our
market positioning review. It complements maintainer policy in
[`RELEASE_PROCESS.md`](RELEASE_PROCESS.md) and the stable surface in
[`PUBLIC_API.md`](PUBLIC_API.md).

**Positioning:** Moltres is not trying to replace Polars, Ibis, or PySpark at everything. The goal
for 1.x is to become the **obvious choice** for *PySpark-style transforms + SQL pushdown + CRUD +
async* on traditional RDBMS workloads (PostgreSQL, MySQL, SQLite, DuckDB).

**Current release:** [1.1.0](https://github.com/eddiethedean/moltres/blob/main/CHANGELOG.md) — public API contracts, async CRUD parity, Records
constructors, optional-export guardrails, canonical I/O documentation.

---

## Competitive gaps we are closing

| Gap | vs | Target phase |
|-----|-----|--------------|
| Multiple paths to the same outcome (`load` / `read` / `scan`, four insert paths) | Ibis, Polars UX | **1.2** |
| Doc/code drift and deprecated examples | Ecosystem trust | **1.2** |
| `explode()` / UNNEST limited dialects | Ibis, PySpark | **1.3** |
| No SQL escape hatch in DataFrame API | Ibis `t.sql()` | **1.3** |
| File reads materialize to temp tables (surprise cost) | Polars, Spark scan | **1.4** |
| No published performance benchmarks | Ibis, Polars credibility | **1.4** |
| Postgres/MySQL depth vs generic SQLAlchemy | SQLAlchemy, Ibis backends | **1.5** |
| Thin community / integration maturity | Polars, Ibis ecosystem | **1.6** |
| Semantic traps (`union`, `column` vs `col`) | PySpark migrants | **2.0** (deprecate in 1.x) |

---

## Release philosophy (1.x)

- **Semver:** 1.x releases are additive and deprecation-driven. Breaking removals land in **2.0**
  with warnings in 1.x per [`RELEASE_PROCESS.md`](RELEASE_PROCESS.md).
- **Theme per release:** One headline user-facing improvement, not a grab bag.
- **Dialect strategy:** Go deep on **PostgreSQL + SQLite** first; treat other SQLAlchemy dialects as
  best-effort until explicitly promoted.
- **Out of scope for 1.x:** Distributed compute, UDFs, MLlib, Structured Streaming, native
  BigQuery/Snowflake engines (investigate in 1.6+, ship in 2.x if pursued).

| Phase | Target window | Theme |
|-------|---------------|-------|
| v1.2 | 2026-07 | API consolidation |
| v1.3 | 2026-09 | Dialect depth + SQL escape hatch |
| v1.4 | 2026-11 | File I/O honesty + benchmarks |
| v1.5 | 2027-01 | PostgreSQL-first backend depth |
| v1.6 | 2027-03 | Community + integrations |

---

## v1.2 — API consolidation (“one obvious way”)

**Goal:** Remove adoption friction from API fragmentation. Finish the work started in 1.1.

**Competitive answer:** “I know exactly which API to use” — matches Polars/Ibis clarity on I/O.

### Deliverables

- [ ] **Enforce canonical I/O paths**
  - `db.load.*` for lazy `DataFrame` file reads (keep)
  - `db.read.records.*` for eager `Records` (keep)
  - Remove or hard-deprecate `db.read.csv/json/...` DataFrame paths (warnings today → removal 2.0)
  - Move `db.scan_*` under documented Polars-only namespace (`db.polars.scan_*` alias + deprecation)
- [ ] **CRUD path clarity**
  - Document return-type split: `db.insert` / `Records.insert_into` → `int`; `df.write.*` → document why
  - Add `values=` alias for `set=` on `update()` (deprecate `set=` for 2.0)
- [ ] **Contract expansion**
  - Extend deprecation policy coverage to `moltres.io.records` and `Database` CRUD methods
  - Contract-test async CRUD and `LazyRecords` factories
- [ ] **Documentation hygiene**
  - Fix all `scripts/validate_examples.py` failures (`OPS_RUNBOOKS.md`, `MOLTRES_VS_PYSPARK_COMPARISON.md`)
  - Migrate examples from `Records(_data=...)` to `Records.from_list(..., database=db)`
- [ ] **PySpark migration safety**
  - Prominent `union` vs `unionAll` callouts in quick start and migration guide
  - “Intentionally absent” section (no `cache`, `repartition`, UDFs) for Spark migrants

### Success metrics

- Zero doc-example validation failures in CI
- `PUBLIC_API.md` is the only required reading for I/O + CRUD entry points
- Deprecation warnings on all non-canonical DataFrame read paths

---

## v1.3 — SQL depth and dialect parity

**Goal:** Close the “Ibis compiles more SQL than Moltres” gap on core analytics transforms.

**Competitive answer:** Feature parity with Ibis on **read/transform** for Postgres/MySQL/SQLite/DuckDB.

### Deliverables

- [ ] **`explode()` / UNNEST** on MySQL and remaining dialect gaps ([`plan_compiler.py`](https://github.com/eddiethedean/moltres/blob/main/src/moltres/sql/plan_compiler.py))
- [ ] **Table-valued / UNNEST functions** ([maintainer notes](https://github.com/eddiethedean/moltres/blob/main/docs/PRIORITY_IMPLEMENTATION.md))
- [ ] **SQL escape hatch** — `df.sql("SELECT ...")` or `db.sql(...)` returning `DataFrame` (Ibis parity)
- [ ] **Dialect capability matrix** — published table: which ops work on which dialect
- [ ] **Compilation errors** — actionable messages with dialect hint and suggested workaround
- [ ] **Semi/anti join** camelCase aliases (`semiJoin`, `antiJoin`) for PySpark migrants

### Success metrics

- `explode()` integration tests pass on sqlite, postgresql, mysql, duckdb
- SQL escape hatch documented with security guidance (parameterization)
- Dialect matrix auto-generated or CI-checked against feature tests

---

## v1.4 — File I/O honesty and performance credibility

**Goal:** Stop losing to Polars/DuckDB on file workflows and prove DB-pushdown value with data.

**Competitive answer:** “Files are staged explicitly; transforms on SQL tables are genuinely pushdown.”

### Deliverables

- [ ] **Rename mental model** — document file path as *stage → temp table → SQL* (not “lazy scan”)
- [ ] **Staging optimizations**
  - Chunked staging with configurable batch size defaults
  - Skip re-staging when same file path + mtime unchanged (session cache)
  - `INSERT INTO ... SELECT` from staged files without Python round-trip where possible
- [ ] **Writer memory posture** — default streaming for large writes; document `.stream()` requirement
- [ ] **Benchmark suite** ([`PERFORMANCE.md`](PERFORMANCE.md))
  - Moltres vs raw SQLAlchemy vs Ibis (PostgreSQL) on TPC-H–style queries
  - Publish results in docs and CI regression threshold (non-blocking initially)
- [ ] **ORC read support** (optional extra) — writer remains out of scope until demand proven

### Success metrics

- Benchmark docs with reproducible `scripts/benchmark_*.py`
- File-read docs pass a “no false lazy claims” review
- Measurable reduction in temp-table staging time for repeated reads

---

## v1.5 — PostgreSQL-first backend depth

**Goal:** Win the backend engineer persona against generic SQLAlchemy + hand-written SQL.

**Competitive answer:** “Best PySpark-like layer for Postgres” — not “generic SQLAlchemy wrapper.”

### Deliverables

- [ ] **PostgreSQL reference dialect**
  - MERGE/upsert edge cases, `FOR UPDATE`/`SKIP LOCKED`, savepoints, isolation level docs
  - `FILTER` clause polish and SQLite/MySQL fallback documentation
- [ ] **DuckDB CI tier** — promote DuckDB to required CI dialect (alongside SQLite)
- [ ] **Observability**
  - Query timing breakdown in performance hooks (compile vs execute)
  - Optional slow-query logging via `MoltresConfig`
- [ ] **SQLAlchemy escape hatch** — `df.to_sqlalchemy()` / compiled statement export documented for interop
- [ ] **Community tier dialects** — SQL Server, Oracle: document “works via SQLAlchemy, not CI-guaranteed”

### Success metrics

- Postgres-specific integration test module with 95%+ feature coverage of public transform API
- DuckDB in default CI matrix
- At least one published “Moltres + Postgres” production case study (see 1.6)

---

## v1.6 — Ecosystem maturity and adoption

**Goal:** Close the “young project” gap with integrations and social proof.

**Competitive answer:** “Safe to adopt in a real FastAPI/dbt/pytest stack.”

### Deliverables

- [ ] **Integration hardening**
  - pytest plugin: stable fixture API, documented in `PUBLIC_API.md`
  - dbt: Python model adapter docs + example project in repo
  - FastAPI: async CRUD cookbook patterns
- [ ] **Interactive ergonomics** — optional `moltres.interactive` preview mode (Ibis-style table display)
- [ ] **Adoption kit**
  - 3 case studies (backend CRUD, Spark downsizing, analytics engineer)
  - Migration guides: SQLAlchemy Core, raw psycopg, PySpark subset
- [ ] **Warehouse investigation spike** (non-committing)
  - Feasibility doc for BigQuery/Snowflake via Ibis-backend bridge or SQLFrame comparison
  - Decision gate for 2.0 backend architecture
- [ ] **Contributor growth** — good-first-issue labels, integration test tiers documented

### Success metrics

- Integration extras have contract tests and version-pinned CI jobs
- ReadTheDocs “Comparisons” section links to updated case studies
- GitHub issue response SLA documented for maintainers

---

## v2.0 preview (breaking window — deprecate throughout 1.x)

Items **not** shipped as breaking changes in 1.x. Start deprecation warnings in 1.2–1.6 where noted.

| Change | Deprecation starts | Rationale |
|--------|------------------|-----------|
| `union()` → ALL rows; `union_distinct()` for DISTINCT | 1.2 warn | PySpark semantic trap |
| `column` → `schema_column` (keep alias) | 1.3 warn | `col` vs `column` confusion |
| Remove `Records._data` / `_database` | 1.1 warn → 2.0 remove | Public constructor shipped 1.1 |
| Remove `db.read.*` DataFrame paths | 1.1 warn → 2.0 remove | Canonical `db.load.*` |
| Remove `set=` on `update()` | 1.2 warn → 2.0 remove | Builtin shadowing |
| Namespace `scan_*` under Polars submodule | 1.2 warn → 2.0 remove | I/O consolidation |

---

## What we are explicitly not building in 1.x

These are competitor strengths Moltres should **not** chase without a 2.0 architecture decision:

- **Distributed cluster execution** (PySpark) — single DB connection is the model
- **Python UDFs** — breaks SQL pushdown guarantee
- **MLlib / model training** — use dedicated ML stack
- **In-process columnar engine** (Polars) — use Polars interface or DuckDB directly
- **20+ native backends** (Ibis) — depth over breadth until 2.0 warehouse work lands

---

## How to influence the roadmap

- **Users:** Open a [GitHub discussion](https://github.com/eddiethedean/moltres/discussions) with your
  persona (backend, analytics, Spark migrant) and workload (dialect, data size, CRUD vs read-only).
- **Contributors:** Pick items marked good-first-issue; align PRs to a release phase above.
- **Maintainers:** Update this file at each minor release; move completed items to `CHANGELOG.md`.

---

## Related documents

- [`PUBLIC_API.md`](PUBLIC_API.md) — stable import surface
- [`WHY_MOLTRES.md`](WHY_MOLTRES.md) — positioning and unique value
- [Priority implementation notes](https://github.com/eddiethedean/moltres/blob/main/docs/PRIORITY_IMPLEMENTATION.md) — feature-level implementation notes
- [`MOLTRES_VS_IBIS_COMPARISON.md`](MOLTRES_VS_IBIS_COMPARISON.md) — Ibis comparison
- [`MOLTRES_VS_PYSPARK_COMPARISON.md`](MOLTRES_VS_PYSPARK_COMPARISON.md) — PySpark comparison
