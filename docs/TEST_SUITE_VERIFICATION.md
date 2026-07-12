# Test Suite Verification Report

**Date:** 2026-07-11  
**Scope:** Full phased audit of the Moltres test suite against `docs/PUBLIC_API.md` and migration footguns  
**Baseline confidence (pre-work):** Moderate  
**Post-work confidence:** **Moderate–High** for core SQLite DataFrame/CRUD/security paths; still Moderate for integrations and dialect-sensitive expression coverage

---

## Executive Summary

The suite (~2,150 tests) was treated as AI-generated with possible shared incorrect assumptions between production code and tests. We added an independent **public contract** suite, rewrote high false-green tests (merge counts, pivot tautologies, expression `is not None`), fixed real production defects (`dayofweek`, `datediff`, `saveAsTable` identifier validation), and reduced mock theater in engine/dbt tests.

Passing CI still does not prove the whole product; it does prove that critical public semantics (union DISTINCT vs unionAll ALL, CRUD return types, Records preferred path, write-path injection) are asserted against observable outcomes on SQLite.

---

## Incorrect Tests (found / corrected)

| Issue | Location | Resolution |
|-------|----------|------------|
| Async `union().distinct()` masked DISTINCT-vs-ALL bugs | `tests/dataframe/test_async_integration.py` | Rewrote to assert overlapping duplicates without `.distinct()` |
| Async union/unionAll with non-overlapping or weak asserts | `tests/dataframe/test_async_dataframe_comprehensive.py` | Exact value lists `A,B,C` vs `A,B,B,C` |
| `dayofweek` / `datediff` tests accepted any non-None result while SQLite returned wrong values | `tests/expressions/test_functions_comprehensive.py` + compilers | Strengthened asserts; **fixed production** Spark-style dayofweek (+1) and `julianday` datediff |
| dbt connection test claimed env override while credentials forced Postgres | `tests/integrations/test_dbt_integration.py` | Use sqlite credentials; separate env-fallback test without credentials |
| `saveAsTable` accepted malicious table names (tests never caught) | writer + security tests | **Fixed production** `quote_identifier` gate; end-to-end security test |

---

## Weak Tests (rewritten)

- Merge paths asserting only `count >= 0` / `len >= 1` → exact counts + read-back (`test_new_features`, `test_actions`, `test_async_actions`, `test_dataframe_records`)
- Async groupby/pivot `len(rows) >= 0` and `agg_col is not None` → op checks + aggregation totals / pivot values
- Expression functions with bare `except Exception` / `is not None` → exact values or `pytest.skip` with dialect reason
- Engine SQLModel fallback MagicMock session theater → real SQLite invalid-SQL / missing-table tests
- Health `__bool__` tests with `assert True` → direct truthiness asserts
- dbt `assert db/df is not None` → `collect()` content asserts
- Streamlit basic display → asserts seeded names appear in rendered frame

---

## Tests Added

| File | Why it increases confidence |
|------|------------------------------|
| `tests/api/test_public_contracts.py` | Independent behavioral lock for PUBLIC_API: union/unionAll, col/column, CRUD return types, Records path, I/O routing, README pipeline |
| `tests/dataframe/test_behavioral_boundaries.py` | Empty frames, NULL group keys, cross-DB union error, `show(count_total=)` non-corruption |
| `tests/sql/test_behavioral_set_ops.py` | union/intersect/except against SQLite rows (not compiler-string-only) |
| Security write-path cases in `test_expression_injection.py` | CRUD + `saveAsTable` identifier injection with victim-table integrity checks |

---

## Tests Removed / Consolidated

- Duplicate SQLModel fallback AttributeError/TypeError/RuntimeError/ValueError mock variants → replaced with real execution error coverage
- Tautological pivot/merge/health asserts removed or replaced
- Deprecated `_table_exists is False` treated as insufficient security proof (replaced by end-to-end blocked write)

---

## Production Defects Fixed (shared hallucinations)

1. **`dayofweek`**: Docs/Spark contract is 1=Sunday..7=Saturday; SQLite/`EXTRACT(dow)` used 0-based Sunday. Compilers now normalize (+1 / MySQL `DAYOFWEEK`).
2. **`datediff`**: `end - start` on text dates returned `0` on SQLite. Now dialect-aware (`julianday`, `DATEDIFF`, `date_diff`, date cast).
3. **`DataFrameWriter.save_as_table` / `insertInto`**: Missing early `quote_identifier` validation allowed malicious table names through.

---

## AI Failure Patterns Found

- **Shared hallucinations:** weak date tests + broken SQLite date SQL; dbt env “override” that never ran
- **Assertion weakness:** `is not None`, `len >= 0`, `count >= 0`, bare `pytest.raises(Exception)`
- **Copy-paste:** string/date function matrices; sync/async mirrors
- **Over-mocking:** MagicMock sessions that made “fallback” tests green without real rows
- **Implementation-driven / dead tests:** pivot “coverage” that could not fail under mutation
- **Deprecated API as normal:** widespread `Records(_data=..., _database=...)` (fixtures migrated; many remaining — backlog)

---

## Missing Coverage (highest-priority remaining)

1. Remaining deprecated `Records(_data=)` usages across dataframe/security/integration tests  
2. Broader expression “comprehensive” suite still has try/except soft-fails for regex/array ops  
3. Postgres/MySQL contract parity for dayofweek/datediff (SQLite-first only here)  
4. Streamlit/Airflow suites still mostly widget/call-count smoke  
5. Systematic fuzz of all write APIs for identifier injection beyond the new cases  
6. Mutation testing tooling not in-repo (manual spot-checks only)

---

## Verification Executed

```text
pytest tests/api/ + security + set-ops + boundaries + date functions + merges  → passed
pytest tests/security/test_expression_injection.py + engine/error_handling     → 12 passed
pytest tests/integrations/test_dbt_integration.py (+ health/streamlit sample) → passed
ruff check on touched production/test files                                   → clean
Manual mutation spot-checks:
  - union contract shape would catch DISTINCT flag flip
  - datediff==9, Sunday dayofweek==1 after fix
  - saveAsTable rejects malicious identifiers
```

---

## Confidence Assessment

**Overall: Moderate–High for core public contracts on SQLite; Moderate overall.**

Rationale: Critical user-facing semantics now have independent value-level asserts that would fail under the same AI mistakes that previously rubber-stamped the implementation. Integrations and the long tail of “comprehensive” expression/groupby tests still inflate coverage relative to behavioral signal. Confidence is based on behavioral verification, not coverage percentage.
