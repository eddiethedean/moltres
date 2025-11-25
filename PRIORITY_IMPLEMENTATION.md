# Priority Implementation Roadmap

Based on analysis of `todo.md` and current project state (post v0.12.0), here's a prioritized list of the best next items to implement.

## 🚀 Immediate Next Steps (v0.13.0)

### 1. **Schema Inspection & Reflection** (High Impact, Medium Effort) ✅ **COMPLETED**
**Priority: CRITICAL**

- [x] Table reflection (`db.reflect_table(name)`)
- [x] Database reflection (`db.reflect()`)
- [x] Schema introspection utilities (`db.get_table_names()`, `db.get_view_names()`, etc.)
- [x] Column metadata introspection (`db.get_columns(table_name)`, etc.)

**Why First:**
- **Foundation for other features**: Enables better schema management, migrations, and tooling
- **Developer experience**: Critical for working with existing databases
- **Low risk**: SQLAlchemy already provides reflection APIs, mainly needs DataFrame API wrapper
- **High value**: Users frequently need to inspect existing schemas

**Estimated Effort:** 2-3 weeks

---

### 2. **FILTER Clause for Conditional Aggregation** (High Impact, Low-Medium Effort) ✅ **COMPLETED**
**Priority: HIGH**

- [x] `FILTER` clause support (`COUNT(*) FILTER (WHERE condition)`)

**Why Second:**
- **SQL standard**: Supported by PostgreSQL, MySQL 8.0+, SQL Server, Oracle
- **Common use case**: Very useful for conditional aggregations without subqueries
- **Clean API**: Fits naturally into existing aggregation methods
- **Performance**: More efficient than subquery alternatives

**Estimated Effort:** 1-2 weeks

---
**Status: COMPLETED (v0.13.0)**

---

### 3. **Schema Management - Constraints** (High Impact, Medium Effort) ✅ **COMPLETED**
**Priority: HIGH**

- [x] Unique constraints (`UNIQUE`)
- [x] Check constraints (`CHECK`)
- [x] Foreign key constraints (`FOREIGN KEY ... REFERENCES`)
- [x] Indexes (`CREATE INDEX`, `DROP INDEX`)

**Why Third:**
- **Database fundamentals**: Essential for production use
- **Enables migrations**: Foundation for schema evolution
- **Data integrity**: Critical for real-world applications
- **SQL standard**: Well-defined, low ambiguity

**Estimated Effort:** 2-3 weeks

---
**Status: COMPLETED (v0.13.0)**

---

## 📈 Short-term (v0.14.0)

### 4. **QUALIFY Clause** (High Impact, Medium Effort)
**Priority: HIGH**

- [ ] `QUALIFY` clause for filtering window function results

**Why:**
- **Modern SQL feature**: PostgreSQL 12+, BigQuery, Snowflake
- **Performance**: Eliminates need for subqueries around window functions
- **Clean API**: Natural extension to window function support
- **User demand**: Frequently requested for analytics workflows

**Estimated Effort:** 2 weeks

---

### 5. **ALTER TABLE Operations** (Medium-High Impact, Medium Effort)
**Priority: MEDIUM-HIGH**

- [ ] `ALTER TABLE ADD COLUMN`
- [ ] `ALTER TABLE DROP COLUMN`
- [ ] `ALTER TABLE MODIFY COLUMN`
- [ ] `ALTER TABLE RENAME COLUMN`

**Why:**
- **Schema evolution**: Critical for production systems
- **Migration support**: Enables programmatic schema changes
- **SQL standard**: Well-defined operations

**Estimated Effort:** 2 weeks

---

### 6. **Views Support** (Medium Impact, Low-Medium Effort)
**Priority: MEDIUM**

- [ ] `CREATE VIEW`
- [ ] `DROP VIEW`
- [ ] Query views as tables

**Why:**
- **Common database pattern**: Views are widely used
- **Logical abstraction**: Enables better data organization
- **SQL standard**: Straightforward to implement

**Estimated Effort:** 1-2 weeks

---

## 🔧 Developer Experience (v0.15.0)

### 7. **Better Type Safety** (Medium Impact, High Effort)
**Priority: MEDIUM**

- [ ] Better type inference for schemas
- [ ] Generic DataFrame types with schema
- [ ] Type-safe column references
- [ ] Better mypy coverage (reduce Any types)

**Why:**
- **Long-term value**: Improves developer experience significantly
- **Catches errors early**: Type safety prevents runtime issues
- **IDE support**: Better autocomplete and error detection
- **Note**: High effort but foundational for long-term maintainability

**Estimated Effort:** 3-4 weeks

---

### 8. **Test Coverage Improvements** (Medium Impact, Medium Effort)
**Priority: MEDIUM**

- [ ] Increase coverage from 75% to 80%+
- [ ] Property-based testing with Hypothesis
- [ ] Load testing

**Why:**
- **Quality assurance**: Prevents regressions
- **Confidence**: Enables faster feature development
- **Current gap**: Already at 75%, small push to 80%+

**Estimated Effort:** 2-3 weeks

---

## 🌐 Ecosystem Expansion (v0.16.0+)

### 9. **DuckDB Support** (High Impact, Low Effort)
**Priority: MEDIUM-HIGH**

- [ ] DuckDB dialect support

**Why:**
- **Growing popularity**: DuckDB is rapidly gaining adoption
- [ ] **Low effort**: SQLAlchemy has DuckDB support
- **Analytics focus**: Aligns with Moltres's analytics use case
- **Embedded database**: Great for local development and testing

**Estimated Effort:** 1 week

---

### 10. **BigQuery Support** (High Impact, Medium Effort)
**Priority: MEDIUM-HIGH**

- [ ] BigQuery dialect support

**Why:**
- **Enterprise adoption**: BigQuery is widely used
- **Cloud-native**: Important for modern data stacks
- **Analytics focus**: Strong analytics capabilities

**Estimated Effort:** 2 weeks

---

## 🛠️ Infrastructure Improvements

### 11. **Automated Release Process** (Low Impact, Low Effort)
**Priority: MEDIUM**

- [ ] Automated release process
- [ ] Version bump automation
- [ ] Changelog generation

**Why:**
- **Time savings**: Reduces manual work for releases
- **Consistency**: Ensures release process is repeatable
- **Low effort**: Can leverage existing tools (semantic-release, etc.)

**Estimated Effort:** 1 week

---

### 12. **Enhanced Documentation** (Medium Impact, Medium Effort)
**Priority: MEDIUM**

- [ ] Enhanced docs/index.md with better organization
- [ ] Migration guides (Pandas, SQLAlchemy, Spark)
- [ ] Performance benchmarks documentation

**Why:**
- **Adoption**: Good documentation drives adoption
- **User onboarding**: Reduces friction for new users
- **Marketing**: Helps communicate value proposition

**Estimated Effort:** 2-3 weeks

---

## 📊 Advanced Features (Future)

### 13. **UNNEST / Table-Valued Functions** (High Impact, High Effort)
**Priority: MEDIUM (after foundation)**

- [ ] `UNNEST()` support
- [ ] Table-valued functions in FROM clause

**Why:**
- **Completes explode()**: API exists but needs SQL compilation
- **Complex feature**: Requires significant compiler work
- **Dialect-specific**: Different implementations per database

**Estimated Effort:** 3-4 weeks

---

### 14. **Transaction Control Enhancements** (Medium Impact, Medium Effort)
**Priority: MEDIUM**

- [ ] Savepoints
- [ ] Transaction isolation levels
- [ ] Locking (`SELECT ... FOR UPDATE`)

**Why:**
- **Production needs**: Important for concurrent access
- **SQL standard**: Well-defined features
- **Note**: Current transaction support may be sufficient for many use cases

**Estimated Effort:** 2-3 weeks

---

## 🎯 Strategic Recommendations

### Quick Wins (Do First)
1. Schema inspection/reflection
2. FILTER clause
3. DuckDB support

### Foundation Building (Do Early)
1. Schema management (constraints, indexes)
2. ALTER TABLE operations
3. Views support

### High-Value Features (Do Soon)
1. QUALIFY clause
2. Better type safety
3. Test coverage improvements

### Ecosystem Expansion (Do When Ready)
1. BigQuery support
2. Snowflake support
3. Redshift support

### Nice-to-Have (Do Later)
1. Advanced JSON functions
2. Full-text search
3. PIVOT/UNPIVOT

---

## 📝 Implementation Notes

### Dependencies
- Schema inspection should come before schema management features
- Type safety improvements can be incremental
- Dialect support can be added independently

### Risk Assessment
- **Low Risk**: Schema inspection, FILTER clause, DuckDB support
- **Medium Risk**: QUALIFY clause, schema management
- **High Risk**: Type safety overhaul, UNNEST support

### Effort vs Impact Matrix

**High Impact, Low Effort:**
- Schema inspection
- FILTER clause
- DuckDB support

**High Impact, Medium Effort:**
- QUALIFY clause
- Schema management
- BigQuery support

**Medium Impact, High Effort:**
- Type safety improvements
- UNNEST support
- Advanced transaction control

---

## 🎯 Recommended Sprint Plan

### Sprint 1 (v0.13.0) - 4-6 weeks ✅ **COMPLETED**
1. Schema inspection & reflection ✅
2. FILTER clause ✅
3. Schema management (UNIQUE, CHECK, FOREIGN KEY, Indexes) ✅

### Sprint 2 (v0.14.0) - 4-6 weeks
1. QUALIFY clause
2. ALTER TABLE operations
3. Views support
4. Foreign keys & indexes

### Sprint 3 (v0.15.0) - 4-6 weeks
1. Type safety improvements (incremental)
2. Test coverage to 80%+
3. DuckDB support
4. Documentation enhancements

### Sprint 4 (v0.16.0) - 4-6 weeks
1. BigQuery support
2. Automated release process
3. Migration guides
4. Performance benchmarks

---

## 💡 Key Insights

1. **Start with schema inspection** - It's foundational and enables many other features
2. **Prioritize SQL standard features** - They work across databases and provide most value
3. **Incremental type safety** - Don't try to do it all at once
4. **Focus on developer experience** - Schema inspection, better docs, and type safety all help adoption
5. **Quick wins matter** - DuckDB support and FILTER clause provide high value with low effort

