# Moltres Integration Features Plan

```{admonition} Archived
:class: warning

This document describes historical integration planning for Moltres.
It is kept for maintainers and is not part of the primary user docs.
```

## Overview
This plan outlines the implementation of integration features for four popular Python packages: Django, Streamlit, Pytest, and Airflow/Prefect. These integrations will make Moltres more user-friendly and robust when used with these frameworks, following the pattern established by the FastAPI integration.

## Goals
- Provide framework-specific utilities for error handling, dependency injection, and common patterns
- Improve developer experience when using Moltres with these frameworks
- Maintain consistency with the existing FastAPI integration pattern
- Ensure all integrations are optional dependencies (graceful degradation)

## Integration 1: Django Integration

### Overview
Django is one of the most popular Python web frameworks. This integration will provide utilities for using Moltres seamlessly within Django applications.

### Features to Implement

#### 1.1 Django Middleware for Error Handling
**File**: `src/moltres/integrations/django.py`

**Features**:
- Middleware class that catches Moltres exceptions and converts them to Django `HttpResponse` with appropriate status codes
- Automatic error message formatting for Django templates
- Integration with Django's logging system

**API**:
```python
from moltres.integrations.django import MoltresExceptionMiddleware

MIDDLEWARE = [
    # ... other middleware
    'moltres.integrations.django.MoltresExceptionMiddleware',
]
```

#### 1.2 Django Database Connection Helpers
**Features**:
- Helper function to create Moltres Database from Django's database connection
- Support for Django's database routing
- Integration with Django's transaction management

**API**:
```python
from moltres.integrations.django import get_moltres_db

# In views
def my_view(request):
    db = get_moltres_db(using='default')  # Supports database routing
    df = db.table("users").select()
    return JsonResponse({'users': df.collect()})
```

#### 1.3 Django Management Commands
**Features**:
- Management command for running DataFrame operations
- Query builder command for interactive exploration
- Data migration helpers using Moltres

**API**:
```bash
# Management command
python manage.py moltres_query "db.table('users').select().where(col('age') > 25)"
```

#### 1.4 Django Template Tags
**Features**:
- Template tags for querying data in templates
- Safe rendering of query results
- Caching support

**API**:
```django
{% load moltres_tags %}
{% moltres_query "users" as users %}
{% for user in users %}
    {{ user.name }}
{% endfor %}
```

### Implementation Steps

**Phase 1.1: Core Django Integration**
- [ ] Create `src/moltres/integrations/django.py`
- [ ] Implement `MoltresExceptionMiddleware`
- [ ] Add error handling for all Moltres exceptions
- [ ] Create tests in `tests/integrations/test_django_integration.py`

**Phase 1.2: Database Connection Helpers**
- [ ] Implement `get_moltres_db()` function
- [ ] Support Django's database routing
- [ ] Integration with Django transactions
- [ ] Add tests for connection helpers

**Phase 1.3: Management Commands**
- [ ] Create `src/moltres/integrations/django/management/commands/moltres_query.py`
- [ ] Implement query execution command
- [ ] Add interactive mode
- [ ] Add tests for management commands

**Phase 1.4: Template Tags**
- [ ] Create `src/moltres/integrations/django/templatetags/moltres_tags.py`
- [ ] Implement `moltres_query` template tag
- [ ] Add caching support
- [ ] Add tests for template tags

**Phase 1.5: Documentation and Examples**
- [ ] Create example file `docs/examples/23_django_integration.py`
- [ ] Update README with Django integration section
- [ ] Create guide `guides/13-django-integration.md`

### Dependencies
- `django>=3.2` (optional)

### Testing Strategy
- Unit tests for middleware, helpers, commands, and template tags
- Integration tests with Django test client
- Test database routing support
- Test transaction management

---

## Integration 2: Streamlit Integration

### Overview
Streamlit is a popular framework for building data applications. This integration will provide components and utilities for using Moltres DataFrames directly in Streamlit apps.

### Features to Implement

#### 2.1 Streamlit DataFrame Components
**File**: `src/moltres/integrations/streamlit.py`

**Features**:
- Custom Streamlit component for displaying Moltres DataFrames
- Automatic integration with `st.dataframe()` and `st.data_editor()`
- Query builder widget for interactive query construction
- Progress indicators for long-running queries

**API**:
```python
import streamlit as st
from moltres.integrations.streamlit import moltres_dataframe, query_builder

db = connect("sqlite:///example.db")

# Display DataFrame with automatic formatting
df = db.table("users").select()
moltres_dataframe(df)  # Enhanced display with query info

# Interactive query builder
query = query_builder(db)
if query:
    results = query.collect()
    st.dataframe(results)
```

#### 2.2 Streamlit Caching Integration
**Features**:
- Automatic caching of DataFrame operations using `@st.cache_data`
- Cache invalidation helpers
- Query result caching with TTL

**API**:
```python
from moltres.integrations.streamlit import cached_query

@cached_query(ttl=3600)
def get_user_stats():
    db = connect("sqlite:///example.db")
    return db.table("users").select().agg(...).collect()
```

#### 2.3 Streamlit Session State Helpers
**Features**:
- Helpers for managing database connections in Streamlit session state
- Automatic connection cleanup
- Multi-database support

**API**:
```python
from moltres.integrations.streamlit import get_db_from_session

# In Streamlit app
if 'db' not in st.session_state:
    st.session_state.db = connect("sqlite:///example.db")

db = get_db_from_session()
df = db.table("users").select()
```

#### 2.4 Streamlit Query Visualization
**Features**:
- Visual query builder component
- SQL query display and explanation
- Query performance metrics display

**API**:
```python
from moltres.integrations.streamlit import visualize_query

df = db.table("users").select().where(col("age") > 25)
visualize_query(df)  # Shows query plan, SQL, and performance metrics
```

### Implementation Steps

**Phase 2.1: Core Streamlit Components**
- [ ] Create `src/moltres/integrations/streamlit.py`
- [ ] Implement `moltres_dataframe()` component
- [ ] Implement `query_builder()` widget
- [ ] Add tests in `tests/integrations/test_streamlit_integration.py`

**Phase 2.2: Caching Integration**
- [ ] Implement `cached_query()` decorator
- [ ] Add cache invalidation helpers
- [ ] Support TTL and custom cache keys
- [ ] Add tests for caching

**Phase 2.3: Session State Helpers**
- [ ] Implement `get_db_from_session()` helper
- [ ] Add connection management utilities
- [ ] Support multi-database scenarios
- [ ] Add tests for session state management

**Phase 2.4: Query Visualization**
- [ ] Implement `visualize_query()` component
- [ ] Add query plan visualization
- [ ] Add performance metrics display
- [ ] Add tests for visualization

**Phase 2.5: Documentation and Examples**
- [ ] Create example file `docs/examples/25_streamlit_integration.py`
- [ ] Update README with Streamlit integration section
- [ ] Create guide `guides/14-streamlit-integration.md`

### Dependencies
- `streamlit>=1.28.0` (optional)

### Testing Strategy
- Unit tests for components and helpers
- Integration tests with Streamlit test framework
- Test caching behavior
- Test session state management

---

## Integration 3: Pytest Integration

### Overview
Pytest is the most popular Python testing framework. This integration will provide fixtures and utilities for testing applications that use Moltres.

### Features to Implement

#### 3.1 Pytest Fixtures for Database Connections
**File**: `src/moltres/integrations/pytest.py`

**Features**:
- Fixtures for creating test databases
- Automatic database setup and teardown
- Support for multiple database backends
- Transaction rollback for test isolation

**API**:
```python
import pytest
from moltres.integrations.pytest import moltres_db, moltres_async_db

def test_user_query(moltres_db):
    # moltres_db is a Database instance with a test database
    db = moltres_db
    db.create_table("users", [...])
    df = db.table("users").select()
    assert len(df.collect()) == 0

@pytest.mark.asyncio
async def test_async_query(moltres_async_db):
    db = await moltres_async_db
    await db.create_table("users", [...])
    df = (await db.table("users")).select()
    results = await df.collect()
    assert len(results) == 0
```

#### 3.2 Pytest Fixtures for Test Data
**Features**:
- Fixtures for loading test data from files
- Helpers for creating test DataFrames
- Data factories for generating test data

**API**:
```python
from moltres.integrations.pytest import test_data, create_test_df

def test_with_data(moltres_db, test_data):
    # test_data fixture loads data from test_data/ directory
    db = moltres_db
    db.create_table("users", test_data["users_schema"])
    Records(_data=test_data["users"], _database=db).insert_into("users")
    
    df = db.table("users").select()
    assert len(df.collect()) == len(test_data["users"])
```

#### 3.3 Pytest Assertions for DataFrames
**Features**:
- Custom assertions for DataFrame comparisons
- Schema validation assertions
- Query result assertions

**API**:
```python
from moltres.integrations.pytest import assert_dataframe_equal

def test_dataframe_comparison(moltres_db):
    df1 = moltres_db.table("users").select()
    df2 = moltres_db.table("users_backup").select()
    
    assert_dataframe_equal(df1, df2)  # Compares schemas and data
```

#### 3.4 Pytest Markers for Database Tests
**Features**:
- Custom markers for database-specific tests
- Skip markers for unsupported databases
- Performance test markers

**API**:
```python
import pytest

@pytest.mark.moltres_db("postgresql")
def test_postgresql_specific_feature(moltres_db):
    # Only runs if PostgreSQL is available
    pass

@pytest.mark.moltres_performance
def test_query_performance(moltres_db):
    # Performance test with timing
    pass
```

#### 3.5 Pytest Plugins for Query Logging
**Features**:
- Plugin to log all SQL queries during tests
- Query count assertions
- Query performance tracking

**API**:
```python
def test_query_logging(moltres_db, query_logger):
    df = moltres_db.table("users").select()
    df.collect()
    
    assert query_logger.count == 1
    assert "SELECT" in query_logger.queries[0]
```

### Implementation Steps

**Phase 3.1: Core Pytest Fixtures**
- [ ] Create `src/moltres/integrations/pytest.py`
- [ ] Implement `moltres_db` fixture
- [ ] Implement `moltres_async_db` fixture
- [ ] Add conftest.py for automatic fixture registration
- [ ] Add tests in `tests/integrations/test_pytest_integration.py`

**Phase 3.2: Test Data Fixtures**
- [ ] Implement `test_data` fixture
- [ ] Implement `create_test_df` helper
- [ ] Add data factory utilities
- [ ] Add tests for test data fixtures

**Phase 3.3: Custom Assertions**
- [ ] Implement `assert_dataframe_equal()` function
- [ ] Implement schema validation assertions
- [ ] Add query result assertions
- [ ] Add tests for assertions

**Phase 3.4: Pytest Markers**
- [ ] Register custom markers in `pytest.ini`
- [ ] Implement database-specific markers
- [ ] Implement performance test markers
- [ ] Add tests for markers

**Phase 3.5: Query Logging Plugin**
- [ ] Implement query logging plugin
- [ ] Add query count tracking
- [ ] Add performance tracking
- [ ] Add tests for query logging

**Phase 3.6: Documentation and Examples**
- [ ] Create example file `docs/examples/26_pytest_integration.py`
- [ ] Update README with Pytest integration section
- [ ] Create guide `guides/15-pytest-integration.md`
- [ ] Add to pytest best practices documentation

### Dependencies
- `pytest>=7.0.0` (optional, but commonly used)

### Testing Strategy
- Unit tests for fixtures and helpers
- Integration tests using pytest itself
- Test fixture isolation and cleanup
- Test marker functionality

---

## Integration 4: Airflow/Prefect Integration

### Overview
Airflow and Prefect are popular workflow orchestration tools. This integration will provide operators/tasks for using Moltres in data pipelines.

### Features to Implement

#### 4.1 Airflow Operators
**File**: `src/moltres/integrations/airflow.py`

**Features**:
- `MoltresQueryOperator` for executing DataFrame operations
- `MoltresToTableOperator` for writing DataFrames to tables
- `MoltresDataQualityOperator` for data validation
- Support for XComs (passing DataFrames between tasks)

**API**:
```python
from airflow import DAG
from moltres.integrations.airflow import (
    MoltresQueryOperator,
    MoltresToTableOperator,
    MoltresDataQualityOperator,
)

with DAG('moltres_pipeline', ...) as dag:
    query_task = MoltresQueryOperator(
        task_id='query_users',
        dsn='postgresql://...',
        query=lambda db: db.table("users").select().where(col("active") == True),
        output_key='active_users',
    )
    
    quality_check = MoltresDataQualityOperator(
        task_id='check_quality',
        dsn='postgresql://...',
        query=lambda db: db.table("users").select(),
        checks=[
            {'column': 'email', 'type': 'not_null'},
            {'column': 'age', 'type': 'range', 'min': 0, 'max': 150},
        ],
    )
    
    write_task = MoltresToTableOperator(
        task_id='write_results',
        dsn='postgresql://...',
        table_name='active_users_summary',
        input_key='active_users',
    )
    
    query_task >> quality_check >> write_task
```

#### 4.2 Prefect Tasks
**File**: `src/moltres/integrations/prefect.py`

**Features**:
- `moltres_query` task for executing DataFrame operations
- `moltres_to_table` task for writing DataFrames
- `moltres_data_quality` task for validation
- Integration with Prefect's result storage

**API**:
```python
from prefect import flow, task
from moltres.integrations.prefect import (
    moltres_query,
    moltres_to_table,
    moltres_data_quality,
)

@flow
def data_pipeline():
    # Query data
    users = moltres_query(
        dsn='postgresql://...',
        query=lambda db: db.table("users").select(),
    )
    
    # Quality check
    quality_result = moltres_data_quality(
        dsn='postgresql://...',
        query=lambda db: db.table("users").select(),
        checks=[...],
    )
    
    # Write results
    if quality_result.passed:
        moltres_to_table(
            dsn='postgresql://...',
            table_name='processed_users',
            data=users,
        )
```

#### 4.3 Data Quality Checks
**Features**:
- Built-in data quality check functions
- Custom check support
- Quality report generation

**API**:
```python
from moltres.integrations.airflow import DataQualityCheck

checks = [
    DataQualityCheck.column_not_null('email'),
    DataQualityCheck.column_range('age', min=0, max=150),
    DataQualityCheck.custom(lambda df: len(df) > 0),
]
```

#### 4.4 ETL Pipeline Helpers
**Features**:
- Extract helpers (read from various sources)
- Transform helpers (DataFrame operations)
- Load helpers (write to destinations)
- Pipeline templates

**API**:
```python
from moltres.integrations.airflow import ETLPipeline

pipeline = ETLPipeline(
    extract=lambda: read_from_source(),
    transform=lambda df: df.select(...).where(...),
    load=lambda df: df.write.save_as_table("target"),
)
```

### Implementation Steps

**Phase 4.1: Airflow Operators**
- [ ] Create `src/moltres/integrations/airflow.py`
- [ ] Implement `MoltresQueryOperator`
- [ ] Implement `MoltresToTableOperator`
- [ ] Implement `MoltresDataQualityOperator`
- [ ] Add XCom support
- [ ] Add tests in `tests/integrations/test_airflow_integration.py`

**Phase 4.2: Prefect Tasks**
- [ ] Create `src/moltres/integrations/prefect.py`
- [ ] Implement `moltres_query` task
- [ ] Implement `moltres_to_table` task
- [ ] Implement `moltres_data_quality` task
- [ ] Add result storage integration
- [ ] Add tests in `tests/integrations/test_prefect_integration.py`

**Phase 4.3: Data Quality Framework**
- [ ] Create `src/moltres/integrations/data_quality.py`
- [ ] Implement built-in check functions
- [ ] Implement custom check support
- [ ] Implement quality report generation
- [ ] Add tests for data quality checks

**Phase 4.4: ETL Pipeline Helpers**
- [ ] Implement ETL pipeline templates
- [ ] Add extract/transform/load helpers
- [ ] Add pipeline validation
- [ ] Add tests for ETL helpers

**Phase 4.5: Documentation and Examples**
- [ ] Create example file `docs/examples/27_airflow_integration.py`
- [ ] Create example file `docs/examples/28_prefect_integration.py`
- [ ] Update README with Airflow/Prefect integration sections
- [ ] Create guide `guides/16-workflow-integration.md`

### Dependencies
- `apache-airflow>=2.5.0` (optional)
- `prefect>=2.0.0` (optional)

### Testing Strategy
- Unit tests for operators and tasks
- Integration tests with Airflow/Prefect test frameworks
- Test XCom and result storage
- Test data quality checks
- Test ETL pipeline execution

---

## Common Patterns Across All Integrations

### Error Handling
All integrations should follow the FastAPI pattern:
- Convert Moltres exceptions to framework-appropriate errors
- Provide helpful error messages with suggestions
- Include context information

### Dependency Management
- All integrations are optional dependencies
- Graceful degradation when frameworks are not installed
- Clear error messages when dependencies are missing

### Testing
- Comprehensive test coverage for each integration
- Integration tests with the actual frameworks
- Test graceful degradation
- Test error handling

### Documentation
- Example files for each integration
- README updates with integration sections
- Guide documents for detailed usage
- API reference documentation

---

## Implementation Priority

### Phase 1: High Priority (Immediate Value)
1. **Pytest Integration** - Most widely used, immediate testing value
2. **Streamlit Integration** - Great for data apps, visual impact

### Phase 2: Medium Priority (High Impact)
3. **Django Integration** - Large user base, significant impact
4. **Airflow/Prefect Integration** - Important for data engineering workflows

---

## Success Criteria

For each integration:
- [ ] All core features implemented and tested
- [ ] Comprehensive test coverage (>90%)
- [ ] Example file demonstrating usage
- [ ] Documentation guide created
- [ ] README updated with integration section
- [ ] Error handling tested and working
- [ ] Graceful degradation when dependencies missing
- [ ] Performance acceptable (no significant overhead)

---

## Timeline Estimate

- **Pytest Integration**: 2-3 days
- **Streamlit Integration**: 3-4 days
- **Django Integration**: 4-5 days
- **Airflow/Prefect Integration**: 5-6 days

**Total**: ~15-18 days of development time

---

## Notes

- All integrations should follow the pattern established by FastAPI integration
- Maintain backward compatibility
- Keep integrations as optional dependencies
- Focus on developer experience and ease of use
- Provide clear error messages and helpful suggestions

