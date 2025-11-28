# Django Integration Guide

This guide demonstrates how to integrate Moltres with Django, providing seamless DataFrame operations within Django applications.

## Overview

Moltres provides comprehensive Django integration with the following features:

1. **Middleware for Error Handling** - Automatic conversion of Moltres exceptions to Django HTTP responses
2. **Database Connection Helpers** - Easy integration with Django's database routing and transaction management
3. **Management Commands** - Command-line utilities for executing Moltres queries
4. **Template Tags** - Query data directly in Django templates with caching support

## Installation

Install Moltres with Django support:

```bash
pip install moltres[django]
```

Or install separately:

```bash
pip install moltres django
```

## Quick Start

### 1. Configure Django Settings

Add Moltres to your Django `settings.py`:

```python
INSTALLED_APPS = [
    # ... other apps
    'moltres.integrations.django',
]

MIDDLEWARE = [
    # ... other middleware
    'moltres.integrations.django.MoltresExceptionMiddleware',
]
```

### 2. Use in Views

```python
from django.http import JsonResponse
from django.views import View
from moltres.integrations.django import get_moltres_db
from moltres import col

class UserListView(View):
    def get(self, request):
        # Get Moltres database connection
        db = get_moltres_db(using='default')
        
        # Query data with Moltres
        df = db.table("users").select().where(col("active") == True)
        results = df.collect()
        
        return JsonResponse({'users': results}, safe=False)
```

### 3. Use in Templates

```django
{% load moltres_tags %}

{% moltres_query "users" as users %}
{% for user in users %}
    <div>{{ user.name }} - {{ user.email }}</div>
{% endfor %}
```

## Middleware for Error Handling

The `MoltresExceptionMiddleware` automatically converts Moltres exceptions to appropriate Django HTTP responses.

### Configuration

Add to `MIDDLEWARE` in `settings.py`:

```python
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.common.CommonMiddleware',
    'moltres.integrations.django.MoltresExceptionMiddleware',  # Add this
    # ... other middleware
]
```

### Error Mapping

The middleware maps Moltres exceptions to HTTP status codes:

- `DatabaseConnectionError` → 503 Service Unavailable
- `ConnectionPoolError` → 503 Service Unavailable
- `QueryTimeoutError` → 504 Gateway Timeout
- `ExecutionError` (not found) → 404 Not Found
- `ExecutionError` (permission) → 403 Forbidden
- `ExecutionError` (syntax error) → 400 Bad Request
- `ExecutionError` (generic) → 500 Internal Server Error
- `CompilationError` → 400 Bad Request
- `ValidationError` → 400 Bad Request
- `ColumnNotFoundError` → 400 Bad Request
- `TransactionError` → 500 Internal Server Error
- `MoltresError` → 500 Internal Server Error

### Example

```python
from django.http import JsonResponse
from moltres.integrations.django import get_moltres_db

def my_view(request):
    db = get_moltres_db(using='default')
    
    # If this raises a Moltres exception, the middleware will
    # automatically convert it to an appropriate HTTP response
    df = db.table("nonexistent_table").select()
    results = df.collect()  # Raises ExecutionError → 404 response
    
    return JsonResponse({'data': results})
```

## Database Connection Helpers

### Basic Usage

The `get_moltres_db()` function creates a Moltres Database instance from Django's database configuration:

```python
from moltres.integrations.django import get_moltres_db
from moltres import col

def my_view(request):
    # Get database connection (uses 'default' database)
    db = get_moltres_db(using='default')
    
    # Use Moltres operations
    df = db.table("users").select().where(col("age") > 25)
    results = df.collect()
    
    return JsonResponse({'users': results}, safe=False)
```

### Database Routing

Moltres supports Django's database routing:

```python
from moltres.integrations.django import get_moltres_db

def my_view(request):
    # Route to read replica for GET requests
    db_alias = 'read_replica' if request.method == 'GET' else 'default'
    db = get_moltres_db(using=db_alias)
    
    df = db.table("users").select()
    results = df.collect()
    
    return JsonResponse({'users': results}, safe=False)
```

### Supported Databases

The integration supports the following Django database backends:

- **SQLite** - `django.db.backends.sqlite3`
- **PostgreSQL** - `django.db.backends.postgresql` or `django.contrib.gis.db.backends.postgis`
- **MySQL** - `django.db.backends.mysql`

For other databases, use `connect()` directly with a DSN string.

### Transaction Management

Moltres integrates with Django's transaction management:

```python
from django.db import transaction
from moltres.integrations.django import get_moltres_db
from moltres.io.records import Records

@transaction.atomic
def create_user_with_orders(request):
    db = get_moltres_db(using='default')
    
    # All operations are part of the same transaction
    Records(
        _data=[{"name": "Alice", "email": "alice@example.com"}],
        _database=db,
    ).insert_into("users")
    
    Records(
        _data=[{"user_id": 1, "product": "Widget", "amount": 100.0}],
        _database=db,
    ).insert_into("orders")
    
    # If any operation fails, all are rolled back
```

## Management Commands

### Basic Usage

Execute Moltres queries from the command line:

```bash
python manage.py moltres_query "db.table('users').select()"
```

### Options

```bash
# Use different database
python manage.py moltres_query "db.table('users').select()" --database=read_replica

# Output as JSON
python manage.py moltres_query "db.table('users').select()" --format=json

# Output as CSV
python manage.py moltres_query "db.table('users').select()" --format=csv

# Interactive mode
python manage.py moltres_query --interactive

# Read query from file
python manage.py moltres_query --file=query.txt
```

### Interactive Mode

Interactive mode allows you to explore data interactively:

```bash
$ python manage.py moltres_query --interactive
Moltres Interactive Query Mode
Type 'exit' or 'quit' to exit
Type 'help' for help

moltres> db.table("users").select()
id | name  | email
---|-------|----------
1  | Alice | alice@example.com
2  | Bob   | bob@example.com

moltres> db.table("users").select().where(col("age") > 25)
id | name | email
---|------|----------
1  | Alice| alice@example.com

moltres> exit
```

### Example Queries

```bash
# Simple select
python manage.py moltres_query "db.table('users').select()"

# With filtering
python manage.py moltres_query "db.table('users').select().where(col('active') == True)"

# With aggregation
python manage.py moltres_query "db.table('orders').select().group_by('status').agg(F.count(col('id')).alias('count'))"

# Complex query
python manage.py moltres_query "db.table('users').select().join(db.table('orders').select(), on=[col('users.id') == col('orders.user_id')])"
```

## Template Tags

### Basic Usage

Load the template tags and query data:

```django
{% load moltres_tags %}

{% moltres_query "users" as users %}
{% for user in users %}
    <div>{{ user.name }} - {{ user.email }}</div>
{% endfor %}
```

### With Filtering

Use a custom query expression:

```django
{% load moltres_tags %}

{% moltres_query query="db.table('users').select().where(col('active') == True)" as active_users %}
<p>Active users: {{ active_users|length }}</p>
```

### Caching

Cache query results for performance:

```django
{% load moltres_tags %}

{# Cache for 1 hour #}
{% moltres_query "users" cache_timeout=3600 as users %}
{% for user in users %}
    {{ user.name }}
{% endfor %}
```

### Custom Cache Key

Use a custom cache key:

```django
{% load moltres_tags %}

{% moltres_query "users" cache_timeout=3600 cache_key="active_users_list" as users %}
```

### Formatting Results

Use the `moltres_format` filter to format results:

```django
{% load moltres_tags %}

{% moltres_query "users" as users %}

{# Count #}
<p>Total users: {{ users|moltres_format:"count" }}</p>

{# First user #}
<p>First user: {{ users|moltres_format:"first" }}</p>

{# Last user #}
<p>Last user: {{ users|moltres_format:"last" }}</p>

{# JSON #}
<pre>{{ users|moltres_format:"json" }}</pre>
```

### Template Tag Options

- `table_name` - Simple table select (e.g., `"users"`)
- `query` - Moltres query expression (e.g., `"db.table('users').select().where(col('age') > 25)"`)
- `database` - Django database alias (default: `"default"`)
- `cache_timeout` - Cache timeout in seconds (optional)
- `cache_key` - Custom cache key (optional, auto-generated if not provided)

## Complete Example

Here's a complete example of a Django app using Moltres:

### settings.py

```python
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'moltres.integrations.django',  # Add Moltres
    'myapp',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'moltres.integrations.django.MoltresExceptionMiddleware',  # Add Moltres middleware
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
```

### views.py

```python
from django.http import JsonResponse
from django.views import View
from moltres.integrations.django import get_moltres_db
from moltres import col
from moltres.expressions import functions as F
from moltres.io.records import Records
from moltres.table.schema import column

class UserListView(View):
    def get(self, request):
        db = get_moltres_db(using='default')
        
        # Query with filtering
        df = db.table("users").select()
        
        # Apply filters from query parameters
        min_age = request.GET.get("min_age")
        if min_age:
            df = df.where(col("age") >= int(min_age))
        
        search = request.GET.get("search")
        if search:
            df = df.where(
                (col("name").like(f"%{search}%")) | 
                (col("email").like(f"%{search}%"))
            )
        
        results = df.collect()
        return JsonResponse({'users': results}, safe=False)

class UserStatsView(View):
    def get(self, request):
        db = get_moltres_db(using='default')
        
        df = db.table("users").select()
        
        # Calculate statistics
        stats_df = df.select(
            F.count(col("id")).alias("total_users"),
            F.avg(col("age")).alias("avg_age"),
            F.min(col("age")).alias("min_age"),
            F.max(col("age")).alias("max_age"),
        )
        
        stats = stats_df.collect()[0]
        return JsonResponse({
            'total_users': stats['total_users'],
            'average_age': round(stats['avg_age'], 2),
            'min_age': stats['min_age'],
            'max_age': stats['max_age'],
        })

class UserCreateView(View):
    def post(self, request):
        import json
        
        db = get_moltres_db(using='default')
        data = json.loads(request.body)
        
        # Ensure table exists (use migrations in production)
        try:
            db.create_table(
                "users",
                [
                    column("id", "INTEGER", primary_key=True),
                    column("name", "TEXT"),
                    column("email", "TEXT"),
                    column("age", "INTEGER"),
                ],
            ).collect()
        except Exception:
            pass  # Table might already exist
        
        # Insert user
        Records(
            _data=[{
                "name": data.get("name"),
                "email": data.get("email"),
                "age": data.get("age"),
            }],
            _database=db,
        ).insert_into("users")
        
        return JsonResponse({'status': 'created'}, status=201)
```

### template.html

```django
{% load moltres_tags %}

<h1>Users</h1>

{% moltres_query "users" cache_timeout=3600 as users %}
<ul>
{% for user in users %}
    <li>{{ user.name }} ({{ user.email }}) - Age: {{ user.age }}</li>
{% endfor %}
</ul>

<p>Total users: {{ users|moltres_format:"count" }}</p>
```

### urls.py

```python
from django.urls import path
from . import views

urlpatterns = [
    path('api/users/', views.UserListView.as_view(), name='user-list'),
    path('api/users/stats/', views.UserStatsView.as_view(), name='user-stats'),
    path('api/users/create/', views.UserCreateView.as_view(), name='user-create'),
]
```

## Best Practices

### 1. Use Database Routing

Route read queries to read replicas:

```python
def my_view(request):
    db_alias = 'read_replica' if request.method == 'GET' else 'default'
    db = get_moltres_db(using=db_alias)
    # ... use db
```

### 2. Cache Template Queries

Cache expensive queries in templates:

```django
{% moltres_query "expensive_query" cache_timeout=3600 as results %}
```

### 3. Use Transactions

Wrap related operations in transactions:

```python
from django.db import transaction

@transaction.atomic
def create_user_with_profile(request):
    db = get_moltres_db(using='default')
    # ... multiple operations
```

### 4. Error Handling

The middleware handles errors automatically, but you can also handle them manually:

```python
from moltres.utils.exceptions import ExecutionError

def my_view(request):
    try:
        db = get_moltres_db(using='default')
        df = db.table("users").select()
        results = df.collect()
    except ExecutionError as e:
        # Handle specific error
        return JsonResponse({'error': str(e)}, status=404)
    return JsonResponse({'users': results}, safe=False)
```

### 5. Use Management Commands for Data Exploration

Use the management command for interactive data exploration:

```bash
python manage.py moltres_query --interactive
```

## Troubleshooting

### ImportError: Django is required

If you see this error, install Django:

```bash
pip install django
```

Or install Moltres with Django support:

```bash
pip install moltres[django]
```

### Database Connection Errors

Ensure your Django database settings are correct:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',  # or postgresql, mysql
        'NAME': 'your_database_name',
        # ... other settings
    }
}
```

### Template Tag Not Found

Ensure you've loaded the template tags:

```django
{% load moltres_tags %}
```

### Management Command Not Found

Ensure Moltres Django integration is in `INSTALLED_APPS`:

```python
INSTALLED_APPS = [
    # ... other apps
    'moltres.integrations.django',
]
```

## See Also

- [FastAPI Integration Guide](11-sqlalchemy-integration.md) - Similar patterns for FastAPI
- [SQLAlchemy Integration Guide](11-sqlalchemy-integration.md) - Using Moltres with SQLAlchemy
- [Examples](../examples/23_django_integration.py) - Complete Django integration examples

