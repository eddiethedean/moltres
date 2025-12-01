"""Django Integration Examples with Moltres

This example demonstrates how to use Moltres with Django following
the standard Django patterns, leveraging Moltres's DataFrame operations
for data querying and manipulation.

Key features:
- Django middleware for automatic error handling
- Database connection helpers with Django database routing
- Management commands for query execution
- Template tags for querying data in templates
- Integration with Django's transaction management

IMPORTANT: This is a demonstration file. To use in a real Django project:
1. Install Django: pip install django
2. Install Moltres with Django support: pip install moltres[django]
3. Add 'moltres.integrations.django' to INSTALLED_APPS
4. Add MoltresExceptionMiddleware to MIDDLEWARE
5. Use get_moltres_db() in views
"""

# Example 1: Django Settings Configuration
# =========================================

try:
    from django.conf import settings

    # Configure Django settings (in a real project, this would be in settings.py)
    if not settings.configured:
        settings.configure(
            DEBUG=True,
            SECRET_KEY="django-insecure-example-key-change-in-production",
            DATABASES={
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": "example_django.db",
                }
            },
            INSTALLED_APPS=[
                "django.contrib.contenttypes",
                "django.contrib.auth",
                "moltres.integrations.django",  # Add Moltres Django integration
            ],
            MIDDLEWARE=[
                "django.middleware.security.SecurityMiddleware",
                "django.middleware.common.CommonMiddleware",
                "moltres.integrations.django.MoltresExceptionMiddleware",  # Add Moltres middleware
            ],
            ROOT_URLCONF="example_django_app",
            USE_TZ=True,
        )

    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    print("Django is not installed. Install with: pip install django")
    print("Or: pip install moltres[django]")


# Example 2: Django Views with Moltres
# =====================================

if DJANGO_AVAILABLE:
    from django.http import JsonResponse
    from django.views import View
    from moltres import col
    from moltres.expressions import functions as F
    from moltres.integrations.django import get_moltres_db
    from moltres.io.records import Records
    from moltres.table.schema import column

    class UserListView(View):
        """Django view using Moltres for data querying.

        This view demonstrates:
        - Using get_moltres_db() to get a Moltres Database instance
        - Querying data with Moltres DataFrame operations
        - Automatic error handling via middleware
        """

        def get(self, request):
            """Get list of users using Moltres."""
            # Get Moltres database connection
            db = get_moltres_db(using="default")

            # Query users with filtering
            df = db.table("users").select()

            # Apply filters from query parameters
            min_age = request.GET.get("min_age")
            if min_age:
                df = df.where(col("age") >= int(min_age))

            search = request.GET.get("search")
            if search:
                df = df.where(
                    (col("name").like(f"%{search}%")) | (col("email").like(f"%{search}%"))
                )

            # Execute query and return results
            results = df.collect()
            return JsonResponse({"users": results}, safe=False)

    class UserStatsView(View):
        """Django view for user statistics using Moltres aggregations."""

        def get(self, request):
            """Get user statistics."""
            db = get_moltres_db(using="default")

            df = db.table("users").select()

            # Calculate statistics
            stats_df = df.select(
                F.count(col("id")).alias("total_users"),
                F.avg(col("age")).alias("avg_age"),
                F.min(col("age")).alias("min_age"),
                F.max(col("age")).alias("max_age"),
            )

            stats = stats_df.collect()[0]
            return JsonResponse(
                {
                    "total_users": stats["total_users"],
                    "average_age": round(stats["avg_age"], 2),
                    "min_age": stats["min_age"],
                    "max_age": stats["max_age"],
                }
            )

    class UserCreateView(View):
        """Django view for creating users using Moltres."""

        def post(self, request):
            """Create a new user."""
            import json

            db = get_moltres_db(using="default")

            # Parse request data
            data = json.loads(request.body)
            name = data.get("name")
            email = data.get("email")
            age = data.get("age")

            # Ensure table exists (in production, use migrations)
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
                # Table might already exist
                pass

            # Insert user
            Records(
                _data=[{"name": name, "email": email, "age": age}],
                _database=db,
            ).insert_into("users")

            return JsonResponse({"status": "created", "name": name}, status=201)


# Example 3: Django Management Command Usage
# =========================================

"""
Django management commands allow executing Moltres queries from the command line.

Usage:
    python manage.py moltres_query "db.table('users').select()"
    python manage.py moltres_query "db.table('users').select().where(col('age') > 25)" --format=json
    python manage.py moltres_query --interactive

Options:
    --database: Django database alias (default: 'default')
    --format: Output format - json, table, or csv (default: table)
    --interactive: Start interactive query mode
    --file: Read query from file
"""

# Example 4: Django Template Tags
# ================================

"""
Django template tags allow querying data directly in templates.

In your template:
    {% load moltres_tags %}
    
    {% moltres_query "users" as users %}
    {% for user in users %}
        <div>{{ user.name }} - {{ user.email }}</div>
    {% endfor %}
    
    {% moltres_query query="db.table('users').select().where(col('active') == True)" cache_timeout=3600 as active_users %}
    <p>Active users: {{ active_users|moltres_format:"count" }}</p>

Template tag options:
    table_name: Simple table select
    query: Moltres query expression
    database: Django database alias (default: 'default')
    cache_timeout: Cache timeout in seconds (optional)
    cache_key: Custom cache key (optional)
"""

# Example 5: Django URL Configuration
# ==================================

if DJANGO_AVAILABLE:
    from django.urls import path

    # URL patterns (in a real project, this would be in urls.py)
    urlpatterns = [
        path("api/users/", UserListView.as_view(), name="user-list"),
        path("api/users/stats/", UserStatsView.as_view(), name="user-stats"),
        path("api/users/create/", UserCreateView.as_view(), name="user-create"),
    ]


# Example 6: Complete Django App Setup
# ====================================

"""
Complete setup for a Django app using Moltres:

1. Install dependencies:
    pip install django moltres[django]

2. Add to settings.py:
    INSTALLED_APPS = [
        # ... other apps
        'moltres.integrations.django',
    ]
    
    MIDDLEWARE = [
        # ... other middleware
        'moltres.integrations.django.MoltresExceptionMiddleware',
    ]

3. Use in views:
    from moltres.integrations.django import get_moltres_db
    from moltres import col
    
    def my_view(request):
        db = get_moltres_db(using='default')
        df = db.table("users").select().where(col("active") == True)
        results = df.collect()
        return JsonResponse({'users': results})

4. Use in templates:
    {% load moltres_tags %}
    {% moltres_query "users" as users %}
    {% for user in users %}
        {{ user.name }}
    {% endfor %}

5. Use management commands:
    python manage.py moltres_query "db.table('users').select()"
"""

# Example 7: Error Handling
# =========================

"""
Moltres exceptions are automatically converted to appropriate HTTP responses
by the MoltresExceptionMiddleware:

- DatabaseConnectionError → 503 Service Unavailable
- QueryTimeoutError → 504 Gateway Timeout
- ExecutionError (not found) → 404 Not Found
- ExecutionError (permission) → 403 Forbidden
- CompilationError → 400 Bad Request
- ValidationError → 400 Bad Request
- ColumnNotFoundError → 400 Bad Request
- TransactionError → 500 Internal Server Error
- MoltresError → 500 Internal Server Error

All errors include helpful messages and suggestions in the JSON response.
"""

# Example 8: Database Routing
# ===========================

"""
Moltres supports Django's database routing:

    # Use default database
    db = get_moltres_db(using='default')
    
    # Use another database
    db = get_moltres_db(using='read_replica')
    
    # In views with database routing
    def my_view(request):
        # Route to appropriate database based on request
        db_alias = 'read_replica' if request.method == 'GET' else 'default'
        db = get_moltres_db(using=db_alias)
        # ... use db
"""

# Example 9: Transaction Management
# =================================

"""
Moltres integrates with Django's transaction management:

    from django.db import transaction
    
    @transaction.atomic
    def create_user_with_orders(request):
        db = get_moltres_db(using='default')
        
        # All operations in this function are part of the same transaction
        Records(_data=[{"name": "Alice"}], _database=db).insert_into("users")
        Records(_data=[{"user_id": 1, "product": "Widget"}], _database=db).insert_into("orders")
        
        # If any operation fails, all are rolled back
"""

if __name__ == "__main__":
    print("Django Integration Examples")
    print("=" * 50)
    print("\nThis file demonstrates Django integration features.")
    print("\nRequired dependencies:")
    print("  pip install django")
    print("  Or: pip install moltres[django]")
    print("\nTo use in a real Django project:")
    print("1. Add 'moltres.integrations.django' to INSTALLED_APPS")
    print("2. Add MoltresExceptionMiddleware to MIDDLEWARE")
    print("3. Use get_moltres_db() in views")
    print("4. Use {% load moltres_tags %} in templates")
    print("\nSee the code comments for detailed examples.")
