"""Django template tags for Moltres.

This module provides template tags for querying data in Django templates.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

try:
    from django import template  # type: ignore[import-untyped]
    from django.core.cache import cache  # type: ignore[import-untyped]
    from django.core.exceptions import ImproperlyConfigured  # type: ignore[import-untyped]
    from django.db import connections  # type: ignore[import-untyped]

    DJANGO_AVAILABLE = True
except ImportError:
    DJANGO_AVAILABLE = False
    template = None
    cache = None
    connections = None
    ImproperlyConfigured = None

logger = logging.getLogger(__name__)

if DJANGO_AVAILABLE:
    register = template.Library()
else:
    # Create a mock register object for when Django is not available
    class MockRegister:
        def simple_tag(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

        def filter(self, *args: Any, **kwargs: Any) -> Any:
            def decorator(func: Any) -> Any:
                return func

            return decorator

    register = MockRegister()


if DJANGO_AVAILABLE:

    @register.simple_tag(takes_context=True)
    def moltres_query(
        context: dict,
        table_name: Optional[str] = None,
        query: Optional[str] = None,
        database: str = "default",
        cache_timeout: Optional[int] = None,
        cache_key: Optional[str] = None,
    ) -> Any:
        """Execute a Moltres query in a Django template.

        This template tag allows you to query data using Moltres directly in Django templates.
        Results can be cached for performance.

        Args:
            context: Django template context
            table_name: Name of the table to query (simple select)
            query: Moltres query expression (e.g., "db.table('users').select().where(col('age') > 25)")
            database: Django database alias (default: 'default')
            cache_timeout: Cache timeout in seconds (None = no caching)
            cache_key: Custom cache key (auto-generated if not provided)

        Returns:
            Query results (list of dicts)

        Example:
            {% load moltres_tags %}
            {% moltres_query "users" as users %}
            {% for user in users %}
                {{ user.name }}
            {% endfor %}

            {% moltres_query query="db.table('users').select().where(col('active') == True)" as active_users %}
        """
        if not DJANGO_AVAILABLE:
            logger.error("Django is not available for moltres_query template tag")
            return []

        # Check cache if caching is enabled
        if cache_timeout is not None:
            if cache_key is None:
                # Generate cache key from query parameters
                cache_key = f"moltres_query:{database}:{table_name or query}:{cache_timeout}"
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        try:
            # Get Moltres database connection
            from moltres.integrations.django import get_moltres_db

            db = get_moltres_db(using=database)
        except ImportError as e:
            logger.error(f"Failed to import Moltres Django integration: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to create Moltres database connection: {e}")
            return []

        try:
            # Execute query
            if query:
                # Execute custom query expression
                from moltres import col
                from moltres.expressions import functions as F

                namespace = {
                    "db": db,
                    "col": col,
                    "F": F,
                    "__builtins__": __builtins__,
                }

                result = eval(query, namespace)  # noqa: S307

                # If result is a DataFrame, collect it
                if hasattr(result, "collect"):
                    results = result.collect()
                else:
                    results = result
            elif table_name:
                # Simple table select
                df = db.table(table_name).select()
                results = df.collect()
            else:
                logger.error("Either 'table_name' or 'query' must be provided")
                return []

            # Cache results if caching is enabled
            if cache_timeout is not None and cache_key:
                cache.set(cache_key, results, cache_timeout)

            return results
        except Exception as e:
            logger.error(f"Moltres query execution failed: {e}", exc_info=True)
            # Return empty list on error to prevent template errors
            return []
else:

    def moltres_query(*args: Any, **kwargs: Any) -> list:
        return []


if DJANGO_AVAILABLE:

    @register.filter
    def moltres_format(value: Any, format_type: str = "json") -> str:
        """Format Moltres query results for display.

        Args:
            value: Query results to format
            format_type: Format type ('json', 'count', 'first', 'last')

        Returns:
            Formatted string

        Example:
            {% moltres_query "users" as users %}
            {{ users|moltres_format:"count" }}
        """
        if not DJANGO_AVAILABLE:
            return str(value)

        import json

        if format_type == "json":
            return json.dumps(value, indent=2, default=str)
        elif format_type == "count":
            if isinstance(value, list):
                return str(len(value))
            return "0"
        elif format_type == "first":
            if isinstance(value, list) and value:
                return str(value[0])
            return ""
        elif format_type == "last":
            if isinstance(value, list) and value:
                return str(value[-1])
            return ""
        else:
            return str(value)
else:

    def moltres_format(value: Any, format_type: str = "json") -> str:
        return str(value)
