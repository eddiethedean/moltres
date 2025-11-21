# Deployment Guide

This guide covers best practices for deploying Moltres in production environments.

## Pre-Deployment Checklist

- [ ] Database connection strings configured securely
- [ ] Connection pooling configured appropriately
- [ ] Query timeouts set
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Performance monitoring enabled
- [ ] Indexes created on frequently queried columns
- [ ] Database backups configured
- [ ] Environment variables set
- [ ] Dependencies pinned

## Configuration

### Environment Variables

Use environment variables for configuration (12-factor app principle):

```bash
# Database connection
export MOLTRES_DSN="postgresql://user:pass@host/dbname"

# Connection pooling
export MOLTRES_POOL_SIZE=10
export MOLTRES_MAX_OVERFLOW=5
export MOLTRES_POOL_TIMEOUT=30
export MOLTRES_POOL_RECYCLE=3600
export MOLTRES_POOL_PRE_PING=true

# Query settings
export MOLTRES_QUERY_TIMEOUT=30.0

# Logging
export MOLTRES_ECHO=false  # Set to true for debugging only
```

### Application Configuration

```python
from moltres import connect
import os

# Load from environment
db = connect(
    dsn=os.environ.get("MOLTRES_DSN"),
    pool_size=int(os.environ.get("MOLTRES_POOL_SIZE", "10")),
    max_overflow=int(os.environ.get("MOLTRES_MAX_OVERFLOW", "5")),
    query_timeout=float(os.environ.get("MOLTRES_QUERY_TIMEOUT", "30.0")),
    pool_pre_ping=True,
    echo=False,  # Disable in production
)
```

## Connection Management

### Connection Pooling

Configure connection pooling based on your workload:

```python
# For web applications (many concurrent requests)
db = connect(
    "postgresql://user:pass@host/dbname",
    pool_size=20,        # Larger pool for concurrency
    max_overflow=10,     # Allow overflow
    pool_timeout=30,     # Wait for connection
    pool_recycle=3600,   # Recycle after 1 hour
    pool_pre_ping=True,  # Verify connections
)

# For batch processing (fewer concurrent requests)
db = connect(
    "postgresql://user:pass@host/dbname",
    pool_size=5,         # Smaller pool
    max_overflow=2,      # Minimal overflow
    pool_timeout=60,     # Longer timeout
    pool_recycle=7200,   # Recycle after 2 hours
    pool_pre_ping=True,
)
```

### Connection Lifecycle

```python
# Application startup
db = connect("postgresql://user:pass@host/dbname")

# Application shutdown
db.close()  # Close all connections
```

## Error Handling

### Production Error Handling

```python
from moltres.utils.exceptions import (
    ExecutionError,
    CompilationError,
    DatabaseConnectionError,
    QueryTimeoutError,
)

def safe_query(df):
    try:
        return df.collect()
    except QueryTimeoutError as e:
        logger.error(f"Query timeout: {e}")
        # Return empty result or cached result
        return []
    except ExecutionError as e:
        logger.error(f"Query failed: {e}")
        # Log and handle gracefully
        raise
    except DatabaseConnectionError as e:
        logger.error(f"Connection failed: {e}")
        # Retry or use fallback
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise
```

### Retry Logic

```python
import time
from functools import wraps

def retry_on_connection_error(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except DatabaseConnectionError:
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    raise
        return wrapper
    return decorator

@retry_on_connection_error(max_retries=3)
def get_users():
    return db.table("users").select().collect()
```

## Logging

### Configure Logging

```python
import logging

# Configure Moltres logging
logging.getLogger("moltres").setLevel(logging.WARNING)

# Log SQL queries (for debugging)
logging.getLogger("moltres.engine").setLevel(logging.DEBUG)
```

### Custom Logging

```python
from moltres.engine import register_performance_hook
import logging

logger = logging.getLogger(__name__)

def log_slow_queries(sql: str, elapsed: float, metadata: dict):
    if elapsed > 1.0:
        logger.warning(
            f"Slow query ({elapsed:.2f}s): {sql[:200]}",
            extra={
                "query_time": elapsed,
                "rowcount": metadata.get("rowcount"),
            }
        )

register_performance_hook("query_end", log_slow_queries)
```

## Monitoring

### Health Checks

```python
def health_check():
    """Check database connectivity."""
    try:
        db.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Metrics Collection

```python
from moltres.engine import register_performance_hook

query_metrics = {
    "total_queries": 0,
    "total_time": 0.0,
    "slow_queries": 0,
}

def collect_metrics(sql: str, elapsed: float, metadata: dict):
    query_metrics["total_queries"] += 1
    query_metrics["total_time"] += elapsed
    if elapsed > 1.0:
        query_metrics["slow_queries"] += 1

register_performance_hook("query_end", collect_metrics)

# Expose metrics (e.g., for Prometheus)
def get_metrics():
    return {
        "queries_total": query_metrics["total_queries"],
        "queries_duration_seconds": query_metrics["total_time"],
        "slow_queries_total": query_metrics["slow_queries"],
    }
```

## Security

### Connection String Security

**Never commit credentials:**

```python
# Bad: Hardcoded credentials
db = connect("postgresql://user:password@host/dbname")  # DON'T DO THIS

# Good: Environment variables
db = connect(os.environ["MOLTRES_DSN"])

# Good: Configuration file (not in version control)
import json
with open("config.json") as f:
    config = json.load(f)
db = connect(config["database"]["dsn"])
```

### SQL Injection Prevention

Moltres prevents SQL injection by using parameterized queries:

```python
# Good: Safe - uses parameterized queries
user_id = 123
df = db.table("users").select().where(col("id") == user_id)

# Bad: Vulnerable (but Moltres doesn't support this anyway)
# Don't use raw SQL with string formatting
```

## Database Setup

### Indexes

Create indexes before deployment:

```python
# Create indexes for production
indexes = [
    "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)",
    "CREATE INDEX IF NOT EXISTS idx_orders_customer_id ON orders(customer_id)",
    "CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)",
    "CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(date)",
]

for index_sql in indexes:
    try:
        db.execute(index_sql)
    except Exception as e:
        logger.warning(f"Failed to create index: {e}")
```

### Database Migrations

Use a migration tool (Alembic, Django migrations, etc.) for schema changes:

```python
# Example: Run migrations before application starts
def run_migrations():
    import subprocess
    subprocess.run(["alembic", "upgrade", "head"])
```

## Deployment Strategies

### Blue-Green Deployment

1. Deploy new version to green environment
2. Test with production database (read-only)
3. Switch traffic to green
4. Keep blue as rollback option

### Canary Deployment

1. Deploy to small percentage of instances
2. Monitor for errors
3. Gradually increase percentage
4. Full rollout if successful

### Rolling Deployment

1. Deploy to one instance at a time
2. Verify health after each deployment
3. Continue until all instances updated

## Scaling

### Horizontal Scaling

- Run multiple application instances
- Each instance has its own connection pool
- Database handles connection limits
- Use load balancer for distribution

### Vertical Scaling

- Increase database resources (CPU, memory)
- Optimize queries and indexes
- Increase connection pool size if needed

### Database Read Replicas

```python
# Primary for writes
write_db = connect("postgresql://user:pass@primary/dbname")

# Replica for reads
read_db = connect("postgresql://user:pass@replica/dbname")

# Use appropriate database
def get_users():
    return read_db.table("users").select().collect()

def create_user(data):
    write_db.table("users").insert(data)
```

## Backup and Recovery

### Database Backups

Ensure database backups are configured:
- Automated daily backups
- Point-in-time recovery
- Test restore procedures

### Application State

- Log important operations
- Use idempotent operations where possible
- Implement retry logic for transient failures

## Performance Tuning

### Query Optimization

1. **Create indexes** on frequently queried columns
2. **Filter early** in query chains
3. **Use LIMIT** for exploratory queries
4. **Monitor slow queries** with performance hooks

See [Performance Guide](./PERFORMANCE.md) for details.

### Connection Pool Tuning

Monitor connection pool metrics:
- Active connections
- Idle connections
- Connection wait time
- Connection errors

Adjust pool size based on metrics.

## Troubleshooting

### Common Issues

1. **Connection pool exhausted**
   - Increase `pool_size` and `max_overflow`
   - Check for connection leaks
   - Reduce connection timeout

2. **Query timeouts**
   - Optimize queries (add indexes)
   - Increase `query_timeout` if appropriate
   - Break queries into smaller chunks

3. **Database connection failures**
   - Check network connectivity
   - Verify credentials
   - Check database server status

See [Debugging Guide](./DEBUGGING.md) for more help.

## Summary

1. **Use environment variables** for configuration
2. **Configure connection pooling** appropriately
3. **Implement error handling** and retry logic
4. **Set up logging** and monitoring
5. **Secure credentials** (never commit)
6. **Create indexes** before deployment
7. **Monitor performance** and adjust
8. **Plan for scaling** (horizontal/vertical)
9. **Test backups** and recovery
10. **Document deployment** procedures

For more information:
- [Performance Guide](./PERFORMANCE.md)
- [Best Practices](./BEST_PRACTICES.md)
- [Debugging Guide](./DEBUGGING.md)
- [Troubleshooting Guide](./TROUBLESHOOTING.md)

