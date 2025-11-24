# Operational Runbooks

This document provides operational procedures for running Moltres in production environments.

## Deployment

### Pre-Deployment Checklist

- [ ] All tests pass (`pytest`)
- [ ] Type checking passes (`mypy src`)
- [ ] Linting passes (`ruff check .`)
- [ ] Documentation is up to date
- [ ] Changelog is updated
- [ ] Version number is bumped
- [ ] Dependencies are reviewed for security issues
- [ ] Performance benchmarks are within acceptable ranges

### Deployment Steps

1. **Build Package**
   ```bash
   python -m build
   ```

2. **Test Installation**
   ```bash
   pip install dist/moltres-*.whl
   python -c "import moltres; print(moltres.__version__)"
   ```

3. **Upload to PyPI**
   ```bash
   twine upload dist/*
   ```

4. **Verify Release**
   ```bash
   pip install --upgrade moltres
   python -c "import moltres; print(moltres.__version__)"
   ```

5. **Create GitHub Release**
   - Tag the release: `git tag vX.Y.Z`
   - Push tag: `git push origin vX.Y.Z`
   - Create release on GitHub with changelog

## Schema Migrations

### Adding New Tables

1. **Create Migration Script**
   ```python
from moltres import connect
from moltres.table.schema import column

db = connect(os.getenv("DATABASE_URL"))

# Create new table
db.create_table(
    "new_table",
    [
        column("id", "INTEGER", primary_key=True),
        column("name", "TEXT"),
    ],
    if_not_exists=True,
).collect()
   ```

2. **Test Migration**
   - Test on staging environment first
   - Verify table structure
   - Test data insertion/retrieval

3. **Apply to Production**
   - Run during maintenance window
   - Monitor for errors
   - Verify table creation

### Modifying Existing Tables

1. **Add Columns**
   ```python
# Use ALTER TABLE (database-specific)
db.sql("ALTER TABLE existing_table ADD COLUMN new_column TEXT").collect()
   ```

2. **Drop Columns**
   ```python
# Use ALTER TABLE (database-specific)
db.sql("ALTER TABLE existing_table DROP COLUMN old_column").collect()
   ```

3. **Rename Columns**
   ```python
# Use ALTER TABLE (database-specific)
db.sql("ALTER TABLE existing_table RENAME COLUMN old_name TO new_name").collect()
   ```

### Migration Best Practices

1. **Backup First**: Always backup database before migrations
2. **Test on Staging**: Test all migrations on staging environment
3. **Rollback Plan**: Have rollback scripts ready
4. **Monitor**: Watch for errors during migration
5. **Document**: Document all schema changes

## Incident Response

### Database Connection Issues

**Symptoms:**
- Connection timeouts
- "Connection refused" errors
- Pool exhaustion errors

**Diagnosis:**
```python
from moltres.utils.health import check_connection_health, check_pool_health

# Check connection health
result = check_connection_health(db)
print(result)

# Check pool health
pool_result = check_pool_health(db)
print(pool_result)
```

**Resolution:**
1. Check database server status
2. Verify network connectivity
3. Check firewall rules
4. Review connection pool settings
5. Increase pool size if needed:
   ```python
   db = connect(
       dsn,
       pool_size=20,  # Increase from default
       max_overflow=40,
   )
   ```

### Query Performance Issues

**Symptoms:**
- Slow query execution
- Timeout errors
- High CPU usage

**Diagnosis:**
```python
# Enable query logging
from moltres import register_performance_hook

def log_slow_queries(sql: str, elapsed: float, metadata: dict):
    if elapsed > 1.0:
        print(f"Slow query ({elapsed:.2f}s): {sql[:200]}")

register_performance_hook("query_end", log_slow_queries)
```

**Resolution:**
1. Analyze slow queries with EXPLAIN
2. Add indexes on frequently queried columns
3. Optimize query patterns
4. Consider query caching
5. Scale database resources if needed

### Memory Issues

**Symptoms:**
- Out of memory errors
- High memory usage
- Process crashes

**Diagnosis:**
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024} MB")
```

**Resolution:**
1. Use streaming for large datasets:
   ```python
# Use streaming mode
df = db.read.csv("large_file.csv", stream=True)
   ```
2. Process data in chunks
3. Close database connections when done
4. Monitor memory usage
5. Scale application resources

### Data Corruption

**Symptoms:**
- Unexpected query results
- Constraint violations
- Data inconsistencies

**Diagnosis:**
```python
# Verify data integrity
result = db.sql("SELECT COUNT(*) FROM table_name").collect()
print(f"Row count: {result[0]['count']}")

# Check for duplicates
result = db.sql("""
    SELECT column_name, COUNT(*) as count
    FROM table_name
    GROUP BY column_name
    HAVING COUNT(*) > 1
""").collect()
```

**Resolution:**
1. Identify corrupted data
2. Restore from backup if needed
3. Fix data inconsistencies
4. Add data validation
5. Implement data integrity checks

## Monitoring

### Key Metrics to Monitor

1. **Query Performance**
   - Average query duration
   - P95/P99 query latency
   - Slow query count

2. **Connection Pool**
   - Active connections
   - Pool utilization
   - Connection wait time

3. **Error Rates**
   - Query error rate
   - Connection error rate
   - Timeout rate

4. **Resource Usage**
   - Memory usage
   - CPU usage
   - Network I/O

### Setting Up Monitoring

```python
from moltres.utils.telemetry import get_metrics_collector, get_structured_logger

# Get metrics
metrics = get_metrics_collector()
print(metrics.get_metrics())

# Structured logging
logger = get_structured_logger()
# Logs are automatically structured JSON
```

### Alerting Thresholds

- **Query Duration**: Alert if P95 > 5 seconds
- **Error Rate**: Alert if error rate > 1%
- **Connection Pool**: Alert if utilization > 80%
- **Memory Usage**: Alert if usage > 80% of available

## Backup and Recovery

### Backup Procedures

1. **Database Backups**
   ```bash
   # PostgreSQL
   pg_dump -h host -U user -d dbname > backup.sql
   
   # MySQL
   mysqldump -h host -u user -p dbname > backup.sql
   
   # SQLite
   sqlite3 database.db ".backup backup.db"
   ```

2. **Application Data Backups**
   - Export critical data to files
   - Store backups in secure location
   - Encrypt sensitive backups

### Recovery Procedures

1. **Restore from Backup**
   ```bash
   # PostgreSQL
   psql -h host -U user -d dbname < backup.sql
   
   # MySQL
   mysql -h host -u user -p dbname < backup.sql
   
   # SQLite
   cp backup.db database.db
   ```

2. **Point-in-Time Recovery**
   - Use database transaction logs
   - Restore to specific timestamp
   - Verify data integrity

## Scaling

### Horizontal Scaling

1. **Read Replicas**
   - Set up read replicas for read-heavy workloads
   - Route read queries to replicas
   - Keep writes on primary

2. **Connection Pooling**
   - Increase pool size for more connections
   - Use connection pooler (e.g., PgBouncer)

### Vertical Scaling

1. **Database Resources**
   - Increase CPU/memory for database server
   - Add faster storage (SSD)
   - Optimize database configuration

2. **Application Resources**
   - Increase application server resources
   - Use multiple application instances
   - Load balance requests

## Maintenance Windows

### Scheduled Maintenance

1. **Database Maintenance**
   - Run VACUUM/OPTIMIZE (database-specific)
   - Update statistics
   - Check for corruption

2. **Application Maintenance**
   - Deploy updates
   - Restart services
   - Clear caches

### Maintenance Checklist

- [ ] Notify users of maintenance window
- [ ] Backup database
- [ ] Put application in maintenance mode
- [ ] Perform maintenance tasks
- [ ] Verify system health
- [ ] Remove maintenance mode
- [ ] Monitor for issues

## Troubleshooting

### Common Issues

1. **"Connection pool exhausted"**
   - Increase pool size
   - Check for connection leaks
   - Use connection pooler

2. **"Query timeout"**
   - Optimize slow queries
   - Increase query timeout
   - Add indexes

3. **"Table does not exist"**
   - Verify table name
   - Check schema/database
   - Verify permissions

4. **"Permission denied"**
   - Check database user permissions
   - Verify table access
   - Check firewall rules

### Debug Mode

Enable debug logging:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("moltres").setLevel(logging.DEBUG)
```

### Getting Help

1. Check documentation
2. Search GitHub issues
3. Review logs
4. Contact support (if available)

