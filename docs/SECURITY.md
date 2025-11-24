# Security Guide

This guide covers security best practices for using Moltres in production environments.

## Secure DSN Handling

### Never Hardcode Credentials

**❌ Bad:**
```python
db = connect("postgresql://user:password@host/dbname")
```

**✅ Good:**
```python
import os
dsn = os.getenv("DATABASE_URL")
db = connect(dsn)
```

### Use Environment Variables

Store database credentials in environment variables:

```bash
# .env file (never commit to version control)
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file
db = connect(os.getenv("DATABASE_URL"))
```

### Use Secret Management Services

For production, use secret management services:

**AWS Secrets Manager:**
```python
import boto3
import json

def get_database_dsn():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='prod/database')
    secret = json.loads(response['SecretString'])
    return secret['dsn']

db = connect(get_database_dsn())
```

**HashiCorp Vault:**
```python
import hvac

def get_database_dsn():
    client = hvac.Client(url='https://vault.example.com')
    secret = client.secrets.kv.v2.read_secret_version(path='database/prod')
    return secret['data']['data']['dsn']

db = connect(get_database_dsn())
```

**Azure Key Vault:**
```python
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential

def get_database_dsn():
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url="https://vault.vault.azure.net/", credential=credential)
    return client.get_secret("database-dsn").value

db = connect(get_database_dsn())
```

### DSN Security Best Practices

1. **Use SSL/TLS**: Always use encrypted connections in production:

```python
dsn = "postgresql://user:pass@host/dbname?sslmode=require"
```

2. **Sanitize Logs**: Never log full DSNs (they contain credentials):

```python
# ❌ Bad
logger.info(f"Connecting to {dsn}")

# ✅ Good
logger.info(f"Connecting to {dsn.split('@')[-1] if '@' in dsn else 'database'}")
```

3. **Rotate Credentials**: Regularly rotate database passwords and update secrets.

4. **Use Read-Only Connections**: When possible, use read-only database users for queries:

```python
# Read-only user
read_dsn = os.getenv("DATABASE_READ_ONLY_URL")
read_db = connect(read_dsn)

# Read-write user (only for mutations)
write_dsn = os.getenv("DATABASE_WRITE_URL")
write_db = connect(write_dsn)
```

## SQL Injection Prevention

Moltres automatically prevents SQL injection through:

1. **Parameterized Queries**: All user input is parameterized
2. **Identifier Validation**: Table and column names are validated
3. **Type Safety**: Type checking prevents injection vectors

### Safe Practices

**✅ Safe - Parameterized Queries:**
```python
# User input is automatically parameterized
user_id = request.args.get('user_id')
df = db.table("users").select().where(col("id") == user_id)
```

**✅ Safe - Raw SQL with Parameters:**
```python
user_id = request.args.get('user_id')
df = db.sql("SELECT * FROM users WHERE id = :user_id", user_id=user_id)
```

**❌ Unsafe - String Concatenation (Don't Do This):**
```python
# NEVER do this - vulnerable to SQL injection
user_id = request.args.get('user_id')
df = db.sql(f"SELECT * FROM users WHERE id = {user_id}")  # DANGEROUS!
```

## Access Control

### Database-Level Security

1. **Principle of Least Privilege**: Grant only necessary permissions:
   ```sql
   -- Read-only user
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
   
   -- Write user (only for specific tables)
   GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE orders TO write_user;
```

2. **Row-Level Security**: Use database row-level security policies:
   ```sql
   -- PostgreSQL example
   CREATE POLICY user_isolation ON users
       FOR ALL
       USING (user_id = current_user);
```

3. **Schema Isolation**: Use separate schemas for different applications:

```python
# Application-specific schema
dsn = "postgresql://user:pass@host/dbname?options=-csearch_path=app_schema"
db = connect(dsn)
```

### Application-Level Security

1. **Validate Input**: Always validate user input before database operations:

```python
def validate_user_id(user_id: str) -> int:
    try:
        uid = int(user_id)
        if uid <= 0:
            raise ValueError("User ID must be positive")
        return uid
    except ValueError:
        raise ValueError("Invalid user ID format")

user_id = validate_user_id(request.args.get('user_id'))
df = db.table("users").select().where(col("id") == user_id)
```

2. **Rate Limiting**: Implement rate limiting for database operations:

```python
from functools import wraps
import time

def rate_limit(calls_per_second: float):
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(10.0)  # Max 10 calls per second
def query_users():
    return db.table("users").select().collect()
```

## Audit Logging

Enable audit logging for sensitive operations:

```python
import logging

audit_logger = logging.getLogger("audit")

def audit_query(sql: str, user: str, params: dict = None):
    """Log query for audit purposes."""
    audit_logger.info(
        "Query executed",
        extra={
            "sql": sql[:500],  # Truncate long queries
            "user": user,
            "params": params,
        }
    )

# Use performance hooks for audit logging
from moltres import register_performance_hook

def audit_hook(sql: str, elapsed: float, metadata: dict):
    audit_query(sql, current_user(), metadata.get("params"))

register_performance_hook("query_end", audit_hook)
```

## Dependency Security

### Regular Updates

Keep dependencies up to date:

```bash
# Check for outdated packages
pip list --outdated

# Update dependencies
pip install --upgrade moltres
```

### Vulnerability Scanning

Use tools to scan for known vulnerabilities:

```bash
# Using safety
pip install safety
safety check

# Using pip-audit
pip install pip-audit
pip-audit
```

### Dependency Pinning

Pin critical dependencies in production:

```toml
# pyproject.toml
dependencies = [
  "SQLAlchemy>=2.0,<3.0",  # Pin to major version
  "typing-extensions>=4.5,<5.0",
]
```

## Network Security

### Use Encrypted Connections

Always use SSL/TLS for remote database connections:

```python
# PostgreSQL
dsn = "postgresql://user:pass@host/dbname?sslmode=require"

# MySQL
dsn = "mysql://user:pass@host/dbname?ssl=true"

# SQL Server
dsn = "mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server&Encrypt=yes"
```

### Firewall Rules

Restrict database access with firewall rules:

1. **Database Firewall**: Only allow connections from application servers
2. **Application Firewall**: Use WAF (Web Application Firewall) for web applications
3. **Network Segmentation**: Isolate database servers in private networks

## Data Handling

### Sensitive Data

1. **Encryption at Rest**: Ensure database encryption is enabled
2. **Encryption in Transit**: Always use SSL/TLS connections
3. **Data Masking**: Mask sensitive data in logs and error messages:

```python
def mask_sensitive_data(data: dict) -> dict:
    """Mask sensitive fields in data."""
    sensitive_fields = ["password", "ssn", "credit_card"]
    masked = data.copy()
    for field in sensitive_fields:
        if field in masked:
            masked[field] = "***"
    return masked
```

### Data Retention

1. **Retention Policies**: Implement data retention policies
2. **Secure Deletion**: Use secure deletion for sensitive data
3. **Backup Security**: Encrypt database backups

## Compliance

### GDPR

For GDPR compliance:

1. **Right to Access**: Provide APIs to export user data
2. **Right to Deletion**: Implement secure data deletion
3. **Data Minimization**: Only collect necessary data
4. **Consent Management**: Track user consent for data processing

### HIPAA

For HIPAA compliance:

1. **Access Controls**: Implement strict access controls
2. **Audit Logs**: Maintain comprehensive audit logs
3. **Encryption**: Encrypt PHI (Protected Health Information) at rest and in transit
4. **Business Associate Agreements**: Ensure BAAs with service providers

## Incident Response

### Security Incident Procedures

1. **Detection**: Monitor for suspicious activity
2. **Containment**: Isolate affected systems
3. **Investigation**: Analyze logs and determine scope
4. **Remediation**: Fix vulnerabilities and restore systems
5. **Communication**: Notify affected parties if required

### Logging Security Events

Log all security-relevant events:

```python
security_logger = logging.getLogger("security")

def log_security_event(event_type: str, details: dict):
    """Log security event."""
    security_logger.warning(
        f"Security event: {event_type}",
        extra=details
    )

# Example: Log failed authentication
log_security_event("auth_failure", {
    "user": username,
    "ip": request.remote_addr,
    "timestamp": time.time(),
})
```

## Best Practices Summary

1. ✅ Never hardcode credentials
2. ✅ Use environment variables or secret management
3. ✅ Always use SSL/TLS for remote connections
4. ✅ Implement least privilege access control
5. ✅ Validate and sanitize all user input
6. ✅ Enable audit logging for sensitive operations
7. ✅ Keep dependencies up to date
8. ✅ Scan for vulnerabilities regularly
9. ✅ Encrypt sensitive data at rest and in transit
10. ✅ Implement proper error handling (don't expose internals)

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security concerns to: [security contact email]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work with you to resolve the issue.
