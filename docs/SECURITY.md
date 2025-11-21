# Security Best Practices

This document outlines security best practices when using Moltres.

## SQL Injection Prevention

Moltres is designed with SQL injection prevention in mind. However, it's important to understand how it works and follow best practices.

### How Moltres Prevents SQL Injection

1. **Parameterized Queries**: All user-provided values are passed as parameters, never directly interpolated into SQL strings.

2. **Identifier Validation**: Table and column names are validated and quoted to prevent injection through identifier names.

3. **Expression Compilation**: Column expressions are compiled safely, with literals properly escaped.

### Best Practices

#### ✅ DO: Use Parameterized Values

```python
from moltres import col, connect

db = connect("sqlite:///example.db")

# ✅ GOOD: User input is passed as a value, not SQL
user_id = get_user_input()  # e.g., "123"
df = db.table("users").select().where(col("id") == user_id)
```

#### ❌ DON'T: Construct SQL Strings Manually

```python
# ❌ BAD: Never do this!
user_id = get_user_input()  # Could be "1; DROP TABLE users;--"
sql = f"SELECT * FROM users WHERE id = {user_id}"  # DANGEROUS!
```

#### ✅ DO: Validate Table/Column Names

```python
# ✅ GOOD: Table names are validated automatically
table_name = get_table_name()  # Validated by Moltres
df = db.table(table_name).select()
```

#### ❌ DON'T: Use User Input as Table/Column Names Without Validation

```python
# ❌ BAD: If you must use dynamic table names, validate them first
table_name = get_user_input()
# Validate that table_name only contains safe characters
if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', table_name):
    raise ValueError("Invalid table name")
df = db.table(table_name).select()
```

### Identifier Validation

Moltres automatically validates SQL identifiers (table names, column names) to prevent injection:

- Empty strings are rejected
- Characters like semicolons, quotes, and backslashes are rejected
- Qualified names (e.g., `schema.table`) are validated per part

If you need to use dynamic identifiers, ensure they come from trusted sources or validate them yourself.

### Connection Strings

**Never** include user-provided values directly in connection strings:

```python
# ❌ BAD
password = get_user_input()
db = connect(f"postgresql://user:{password}@host/db")

# ✅ GOOD: Use environment variables or secure config management
import os
db = connect(os.environ["DATABASE_URL"])
```

### File Paths

When reading files, validate paths to prevent directory traversal:

```python
# ✅ GOOD: Validate file paths
from pathlib import Path

user_path = get_user_input()
path = Path(user_path).resolve()
if not str(path).startswith(str(Path("/safe/directory").resolve())):
    raise ValueError("Invalid file path")
records = db.load.csv(str(path))
```

## Authentication and Authorization

Moltres does not handle authentication or authorization. These must be handled at the database level:

1. **Database Users**: Use database users with minimal required permissions
2. **Connection Pooling**: Configure connection pools appropriately for your security needs
3. **Network Security**: Use encrypted connections (SSL/TLS) for remote databases

## Logging and Monitoring

Be careful about logging sensitive data:

```python
# ⚠️ CAUTION: Query logging may expose sensitive data
db = connect("sqlite:///example.db", echo=True)  # Logs all SQL

# ✅ BETTER: Disable logging in production or use log filtering
db = connect("sqlite:///example.db", echo=False)
```

## Error Messages

Error messages may contain SQL or table names. In production, consider:

1. Logging detailed errors server-side
2. Returning generic error messages to clients
3. Not exposing internal table/column names in public APIs

## Recommendations Summary

1. ✅ Always use Moltres APIs, never construct SQL manually
2. ✅ Validate user input before using it in queries
3. ✅ Use parameterized queries (Moltres does this automatically)
4. ✅ Keep database credentials secure (use environment variables)
5. ✅ Use database-level authentication and authorization
6. ✅ Enable SSL/TLS for remote database connections
7. ✅ Be cautious with logging in production
8. ✅ Keep Moltres and dependencies up to date

## Reporting Security Issues

If you discover a security vulnerability, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email security concerns to: odosmatthews@gmail.com
3. Include details about the vulnerability and steps to reproduce

We take security seriously and will respond promptly to security reports.

