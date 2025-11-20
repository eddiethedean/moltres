"""Dataset readers."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

from ..table.table import Database


def read_table(db: Database, table_name: str, columns: Optional[Iterable[str]] = None):
    handle = db.table(table_name)
    df = handle.select()
    if columns:
        df = df.select(*columns)
    return df.collect()


def read_sql(db: Database, sql: str, params: Optional[Dict[str, object]] = None):
    return db.execute_sql(sql, params=params).rows
