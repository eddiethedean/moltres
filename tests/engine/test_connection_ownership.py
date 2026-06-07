"""Tests for engine ownership and close() behavior."""

from __future__ import annotations

from sqlalchemy import create_engine

from moltres import connect


def test_close_does_not_dispose_injected_engine() -> None:
    engine = create_engine("sqlite:///:memory:")
    db = connect(engine=engine)
    db.close()
    # Engine should still be usable after closing the Database handle
    with engine.connect() as conn:
        conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    engine.dispose()
