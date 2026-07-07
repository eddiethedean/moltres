"""Tests for filesystem path validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from moltres.utils.exceptions import ValidationError
from moltres.utils.path_security import validate_file_path


def test_validate_file_path_allows_path_under_root(tmp_path: Path) -> None:
    data_file = tmp_path / "data.csv"
    data_file.write_text("a,b\n1,2\n")
    resolved = validate_file_path(data_file, allowed_paths=(tmp_path,))
    assert resolved == data_file.resolve()


def test_validate_file_path_rejects_escape(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("secret")

    with pytest.raises(ValidationError, match="outside allowed"):
        validate_file_path(outside, allowed_paths=(allowed,))


def test_connect_allowed_paths_enforced(tmp_path: Path) -> None:
    from moltres import connect

    allowed = tmp_path / "data"
    allowed.mkdir()
    csv_file = allowed / "users.csv"
    csv_file.write_text("id,name\n1,Alice\n")

    db = connect(f"sqlite:///{tmp_path / 'test.db'}", allowed_paths=(str(allowed),))
    try:
        df = db.load.csv(str(csv_file))
        rows = df.collect()
        assert len(rows) == 1
        assert rows[0]["name"] == "Alice"

        outside = tmp_path / "evil.csv"
        outside.write_text("x\n")
        with pytest.raises(ValidationError):
            db.load.csv(str(outside))
    finally:
        db.close()
