"""SQL plan handles for :class:`MoltresExecutionEngine`.

``SqlPlan`` exposes ``schema_descriptors()`` like the native Rust plan so
:class:`pydantable.DataFrame` can derive field types after each transform.
"""

from __future__ import annotations

import types
from dataclasses import dataclass, field
from typing import Any, Mapping, Union, get_args, get_origin

from sqlalchemy import asc, desc, select
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import FromClause

UnionType: Any = getattr(types, "UnionType", None)

_NoneType = type(None)


def _split_optional(annotation: Any) -> tuple[Any, bool]:
    """Return (inner, nullable) for ``T | None`` / ``Optional[T]``."""
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Union or (UnionType is not None and origin is UnionType):
        if _NoneType in args:
            rest = tuple(a for a in args if a is not _NoneType)
            if len(rest) == 1:
                return rest[0], True
    return annotation, False


def annotation_to_descriptor(annotation: Any) -> dict[str, Any]:
    """Map a Python annotation to a pydantable dtype descriptor dict."""
    if annotation is None:
        return {"base": "unknown", "nullable": False}

    inner, nullable = _split_optional(annotation)

    if inner is int:
        return {"base": "int", "nullable": nullable}
    if inner is float:
        return {"base": "float", "nullable": nullable}
    if inner is bool:
        return {"base": "bool", "nullable": nullable}
    if inner is str:
        return {"base": "str", "nullable": nullable}
    if inner is bytes:
        return {"base": "binary", "nullable": nullable}

    return {"base": "unknown", "nullable": nullable}


@dataclass(frozen=True)
class SqlRootData:
    """Root relation for SQL execution (typically a bound SQLAlchemy table)."""

    table: FromClause
    name: str = "root"


@dataclass(frozen=True)
class SqlPlan:
    """Lazy SQL plan state mirrored to pydantable's logical plan API."""

    columns: tuple[str, ...]
    field_types: dict[str, Any]
    order_by: tuple[tuple[str, bool, bool], ...] = ()
    offset: int | None = None
    limit: int | None = None
    distinct: bool = False
    coalesce_values: dict[str, Any] = field(default_factory=dict)
    drop_nulls_predicates: tuple[tuple[tuple[str, ...] | None, str, int | None], ...] = ()

    def schema_descriptors(self) -> dict[str, dict[str, Any]]:
        return {name: annotation_to_descriptor(self.field_types.get(name)) for name in self.columns}

    def _col(self, table: FromClause, name: str) -> ColumnElement[Any]:
        try:
            return table.c[name]
        except KeyError as e:
            raise ValueError(f"Unknown column {name!r} for SQL root") from e

    def build_select(self, root: SqlRootData) -> Select[Any]:
        from sqlalchemy import and_, literal, or_
        from sqlalchemy.sql.functions import coalesce

        t = root.table
        proj: list[ColumnElement[Any]] = []
        for name in self.columns:
            c = self._col(t, name)
            if name in self.coalesce_values:
                proj.append(coalesce(c, literal(self.coalesce_values[name])).label(name))
            else:
                proj.append(c.label(name))

        stmt = select(*proj).select_from(t)

        for subset, how, threshold in self.drop_nulls_predicates:
            if threshold is not None:
                raise ValueError("SQL engine: drop_nulls threshold= is not supported yet")
            key_cols = list(subset) if subset is not None else list(self.columns)
            if how == "any":
                cond = and_(*(self._col(t, c).is_not(None) for c in key_cols))
            elif how == "all":
                cond = or_(*(self._col(t, c).is_not(None) for c in key_cols))
            else:
                raise ValueError(f"Invalid how= for drop_nulls: {how!r}")
            stmt = stmt.where(cond)

        if self.distinct:
            stmt = stmt.distinct()

        for key, is_desc, nulls_last in self.order_by:
            c = self._col(t, key)
            o = desc(c) if is_desc else asc(c)
            if nulls_last:
                try:
                    o = o.nulls_last()
                except AttributeError:  # pragma: no cover
                    pass
            stmt = stmt.order_by(o)

        if self.offset is not None:
            stmt = stmt.offset(self.offset)
        if self.limit is not None:
            stmt = stmt.limit(self.limit)
        return stmt


def sql_plan_from_field_types(field_types: Mapping[str, Any]) -> SqlPlan:
    columns = tuple(field_types.keys())
    return SqlPlan(columns=columns, field_types=dict(field_types))
