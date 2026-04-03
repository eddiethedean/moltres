"""Pydantable :class:`~pydantable_protocol.ExecutionEngine` built on Moltres SQL stack."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, cast

from moltres_core.embedded_protocol import (
    EngineCapabilities,
    UnsupportedEngineOperationError,
)
from sqlalchemy import and_, func, select

from moltres_core.plan import (
    SqlPlan,
    SqlRootData,
    annotation_to_descriptor,
    sql_plan_from_field_types,
)
from moltres_core.sql.connection import ConnectionManager
from moltres_core.sql.execution import QueryExecutor

ConfigT = Any


def _unsupported(name: str) -> None:
    raise UnsupportedEngineOperationError(f"Moltres SQL engine does not support {name!r} yet")


def _records_to_columns(records: list[dict[str, Any]], column_order: list[str]) -> dict[str, list[Any]]:
    if not records:
        return {c: [] for c in column_order}
    return {c: [row.get(c) for row in records] for c in column_order}


def _descriptor_from_value(sample: Any) -> dict[str, Any]:
    if sample is None:
        return {"base": "unknown", "nullable": True}
    if isinstance(sample, bool):
        return {"base": "bool", "nullable": False}
    if isinstance(sample, int):
        return {"base": "int", "nullable": False}
    if isinstance(sample, float):
        return {"base": "float", "nullable": False}
    if isinstance(sample, str):
        return {"base": "str", "nullable": False}
    return {"base": "unknown", "nullable": True}


def _schema_descriptors_from_rows(
    rows: list[dict[str, Any]], columns: Sequence[str]
) -> dict[str, dict[str, Any]]:
    if not rows:
        return {c: {"base": "unknown", "nullable": True} for c in columns}
    first = rows[0]
    return {c: _descriptor_from_value(first.get(c)) for c in columns}


def _copy_plan_state(src: SqlPlan, **kwargs: Any) -> SqlPlan:
    data = {
        "columns": src.columns,
        "field_types": dict(src.field_types),
        "order_by": src.order_by,
        "offset": src.offset,
        "limit": src.limit,
        "distinct": src.distinct,
        "coalesce_values": dict(src.coalesce_values),
        "drop_nulls_predicates": src.drop_nulls_predicates,
    }
    data.update(kwargs)
    return SqlPlan(**data)


class MoltresPydantableEngine:
    """Execution engine translating a subset of pydantable plans into SQL."""

    __slots__ = ("_executor", "_capabilities")

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config: ConfigT,
        *,
        capabilities: EngineCapabilities | None = None,
    ) -> None:
        self._executor = QueryExecutor(connection_manager, config)
        self._capabilities = capabilities or EngineCapabilities(
            backend="custom",
            extension_loaded=False,
            has_execute_plan=True,
            has_async_execute_plan=True,
            has_async_collect_plan_batches=False,
            has_sink_parquet=False,
            has_sink_csv=False,
            has_sink_ipc=False,
            has_sink_ndjson=False,
            has_collect_plan_batches=True,
            has_execute_join=True,
            has_execute_groupby_agg=True,
        )

    @classmethod
    def from_database(cls, db: Any) -> MoltresPydantableEngine:
        """Build from a Moltres :class:`moltres.table.table.Database` instance."""
        return cls(db.connection_manager, db.config.engine)

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._capabilities

    @staticmethod
    def _require_plan(plan: Any) -> SqlPlan:
        if not isinstance(plan, SqlPlan):
            raise TypeError(f"Expected SqlPlan, got {type(plan).__name__}")
        return plan

    @staticmethod
    def _require_sql_root(data: Any) -> SqlRootData:
        if not isinstance(data, SqlRootData):
            raise TypeError(
                "Moltres SQL engine expects root data as moltres_core.SqlRootData "
                f"(got {type(data).__name__})"
            )
        return data

    @staticmethod
    def _is_column_dict(data: Any) -> bool:
        if not isinstance(data, Mapping):
            return False
        if not data:
            return True
        return all(isinstance(k, str) and isinstance(v, list) for k, v in data.items())

    def _execute_plan_on_column_dict(
        self,
        plan: SqlPlan,
        data: Mapping[str, list[Any]],
        *,
        as_python_lists: bool,
    ) -> Any:
        """Materialize transforms for eager in-memory roots (post ``execute_*``)."""

        def _n_rows(cols: Mapping[str, list[Any]]) -> int:
            if not cols:
                return 0
            return len(next(iter(cols.values())))

        n = _n_rows(data)
        if n and any(len(v) != n for v in data.values()):
            raise ValueError("In-memory root columns have mismatched lengths")

        work: dict[str, list[Any]] = {str(k): [x for x in v] for k, v in data.items()}
        for c in plan.columns:
            if c not in work:
                raise ValueError(f"In-memory root missing column {c!r}")

        for c, fill_v in plan.coalesce_values.items():
            work[c] = [fill_v if x is None else x for x in work[c]]

        keep_rows = list(range(n))
        for subset, how, threshold in plan.drop_nulls_predicates:
            if threshold is not None:
                _unsupported("in-memory drop_nulls with threshold=")
            key_cols = list(subset) if subset is not None else list(plan.columns)
            new_keep: list[int] = []
            for i in keep_rows:
                vals = [work[c][i] for c in key_cols]
                if how == "any":
                    if all(v is not None for v in vals):
                        new_keep.append(i)
                elif how == "all":
                    if any(v is not None for v in vals):
                        new_keep.append(i)
                else:
                    raise ValueError(f"Invalid how= for drop_nulls: {how!r}")
            keep_rows = new_keep

        if plan.distinct:
            seen: set[tuple[Any, ...]] = set()
            new_keep: list[int] = []
            for i in keep_rows:
                key = tuple(work[c][i] for c in plan.columns)
                if key in seen:
                    continue
                seen.add(key)
                new_keep.append(i)
            keep_rows = new_keep

        if plan.order_by:
            rows_sorted: list[dict[str, Any]] = [
                {c: work[c][i] for c in plan.columns} for i in keep_rows
            ]
            for key, desc, nulls_last in reversed(plan.order_by):
                rows_sorted.sort(
                    key=lambda r, k=key, nl=nulls_last: (
                        (r[k] is None) if nl else (r[k] is not None),
                        r[k],
                    ),
                    reverse=desc,
                )
            for c in plan.columns:
                work[c] = [r[c] for r in rows_sorted]
        else:
            for c in plan.columns:
                work[c] = [work[c][i] for i in keep_rows]

        if plan.offset is not None or plan.limit is not None:
            off = int(plan.offset or 0)
            lim = plan.limit
            for c in plan.columns:
                work[c] = work[c][off : None if lim is None else off + int(lim)]

        out = {c: work[c] for c in plan.columns}
        if as_python_lists:
            return out
        primary = plan.columns[0]
        nrow = len(out[primary]) if plan.columns else 0
        return [{c: out[c][i] for c in plan.columns} for i in range(nrow)]

    # --- planning ---------------------------------------------------------

    def make_plan(self, field_types: Any) -> SqlPlan:
        if not isinstance(field_types, Mapping):
            raise TypeError("make_plan expects a mapping of column -> dtype")
        return sql_plan_from_field_types(cast(Mapping[str, Any], field_types))

    def has_async_execute_plan(self) -> bool:
        return bool(self._capabilities.has_async_execute_plan)

    def has_async_collect_plan_batches(self) -> bool:
        return bool(self._capabilities.has_async_collect_plan_batches)

    def make_literal(self, *, value: Any) -> Any:
        _unsupported("make_literal (no expression runtime)")

    def plan_with_columns(self, plan: Any, columns: dict[str, Any]) -> Any:
        _unsupported("plan_with_columns (expression runtime required)")

    def expr_is_global_agg(self, expr: Any) -> bool:
        return False

    def expr_global_default_alias(self, expr: Any) -> Any:
        return None

    def plan_global_select(self, plan: Any, items: list[tuple[str, Any]]) -> Any:
        _unsupported("plan_global_select (expression runtime required)")

    def plan_select(self, plan: Any, projects: list[str]) -> Any:
        p = self._require_plan(plan)
        for c in projects:
            if c not in p.field_types:
                raise ValueError(f"plan_select: unknown column {c!r}")
        cols = tuple(projects)
        ft = {c: p.field_types[c] for c in cols}
        return _copy_plan_state(p, columns=cols, field_types=ft)

    def plan_filter(self, plan: Any, condition_expr: Any) -> Any:
        _unsupported("plan_filter (expression runtime / SQL translation required)")

    def plan_sort(
        self,
        plan: Any,
        keys: list[str],
        desc: list[bool],
        nulls_last: list[bool],
        maintain_order: bool,
    ) -> Any:
        _ = maintain_order
        p = self._require_plan(plan)
        order: list[tuple[str, bool, bool]] = []
        nl_pad = nulls_last or [False] * len(keys)
        for i, k in enumerate(keys):
            if k not in p.field_types:
                raise ValueError(f"plan_sort: unknown column {k!r}")
            order.append((k, desc[i], nl_pad[i] if i < len(nl_pad) else False))
        return _copy_plan_state(p, order_by=tuple(order))

    def plan_unique(
        self,
        plan: Any,
        subset: list[str] | None,
        keep: str,
        maintain_order: bool,
    ) -> Any:
        _ = keep, maintain_order
        p = self._require_plan(plan)
        if subset is not None:
            missing = [c for c in subset if c not in p.field_types]
            if missing:
                raise ValueError(f"plan_unique: unknown columns {missing}")
        return _copy_plan_state(p, distinct=True)

    def plan_duplicate_mask(self, plan: Any, subset: list[str] | None, keep: str) -> Any:
        _unsupported("plan_duplicate_mask")

    def plan_drop_duplicate_groups(self, plan: Any, subset: list[str] | None) -> Any:
        _unsupported("plan_drop_duplicate_groups")

    def plan_drop(self, plan: Any, columns: list[str]) -> Any:
        p = self._require_plan(plan)
        drop = set(columns)
        new_cols = tuple(c for c in p.columns if c not in drop)
        if not new_cols:
            raise ValueError("plan_drop removed all columns")
        ft = {c: p.field_types[c] for c in new_cols}
        return _copy_plan_state(p, columns=new_cols, field_types=ft)

    def plan_rename(self, plan: Any, rename_map: Mapping[str, str]) -> Any:
        p = self._require_plan(plan)
        rmap = dict(rename_map)
        if len(set(rmap.values())) != len(rmap.values()):
            raise ValueError("plan_rename: output names must be unique")
        new_cols: list[str] = []
        ft: dict[str, Any] = {}
        for c in p.columns:
            new_c = rmap.get(c, c)
            new_cols.append(new_c)
            ft[new_c] = p.field_types[c]
        return _copy_plan_state(p, columns=tuple(new_cols), field_types=ft)

    def plan_slice(self, plan: Any, offset: int, length: int) -> Any:
        p = self._require_plan(plan)
        return _copy_plan_state(p, offset=int(offset), limit=int(length))

    def plan_with_row_count(self, plan: Any, name: str, offset: int) -> Any:
        _unsupported("plan_with_row_count (window functions not implemented)")

    def plan_fill_null(
        self,
        plan: Any,
        subset: list[str] | None,
        value: Any,
        strategy: str | None,
    ) -> Any:
        if strategy is not None and strategy not in ("", "literal"):
            _unsupported(f"plan_fill_null strategy={strategy!r}")
        p = self._require_plan(plan)
        cols = list(subset) if subset is not None else list(p.columns)
        coalesce = dict(p.coalesce_values)
        for c in cols:
            if c not in p.field_types:
                raise ValueError(f"plan_fill_null: unknown column {c!r}")
            coalesce[c] = value
        return _copy_plan_state(p, coalesce_values=coalesce)

    def plan_drop_nulls(
        self,
        plan: Any,
        subset: list[str] | None,
        how: str,
        threshold: int | None,
    ) -> Any:
        p = self._require_plan(plan)
        pred = tuple(p.drop_nulls_predicates) + ((tuple(subset) if subset else None, how, threshold),)
        return _copy_plan_state(p, drop_nulls_predicates=pred)

    def plan_rolling_agg(
        self,
        plan: Any,
        column: str,
        window_size: int,
        min_periods: int,
        op: str,
        out_name: str,
        partition_by: Sequence[str] | None = None,
    ) -> Any:
        _unsupported("plan_rolling_agg")

    # --- execution --------------------------------------------------------

    def execute_plan(
        self,
        plan: Any,
        data: Any,
        *,
        as_python_lists: bool = False,
        streaming: bool = False,
        error_context: str | None = None,
    ) -> Any:
        _ = streaming, error_context
        p = self._require_plan(plan)
        if self._is_column_dict(data):
            return self._execute_plan_on_column_dict(
                p, cast(Mapping[str, list[Any]], data), as_python_lists=as_python_lists
            )
        root = self._require_sql_root(data)
        stmt = p.build_select(root)
        res = self._executor.fetch(stmt)
        rows = res.rows
        if rows is None:
            return None if not as_python_lists else {c: [] for c in p.columns}
        if isinstance(rows, list) and (not rows or isinstance(rows[0], dict)):
            records = cast(list[dict[str, Any]], rows)
            col_order = list(p.columns)
            if not as_python_lists:
                return records
            return _records_to_columns(records, col_order)
        _unsupported(f"execute_plan with fetch_format={self._executor._config.fetch_format!r}")

    async def async_execute_plan(
        self,
        plan: Any,
        data: Any,
        *,
        as_python_lists: bool = False,
        streaming: bool = False,
        error_context: str | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self.execute_plan,
            plan,
            data,
            as_python_lists=as_python_lists,
            streaming=streaming,
            error_context=error_context,
        )

    async def async_collect_plan_batches(
        self,
        plan: Any,
        root_data: Any,
        *,
        batch_size: int = 65_536,
        streaming: bool = False,
    ) -> list[Any]:
        _unsupported("async_collect_plan_batches")

    def collect_batches(
        self,
        plan: Any,
        root_data: Any,
        *,
        batch_size: int = 65_536,
        streaming: bool = False,
    ) -> list[Any]:
        _ = batch_size, streaming
        # Single batch of python column dicts — matches lazy streaming contract loosely.
        cols = self.execute_plan(plan, root_data, as_python_lists=True)
        return [cols]

    def execute_join(
        self,
        left_plan: Any,
        left_root_data: Any,
        right_plan: Any,
        right_root_data: Any,
        left_on: Sequence[str],
        right_on: Sequence[str],
        how: str,
        suffix: str,
        *,
        validate: str | None = None,
        coalesce: bool | None = None,
        join_nulls: bool | None = None,
        maintain_order: str | None = None,
        allow_parallel: bool | None = None,
        force_parallel: bool | None = None,
        as_python_lists: bool = False,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _ = validate, join_nulls, maintain_order, allow_parallel, force_parallel, streaming
        if how != "inner":
            _unsupported(f"execute_join how={how!r} (only inner joins are implemented)")
        if coalesce is True and list(left_on) != list(right_on):
            _unsupported("execute_join coalesce=True with differing key names")

        lp = self._require_plan(left_plan)
        rp = self._require_plan(right_plan)
        lroot = self._require_sql_root(left_root_data)
        rroot = self._require_sql_root(right_root_data)
        lt = lroot.table
        rt = rroot.table
        if len(left_on) != len(right_on):
            raise ValueError("left_on / right_on length mismatch")
        conds = [lt.c[l] == rt.c[r] for l, r in zip(left_on, right_on, strict=True)]
        join_on = and_(*conds)

        j = lt.join(rt, join_on, isouter=False)
        proj: list[Any] = []
        out_labels: list[str] = []
        field_types: dict[str, Any] = {}

        coalesce_keys = (
            coalesce is not False
            and list(left_on) == list(right_on)
            and len(left_on) > 0
        )

        for c in lp.columns:
            if coalesce_keys and c in left_on:
                from sqlalchemy.sql.functions import coalesce as sa_coalesce

                rk = right_on[left_on.index(c)]
                proj.append(sa_coalesce(lt.c[c], rt.c[rk]).label(c))
            else:
                proj.append(lt.c[c].label(c))
            out_labels.append(c)
            field_types[c] = lp.field_types[c]

        key_set = set(left_on)
        for c in rp.columns:
            if coalesce_keys and c in right_on and c in key_set:
                continue
            label = c
            if label in field_types:
                if not suffix:
                    raise ValueError(f"join column collision for {c!r}; pass a non-empty suffix=")
                label = f"{c}{suffix}"
            proj.append(rt.c[c].label(label))
            out_labels.append(label)
            field_types[label] = rp.field_types[c]

        stmt = select(*proj).select_from(j)
        res = self._executor.fetch(stmt)
        rows_raw = res.rows
        if not isinstance(rows_raw, list) or (rows_raw and not isinstance(rows_raw[0], dict)):
            _unsupported("execute_join requires fetch_format=records")

        records = cast(list[dict[str, Any]], rows_raw)
        descriptors = {k: annotation_to_descriptor(v) for k, v in field_types.items()}
        first_row: dict[str, Any] = {lbl: records[0].get(lbl) for lbl in out_labels} if records else {}
        for lbl in out_labels:
            if lbl in descriptors and lbl in first_row:
                descriptors[lbl] = _descriptor_from_value(first_row[lbl])

        mem_root = {lbl: [row.get(lbl) for row in records] for lbl in out_labels}
        if as_python_lists:
            return mem_root, descriptors
        return records, descriptors

    def execute_groupby_agg(
        self,
        plan: Any,
        root_data: Any,
        by: Sequence[str],
        aggregations: Any,
        *,
        maintain_order: bool = False,
        drop_nulls: bool = True,
        as_python_lists: bool = False,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _ = maintain_order, streaming
        p = self._require_plan(plan)
        root = self._require_sql_root(root_data)
        t = root.table
        for k in by:
            if k not in p.field_types:
                raise ValueError(f"groupby: unknown key column {k!r}")

        if not isinstance(aggregations, Mapping):
            raise TypeError("aggregations must be a mapping")

        group_cols = [t.c[k] for k in by]
        sel: list[Any] = list(group_cols)

        for out_name, spec in aggregations.items():
            if not isinstance(spec, tuple) or len(spec) != 2:
                raise TypeError("aggregation spec must be (op, column)")
            op, incol = spec
            if incol not in p.field_types:
                raise ValueError(f"groupby: unknown aggregated column {incol!r}")
            col = t.c[incol]
            opl = str(op).lower()
            if opl == "count":
                agg_expr = func.count(col)
            elif opl == "sum":
                agg_expr = func.sum(col)
            elif opl == "mean":
                agg_expr = func.avg(col)
            elif opl == "min":
                agg_expr = func.min(col)
            elif opl == "max":
                agg_expr = func.max(col)
            else:
                _unsupported(f"groupby agg op={op!r}")

            sel.append(agg_expr.label(out_name))

        stmt = select(*sel).select_from(t).group_by(*group_cols)
        if drop_nulls:
            for k in by:
                stmt = stmt.where(t.c[k].is_not(None))

        res = self._executor.fetch(stmt)
        rows_raw = res.rows
        if not isinstance(rows_raw, list) or (rows_raw and not isinstance(rows_raw[0], dict)):
            _unsupported("execute_groupby_agg requires fetch_format=records")

        records = cast(list[dict[str, Any]], rows_raw)
        out_cols = list(by) + [str(k) for k in aggregations.keys()]
        descriptors: dict[str, dict[str, Any]] = {k: annotation_to_descriptor(p.field_types[k]) for k in by}
        for out_name, spec in aggregations.items():
            _, incol = spec
            descriptors[out_name] = annotation_to_descriptor(p.field_types[incol])

        first_row = {c: records[0].get(c) for c in out_cols} if records else {}
        for c in out_cols:
            if c in first_row:
                descriptors[c] = _descriptor_from_value(first_row[c])

        mem_root = {c: [row.get(c) for row in records] for c in out_cols}
        if as_python_lists:
            return mem_root, descriptors
        return records, descriptors

    def execute_concat(
        self,
        left_plan: Any,
        left_root_data: Any,
        right_plan: Any,
        right_root_data: Any,
        how: str,
        *,
        as_python_lists: bool = False,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported(f"execute_concat how={how!r}")

    def execute_except_all(
        self,
        left_plan: Any,
        left_root_data: Any,
        right_plan: Any,
        right_root_data: Any,
        *,
        as_python_lists: bool = False,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported("execute_except_all")

    def execute_intersect_all(
        self,
        left_plan: Any,
        left_root_data: Any,
        right_plan: Any,
        right_root_data: Any,
        *,
        as_python_lists: bool = False,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported("execute_intersect_all")

    def execute_melt(
        self,
        plan: Any,
        root_data: Any,
        id_vars: Sequence[str],
        value_vars: Sequence[str] | None,
        variable_name: str,
        value_name: str,
        *,
        as_python_lists: bool = False,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported("execute_melt")

    def execute_pivot(
        self,
        plan: Any,
        root_data: Any,
        index: Sequence[str],
        columns: str,
        values: Sequence[str],
        aggregate_function: str,
        *,
        pivot_values: Sequence[Any] | None = None,
        sort_columns: bool = False,
        separator: str = "_",
        as_python_lists: bool = False,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported("execute_pivot")

    def execute_explode(
        self,
        plan: Any,
        root_data: Any,
        columns: Sequence[str],
        *,
        streaming: bool = False,
        outer: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported("execute_explode")

    def execute_posexplode(
        self,
        plan: Any,
        root_data: Any,
        list_column: str,
        pos_name: str,
        value_name: str,
        *,
        streaming: bool = False,
        outer: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported("execute_posexplode")

    def execute_unnest(
        self,
        plan: Any,
        root_data: Any,
        columns: Sequence[str],
        *,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported("execute_unnest")

    def execute_rolling_agg(
        self,
        plan: Any,
        root_data: Any,
        on: str,
        column: str,
        window_size: int | str,
        op: str,
        out_name: str,
        by: Sequence[str] | None,
        min_periods: int,
    ) -> tuple[Any, Any]:
        _unsupported("execute_rolling_agg")

    def execute_groupby_dynamic_agg(
        self,
        plan: Any,
        root_data: Any,
        index_column: str,
        every: str,
        period: str | None,
        by: Sequence[str] | None,
        aggregations: Any,
        *,
        as_python_lists: bool = False,
        streaming: bool = False,
    ) -> tuple[Any, Any]:
        _unsupported("execute_groupby_dynamic_agg")

    def write_parquet(
        self,
        plan: Any,
        root_data: Any,
        path: str,
        *,
        streaming: bool = False,
        write_kwargs: dict[str, Any] | None = None,
        partition_by: list[str] | tuple[str, ...] | None = None,
        mkdir: bool = True,
    ) -> None:
        _unsupported("write_parquet")

    def write_csv(
        self,
        plan: Any,
        root_data: Any,
        path: str,
        *,
        streaming: bool = False,
        separator: int = ord(","),
        write_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _unsupported("write_csv")

    def write_ipc(
        self,
        plan: Any,
        root_data: Any,
        path: str,
        *,
        streaming: bool = False,
        compression: str | None = None,
        write_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _unsupported("write_ipc")

    def write_ndjson(
        self,
        plan: Any,
        root_data: Any,
        path: str,
        *,
        streaming: bool = False,
        write_kwargs: dict[str, Any] | None = None,
    ) -> None:
        _unsupported("write_ndjson")
