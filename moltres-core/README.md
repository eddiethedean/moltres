# moltres-core

SQL execution utilities shared by [Moltres](https://github.com/eddiethedean/moltres) and an optional
[`pydantable`](https://pypi.org/project/pydantable/)-compatible **MoltresPydantableEngine**
implementing the `ExecutionEngine` protocol.

The `pydantable_protocol` API is **vendored** under `moltres_core.embedded_protocol`
(based on the `pydantable-protocol` package in the pydantable monorepo) so this wheel
stays installable before `pydantable-protocol` hits PyPI.

See the parent repository README and `docs/PYDANTABLE_ENGINE.md` for usage.
