# ADR 002: Secure tenant API is primary

**Status:** Accepted; removal-timeline clause superseded by ADR-003

New examples and evaluation use `TenantGraphRAG`. The namespace-only engine
lacked database-enforced tenancy and document provenance.

> **Superseded:** this ADR originally stated "kept for v0.7 compatibility...
> no breaking removal occurs before a major release." See
> [ADR-003](003-remove-legacy-engine.md): the legacy engine has since been
> removed entirely, ahead of that timeline, because no external callers
> ever depended on it.
