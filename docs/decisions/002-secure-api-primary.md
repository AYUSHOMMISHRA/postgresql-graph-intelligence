# ADR 002: Secure tenant API is primary

**Status:** Accepted; removal-timeline clause superseded by ADR-003

New examples and evaluation use `TenantGraphRAG`. The namespace-only engine
lacked database-enforced tenancy and document provenance.

> **Superseded:** this ADR originally stated "kept for v0.7 compatibility...
> no breaking removal occurs before a major release." See
> [ADR-003](003-remove-legacy-engine.md): the legacy engine has since been
> removed entirely, ahead of that timeline, because no external callers
> ever depended on it.
>
> **Clarification (0.2.0 release):** "v0.7 compatibility" above referred to
> a planned removal-timeline milestone, not an actually released package
> version — `pyproject.toml`/the package's version history never reached
> `0.7.0`. The committed `uv.lock` briefly carried a stale `0.7.0` value for
> the local package's own version metadata (never matching any tag or
> `pyproject.toml`), which has been corrected as part of the `0.2.0` release
> that also removed the last residue of this ADR's compatibility shim (see
> [ADR-003](003-remove-legacy-engine.md)'s addendum).
