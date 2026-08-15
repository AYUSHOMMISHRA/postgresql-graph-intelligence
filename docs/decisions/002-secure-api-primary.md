# ADR 002: Secure tenant API is primary

**Status:** Accepted

New examples and evaluation use `TenantGraphRAG`. The namespace-only engine is
kept for v0.7 compatibility but lacks database-enforced tenancy and document
provenance. No breaking removal occurs before a major release.
