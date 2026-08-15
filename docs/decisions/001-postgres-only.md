# ADR 001: PostgreSQL-only retrieval storage

**Status:** Accepted

Store documents, vectors, graph edges, provenance, and tenancy controls in one
PostgreSQL deployment. This removes cross-store consistency and deployment
work. The trade-off is weaker arbitrary graph analytics and a requirement to
bound recursive traversal.
