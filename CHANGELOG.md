# Changelog

All notable changes to `postgres-graph-rag` are documented here.

## 0.1.0 — Project consolidation

This release consolidates the current project direction under Ayush Mishra's
ownership and presents the production-oriented PostgreSQL GraphRAG architecture.

### Added

- PostgreSQL RLS-secured tenant API with transaction-local tenant context.
- Evidence-aware documents, chunks, entity mentions, and edge mentions.
- Lease-based, provider-aware extraction caching and retryable ingestion.
- Hybrid lexical/vector retrieval with bounded multi-hop graph traversal.
- Explainable connection paths with document and chunk citations.
- Community detection and community summaries.
- Offline evaluation, benchmark tooling, MCP integration, and observability hooks.

### Improved

- Async connection pooling, bulk writes, idempotent ingestion, and metadata filters.
- Entity resolution using normalization, trigram candidates, and embedding confirmation.
- Documentation for architecture, security, operations, evaluation, and known limits.

### Before publishing

- Transfer the GitHub repository to the project owner's account or organization.
- Run the full PostgreSQL-backed test and benchmark suite.
- Complete the security and provenance hardening roadmap.
