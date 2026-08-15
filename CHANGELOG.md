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

### Fixed — ingestion and retrieval correctness (schema v5 → v7)

- **Permanent retry hole.** `content_hash` was previously committed (its own
  transaction) before chunking/embedding ran; a failure afterward left the
  hash pointing at content whose chunks never existed, so a retry with
  identical text saw no change and skipped forever. `content_hash` and
  `document_chunks` now publish atomically in one transaction, guarded by a
  per-document advisory lock (`lock_document_for_publication`) so concurrent
  publishers of the same `source_id` can't race each other into both
  reporting a fresh publish for identical content.
- **`add_record()` partial visibility.** Entity node resolution now happens
  inside the same transaction as document/chunk publication, instead of
  before it — a failure rolls the node back too.
- **Graph enrichment is now an observable, separate stage.** `documents`
  gains `extraction_status` (`pending`/`ready`/`partial`/`failed`,
  CHECK-constrained), `extraction_error`, `extraction_attempts`. Status
  writes are compare-and-set against the exact revision hash the work was
  done for (`expected_content_hash`), so a worker finishing extraction after
  a newer revision has already been published can't overwrite that
  revision's status — verified for both the direct-ingestion path and the
  `retry_failed_chunks()` sweep (which binds to the hash captured when
  candidates were fetched, not one re-read at write time).
- **Stale-extraction race.** A chunk whose extraction finishes after a
  concurrent update has already replaced it is detected
  (`filter_existing_chunk_ids`) before its facts are wired to the wrong
  revision.
- **Unsupported edges no longer traversable.** An evidence-backed edge is
  created before its first mention is recorded; a failure in between could
  leave a zero-support edge that `weight >= min_weight` (default `0.0`)
  would not filter out. `prune_unsupported_edges()` cleans these up, and
  read-time traversal (both the recursive expansion and the induced-edge
  query) independently enforces `support_count > 0 OR manual_weight > 0` so
  correctness never depends entirely on that cleanup having already run.
- **`merge_entities()` evidence loss.** Repointing an edge onto a merged
  node previously carried over only `weight`, never `manual_weight`,
  `support_count`, or `edge_mentions` — silently discarding a merged edge's
  accumulated evidence. Now repoints mentions and recomputes support
  explicitly.
- **Induced-edge query had no filters.** `traverse_graph()`'s edge-loading
  step (edges between already-reached nodes) applied none of the recursive
  expansion's own filters — no support check, `min_weight`, or relation
  allow/deny list — so a caller-excluded or zero-support edge between two
  otherwise-reachable nodes could still surface in returned context.

### Before publishing

- Transfer the GitHub repository to the project owner's account or organization.
- Run the full PostgreSQL-backed test and benchmark suite.
- Complete the security and provenance hardening roadmap.
