# Operations

## Deployment roles

Run `setup_secure()` with an admin connection. Run ingestion and retrieval only
with the restricted runtime connection. `health` reports secure-schema settings
rather than merely opening the legacy pool.

## Schema compatibility

Migration v4 records embedding dimension, vector type, schema version, and a
migration checksum. Setup inspects an existing graph embedding column before
domain writes. A mismatch raises `SchemaCompatibilityError` with both types.

Embedding dimensions are not safely mutable in place. Create a new schema,
re-ingest source documents with the new model, validate it, and switch traffic.

## Recovery

- Failed extraction leaves chunks active for semantic/lexical retrieval.
- `retry_failed_chunks()` replays unfinished chunks from stored content.
- Expired extraction leases can be reclaimed; active leases prevent duplicate
  provider calls.
- Document deletion cascades chunks and mention evidence. Shared entities remain.

## Observability

Register an `EventBus` sink and track ingestion, extraction, embeddings, hybrid
search, traversal, answers, cache hits, failures, duration, tokens, grounded
answers, and citation counts. Built-in events contain identifiers and counts,
not raw prompts or document text.

## Performance workflow

Use an isolated database/schema and run the scale benchmark with a fixed seed.
Record hardware, PostgreSQL/pgvector versions, Git SHA, cold/warm state, query
plans, and p50/p95/p99. Do not treat results from a synthetic random graph as a
retrieval-quality benchmark.
