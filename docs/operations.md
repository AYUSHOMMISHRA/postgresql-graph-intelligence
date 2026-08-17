# Operations

## Deployment roles

Run `setup_secure()` with an admin connection. Run ingestion and retrieval only
with the restricted runtime connection. `health` reports secure-schema settings
directly rather than a bare connectivity check.

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

`benchmarks/bench_scale.py` measured only the legacy single-tenant engine and
was removed along with it (see `docs/decisions/003-remove-legacy-engine.md`).
`benchmarks/bench_scale_secure.py` replaces it: every timed query runs
through `SecureGraphStore.tenant_connection()`, reflecting RLS
policy-evaluation and transaction-local tenant-context overhead the legacy
tool never measured. Use an isolated database/schema and run with a fixed
seed (already the default: seed `1729`).

Published 1K/10K-node numbers against a properly isolated schema have not
yet been run and recorded — the tool exists and is smoke-tested at small
scale, but a real run at production-representative scale is separate,
tracked follow-up work. Until that lands, treat secure-path scale
characteristics at high per-namespace node counts as unmeasured. Do not
treat results from a synthetic random graph as a retrieval-quality
benchmark.
