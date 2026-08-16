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
A `SecureGraphStore`/`TenantGraphRAG`-based replacement — one that reflects
RLS policy-evaluation and transaction-local tenant-context overhead — is
planned but does not exist yet. Until it lands, treat secure-path scale
characteristics at high per-namespace node counts as unmeasured rather than
assuming the legacy engine's prior numbers still apply. Do not treat results
from a synthetic random graph as a retrieval-quality benchmark.
