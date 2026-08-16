
> 🏗️ Early version. Interface can evolve quickly. **Star** the repo to be updated about new changes, as we work our way through the [roadmap](#️-roadmap--future-vision).

# postgres-graph-rag 🐘🕸️

### High-Precision GraphRAG. Native to PostgreSQL.

This is Ayush Mishra's PostgreSQL-native GraphRAG project: a focused effort to
combine vector retrieval, multi-hop graph traversal, evidence-backed citations,
and database-enforced tenancy in one asynchronous Python library. The project
prioritizes explainable retrieval, operational safety, and deployable Postgres
infrastructure over opaque agent loops or a separate graph database.

> **Primary API:** all deployments use the RLS-secured
> `setup_secure()` → `for_tenant()` → `add_document()`/`retrieve()`/`answer()`
> path — see [Multi-Tenancy & Security](#multi-tenancy--security).

Technical documentation: [Architecture](docs/architecture.md) ·
[Evaluation](docs/evaluation.md) · [Security](docs/security.md) ·
[Operations](docs/operations.md) ·
[Engineering Reviews](docs/reviews/) ·
[CTO demo](#production-pilot-demo)

Most RAG systems are Flatlanders. They use vector similarity to find related text, but they are fundamentally blind to **relationships**. If you ask your RAG "How is Person A connected to Project B through their shared dependencies?", standard vector search fails because the answer isn't in a single chunk—it's in the **links** between them.

**Postgres Graph RAG** bridges this reasoning gap by turning your existing PostgreSQL database into a structured knowledge engine.

### Why this exists:
1.  **Infrastructure Nightmare:** Building "Smart RAG" usually means adding a Graph DB (Neo4j) to your stack. Now you have a distributed systems nightmare: keeping your Relational DB, Vector DB, and Graph DB in sync.
2.  **Flatland Problem:** Vector similarity is just probabilistic matching. It doesn't understand hierarchy, causality, or directed relationships (e.g., "A leads B" vs "B leads A"). Traversal here defaults to undirected (matching how most extracted facts are used in practice), but you can opt into strict directed traversal per query — see [Directed & Filtered Traversal](#5-directed--filtered-traversal) below.
3.  **Batch Bottleneck:** Existing GraphRAG research (like Microsoft's) is batch-heavy and token-expensive. It can't handle real-time, incremental updates.

### Postgres-Native Solution:
This library is built for **Postgres Maximalists**. It leverages the engine you already trust to do the heavy lifting:
*   **Recursive Retrieval:** Instead of expensive LLM-agent loops, we use **SQL Recursive CTEs** to perform multi-hop reasoning, scoring each node by seed relevance decayed over hop distance and edge weight as it walks.
*   **Atomic Publication, Eventual Graph Enrichment:** Vectors, nodes, and relationships live in one ACID-compliant engine — no separate Graph DB to keep in sync. A document's hash and its chunks are published atomically in one short transaction, so readers (and retries) only ever see the complete previous revision or the complete new one, never a hash pointing at chunks that don't exist. Graph extraction (an LLM call that can take seconds to minutes) deliberately runs *outside* that transaction as a separate stage afterward: a document is text-searchable the instant it's published and becomes graph-ready shortly after, tracked via each document's `extraction_status` (`pending` / `ready` / `partial` / `failed`). This is not a batch pipeline with sync lag measured in minutes-to-hours — extraction typically finishes within the same request — but it is asynchronous relative to publication, not part of the same transaction.
*   **Flexible Metadata:** Arbitrary application data lives in `JSONB`. The secure schema records its version, migration checksum, vector type, and embedding dimension; incompatible dimensions fail during setup instead of during a later write. Dimension changes still require re-embedding into a compatible schema.

---

## Core Philosophy
- **Infrastructure:** Postgres is the only database (via `pgvector` + `pg_trgm`).
- **Intelligence:** Hosted LLMs (OpenAI or Gemini) for extraction. Freely configurable through `postgres_graph_rag/models.py`.
- **Simplicity:** Native Async Python + SQL.
- **Scalability:** High-performance connection pooling, bulk single-round-trip writes, and namespace-aware design. Postgres Row-Level Security-enforced multi-tenancy (`for_tenant()`, see [Multi-Tenancy & Security](#multi-tenancy--security)) is the only engine; `namespace` is a partition *within* a tenant, not a substitute for one.

> **Known scaling limitation:** all namespaces currently share one `graph_nodes`/`graph_edges` table pair. At the scales exercised so far (1K–10K nodes/namespace), Postgres' planner prefers a namespace-filtered bitmap scan + sort over the HNSW ANN index, because per-namespace row counts are small relative to the whole table. This is fine at that scale; if you operate many large tenants in one deployment, benchmark your own shape before assuming the HNSW index is doing the work (see [Benchmarks](#benchmarks) for the current state of scale-benchmarking tooling).

---

## Installation

```bash
pip install postgres-graph-rag
```

---

## Getting Started (secure API)

Setup requires an administrative DSN once. Application requests use a separate,
restricted role; tenant isolation is enforced by PostgreSQL Row-Level Security.

```python
import os
import uuid
from postgres_graph_rag import PostgresGraphRAG

ADMIN_DSN = os.environ["POSTGRES_URL"]
RUNTIME_DSN = os.environ["PGR_RUNTIME_URL"]
TENANT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")

rag = PostgresGraphRAG(
    runtime_url=RUNTIME_DSN,
    openai_api_key=os.environ["OPENAI_API_KEY"],
)

async def quick_start():
    await rag.setup_secure(
        admin_url=ADMIN_DSN,
        runtime_role="pgr_runtime",
        runtime_password=os.environ["PGR_RUNTIME_PASSWORD"],
    )
    engine = rag.for_tenant(TENANT_ID)
    await engine.add_document(
        "checkout-service depends_on auth-service.",
        namespace="architecture",
        source_id="service-catalog",
    )
    result = await engine.retrieve(
        "What does checkout-service depend on?",
        namespace="architecture",
    )
    answer = await engine.answer(
        "What does checkout-service depend on?",
        namespace="architecture",
    )
    print(result.trace)
    print(answer.answer, answer.citations)
    await rag.close()
```

For a no-key, deterministic setup use `postgres-graph-rag-demo`; see
[Operations](docs/operations.md).

---

## Advanced Usage & Modes

### 1. The Production Way: Async Context Manager
For applications (like FastAPI or background workers), use the `async with` pattern to ensure the connection pool is always closed correctly, even if errors occur.

```python
async with PostgresGraphRAG(runtime_url=RUNTIME_DSN, openai_api_key=KEY) as rag:
    engine = rag.for_tenant(TENANT_ID)
    await engine.add_document("The M4 chip uses ARM architecture.", namespace="notes", source_id="doc-1")
    # No need to call rag.close(), it happens automatically!
```

### 2. Custom Chunking (Inversion of Control)
Don't like the default character splitter? Inject your own. You can pass any callable that takes a string and returns a list of strings.

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Create your favorite chunker
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Inject it into the library
rag = PostgresGraphRAG(
    openai_api_key=KEY,
    chunker=splitter.split_text # Just pass the method
)
```

### 3. Custom Provider Configuration
You can control exactly which models are used for extraction and embeddings.

```python
from postgres_graph_rag.models import ProviderConfig

custom_config: ProviderConfig = {
    "extraction_model": "gpt-5-nano-2025-08-07",
    "embedding_model": "text-embedding-3-large",
    "dimension": 3072 # Must match the model's output
}

rag = PostgresGraphRAG(..., config=custom_config)
```

> **pgvector dimension limits:** HNSW indexes only support up to 2000 dimensions on the standard `vector` type. Above that (like the 3072-dim example above), `setup_database()` automatically switches the embedding column to pgvector's half-precision `halfvec` type, which extends HNSW support to 4000 dimensions. Beyond 4000 dimensions there is no ANN index available at all — `vector_search` falls back to an exact sequential scan, which does not scale past small graphs; a warning is logged when this happens. See [pgvector's HNSW docs](https://github.com/pgvector/pgvector#hnsw).

### 4. Namespaces: partitioning within a tenant
`namespace` isolates data within the same tenant's tables — a project, a document collection, a topic area. It is **not** a tenancy boundary: real isolation between untrusted parties is `for_tenant()`'s job (Postgres Row-Level Security, see [Multi-Tenancy & Security](#multi-tenancy--security)), not `namespace` alone.

```python
engine = rag.for_tenant(TENANT_ID)

# Two projects under the same tenant, kept separate by namespace
await engine.add_document("The Q3 roadmap prioritizes checkout latency.", namespace="project_a", source_id="doc-1")
await engine.add_document("The Q3 roadmap prioritizes onboarding flow.", namespace="project_b", source_id="doc-1")

# Queries are scoped to one namespace
result = await engine.retrieve("What does the roadmap prioritize?", namespace="project_a")
```

### 5. Directed & Filtered Traversal
Retrieval and ingestion behavior are tunable via `retrieval_config` / `ingestion_config`, or per-call keyword overrides on `engine.retrieve()` / `engine.answer()`:

```python
# Only follow edges in their extracted source -> target direction, only
# across "depends_on"/"uses" relations, and ignore weak/low-confidence edges.
result = await engine.retrieve(
    "What does the checkout service depend on?",
    namespace="architecture",
    directed=True,
    relation_types=["depends_on", "uses"],
    min_weight=0.5,
    hops=3,
)
```

Each edge's `weight` increases automatically every time the same
`(source, target, relation)` triple is re-extracted from new text, so
relationships repeated across many documents naturally outrank one-off
mentions during traversal scoring.

### 6. Ingestion Reliability Controls
```python
from postgres_graph_rag.models import IngestionConfig

rag = PostgresGraphRAG(
    openai_api_key=KEY,
    ingestion_config=IngestionConfig(
        max_concurrent_extractions=8,   # bounded concurrency, not unbounded fan-out
        max_extraction_retries=3,       # exponential backoff per chunk
        skip_duplicate_chunks=True,     # content-hash based idempotency
        fuzzy_entity_resolution=True,   # pg_trgm + embedding confirmation
        fuzzy_trgm_threshold=0.4,
        fuzzy_embedding_threshold=0.90,
    ),
)
```

A chunk that keeps failing extraction after retries is logged and skipped —
it does not abort the rest of the batch, and it is **not** marked as
ingested, so a later retry of the same source will retry just that chunk.

> **Known limitation — chunk idempotency has no document identity.** Skip-if-seen
> is keyed on `(namespace, chunk_hash)` only. If identical text appears in two
> different documents in the same namespace, the second occurrence is treated
> as already-ingested and its extraction is skipped. The resulting graph is
> still correct (same text → same triplets → same nodes/edges), but there is
> currently no way to answer "which documents mention X" or to delete one
> document's contribution without risking another's. Fixing this properly
> needs a `documents`/`chunks`/`mentions` evidence schema, which is a bigger,
> deliberately-deferred change rather than a quick patch.

### 7. Grounding modes: entailment verification, not just citation validity

`answer()`'s default behavior (`grounding_mode="citation_only"`, unchanged)
checks that every citation marker names a chunk that was actually
retrieved — never whether that chunk's text *supports* the claim it's
attached to. A reversed relationship, a swapped predicate, or a wrong
number can all cite real evidence and still pass. Two additional modes
run each sentence's claim through an entailment verifier instead:

```python
result = await engine.answer(
    "What does checkout-service depend on?",
    namespace="architecture",
    grounding_mode="verified",  # or "verified_strict"
)
result.grounding_status  # "verified" | "partially_verified" | "citation_valid_only"
                          # | "contradicted" | "insufficient" | "verification_failed" | "abstained"
result.claims             # the atomic claims the answer was split into
result.verifications      # each claim's verdict, supporting quote, reason_code
```

- **`citation_only`** (default): today's behavior, unchanged.
- **`verified`**: drops claims the verifier can't confirm and keeps the rest — `grounding_status` is `"verified"` if every claim survived, `"partially_verified"` if some were dropped.
- **`verified_strict`**: abstains the whole answer (the standard "Insufficient evidence..." text) if *any* claim isn't fully supported.

Verification runs in layers, cheapest first, and a claim that's already
decidable never reaches a model call:

1. Citation existence — the same check `citation_only` does.
2. Identifier/relevance — does the cited text even mention the claim's named entities?
3. A located literal (or formatting-normalized) quote, or an explicit numeric/date mismatch, or a negation in some other retrieved chunk — these three can already positively confirm or reject a claim deterministically.
4. Only a claim none of the above can decide reaches a batched model call — one bounded provider call covering every such claim in the answer, never one call per claim.

A verifier failure (provider error, timeout, malformed response) reports
`grounding_status="verification_failed"` and abstains — it is never
silently treated as `"verified"`. Server-side quote validation means a
model-proposed supporting quote that isn't actually a literal substring of
the cited evidence is rejected, not trusted.

See `benchmarks/grounding/` for the evaluation harness this was built
against and `CHANGELOG.md` for a measured before/after: contradicted-claim
escape rate 100% → 0% from the deterministic layers alone (steps 1-3
above), at an expected, documented cost to recall that the model layer
(step 4) exists to recover.

---

## Benchmarks

Scale-benchmarking tooling (`benchmarks/bench_scale.py`) previously measured
the single-tenant engine directly; that engine has since been removed (see
[CHANGELOG.md](CHANGELOG.md)), so the tooling was removed with it rather
than left broken. `benchmarks/bench_scale_secure.py` is its
`SecureGraphStore`/`TenantGraphRAG`-based replacement: every query it times
runs through `tenant_connection()`, the same transaction-scoped
`set_config()` + RLS-policy-evaluation path a real request takes, so its
numbers include the overhead the legacy tool's never did.

```bash
uv run python -m benchmarks.bench_scale_secure --admin-url "$POSTGRES_URL" --scales 1000,10000
```

Published scale numbers at 1K/10K nodes against a properly isolated schema
have not yet been run and recorded here — the tool exists and is smoke-
tested, but doing a real run and publishing its output is separate,
tracked follow-up work. Until that lands, treat scale characteristics at
high per-namespace node counts as unmeasured rather than assuming any
number is known.

---

## 🗺️ Roadmap & Future Vision

This project follows the **"Postgres Maximalism"** philosophy: Stop building new infrastructure and start using the full power of the database you already own.

### ✅ Phase 1: Foundation
- [x] **Postgres-Native Schema:** Migration-proof design using JSONB and namespacing.
- [x] **Recursive Reasoning:** Multi-hop graph traversal implemented via SQL Recursive CTEs.
- [x] **Bulk Ingestion:** Single-round-trip multi-row upserts for nodes and edges (no per-row loops).
- [x] **Hosted LLM Extraction:** Native support for OpenAI and Google Gemini for high-speed, low-cost tagging.
- [x] **Async Architecture:** Production-ready with high-performance connection pooling.

### ✅ Phase 2: High-Precision Retrieval (Current Release)
- [x] **Directed & Filtered Traversal:** Optional strict source→target direction, relation allow/deny lists, and minimum-weight thresholds.
- [x] **Explainable Scoring:** Every retrieved node carries a hop distance and a score (seed relevance × per-hop decay × edge weight), exposed on `retrieve()`'s structured result instead of a flattened string.
- [x] **Layered Entity Resolution:** Whitespace normalization → exact match → `pg_trgm` trigram candidate generation confirmed by embedding cosine similarity, so lookalike names don't merge without semantic confirmation. Adversarial cases are covered in `tests/test_tenancy.py`.
- [x] **Relationship Scoring:** Edge weight increments automatically on repeated mention of the same `(source, target, relation)` triple.
- [x] **Idempotent Ingestion:** Chunk-content hashing skips re-running (paid) LLM extraction when a source is re-ingested unchanged.
- [x] **Ingestion Reliability:** Bounded-concurrency extraction with retry/backoff; a chunk that keeps failing is skipped and logged, not fatal to the batch.

### ✅ Phase 3: Evidence-Grounded Ingestion, Hybrid Retrieval & Multi-Tenancy (Current Release)
- [x] **Hybrid Search (Postgres FTS + Vector):** `document_chunks.tsv` (a generated `tsvector`, GIN-indexed) combined with `pgvector` cosine search, fused via Reciprocal Rank Fusion. This is genuinely hybrid full-text + vector — **not Okapi BM25** (Postgres's built-in text ranking isn't BM25); see `SecureGraphStore.hybrid_search()`.
- [x] **Evidence-Grounded Schema:** New `documents` / `document_chunks` / `entity_mentions` tables give real provenance — "which documents mention entity X" is an actual query (`get_mentioning_documents()`), not a limitation to document away. Deleting a document cascades its chunks/mentions but preserves entities/edges still supported by other documents.
- [x] **Lease-Based Extraction Cache:** `chunk_extractions` claims a lease before calling the LLM for a given content hash; concurrent ingestion of the same content only pays for one extraction call, with automatic reclaim of expired/failed leases.
- [x] **Tenant Isolation via Postgres Row-Level Security:** `tenant_id` is enforced at the database level, not just by application-level `WHERE` clauses — fail-closed policies (`current_setting(..., true)`), transaction-local tenant context (safe under connection pooling/reuse), and a dedicated non-superuser, `NOBYPASSRLS` runtime role. See [Multi-Tenancy & Security](#multi-tenancy--security) below.
- [x] **Legacy Data Migration:** Pre-multi-tenancy `public.graph_nodes`/`graph_edges` data (from a prior single-tenant deployment) backfills into the new schema under a fixed legacy tenant, preserving IDs.
- [x] **Metadata Pruning:** `hybrid_search`/`traverse_graph` take a typed `metadata_filter` DSL (`postgres_graph_rag/filters.py`) supporting `eq`/`neq`/`gt`/`gte`/`lt`/`lte`/`in` in addition to the original JSONB-containment form — "only traverse relationships from documents updated in the last 90 days" is now literally expressible and tested, not just containment on exact key/value pairs.
- [ ] **Zero-mention node pruning:** Fully-orphaned entities (no remaining mentions from any document) are left in place rather than proactively swept — a reasonable follow-up if storage growth from orphans matters in practice.

### ✅ Phase 4: Global Intelligence (Current Release)
- [x] **SQL-Native Community Detection:** Weighted label propagation over `graph_edges`, computed as a sequence of small deterministic SQL updates (`communities.py`) — not a bundled graph library. Ships as *asynchronous* (Gauss-Seidel-order) propagation, not the more obvious-looking fully-synchronous version: synchronous updates provably oscillate forever on simple structures (two nodes joined by one edge keep swapping each other's label every round) — caught by an actual failing test before being trusted, not assumed correct from the algorithm description. Runs are advisory-lock-protected per `(tenant, namespace)`, immutable once written, and skipped when the namespace isn't dirty (`refresh_communities(force=True)` to override).
- [x] **Global Summarization:** One LLM call per community over its member entities and a sample of supporting chunks (map-only; see limitation below), with evidence-hash-based reuse so re-summarizing unchanged communities doesn't re-call the LLM. `query_global()` ranks existing summaries against a question by lexical overlap — it returns evidence for a caller to synthesize an answer from, it does not itself call an LLM to produce a final answer.
- [x] **Graph/Ingestion/Retrieval Observability:** Typed `Event`s (`observability.py`) fanned out to pluggable sinks — a logging sink, an in-memory per-tenant/namespace `UsageAggregator`, and an optional OpenTelemetry adapter (only active if `opentelemetry` is installed) — covering ingestion, chunking, cache claims/hits, extraction, retrieval (hybrid + traversal separately), and community jobs. Never a hard dependency on a specific backend.
- [x] **Reasoning Path Explainability:** `explain_connection()` reconstructs the actual path between two named entities (`A --leads--> B --designed--> C`), not just a bag of scored neighbors — a dedicated recursive query (`SecureGraphStore.find_path`) separate from the general multi-seed traversal, since a broad retrieval doesn't have one meaningful source/target pair to draw a single path between. Verified to correctly ignore decoy branches and to return an explicit `None` (not a guess) when nothing connects within the hop budget.
- [x] **Relationship-Level Evidence:** `edge_mentions` records the exact document chunk that asserted each extracted relationship. Conflicting edges remain separate and separately citable; deleting a document removes only its edge evidence through the chunk cascade. Deterministic `add_triplets()` edges remain metadata-sourced because they do not create synthetic documents.
- [x] **Bounded Top-N Paths:** `SecureGraphStore.find_paths()` searches across multiple source and target IDs and returns globally ranked complete paths, with the existing hard hop/fan-out limits and per-edge document evidence attached. `explain_connection()` remains the ergonomic single-pair wrapper.
- [ ] **Hierarchical map/reduce summarization:** Community summarization samples a bounded number of chunks per community (`max_chunks_per_community`); a community whose evidence exceeds that isn't map/reduced down further, it's just truncated. Fine at the scales this has been exercised at, a real limitation for very large communities.

### ✅ Phase 5: Agentic Integration (Current Release, RLS carried over from Phase 3)
- [x] **MCP Server Support:** `postgres_graph_rag.mcp_server` exposes the engine as MCP tools over stdio (default, one statically-configured tenant per process) and Streamable HTTP (requires a `tenant_resolver` callable, or an explicit dev-only opt-in restricted to a loopback bind — verified to actually refuse a public unauthenticated bind, not just documented as refusing). Read-only tools (`retrieve`, `get_entity`, `find_paths`, `explain_connection`, `list_communities`, `get_community_summary`, `capabilities`, `health`) are always registered; mutation tools (`ingest_documents`, `delete_document`, `merge_entities`, `refresh_communities`, `summarize_communities`) require `enable_mutations=True`. Console entry point: `postgres-graph-rag-mcp`.
- [x] **Administrative entity-resolution correction:** `merge_entities()` fixes a missed automatic merge by repointing edges and mention provenance onto the target and removing the source. `split_entity()` (undoing a merge) is explicitly **not implemented** — this schema's `entity_mentions` table doesn't retain which original alias a mention came from once merged, so there's no real operation to build; documented as a gap, not faked.
- [x] **Telemetry cost/token tracking:** `LLMExtractor` captures real token usage from OpenAI/Gemini responses (`last_usage`, a `contextvar` — not a plain instance attribute, since concurrent extraction calls share one extractor and a plain attribute gets clobbered under concurrency; verified empirically) and threads it into `EMBEDDING_COMPLETED`/`EXTRACTION_COMPLETED`/`COMMUNITY_SUMMARIZE_COMPLETED` events, so `UsageAggregator.snapshot()` reports real, non-zero token totals. Google's embedding API reports no usage at all (a real SDK limitation, not something skipped) — `last_usage` is `None` in that case rather than a fabricated 0.
- [ ] **OAuth authorization server for HTTP transport:** `mcp_server.py` enforces *that* a tenant-resolving authentication step exists before serving HTTP requests, but does not implement token issuance/client registration/consent flows itself — a deployer supplies their own `tenant_resolver` on top of whatever auth their deployment already has.
- [ ] **Versioned Migration Framework:** `migrate_schema()` now records its version in a `schema_migrations` table and takes a session-level advisory lock so two concurrent migrations fail fast instead of racing on DDL — but it's still one idempotent `CREATE ... IF NOT EXISTS` script, not tracked, individually-appliable, reversible migrations with upgrade/rollback history.

---

### 💡 User-Driven Priorities
We prioritize features that reduce **Operational Overhead**. If you need a feature that further consolidates the "Standard Stack" (Vector + Graph + Relational) into Postgres, open an issue!

**Launch Status:** 🚀 Evidence-grounded hybrid retrieval, relationship-level citations, bounded top-N paths, RLS-backed multi-tenancy, SQL-native community detection, observability hooks, and MCP server integration are all live. Remaining larger gaps include a fully step-wise migration framework and orphan/relationship-weight pruning.

## Production pilot demo

The repository includes a deterministic, offline incident-investigation walkthrough. It exercises the same tenant/RLS, evidence, retrieval, and MCP-facing code paths without requiring an LLM key, so a demo is repeatable and safe to run offline.

```bash
cp .env.example .env
docker compose up -d postgres

# One-time schema and restricted runtime-role setup (admin DSN only).
uv run postgres-graph-rag-demo --admin-url "$POSTGRES_URL" setup
uv run postgres-graph-rag-demo --admin-url "$POSTGRES_URL" ingest

# Compare vector-only, hybrid, and hybrid+graph retrieval.
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" evaluate --output evaluation.json
uv run postgres-graph-rag-demo --runtime-url "$PGR_RUNTIME_URL" query \
  "Which team owns the dependency of checkout-service?" --mode hybrid_graph
```

Retrieval modes are explicit: `vector` is the lexical/vector baseline, `hybrid` fuses Postgres full-text and vector ranks, and `hybrid_graph` additionally performs bounded multi-hop traversal. `OfflineExtractor` is exported for local fixtures; production deployments can inject the existing OpenAI/Gemini extractor instead. The migration is forward-compatible and adds evidence support counts/manual weights without dropping existing data.

---

## Multi-Tenancy & Security

`postgres_graph_rag.tenancy` (`SecureGraphStore`) is the only engine.

```python
from postgres_graph_rag import PostgresGraphRAG

rag = PostgresGraphRAG(
    runtime_url=RUNTIME_DSN,         # the restricted role's connection string, used for all tenant queries
    openai_api_key=KEY,
)

# One-time (or idempotent re-run) migration: creates the postgres_graph_rag
# schema, RLS policies, and the runtime role. Must be called with an
# admin/superuser connection string, never the runtime role's -- passed
# directly to setup_secure(), not to the constructor above.
await rag.setup_secure(
    admin_url=ADMIN_DSN,
    runtime_role="pgr_runtime",
    runtime_password="change-me",
    migrate_legacy_data=False,  # True to backfill pre-existing public.graph_nodes/edges
)

tenant = rag.for_tenant(tenant_id)   # tenant_id is never a per-call argument again
await tenant.add_document("Apple released the M4 chip.", namespace="research", source_id="doc-1")
result = await tenant.retrieve("What did Apple release?", namespace="research")
print(result.to_context_string())
```

**Why RLS, not just an application-level `WHERE tenant_id = ...`:** a forgotten filter in one query path is a cross-tenant data leak. Postgres enforces the tenant boundary at the database level instead:

- Every domain table (`documents`, `document_chunks`, `chunk_extractions`, `graph_nodes`, `graph_edges`, `entity_mentions`, `edge_mentions`) has a fail-closed RLS policy: `tenant_id = current_setting('postgres_graph_rag.tenant_id', true)`. A session with **no** tenant context set gets `NULL`, and `tenant_id = NULL` is never true — so missing context denies everything rather than accidentally granting it.
- The tenant GUC is set with `set_config(..., true)` — **transaction-local**, not session-local — because connections are pooled and reused across tenants; `tenant_connection()` sets it, yields the connection for exactly one transaction, and commits/rolls back before the connection can be checked back into the pool for someone else's tenant. This is tested explicitly (`test_connection_reuse_across_tenants_does_not_leak`) by reusing one physical connection across two tenants.
- RLS only restricts non-owner, non-`BYPASSRLS` roles. `setup_secure()` creates a dedicated runtime role with `NOSUPERUSER NOBYPASSRLS`, separate from the admin/owner role that runs migrations — connecting as the admin role for regular queries would make every RLS policy a no-op. `test_runtime_role_cannot_bypass_rls_attributes` guards against this regressing silently.

**What this does *not* protect against:** RLS protects trusted backend code from a missing tenant filter. It does not protect against an end user who has the database credential directly and can call `set_config` themselves — the runtime role's credential must stay server-side, never handed to a client.

### Migrating data from a prior single-tenant deployment

If you have data under the old single-tenant schema (`public.graph_nodes`/
`public.graph_edges`, from a prior single-tenant deployment), `setup_secure()`
can backfill it into the secure schema under a fixed legacy tenant,
preserving node/edge IDs:

```python
await rag.setup_secure(
    admin_url=ADMIN_DSN,
    runtime_role="pgr_runtime",
    runtime_password=os.environ["PGR_RUNTIME_PASSWORD"],
    migrate_legacy_data=True,
)

from postgres_graph_rag.tenancy import LEGACY_TENANT_ID
legacy_engine = rag.for_tenant(LEGACY_TENANT_ID)
result = await legacy_engine.retrieve("What does checkout-service depend on?", namespace="legacy-ns")
```

Migrated nodes and edges become graph-only — there's no source document to
backfill a chunk/mention for, since the legacy schema never recorded
provenance. Migrated edges carry their old `weight` forward as
`manual_weight` (not left at its default), so they remain traversable
under the secure path's evidence-support filtering; this is covered by
`tests/test_tenancy.py::test_legacy_data_migration_preserves_ids`.

---

## Communities & Global Summarization

Weighted label propagation groups related entities into communities, entirely in Postgres (`postgres_graph_rag.communities`), then optionally summarizes each one:

```python
from postgres_graph_rag.communities import CommunityEngine

engine = CommunityEngine(rag._secure_store)  # or construct SecureGraphStore yourself

report = await engine.refresh_communities(tenant_id, namespace="research")
# {"skipped": False, "run_id": "...", "converged": True, "iterations": 3,
#  "node_count": 42, "community_count": 5, "duration_ms": 118}

communities = await engine.list_communities(tenant_id, namespace="research")
# [{"community_id": "...", "member_count": 7, "members": ["Apple", "M4", ...]}, ...]

summaries = await engine.summarize_communities(tenant_id, namespace="research", extractor=rag.extractor)
# One LLM call per community not already summarized for its exact current evidence.

evidence = await engine.query_global(tenant_id, namespace="research", question="What are the major themes?")
# Ranks existing summaries by lexical overlap with the question — evidence for
# YOU to synthesize an answer from; it does not call an LLM to produce one itself.
```

`refresh_communities()` is a no-op (`{"skipped": True}`) unless the namespace has changed since the last run (tracked automatically whenever `resolve_and_upsert_nodes`/`upsert_edges` write) — pass `force=True` to recompute anyway. This is meant to run as an explicit, externally-scheduled job (a cron, a management command), not on the request path.

**A real bug this caught, worth knowing about if you extend the algorithm:** the first implementation ran fully-synchronous label propagation (every node updates at once from the previous round's snapshot). It doesn't converge — two nodes joined by a single edge each only ever see the *other's* label as a candidate and swap every round, forever. Fixed by switching to asynchronous (Gauss-Seidel) sequential updates within each pass, which is the standard fix for this well-known pathology and provably avoids it, at the cost of one round trip per node per pass rather than one per pass. Caught by an actual failing test (`test_two_connected_nodes_merge_into_one_community`) before being trusted, not assumed correct from the algorithm description.

---

## Observability

Typed events fan out to whatever sinks you register — no hard dependency on a specific backend:

```python
from postgres_graph_rag.observability import EventBus, LoggingSink, UsageAggregator, make_otel_sink

usage = UsageAggregator()
sinks = [LoggingSink(), usage]
if otel_sink := make_otel_sink():  # None if `opentelemetry` isn't installed
    sinks.append(otel_sink)
bus = EventBus(sinks=sinks)

tenant = rag.for_tenant(tenant_id, event_bus=bus)
await tenant.add_document(...)
await tenant.retrieve(...)

usage.snapshot(str(tenant_id), "research")
# {"counts": {"ingestion.completed": 1, "extraction.completed": 3,
#             "cache.hit": 1, "retrieval.completed": 1, ...}, "tokens": 0}
```

Event kinds cover ingestion/chunking, cache claims/hits/in-progress, extraction success/failure, hybrid + traversal retrieval stages, and community jobs (see `observability.py` for the full list). A sink's own exception is caught and logged, never propagated — observability can't break the operation it's observing. **Privacy convention** (not a runtime-enforced filter): every built-in call site in this codebase puts only counts/durations/ids into `Event.attributes`, never raw document/chunk text, prompts, embeddings, or connection strings.

---

## MCP Server

Optional dependency: `pip install "postgres-graph-rag[mcp]"`. Exposes the tenant-aware engine as MCP tools:

```bash
# stdio (local IDE/agent use) — exactly one, statically-configured tenant per process
postgres-graph-rag-mcp \
  --runtime-url postgresql://pgr_runtime:...@... \
  --openai-api-key sk-... --tenant-id 11111111-1111-1111-1111-111111111111

# Streamable HTTP (remote) — refuses to start without a tenant_resolver unless you
# explicitly opt into --allow-unauthenticated-dev on a loopback bind
postgres-graph-rag-mcp --transport http --host 127.0.0.1 --port 8000 --allow-unauthenticated-dev
```

Read-only tools (`retrieve`, `get_entity`, `find_paths`, `list_communities`, `get_community_summary`, `capabilities`, `health`) are always registered. Mutation tools (`ingest_documents`, `delete_document`, `merge_entities`, `refresh_communities`, `summarize_communities`) require `--enable-mutations`.

For programmatic embedding (e.g. inside your own server) rather than the CLI:

```python
from postgres_graph_rag.mcp_server import build_server

server = build_server(rag, stdio_tenant_id=tenant_id, enable_mutations=True)
await server.run_stdio_async()
```

**HTTP tenant resolution is deployer-supplied, by design.** `build_server(..., tenant_resolver=my_fn)` takes a callable mapping an authenticated request `Context` to a `tenant_id` — this module does not implement an OAuth authorization server (token issuance, client registration, consent flows) itself; that's whatever auth your deployment already has in front of it. What it does enforce, and what's tested (`test_mcp_http_refuses_unauthenticated_public_bind`): HTTP mode refuses to start without either a `tenant_resolver` or an explicit `allow_unauthenticated_dev=True` on a loopback address, so an accidentally-public, unauthenticated multi-tenant endpoint isn't the default failure mode.

`split_entity()` (undoing a `merge_entities()` call) is intentionally not implemented or exposed as a tool — see the note on `merge_entities` in `tenancy.py`: reversing a merge needs mention-level provenance (which alias each mention originally came from) that this schema doesn't retain once a merge has happened.

---

## Structured Data Ingestion

Not everything worth putting in the graph is unstructured text. `add_record()` and `add_triplets()` create entities and relationships **deterministically** — no LLM call, so no cost and no hallucination risk for facts you already have structurally (e.g. a database row):

```python
# One structured record -> one entity. Also creates a hybrid-searchable
# chunk from a deterministic text rendering of the record, but the LLM
# extractor is never called.
await tenant.add_record(
    namespace="company",
    source_id="employee:123",
    entity_type="employee",
    record={"name": "Alice", "department": "Payments", "role": "Manager"},
)

# Deterministic relationships. Subjects/objects resolve through the same
# layered entity resolution as LLM-extracted triplets, so "Alice" here
# resolves to the exact same node add_record() just created — not a duplicate.
await tenant.add_triplets(
    [{"subject": "Alice", "predicate": "works_in", "object": "Payments",
      "metadata": {"source_id": "employee:123"}}],
    namespace="company",
)
```

Both are idempotent the same way `add_document()` is (`add_record` compares a content hash of the rendered record; re-calling with an unchanged record is a no-op). Use `add_document()` for prose that needs an LLM to find the facts in it; use `add_record()`/`add_triplets()` when you already know the facts and just need them in the graph.

**Failed ingestion is retryable without keeping the source text around.** A document's `content_hash` and its `document_chunks` are published together, atomically, in one short transaction — *before* extraction is attempted. A chunk whose extraction subsequently fails is still durably stored and still findable via hybrid search, just without graph entities yet. `retry_failed_chunks(namespace, source_id=None)` re-attempts extraction for exactly those chunks, reading their content back from the database rather than requiring the caller to still have the original document text:

```python
report = await tenant.add_document(text, namespace="research", source_id="doc-1")
# {"skipped": False, "chunks": 2, "triplets": 1, "extraction_status": "partial", "status_applied": True}
# ... if extraction failed for some chunks (provider outage, rate limit, etc.) ...
retry_report = await tenant.retry_failed_chunks(namespace="research")
# {"retried": 2, "triplets": 3}
```

This was a real gap, not a hypothetical one, fixed in two stages:
- Originally, a chunk wasn't written to `document_chunks` until *after* extraction succeeded, so a failed chunk had no row to retry against at all — the only way to recover was re-calling `add_document()` with the original text from scratch.
- Later, `content_hash` was committed (in its own transaction) *before* chunking/embedding ran at all: if either failed, the hash was already updated, so retrying with the identical text saw no content change and skipped permanently — again with no chunk rows to recover. `content_hash` and `document_chunks` are now published in the same transaction, guarded by a per-document advisory lock so concurrent publishers of the same `source_id` can't race each other into both reporting a fresh publish for identical content — a failure anywhere before that commit leaves the previous revision, hash and chunks together, exactly as it was.

**Graph extraction is a separate, eventually-consistent stage, not part of that transaction.** It involves an LLM call that can take seconds to minutes, so it deliberately runs *after* the atomic publish rather than inside it — holding a database transaction open that long would be worse than the problem it solves. Each document's `extraction_status` (`pending` / `ready` / `partial` / `failed`, visible in `add_document()`'s report and updated again after `retry_failed_chunks()`) is how a caller tells "text-searchable, graph still catching up" apart from "fully graph-ready" instead of the two looking identical from outside.

A document superseded by a concurrent update mid-extraction is handled at every layer this touches, not just the obvious one:
- Entity/edge wiring: a chunk whose extraction finishes after a newer revision has already replaced it is detected (its chunk row is simply gone), and its facts are never attached to the wrong revision.
- Graph edges: any edge that would otherwise be left with zero supporting evidence because of this is pruned rather than left silently traversable — and read-time traversal enforces the same "must have real support" filter independently, rather than depending entirely on that pruning having already run.
- Status itself: the status *write* is a compare-and-set against the exact document revision the work was actually done for (`set_extraction_status(..., expected_content_hash=...)`), not against "whatever the document's hash happens to be right now" — otherwise a slow worker finishing after a newer revision was published could overwrite that newer revision's status with a verdict about the old one's chunks. `add_document()`'s report reflects this honestly: `extraction_status` is what *this call's own* processing concluded, and `status_applied` (`False` when superseded) tells you whether that verdict actually got written to the document's current status — it does not mean the document's current status.

Retrieval results also carry the real `source_id` (`tenant.retrieve(...).chunks[0].source_id`) instead of `None` — citations like `[doc-42#0]` in `to_context_string()` now actually point somewhere.

**Metadata filtering** narrows both hybrid search and graph traversal to chunks/entities whose metadata is a JSONB superset of a filter dict:

```python
result = await tenant.retrieve(
    "What's the onboarding policy?",
    namespace="research",
    metadata_filter={"source": "handbook"},  # only chunks/entities tagged this way
)
```

`metadata_filter` also accepts a list of typed clauses for range/comparison filtering, not just containment — this is what makes "only documents updated in the last 90 days" possible, which plain `@>` containment structurally can't express:

```python
result = await tenant.retrieve(
    "What's the current policy?",
    namespace="research",
    metadata_filter=[{"field": "updated_at", "op": "gte", "value": "2026-05-16T00:00:00Z"}],
)
```

Supported ops: `eq`, `neq`, `gt`, `gte`, `lt`, `lte`, `in`. Numeric Python values compare numerically, ISO-8601-looking strings compare as timestamps, everything else falls back to text comparison; multiple clauses in the list are ANDed. Field names are always passed as bind parameters (never string-interpolated), so there's no SQL-injection surface regardless of what's passed as a field or value — see `postgres_graph_rag/filters.py`.

---

## Explainability: Reasoning Paths

Retrieval and traversal tell you *what's* related and *how strongly* — `explain_connection()` tells you *why*, as an actual reconstructed path between two named entities, not just a bag of scored neighbors:

```python
explained = await tenant.explain_connection("Johny Srouji", "ARM", namespace="research", max_hops=3)
print(str(explained))
# Johny Srouji --leads--> Hardware Team --designed--> M4 Chip --uses--> ARM

explained.steps    # [PathStep(node="Johny Srouji", relation=None), PathStep(node="Hardware Team", relation="leads"), ...]
explained.score    # combined edge-strength score of this specific path
explained.hop_distance
```

Returns `None` — an explicit, honest answer — if either entity can't be resolved or no path exists within `max_hops`; it does not fall back to an unrelated or partial result. This is a separate, purpose-built traversal (`SecureGraphStore.find_path`) from the general multi-seed `traverse_graph`/`retrieve` path: a broad multi-seed retrieval doesn't have one meaningful source/target pair to reconstruct a single path between, so bolting path-reconstruction onto it would mean picking an arbitrary path to report. Exposed as the `explain_connection` MCP tool too.

For multiple endpoints, use bounded top-N path search:

```python
paths = await tenant.find_paths(
    source_entities=["Alice", "Bob"],
    target_entities=["Orders", "Payments"],
    namespace="research",
    max_hops=3,
    top_k=5,
)
for path in paths:
    print(path["source_id"], "->", path["target_id"], path["score"])
    for step in path["path"][1:]:
        print(step["relation"], step["node"], step["evidence"])
```

`find_paths()` returns the globally best K complete routes across the supplied source/target sets. It retains the existing hard hop and neighbor limits, and every traversed edge includes the documents/chunks recorded in `edge_mentions`. This is intentionally not a general alternate-route enumerator or a one-best-path-per-target contract.

---

## Development

If you want to contribute or run the tests locally:

```bash
# Clone the repo and sync dependencies (dev extra pulls in mcp too)
uv sync --extra dev

# Bring up Postgres + pgvector
docker compose up -d

# Run tests — unit tests always run; the RLS/multi-tenancy/communities/
# observability/MCP tests in test_tenancy.py run automatically against
# POSTGRES_URL (no LLM key needed; needs a POSTGRES_URL with permission to
# CREATE ROLE); the MCP tests there are skipped if the `mcp` package isn't
# installed.
#
# Note: test_tenancy.py drops and recreates the `postgres_graph_rag` schema
# and `pgr_test_runtime`/`pgr_runtime` roles — safe against a fresh dev
# database, but don't point it at a database that already hosts real
# tenant data under that schema name (see the note at the top of
# tests/test_tenancy.py).
uv run pytest
```
