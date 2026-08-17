# PostgreSQL GraphRAG Architecture — A Practical Guide

This guide explains the project from a user’s point of view first, then adds the technical details needed by developers.

**Reading path:** Sections 1–7 explain the system. Stop there for a product-level understanding. Sections 8 onward are reference material for developers who need settings, diagnostics, and source files.

## 1. Start with one simple idea

The system turns documents into two searchable forms:

1. **Text evidence** — document chunks that can be searched by words and meaning.
2. **A relationship graph** — entities connected by facts extracted from those chunks.

When someone asks a question, PostgreSQL finds the best text, optionally follows related graph connections, and sends the resulting evidence to the configured AI provider. The provider writes an answer with citations.

```text
Documents
   ├── searchable text chunks
   └── entities and relationships
                 │
Question ──> PostgreSQL retrieval ──> evidence ──> cited answer
```

PostgreSQL is the only database. It provides relational storage, full-text search, vector search through pgvector, graph storage, recursive traversal, and tenant isolation.

## 2. A running example

Suppose we add two documents:

```text
Document 1: The Checkout API sends login requests to the Identity Gateway.
Document 2: The Platform Team maintains the Identity Gateway.
```

The system stores the original text and extracts relationships such as:

```text
Checkout API ──sends login requests to──> Identity Gateway
Platform Team ──maintains───────────────> Identity Gateway
```

Now ask:

```text
Which team maintains the service used by Checkout API login requests?
```

The system:

1. Finds both relevant text chunks.
2. Finds entities mentioned in those chunks.
3. Follows their graph relationships.
4. Gives the text and relationships to the answer provider.
5. Returns a cited answer:

```text
The Platform Team maintains the Identity Gateway, which handles the
Checkout API login requests [document-1#0][document-2#0].
```

`document-1#0` means “chunk 0—the first chunk—of document-1.” A longer document may produce `#0`, `#1`, `#2`, and so on.

## 3. Important terms

| Term | Simple meaning |
|---|---|
| Document | The complete text supplied by the user. |
| Chunk | A smaller searchable piece of a document. |
| Embedding | Numbers representing the meaning of text. Similar meanings produce nearby vectors. |
| Entity/node | A named thing in the graph, such as `Identity Gateway`. |
| Relationship/edge | A fact connecting two entities, such as `Platform Team → maintains → Identity Gateway`. |
| Seed | An entity where graph traversal begins. |
| Hop | One relationship step through the graph. |
| Evidence | Retrieved chunks and graph context used to answer the question. |
| Citation | A marker identifying the exact source chunk supporting an answer. |

## 4. What happens when a document is added?

The public operation is:

```python
await engine.add_document(
    text=document_text,
    namespace="my-project",
    source_id="design-doc-1",
)
```

### Step 1 — Validate it

The engine validates the text, namespace, and source ID. It hashes the document content.

- Same source ID and same text: skip the duplicate work.
- Same source ID and changed text: publish a new revision.

The new hash and chunks are committed together, so readers see a complete old revision or a complete new revision—not a half-written document.

### Step 2 — Split it into chunks

Each chunk stores:

```text
document ID
source ID
position in the document (ordinal)
content
metadata
embedding
```

The ordinal later becomes the number after `#` in a citation.

### Step 3 — Create embeddings

The provider converts every chunk into a vector. PostgreSQL stores it with pgvector. This allows meaning-based search even when the question uses different words from the document.

### Step 4 — Extract relationships

A live provider can turn:

```text
The Platform Team maintains the Identity Gateway.
```

into:

```json
{
  "subject": "Platform Team",
  "predicate": "maintains",
  "object": "Identity Gateway"
}
```

Extraction is cached and retried. Concurrent calls are limited so a large document cannot start unlimited paid requests.

Offline mode supports deterministic fixture patterns. It is useful for tests and demonstrations, but it is not general natural-language extraction or answer generation.

### Step 5 — Store graph evidence

Subjects and objects become graph nodes; predicates become edges. The system records which chunks mentioned each node and edge.

An edge’s effective weight is:

```text
weight = manual_weight + support_count
```

- `support_count` grows when evidence chunks support the edge.
- `manual_weight` supports relationships added directly by the caller.
- An edge needs evidence support or positive manual weight before traversal can use it.

Text becomes searchable as soon as chunks are published. Graph extraction runs afterward, so a document can be text-searchable before it is fully graph-ready. Its extraction status reports `ready`, `partial`, or `failed`.

## 5. How a question is retrieved

The public call is:

```python
result = await engine.retrieve(
    question,
    namespace,
    mode="hybrid_graph",
    top_k=5,
    hops=2,
)
```

The question is embedded once. The selected mode then determines the search strategy.

### Mode 1 — `vector`

Vector mode searches by meaning:

```sql
ORDER BY embedding <=> query_embedding ASC
```

Lower distance means more similar. This mode can match “operates the service” with “manages the gateway” even though the exact words differ.

Use it as the semantic-search baseline.

### Mode 2 — `hybrid`

Hybrid mode combines two independent searches:

1. **Lexical search:** PostgreSQL full-text search finds matching words and identifiers.
2. **Semantic search:** pgvector finds similar meaning.

The raw scores cannot be added directly because a PostgreSQL text score and a vector distance use different scales. The project combines their ranks using Reciprocal Rank Fusion (RRF).

```text
RRF contribution = 1 / (60 + rank)
```

If a chunk appears in both result lists:

```text
rrf_score = 1 / (60 + lexical_rank)
          + 1 / (60 + semantic_rank)
```

If it appears in only one list, the missing side contributes zero. PostgreSQL uses a `FULL OUTER JOIN`, so a chunk found by either search remains eligible.

Example:

| Chunk | Word-search rank | Meaning-search rank | RRF result |
|---|---:|---:|---:|
| A | 1 | 2 | `1/61 + 1/62 = 0.0325` |
| B | 3 | 1 | `1/63 + 1/61 = 0.0323` |
| C | 2 | absent | `1/62 = 0.0161` |

A and B rank highly because both methods found them.

The default candidate flow is:

```text
up to 40 lexical candidates
up to 40 semantic candidates
             │
             ▼
          RRF fusion
             │
             ▼
up to 10 fused rows when top_k=5
             │
             ▼
5 final chunks
```

The reason for each value is listed once in the configuration reference in Section 8. These are starting values, not proven universal optima.

### Mode 3 — `hybrid_graph`

This runs hybrid retrieval first, then expands the graph from entities mentioned in selected chunks.

It can return:

```text
chunks  direct textual evidence
nodes   related entities
edges   traversal-selected relationships
trace   timings, seed scores, context size, and truncation
```

The graph adds connected evidence; it does not replace the retrieved text.

### Where the graph is stored

There is no separate graph database. PostgreSQL stores the graph in ordinary
tables inside the tenant-aware schema:

| Table | What it stores | Why it matters |
|---|---|---|
| `documents` | One row per source document, including `source_id`, content hash, namespace, and extraction status. | Tracks the document revision and whether graph extraction finished. |
| `document_chunks` | The searchable text pieces, their ordinal (`0`, `1`, `2`…), metadata, and embedding. | Supplies text evidence and the vector/lexical search input. |
| `graph_nodes` | Entities such as `Checkout API`, `Identity Gateway`, or `Platform Access`. | Gives each named thing one graph ID. |
| `graph_edges` | A directed fact: source node, target node, relation, weight, and namespace. | Represents facts such as `Platform Access → maintains → Identity Gateway`. |
| `entity_mentions` | Which chunk mentioned which node. | Connects retrieved text to graph seed nodes. |
| `edge_mentions` | Which chunk asserted which edge, with optional confidence. | Preserves provenance for graph facts. |

Every tenant-owned row carries `tenant_id`; graph rows also carry
`namespace`. Foreign keys keep an edge from pointing to a missing node, and
row-level security prevents one tenant from reading another tenant's rows.
Indexes on tenant/namespace, edge endpoints, and node names keep the common
lookups bounded.

When the extractor finds this sentence:

```text
The Platform Access team maintains the Identity Gateway.
```

the storage relationship is conceptually:

```text
document_chunks  ──mentions──> entity_mentions ──> graph_nodes
document_chunks  ──asserts───> edge_mentions   ──> graph_edges
graph_nodes      ──source/target──> graph_edges
```

The edge weight is derived from evidence support and optional manual weight:

```text
edge weight = support_count + manual_weight
```

Unsupported zero-weight edges are excluded from traversal. This prevents a
partially written or unverified relationship from becoming graph evidence.

#### A small storage example

For the Identity Gateway example, the real database uses UUIDs. The shortened
IDs below make the relationships easier to see:

```text
documents
────────────────────────────────────────────────────────────
id       source_id      namespace        extraction_status
doc-2    studio-doc-2   playground-ab12  ready
doc-3    studio-doc-3   playground-ab12  ready
```

The source text is split into searchable chunks:

```text
document_chunks
────────────────────────────────────────────────────────────
id       document_id  source_id      ordinal  content
chunk-2  doc-2        studio-doc-2   0        "The Identity Gateway..."
chunk-3  doc-3        studio-doc-3   0        "Customers reported..."
```

Entities become nodes:

```text
graph_nodes
────────────────────────────────────────────
id      content
node-1  Checkout
node-2  Identity Gateway
node-3  Platform Access
node-4  Secrets Vault
node-5  Secondary region
node-6  SRE on-call engineer
```

Facts become directed edges:

```text
graph_edges
────────────────────────────────────────────────────────────
source  relation       target              weight  support_count
node-3  maintains      node-2              1.0     1
node-2  uses           node-4              1.0     1
node-6  redirected_to  node-5              1.0     1
```

Conceptually, the graph looks like this:

```text
Platform Access ──maintains────> Identity Gateway ──uses──> Secrets Vault
SRE engineer ──redirected_to───> Secondary region
```

The mention tables connect the text back to the graph:

```text
entity_mentions
──────────────────────────────
chunk_id  node_id
chunk-2   node-2       # Identity Gateway
chunk-2   node-3       # Platform Access
chunk-2   node-4       # Secrets Vault
chunk-3   node-2       # Identity Gateway
chunk-3   node-5       # Secondary region
chunk-3   node-6       # SRE on-call engineer

edge_mentions
──────────────────────────────
chunk_id  edge_id  confidence
chunk-2   edge-1   0.95
chunk-2   edge-2   0.91
chunk-3   edge-3   0.94
```

Therefore, a citation such as:

```text
[studio-doc-2#0]
```

means `source_id='studio-doc-2'` and `ordinal=0` in
`document_chunks`. It identifies the exact text chunk that supports the
answer. The actual database also includes `tenant_id`, UUID foreign keys,
metadata, timestamps, and row-level-security checks on these rows.

### Hybrid + Graph search: the SQL algorithm

`hybrid_graph` is a deterministic SQL pipeline with one model operation at the
start: the provider creates an embedding for the question. PostgreSQL then
does the ranking and graph work in bounded queries.

#### 1. Find candidate chunks in two ways

The SQL creates two ranked lists over active chunks in the current tenant and
namespace:

```sql
-- Word/identifier search
SELECT id,
       ts_rank_cd(tsv, websearch_to_tsquery('simple', :query)) AS lexical_score,
       row_number() OVER (ORDER BY ts_rank_cd(...) DESC) AS lexical_rank
FROM document_chunks
WHERE tsv @@ websearch_to_tsquery('simple', :query)
ORDER BY lexical_score DESC
LIMIT 40;
```

```sql
-- Meaning search; pgvector cosine distance is lower-is-better
SELECT id,
       embedding <=> :question_embedding AS distance,
       row_number() OVER (ORDER BY embedding <=> :question_embedding) AS semantic_rank
FROM document_chunks
ORDER BY distance ASC
LIMIT 40;
```

The real query also applies `tenant_id`, `namespace`, `status='active'`, and
metadata filters. The first list is lexical PostgreSQL full-text search; it
is not BM25. The second uses pgvector distance.

#### 2. Combine ranks with Reciprocal Rank Fusion

The raw lexical score and vector distance are not added because they have
different units. SQL joins the two lists by chunk ID with a `FULL OUTER JOIN`:

```sql
rrf_score = COALESCE(1.0 / (60 + lexical_rank), 0)
          + COALESCE(1.0 / (60 + semantic_rank), 0)
```

`60` is the RRF constant. It softens the effect of rank differences. A chunk
that appears in both lists receives two contributions; a chunk in only one
list remains eligible with one contribution. The fused rows are ordered by
`rrf_score` and the engine keeps the final `top_k` chunks (five by default).

#### 3. Turn retrieved chunks into graph seeds

For `hybrid_graph`, the engine looks up `entity_mentions` for the selected
chunks. These mentioned nodes are the traversal seeds. It does not start from
every node in the database.

```sql
SELECT DISTINCT node_id
FROM entity_mentions
WHERE tenant_id = :tenant_id
  AND chunk_id = ANY(:selected_chunk_ids);
```

The seed score comes from the supporting chunk's fused retrieval score. The
strongest seed is normalized to `1.0`; other seed scores are proportional to
it. Playground and Studio select up to three chunks as seeds, while the core
library default is one.

#### 4. Walk the graph with a recursive CTE

PostgreSQL's `WITH RECURSIVE` is the graph-traversal algorithm. The first
query term inserts the seeds at depth `0`. The recursive term joins each
current node to eligible edges and inserts its neighbors at the next depth.
The example below shows the forward direction; the production query uses a
small lateral expression to choose the reverse endpoint too when
`directed=false`:

```sql
WITH RECURSIVE graph_expansion AS (
    -- Starting points
    SELECT n.id,
           0 AS depth,
           ARRAY[n.id] AS visited,
           :seed_score AS score
    FROM graph_nodes AS n
    WHERE n.id = ANY(:seed_ids)
      AND n.tenant_id = :tenant_id
      AND n.namespace = :namespace

    UNION ALL

    -- One more relationship step
    SELECT neighbor.id,
           current.depth + 1,
           current.visited || neighbor.id,
           current.score
             * :score_decay
             * (1 - exp(-edge.weight))
    FROM graph_expansion AS current
    JOIN graph_edges AS edge
      ON edge.source_node_id = current.id
    JOIN graph_nodes AS neighbor
      ON neighbor.id = edge.target_node_id
    WHERE current.depth < :max_hops
      AND NOT (neighbor.id = ANY(current.visited))
      AND edge.weight >= :min_weight
      AND (edge.support_count > 0 OR edge.manual_weight > 0)
)
SELECT id,
       MIN(depth) AS hop_distance,
       MAX(score) AS score
FROM graph_expansion
GROUP BY id;
```

The production SQL additionally applies relation allow/deny lists, metadata
filters, tenant checks, direction rules, and the per-node neighbor limit.
When `directed=false`, it can follow an edge from either endpoint; when
`directed=true`, it follows only the stored source-to-target direction.

The `visited` array prevents cycles such as `A → B → C → A`. The hard hop
limit is five, even if a caller sends a larger value. The score formula makes
longer and weaker paths less relevant:

```text
next score = current score × 0.7 × (1 - e^(-edge weight))
```

This is bounded recursive graph search, not an LLM agent loop and not PageRank.
It does not repeatedly ask a model which node to visit.

#### 5. Return graph context safely

After traversal, SQL fetches supported edges whose endpoints are among the
reached nodes. The engine then applies deterministic limits:

```text
maximum graph nodes in context  = 25
maximum graph edges in context  = 50
maximum evidence budget         = 4,000 estimated tokens
```

The final answer prompt contains the selected text chunks plus the bounded
graph nodes and edges. A graph relationship shown as “Traversal-selected
relationships” means it was included by this seeded traversal. It does not
claim that every edge asserted by every retrieved chunk was displayed; direct
evidence provenance is retained separately through `edge_mentions`.

In short:

```text
question embedding
   → lexical SQL + vector SQL
   → RRF rank fusion
   → top text chunks
   → mentioned entity seeds
   → recursive CTE graph walk
   → bounded nodes/edges/context
   → provider answer with chunk citations
```

## 6. Graph traversal in plain language

Suppose the seed is `Checkout API`:

```text
Checkout API ──> Identity Gateway ──> Key Vault ──> Security Team
    hop 0              hop 1             hop 2           hop 3
```

| `hops` | Traversal result |
|---:|---|
| 0 | Seed entities only; no neighbor expansion. |
| 1 | Direct neighbors. |
| 2 | Neighbors of neighbors. |
| 3 | One more relationship level. |

`hops` is graph distance, not the number of nodes returned. A single node can have many neighbors.

The engine accepts `0–5`, and the PostgreSQL-facing store independently enforces the hard maximum of 5. Traversal tracks visited nodes so cycles such as `A → B → C → A` do not repeat forever.

### Where graph seeds come from

Graph traversal needs entity IDs, not text chunks. The engine:

1. Selects the configured number of retrieved chunks.
2. Finds entities mentioned by those chunks.
3. Gives each entity the best score of its supporting chunks.
4. Normalizes the strongest seed to `1.0`.
5. Starts traversal from those entities.

Core library and MCP default to one seed chunk. Playground and Studio use:

```python
graph_seed_chunks = min(3, top_k)
```

This improves the chance of finding facts across several documents while keeping traversal bounded. It is not a guarantee that every relationship from every retrieved chunk will appear.

## 7. How graph scores work

The seed begins with a normalized relevance score, usually `1.0`. Each graph step reduces the score:

```text
child_score = parent_score
            × score_decay
            × (1 - exp(-edge_weight))
```

With `score_decay=0.7` and `edge_weight=1.0`:

```text
Seed entity               1.0000
First relationship away  0.4425
Second relationship away 0.1958
```

Think of relevance like light fading as it travels away from the original evidence.

- Lower decay, such as `0.3`: favor nearby facts strongly.
- Default `0.7`: balanced starting point.
- Higher decay, near `1.0`: preserve longer chains but allow more noise.

`score_decay` never deletes an edge. It only changes ordering. Direction, relation filters, minimum weight, hop limits, neighbor limits, and context limits decide whether an item remains eligible.

Returned edges receive a separate display-order score:

```text
edge_score = average(source_node_score, target_node_score)
           × (1 - exp(-edge_weight))
```

Node score controls traversal relevance; edge score orders the relationships shown afterward.

## 8. Developer reference: why every setting exists

The document cleanup removed repeated explanations, not runtime attributes. Every current retrieval setting is listed below. When a caller omits one, the engine uses the default from `postgres_graph_rag/models.py`.

### Retrieval and graph settings

| Setting | Current default | Why we need it | What happens without this control? | Why this value/default? |
|---|---:|---|---|---|
| `mode` | `hybrid_graph` | Chooses meaning search, word-plus-meaning search, or search plus graph. | The engine would not know which search process to run. | `hybrid_graph` enables every retrieval feature. Callers can choose a simpler mode. |
| `top_k` | `5` | Limits final text evidence. | Returning everything would create noisy, slow, expensive prompts; keeping too little could miss facts. | Five is a small practical evidence set. Valid range is 1–100; this is a starting default, not a universal optimum. |
| `hops` | `2` | Limits relationship steps from each starting entity. | Traversal could spread through much of the graph or stop before finding a two-step answer. | Two supports common connected questions without travelling too far. Both engine and store cap it at 5. |
| `directed` | `False` | Decides whether `A → B` may also be explored from B toward A. | A fixed behavior would either miss reverse-neighborhood questions or incorrectly reverse directional meaning. | `False` improves the chance of finding nearby entities. Use `True` when edge direction must be strict. |
| `relation_types` | `None` | Allows only selected relationship names. | A caller could not focus traversal on relationships such as only `depends_on`. | `None` allows every valid relationship unless the caller asks for a smaller set. |
| `exclude_relation_types` | `None` | Blocks selected relationship names. | Known noisy relationships could not be hidden for one query without deleting them. | `None` keeps all relationships unless the caller explicitly excludes one. |
| `min_weight` | `0.0` | Removes edges below a chosen evidence/manual-weight threshold. | A caller could not request only stronger relationships. | `0.0` keeps every supported or manually weighted edge; a separate support check still rejects unsupported zero-value edges. |
| `score_decay` | `0.7` | Makes distant entities less important after each step. | Far-away facts could rank like facts directly connected to the evidence. | `0.7` keeps useful chains while reducing unrelated results. Test it against the real dataset before changing it. |
| `graph_seed_chunks` | `1` | Limits how many retrieved chunks provide starting entities. | Expanding every chunk could create too many unrelated starting points. | Core/MCP start conservatively with one. Playground/Studio use up to three for multi-document questions. |
| `max_neighbors_per_node` | `20` | Limits relationships explored from one entity. | A highly connected entity could expand hundreds of neighbors and make traversal unpredictable. | Twenty is a practical safety limit that still allows a useful local graph. |
| `max_context_nodes` | `25` | Caps graph entities sent to answer generation. | A large graph could overwhelm the useful text evidence. | Twenty-five is a practical prompt-size starting limit, not a quality guarantee. |
| `max_context_edges` | `50` | Caps relationships sent to answer generation. | The prompt could contain a large relationship dump. | Fifty allows more relationships than nodes because nodes can connect through several useful edges. |
| `max_context_tokens` | `4000` | Sets the total approximate evidence budget. | Prompt size, provider cost, and latency could grow without a predictable bound. | Four thousand leaves room for instructions and the generated answer in typical model contexts. Tune it for the chosen provider. |
| `metadata_filter` | `None` | Restricts eligible chunks and nodes by project, environment, date, or other metadata. | Retrieval would always search the full tenant namespace. | `None` avoids silently hiding data. Callers add filters only when their data model requires them. |

### Internal hybrid-search settings

These values live in `SecureGraphStore.hybrid_search()`. Normal callers do not currently pass them through `TenantGraphRAG.retrieve()`.

| Setting | Current default | Why we need it | What happens without this control? | Why this value/default? |
|---|---:|---|---|---|
| `lexical_candidates` | `40` | Limits the word-search pool before combining results. | An unlimited list would add unnecessary database work. | Forty looks wider than the final five without searching the entire dataset. |
| `semantic_candidates` | `40` | Limits the meaning-search pool before combining results. | An unlimited list would cost more; a very small list could miss useful matches. | Forty gives word and meaning search the same opportunity. |
| `rrf_k` | `60` | Controls how much rank differences affect the combined score. | Raw word-search scores and vector distances cannot safely be added because they use different scales. | Sixty is the usual RRF starting value. It prevents small rank differences from having too much influence. |
| `top_chunks` | `10` in the store method | Limits rows returned after fusion. | The fused list could be larger than the engine needs. | The engine passes `top_k × 2`; with default `top_k=5`, this is 10, giving a small buffer before the final five. |

### Ingestion settings

These defaults also come from `postgres_graph_rag/models.py`.

| Setting | Current default | Why we need it | What happens without this control? | Why this value/default? |
|---|---:|---|---|---|
| `max_concurrent_extractions` | `5` | Limits simultaneous provider extraction calls. | A large document could create a burst of paid calls, rate-limit errors, and memory pressure. | Five provides useful parallelism without making concurrency aggressive. |
| `max_extraction_retries` | `3` | Retries temporary extraction failures. | A short provider/network failure would permanently fail that chunk; unlimited retries could hang ingestion and increase cost. | Three gives two recovery opportunities after the first attempt while keeping work bounded. |
| `retry_base_delay` | `1.0` second | Provides exponential waiting between retries. | Immediate retries could repeatedly hit the same rate limit or outage. | One second produces a short `1s`, then `2s` backoff before later attempts. |
| `skip_duplicate_chunks` | `True` | Intended to control duplicate-chunk work. | **Current reality:** the engine does not read this setting, so changing or omitting it has no runtime effect. Document hashes and extraction caching already handle other forms of duplicate work. | This appears to be obsolete configuration and should be verified for removal in a dedicated cleanup change. |
| `fuzzy_entity_resolution` | `True` | Lets small name variations resolve to one entity. | Variations could create duplicate nodes; overly broad matching could combine different entities. | Enabled because name variation is common. Numbered identifiers are protected from this matching. |
| `fuzzy_trgm_threshold` | `0.4` | Requires names to look similar before considering a merge. | Every existing entity could become a costly and unsafe candidate. | `0.4` is only the first filter; meaning similarity must also pass before merging. |
| `fuzzy_embedding_threshold` | `0.90` | Requires names to have very similar meaning before merging. | A low threshold could merge different entities and create false graph paths. | `0.90` is deliberately strict and works together with the name-similarity check. |

The context token estimate is deterministic and dependency-free:

```python
estimated_tokens = max(1, (len(text) + 3) // 4)
```

When an item cannot fit, `trace.context_truncated` becomes `True`.

## 9. What the retrieval trace tells you

Trace values describe a request; they do not change its behavior and therefore have no configuration defaults.

| Field | What it tells us | What would be harder without it? |
|---|---|---|
| `mode` | Retrieval mode used. | We could misread a vector result as a graph-expanded result. |
| `embedding_ms` | Time spent embedding the question. | Provider latency and PostgreSQL latency would be mixed together. |
| `search_ms` | Time spent retrieving text chunks. | Search/index regressions would be harder to isolate. |
| `traversal_ms` | Time spent expanding the graph; zero for non-graph modes. | We could not measure the extra cost of graph retrieval. |
| `search_candidates` | Search rows available before final chunk selection. | Missed results caused by a small candidate set would be less visible. |
| `seed_scores` | Normalized relevance assigned to seed entities. | We could not explain which entities started traversal or why. |
| `context_tokens` | Estimated evidence tokens retained. | Prompt size and cost would be harder to understand. |
| `context_truncated` | Whether evidence was skipped because of the context budget. | The system could appear to have sent all discovered evidence when it did not. |

These fields help separate provider latency, PostgreSQL search latency, graph cost, and context-budget problems.

## 10. How answers and citations work

`engine.answer()` performs this sequence:

```text
retrieve evidence
   → stop if there is no usable evidence
   → build a prompt from chunks, nodes, and edges
   → ask the provider to answer
   → validate citation markers
   → return the answer, citations, evidence, status, and usage
```

Retrieved chunks are presented as:

```text
[studio-doc-1#0] The Checkout API sends login requests...
[studio-doc-2#0] The Platform Team maintains...
```

The model is instructed to put markers after factual sentences. A citation is accepted only when it points to a retrieved chunk.

The normal Studio display may show:

```text
Answer: sentence-level citation markers
Citations: unique list of every cited source
Evidence: the retrieved passages
Relationships: graph context selected by traversal
```

`citation_valid_only` proves that citation markers resolve to retrieved evidence. It does not prove that every claim is logically entailed by the citation. The optional `verified` and `verified_strict` modes perform stronger claim-level checks and may abstain more often.

Common result states:

| Status/reason | Meaning |
|---|---|
| `citation_valid_only` | Markers point to retrieved chunks; entailment was not checked. |
| `verified` / `partially_verified` | Claim-level verification found support. |
| `no_evidence_retrieved` | Search found no usable chunks. |
| `missing_query_anchor` | A named identifier from the question was missing from evidence. |
| `empty_model_response` | The provider returned no visible answer. |
| `invalid_or_missing_citation` | The answer did not contain valid markers. |
| `model_abstained` | The provider chose not to answer. |
| `verification_failed` | Strong verification could not safely approve the answer. |

## 11. Tenant isolation and database permissions

Every runtime operation carries:

```text
tenant_id + namespace
```

- `tenant_id` isolates customers or Studio sessions.
- `namespace` separates datasets inside a tenant.
- PostgreSQL row-level security and tenant-scoped SQL protect documents, chunks, nodes, and edges.

Two DSNs have different jobs:

```text
POSTGRES_URL      admin DSN used only for setup and migrations
PGR_RUNTIME_URL   restricted DSN used for ingestion and retrieval
```

Studio, Playground, normal library calls, and MCP should use only the runtime DSN. Studio intentionally exposes no setup or reset operation.

## 12. Providers and vector dimensions

The provider supplies:

1. Relationship extraction.
2. Embeddings.
3. Optional answer generation.

| Provider | Project dimension | General answer generation |
|---|---:|---|
| Offline | 1536 | No; deterministic fixture behavior only. |
| OpenAI | 1536 | Yes. |
| Gemini | 3072 | Yes. |
| LiteLLM Gateway | Configured by `LITELLM_EMBEDDING_DIMENSION` | Yes, through an OpenAI-compatible gateway. |

One schema has one fixed vector dimension. A Studio process therefore uses one provider. Matching dimensions do not make two providers or embedding models compatible; they can still use unrelated embedding spaces. LiteLLM also needs a gateway URL, virtual key, chat alias, embedding alias, and the embedding alias's exact output dimension.

## 13. CLI, Browser Studio, and MCP

All interfaces use the same tenant engine:

```text
CLI Playground ─────┐
Browser Studio ─────┼──> playground_service.py ──> TenantGraphRAG
Library / MCP ──────┘
```

- **CLI Playground:** good for scripting, one-shot tests, JSON output, and optional best-effort cleanup.
- **Browser Studio:** good for pasting several documents, comparing retrieval modes, and presenting evidence visually.
- **MCP server:** exposes retrieval and related graph operations as tools.

Studio uses a random tenant and namespace per browser session, keeps a bounded expiring session registry, and binds to loopback by default. Its provider is fixed when the process starts.

“Compare modes” performs retrieval without answer-generation calls. Generating an answer for all three modes can make three paid provider calls.

## 14. Which parts are SQL algorithms?

The system uses several deterministic algorithms inside PostgreSQL:

| Algorithm | PostgreSQL mechanism | Purpose |
|---|---|---|
| Lexical ranking | `ts_rank_cd` with `tsvector` | Find matching words and identifiers. |
| Semantic ranking | pgvector `<=>` distance | Find similar meaning. |
| Rank fusion | SQL RRF calculation | Combine lexical and semantic ranks. |
| Graph traversal | `WITH RECURSIVE` CTE | Follow supported relationships up to the hop limit. |
| Cycle protection | Visited-node array | Prevent a traversal path from repeating nodes. |
| Tenant filtering | RLS plus tenant/namespace predicates | Prevent cross-tenant access. |

AI is used for language understanding: extracting relationships from ordinary prose and composing an answer. Search fusion, hop limits, graph score propagation, filtering, and citation-marker validation are deterministic.

## 15. Source map

| File | What to inspect there |
|---|---|
| [`postgres_graph_rag/models.py`](postgres_graph_rag/models.py) | Provider settings and retrieval defaults. |
| [`postgres_graph_rag/core.py`](postgres_graph_rag/core.py) | Main object creation and secure schema setup. |
| [`postgres_graph_rag/tenant_engine.py`](postgres_graph_rag/tenant_engine.py) | Ingestion, retrieval, context construction, and answering. |
| [`postgres_graph_rag/tenancy.py`](postgres_graph_rag/tenancy.py) | PostgreSQL queries, RLS-aware access, hybrid search, and recursive traversal. |
| [`postgres_graph_rag/extractor.py`](postgres_graph_rag/extractor.py) | OpenAI, Gemini, and OpenAI-compatible LiteLLM provider calls and structured extraction. |
| [`postgres_graph_rag/playground_service.py`](postgres_graph_rag/playground_service.py) | Shared CLI/Studio workflow. |
| [`postgres_graph_rag/demo.py`](postgres_graph_rag/demo.py) | CLI commands. |
| [`postgres_graph_rag/studio_app.py`](postgres_graph_rag/studio_app.py) | Browser routes and session handling. |
| [`postgres_graph_rag/grounding.py`](postgres_graph_rag/grounding.py) | Grounding types and statuses. |
| [`postgres_graph_rag/verification.py`](postgres_graph_rag/verification.py) | Citation and claim verification policies. |
