# Evaluation

## Hypothesis

Hybrid graph retrieval should materially improve questions that require facts
from two or more incident documents while preserving acceptable latency and
not increasing false connections on negative questions.

## Versioned benchmark

`incident-benchmark-v1` contains 60 documents and 120 labeled questions:

| Category | Count | Purpose |
|---|---:|---|
| Semantic single-hop | 30 | Embedding baseline |
| Identifier/lexical | 30 | PostgreSQL FTS contribution |
| Multi-hop | 40 | Graph contribution |
| Negative | 20 | False-connection pressure |

The cases are generated from deterministic, versioned templates in
`postgres_graph_rag.evaluation`. Scoring uses expected entities, relationships,
and source documents; it does not count a substring anywhere in an unstructured
context as a complete answer.

## Run

```bash
postgres-graph-rag-demo --admin-url "$POSTGRES_URL" setup
postgres-graph-rag-eval --ingest --output evaluation.json
```

The report compares `vector`, `hybrid`, and `hybrid_graph` using Recall@5,
Precision@5, MRR, path accuracy, source recall, negative false-connection rate,
context tokens, and p50/p95/p99 latency.

The latest tracked run is recorded in [CHANGELOG.md](../CHANGELOG.md).

## Acceptance gates

- Multi-hop Recall@5: hybrid+graph ≥ vector + 20 percentage points.
- Overall Recall@5 regression: no more than 5 points.
- Negative false-connection rate: ≤10%.
- Warm p95 at 10K entities: <300 ms on the declared reference environment.
- All citations resolve to retrieved chunks.

If a gate fails, publish the failure. Do not tune the dataset after inspecting
results; create a new dataset version with a written rationale.

## Real-provider canary

The deterministic benchmark is suitable for CI but does not validate LLM
extraction. A manually triggered canary should label 20–30 real, anonymized
incident documents and report triplet precision/recall, unsupported edges,
entity duplication, tokens, cost, and latency. Keep the key outside the repo and
cap spend at USD 10.
